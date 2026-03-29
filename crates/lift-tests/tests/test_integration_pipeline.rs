/// Integration tests: full pipeline parse → verify → analyse → optimise → predict → export
use lift_core::context::Context;
use lift_ast::*;

fn parse_and_build(src: &str) -> Context {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().to_vec();
    assert!(lexer.errors().is_empty(), "lexer errors: {:?}", lexer.errors());
    let mut parser = Parser::new(tokens);
    let program = parser.parse().expect("parse failed");
    let mut ctx = Context::new();
    let mut builder = IrBuilder::new();
    builder.build_program(&mut ctx, &program).expect("build failed");
    ctx
}

// ═══════════════════════════════════════════════════
//  END-TO-END: MLP
// ═══════════════════════════════════════════════════

#[test]
fn test_pipeline_mlp_full() {
    let src = r#"
#dialect tensor
module @mlp {
    func @forward(%x: tensor<1x784xf32>, %w1: tensor<784x256xf32>, %b1: tensor<256xf32>, %w2: tensor<256x10xf32>, %b2: tensor<10xf32>) -> tensor<1x10xf32> {
        %h1 = "tensor.matmul"(%x, %w1) : (tensor<1x784xf32>, tensor<784x256xf32>) -> tensor<1x256xf32>
        %h2 = "tensor.add"(%h1, %b1) : (tensor<1x256xf32>, tensor<256xf32>) -> tensor<1x256xf32>
        %h3 = "tensor.relu"(%h2) : (tensor<1x256xf32>) -> tensor<1x256xf32>
        %h4 = "tensor.matmul"(%h3, %w2) : (tensor<1x256xf32>, tensor<256x10xf32>) -> tensor<1x10xf32>
        %h5 = "tensor.add"(%h4, %b2) : (tensor<1x10xf32>, tensor<10xf32>) -> tensor<1x10xf32>
        %out = "tensor.softmax"(%h5) : (tensor<1x10xf32>) -> tensor<1x10xf32>
        return %out
    }
}
"#;
    let mut ctx = parse_and_build(src);

    // 1. Verify
    assert!(lift_core::verifier::verify(&ctx).is_ok());

    // 2. Analyse
    let report = lift_sim::analyze_module(&ctx);
    assert_eq!(report.num_ops, 7);
    assert_eq!(report.num_tensor_ops, 6);
    assert_eq!(report.num_quantum_ops, 0);
    assert!(report.total_flops > 0, "MLP must have nonzero FLOPs: {}", report.total_flops);
    assert!(report.total_memory_bytes > 0);
    // matmul1: 2*1*784*256 = 401408, matmul2: 2*1*256*10 = 5120
    assert!(report.total_flops >= 401408 + 5120);

    // 3. Predict
    let cost = lift_sim::cost::CostModel::a100();
    let prediction = lift_predict::predict_performance(&report, &cost);
    assert!(prediction.predicted_time_ms > 0.0);
    assert!(prediction.arithmetic_intensity > 0.0);

    // 4. Optimise
    let mut pm = lift_core::PassManager::new();
    pm.add_pass(Box::new(lift_opt::Canonicalize));
    pm.add_pass(Box::new(lift_opt::ConstantFolding));
    pm.add_pass(Box::new(lift_opt::TensorFusion));
    pm.add_pass(Box::new(lift_opt::DeadCodeElimination));
    let results = pm.run_all(&mut ctx);
    let fusion_result = results.iter().find(|(n, _)| n == "tensor-fusion").unwrap();
    assert_eq!(fusion_result.1, lift_core::PassResult::Changed);

    // 5. Export LLVM
    let llvm = lift_export::LlvmExporter::new().export(&ctx).unwrap();
    assert!(llvm.contains("@forward"));

    // 6. Print
    let printed = lift_core::printer::print_ir(&ctx);
    assert!(printed.contains("module @mlp"));
    assert!(printed.contains("tensor.fused_matmul_bias_relu"));
}

// ═══════════════════════════════════════════════════
//  END-TO-END: QUANTUM BELL STATE
// ═══════════════════════════════════════════════════

#[test]
fn test_pipeline_quantum_bell() {
    let src = r#"
#dialect quantum
module @bell {
    func @create(%q0: qubit, %q1: qubit) -> (qubit, qubit) {
        %q2 = "quantum.h"(%q0) : (qubit) -> qubit
        %q3, %q4 = "quantum.cx"(%q2, %q1) : (qubit, qubit) -> (qubit, qubit)
        return %q3, %q4
    }
}
"#;
    let ctx = parse_and_build(src);
    assert!(lift_core::verifier::verify(&ctx).is_ok());

    let qa = lift_sim::analyze_quantum_ops(&ctx);
    assert_eq!(qa.gate_count, 2);
    assert_eq!(qa.one_qubit_gates, 1);
    assert_eq!(qa.two_qubit_gates, 1);
    assert_eq!(qa.num_qubits_used, 2);
    assert!(qa.estimated_fidelity > 0.98);

    let qcost = lift_sim::cost::QuantumCostModel::superconducting_default();
    let qpred = lift_predict::predict_quantum(&qa, &qcost, 0.01);
    assert!(qpred.estimated_fidelity > 0.95);
    assert!(qpred.circuit_time_us > 0.0);
    assert!(qpred.num_shots_for_precision > 0);

    let qasm = lift_export::QasmExporter::new().export(&ctx).unwrap();
    assert!(qasm.contains("OPENQASM 3.0;"));
    assert!(qasm.contains("qubit[2] q;"));
    assert!(qasm.contains("h q["));
    assert!(qasm.contains("cx q["));
}

// ═══════════════════════════════════════════════════
//  END-TO-END: GHZ STATE
// ═══════════════════════════════════════════════════

#[test]
fn test_pipeline_quantum_ghz() {
    let src = r#"
#dialect quantum
module @ghz {
    func @create(%q0: qubit, %q1: qubit, %q2: qubit, %q3: qubit) -> (qubit, qubit, qubit, qubit) {
        %a = "quantum.h"(%q0) : (qubit) -> qubit
        %b, %c = "quantum.cx"(%a, %q1) : (qubit, qubit) -> (qubit, qubit)
        %d, %e = "quantum.cx"(%c, %q2) : (qubit, qubit) -> (qubit, qubit)
        %f, %g = "quantum.cx"(%e, %q3) : (qubit, qubit) -> (qubit, qubit)
        return %b, %d, %f, %g
    }
}
"#;
    let ctx = parse_and_build(src);
    assert!(lift_core::verifier::verify(&ctx).is_ok());

    let qa = lift_sim::analyze_quantum_ops(&ctx);
    assert_eq!(qa.gate_count, 4);
    assert_eq!(qa.one_qubit_gates, 1);
    assert_eq!(qa.two_qubit_gates, 3);
    assert_eq!(qa.num_qubits_used, 4);
    assert!(qa.estimated_fidelity > 0.96 && qa.estimated_fidelity < 0.98,
        "4-qubit GHZ fidelity: {}", qa.estimated_fidelity);
}

// ═══════════════════════════════════════════════════
//  END-TO-END: GATE CANCELLATION
// ═══════════════════════════════════════════════════

#[test]
fn test_pipeline_gate_cancellation() {
    let src = r#"
#dialect quantum
module @cancel {
    func @hh(%q: qubit) -> qubit {
        %a = "quantum.h"(%q) : (qubit) -> qubit
        %b = "quantum.h"(%a) : (qubit) -> qubit
        return %b
    }
}
"#;
    let mut ctx = parse_and_build(src);
    assert!(lift_core::verifier::verify(&ctx).is_ok());

    let qa_before = lift_sim::analyze_quantum_ops(&ctx);
    assert_eq!(qa_before.gate_count, 2);

    let mut pm = lift_core::PassManager::new();
    pm.add_pass(Box::new(lift_opt::GateCancellation));
    let results = pm.run_all(&mut ctx);
    assert_eq!(results[0].1, lift_core::PassResult::Changed);

    let qa_after = lift_sim::analyze_quantum_ops(&ctx);
    assert!(qa_after.gate_count < qa_before.gate_count);
}

// ═══════════════════════════════════════════════════
//  END-TO-END: ATTENTION
// ═══════════════════════════════════════════════════

#[test]
fn test_pipeline_attention() {
    let src = r#"
#dialect tensor
module @transformer {
    func @self_attention(%q: tensor<1x128x64xf32>, %k: tensor<1x128x64xf32>, %v: tensor<1x128x64xf32>, %w: tensor<64xf32>) -> tensor<1x128x64xf32> {
        %attn = "tensor.attention"(%q, %k, %v) : (tensor<1x128x64xf32>, tensor<1x128x64xf32>, tensor<1x128x64xf32>) -> tensor<1x128x64xf32>
        %out = "tensor.layernorm"(%attn, %w) : (tensor<1x128x64xf32>, tensor<64xf32>) -> tensor<1x128x64xf32>
        return %out
    }
}
"#;
    let ctx = parse_and_build(src);
    assert!(lift_core::verifier::verify(&ctx).is_ok());

    let report = lift_sim::analyze_module(&ctx);
    assert_eq!(report.num_tensor_ops, 2);
    assert!(report.total_flops > 0);
    assert!(report.total_memory_bytes > 0);

    let cost = lift_sim::cost::CostModel::a100();
    let pred = lift_predict::predict_performance(&report, &cost);
    assert!(pred.predicted_time_ms > 0.0);
}

// ═══════════════════════════════════════════════════
//  COST MODEL COMPARISON A100 vs H100
// ═══════════════════════════════════════════════════

#[test]
fn test_cost_model_a100_vs_h100() {
    let a100 = lift_sim::cost::CostModel::a100();
    let h100 = lift_sim::cost::CostModel::h100();

    let flops = 1_000_000_000_000u64;
    let bytes = 10_000_000_000u64;

    let a100_time = a100.roofline_time_ms(flops, bytes);
    let h100_time = h100.roofline_time_ms(flops, bytes);
    assert!(h100_time < a100_time);

    let speedup = a100.compute_time_ms(flops) / h100.compute_time_ms(flops);
    assert!(speedup > 2.5 && speedup < 4.0, "H100 speedup: {:.2}x", speedup);
}

#[test]
fn test_cost_model_memory_bound_detection() {
    let a100 = lift_sim::cost::CostModel::a100();
    assert!(!a100.is_compute_bound(1_000, 1_000_000_000));
    assert!(a100.is_compute_bound(1_000_000_000_000, 1_000));
}

#[test]
fn test_cost_model_gpu_memory_fit() {
    let a100 = lift_sim::cost::CostModel::a100();
    assert!(a100.fits_in_memory(40_000_000_000));
    assert!(!a100.fits_in_memory(100_000_000_000));
    assert_eq!(a100.num_gpus_needed(80_000_000_000), 1);
    assert_eq!(a100.num_gpus_needed(160_000_000_000), 2);
    assert_eq!(a100.num_gpus_needed(320_000_000_000), 4);
}

// ═══════════════════════════════════════════════════
//  QUANTUM COST MODEL
// ═══════════════════════════════════════════════════

#[test]
fn test_quantum_cost_superconducting() {
    let qcm = lift_sim::cost::QuantumCostModel::superconducting_default();
    let f1 = qcm.circuit_fidelity(1, 0);
    assert!((f1 - 0.999).abs() < 1e-6);
    let f_big = qcm.circuit_fidelity(100, 50);
    let expected = 0.999f64.powi(100) * 0.99f64.powi(50);
    assert!((f_big - expected).abs() < 1e-10);
}

#[test]
fn test_quantum_decoherence() {
    let qcm = lift_sim::cost::QuantumCostModel::superconducting_default();
    let f_short = qcm.decoherence_fidelity(0.1);
    assert!(f_short > 0.99);
    let f_long = qcm.decoherence_fidelity(50.0);
    assert!(f_long < f_short);
}

// ═══════════════════════════════════════════════════
//  BUDGET CHECKING
// ═══════════════════════════════════════════════════

#[test]
fn test_budget_checks() {
    let budget = lift_sim::cost::Budget {
        max_flops: Some(1_000_000),
        max_memory_bytes: Some(80_000_000_000),
        max_time_ms: None,
        min_fidelity: Some(0.95),
        max_circuit_depth: None,
    };
    assert!(budget.check_flops(500_000).is_ok());
    assert!(budget.check_flops(2_000_000).is_err());
    assert!(budget.check_memory(40_000_000_000).is_ok());
    assert!(budget.check_memory(100_000_000_000).is_err());
    assert!(budget.check_fidelity(0.99).is_ok());
    assert!(budget.check_fidelity(0.90).is_err());
}

// ═══════════════════════════════════════════════════
//  CONFIG PARSING
// ═══════════════════════════════════════════════════

#[test]
fn test_config_full_parse() {
    let src = r#"
[target]
backend = "cuda"
device = "H100"
precision = "fp16"

[budget]
max_flops = 1000000000000
max_memory_bytes = 80000000000
max_time_ms = 100.0
min_fidelity = 0.95

[optimisation]
level = O3
max_iterations = 20

[simulation]
shape_propagation = true
flop_counting = true
memory_analysis = true
noise_simulation = true

[quantum]
topology = "grid"
num_qubits = 127
shots = 8192
"#;
    let config = lift_config::ConfigParser::new().parse(src).unwrap();
    assert_eq!(config.target.backend, "cuda");
    assert_eq!(config.target.device, Some("H100".into()));
    assert_eq!(config.optimisation.level, lift_config::OptLevel::O3);
    assert_eq!(config.optimisation.max_iterations, 20);
    assert!(config.quantum.is_some());
    assert_eq!(config.quantum.as_ref().unwrap().num_qubits, 127);
    assert_eq!(config.quantum.as_ref().unwrap().shots, Some(8192));
}

#[test]
fn test_config_default() {
    let config = lift_config::LithConfig::default();
    assert_eq!(config.target.backend, "llvm");
    assert_eq!(config.optimisation.level, lift_config::OptLevel::O2);
    assert!(config.quantum.is_none());
}
