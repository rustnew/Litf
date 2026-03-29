/// Benchmark & stress tests: known model FLOPs, memory footprints, scaling, edge cases
use lift_core::context::Context;
use lift_core::types::*;
use lift_core::attributes::*;
use lift_core::location::Location;
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

fn mk(shape: Vec<usize>, dtype: DataType) -> TensorTypeInfo {
    TensorTypeInfo {
        shape: shape.into_iter().map(Dimension::Constant).collect(),
        dtype,
        layout: MemoryLayout::Contiguous,
    }
}

// ═══════════════════════════════════════════════════
//  RESNET-50
// ═══════════════════════════════════════════════════

#[test]
fn test_resnet50_bottleneck_flops() {
    use lift_tensor::ops::TensorOp;
    use lift_tensor::shape::ShapeInference;

    let x = mk(vec![1, 256, 56, 56], DataType::FP32);
    let w1 = mk(vec![64, 256, 1, 1], DataType::FP32);
    let f1 = ShapeInference::compute_flops(&TensorOp::Conv2D, &[&x, &w1]).unwrap();
    assert_eq!(f1, 2 * 1 * 64 * 56 * 56 * 256 * 1 * 1);

    let h = mk(vec![1, 64, 56, 56], DataType::FP32);
    let w2 = mk(vec![64, 64, 3, 3], DataType::FP32);
    let f2 = ShapeInference::compute_flops(&TensorOp::Conv2D, &[&h, &w2]).unwrap();
    let out_h = 56 - 3 + 1;
    assert_eq!(f2, 2 * 1 * 64 * out_h * out_h * 64 * 3 * 3);

    let total = f1 + f2;
    assert!(total > 100_000_000);
}

#[test]
fn test_resnet50_total_flops_subset() {
    use lift_tensor::ops::TensorOp;
    use lift_tensor::shape::ShapeInference;

    let layers: Vec<(Vec<usize>, Vec<usize>)> = vec![
        (vec![1, 3, 224, 224], vec![64, 3, 7, 7]),
        (vec![1, 64, 56, 56], vec![64, 64, 1, 1]),
        (vec![1, 64, 56, 56], vec![64, 64, 3, 3]),
        (vec![1, 128, 28, 28], vec![128, 128, 3, 3]),
        (vec![1, 256, 14, 14], vec![256, 256, 3, 3]),
        (vec![1, 512, 7, 7], vec![512, 512, 3, 3]),
    ];

    let mut total = 0u64;
    for (inp, ker) in &layers {
        let i = mk(inp.clone(), DataType::FP32);
        let k = mk(ker.clone(), DataType::FP32);
        if let Some(f) = ShapeInference::compute_flops(&TensorOp::Conv2D, &[&i, &k]) {
            total += f;
        }
    }
    assert!(total > 500_000_000, "ResNet-50 subset > 500M: {}", total);
}

// ═══════════════════════════════════════════════════
//  GPT-2
// ═══════════════════════════════════════════════════

#[test]
fn test_gpt2_small_single_layer() {
    use lift_tensor::ops::TensorOp;
    use lift_tensor::shape::ShapeInference;

    let seq = 1024usize;
    let hidden = 768usize;
    let head_dim = 64usize;
    let heads = 12u64;
    let ffn = 4 * hidden;

    let x = mk(vec![seq, hidden], DataType::FP32);
    let wqkv = mk(vec![hidden, hidden], DataType::FP32);
    let qkv_flops = ShapeInference::compute_flops(&TensorOp::MatMul, &[&x, &wqkv]).unwrap();
    let total_qkv = 3 * qkv_flops;

    let q = mk(vec![seq, head_dim], DataType::FP32);
    let kt = mk(vec![head_dim, seq], DataType::FP32);
    let qk_flops = ShapeInference::compute_flops(&TensorOp::MatMul, &[&q, &kt]).unwrap();
    let total_qk = heads * qk_flops;

    let attn = mk(vec![seq, seq], DataType::FP32);
    let v = mk(vec![seq, head_dim], DataType::FP32);
    let av_flops = ShapeInference::compute_flops(&TensorOp::MatMul, &[&attn, &v]).unwrap();
    let total_av = heads * av_flops;

    let wffn1 = mk(vec![hidden, ffn], DataType::FP32);
    let ffn1 = ShapeInference::compute_flops(&TensorOp::MatMul, &[&x, &wffn1]).unwrap();
    let xffn = mk(vec![seq, ffn], DataType::FP32);
    let wffn2 = mk(vec![ffn, hidden], DataType::FP32);
    let ffn2 = ShapeInference::compute_flops(&TensorOp::MatMul, &[&xffn, &wffn2]).unwrap();

    let layer = total_qkv + total_qk + total_av + qkv_flops + ffn1 + ffn2;
    assert!(layer > 1_500_000_000, "GPT-2 layer > 1.5G: {}", layer);

    let total_12 = 12 * layer;
    assert!(total_12 > 15_000_000_000, "GPT-2 full > 15G: {}", total_12);
}

// ═══════════════════════════════════════════════════
//  LLaMA-7B MEMORY
// ═══════════════════════════════════════════════════

#[test]
fn test_llama7b_memory_estimate() {
    let hidden: u64 = 4096;
    let ffn: u64 = 11008;
    let layers: u64 = 32;

    let qkvo = 4 * hidden * hidden * 2; // FP16
    let ffn_b = 3 * hidden * ffn * 2;
    let ln = 2 * hidden * 2;
    let per_layer = qkvo + ffn_b + ln;
    let total = layers * per_layer;
    let gb = total as f64 / 1e9;
    assert!(gb > 10.0 && gb < 20.0, "LLaMA-7B FP16: {:.2} GB", gb);

    let a100 = lift_sim::cost::CostModel::a100();
    assert!(a100.fits_in_memory(total as u64));
}

// ═══════════════════════════════════════════════════
//  PRECISION TRADEOFF
// ═══════════════════════════════════════════════════

#[test]
fn test_precision_fp32_vs_fp16() {
    use lift_tensor::ops::TensorOp;
    use lift_tensor::shape::ShapeInference;

    let a32 = mk(vec![512, 768], DataType::FP32);
    let b32 = mk(vec![768, 768], DataType::FP32);
    let a16 = mk(vec![512, 768], DataType::FP16);
    let b16 = mk(vec![768, 768], DataType::FP16);

    let f32_ = ShapeInference::compute_flops(&TensorOp::MatMul, &[&a32, &b32]).unwrap();
    let f16_ = ShapeInference::compute_flops(&TensorOp::MatMul, &[&a16, &b16]).unwrap();
    assert_eq!(f32_, f16_, "FLOP count is precision-independent");

    let m32 = ShapeInference::compute_memory_bytes(&TensorOp::MatMul, &[&a32, &b32]).unwrap();
    let m16 = ShapeInference::compute_memory_bytes(&TensorOp::MatMul, &[&a16, &b16]).unwrap();
    assert_eq!(m32, 2 * m16, "FP32 = 2x FP16 memory");

    let a100 = lift_sim::cost::CostModel::a100();
    let t32 = a100.memory_time_ms(m32);
    let t16 = a100.memory_time_ms(m16);
    assert!((t32 / t16 - 2.0).abs() < 0.01);
}

#[test]
fn test_arithmetic_intensity_comparison() {
    let a100 = lift_sim::cost::CostModel::a100();

    let elem_ai = a100.arithmetic_intensity(1024 * 1024, 1024 * 1024 * 8);
    assert!(elem_ai < 1.0, "elementwise AI < 1: {}", elem_ai);
    assert!(!a100.is_compute_bound(1024 * 1024, 1024 * 1024 * 8));

    let mm_flops = 2 * 4096 * 4096 * 4096u64;
    let mm_bytes = (4096 * 4096 * 3 * 4) as u64;
    let mm_ai = a100.arithmetic_intensity(mm_flops, mm_bytes);
    assert!(mm_ai > 100.0, "matmul AI > 100: {}", mm_ai);
    assert!(a100.is_compute_bound(mm_flops, mm_bytes));
}

// ═══════════════════════════════════════════════════
//  STRESS TESTS
// ═══════════════════════════════════════════════════

#[test]
fn test_stress_1000_ops() {
    let mut ctx = Context::new();
    let ty = ctx.make_tensor_type(vec![Dimension::Constant(64)], DataType::FP32, MemoryLayout::Contiguous);
    let block = ctx.create_block();
    let mut val = ctx.create_block_arg(block, ty);
    for _ in 0..1000 {
        let (op, res) = ctx.create_op("tensor.relu", "tensor", vec![val], vec![ty], Attributes::new(), Location::unknown());
        ctx.add_op_to_block(block, op);
        val = res[0];
    }
    assert_eq!(ctx.ops.len(), 1000);
    assert!(lift_core::verifier::verify(&ctx).is_ok());
    let report = lift_sim::analyze_module(&ctx);
    assert_eq!(report.num_ops, 1000);
    assert_eq!(report.num_tensor_ops, 1000);
}

#[test]
fn test_stress_deep_quantum() {
    let mut ctx = Context::new();
    let q_ty = ctx.make_qubit_type();
    let block = ctx.create_block();
    let mut q = ctx.create_block_arg(block, q_ty);
    for i in 0..500 {
        let gate = if i % 2 == 0 { "quantum.h" } else { "quantum.t" };
        let (op, res) = ctx.create_op(gate, "quantum", vec![q], vec![q_ty], Attributes::new(), Location::unknown());
        ctx.add_op_to_block(block, op);
        q = res[0];
    }
    let qa = lift_sim::analyze_quantum_ops(&ctx);
    assert_eq!(qa.gate_count, 500);
    assert!(qa.estimated_fidelity > 0.5 && qa.estimated_fidelity < 0.7,
        "500-gate fidelity: {}", qa.estimated_fidelity);
}

#[test]
fn test_stress_string_interning_10k() {
    let mut ctx = Context::new();
    let mut ids = Vec::with_capacity(10000);
    for i in 0..10000 { ids.push(ctx.intern_string(&format!("str_{}", i))); }
    for i in 0..10000 { assert_eq!(ctx.resolve_string(ids[i]), format!("str_{}", i)); }
    for i in 0..10000 { assert_eq!(ctx.intern_string(&format!("str_{}", i)), ids[i]); }
}

// ═══════════════════════════════════════════════════
//  EDGE CASES
// ═══════════════════════════════════════════════════

#[test]
fn test_edge_1d_tensor() {
    let a = mk(vec![1], DataType::FP32);
    let b = mk(vec![1], DataType::FP32);
    let out = lift_tensor::shape::ShapeInference::infer_output_shape(
        &lift_tensor::ops::TensorOp::Add, &[&a, &b],
    ).unwrap();
    assert_eq!(out[0].shape[0].static_value(), Some(1));
}

#[test]
fn test_edge_empty_module() {
    let ctx = parse_and_build("module @empty {}");
    assert!(lift_core::verifier::verify(&ctx).is_ok());
    let report = lift_sim::analyze_module(&ctx);
    assert_eq!(report.num_ops, 0);
    assert_eq!(report.total_flops, 0);
}

#[test]
fn test_edge_quantum_no_gates() {
    let ctx = Context::new();
    let qa = lift_sim::analyze_quantum_ops(&ctx);
    assert_eq!(qa.gate_count, 0);
    assert_eq!(qa.estimated_fidelity, 1.0);
}

#[test]
fn test_edge_zero_flop_reshape() {
    use lift_tensor::ops::TensorOp;
    use lift_tensor::shape::ShapeInference;
    let a = mk(vec![4, 8], DataType::FP32);
    let flops = ShapeInference::compute_flops(&TensorOp::Reshape, &[&a]);
    assert!(flops.is_none() || flops == Some(0));
}
