/// Comprehensive tests for lift-opt: DCE, constant folding, tensor fusion, gate cancel, canonicalize
use lift_core::*;
use lift_core::context::Context;
use lift_core::types::*;
use lift_core::attributes::*;
use lift_core::location::Location;

// ═══════════════════════════════════════════════════
//  DEAD CODE ELIMINATION
// ═══════════════════════════════════════════════════

#[test]
fn test_dce_removes_unused() {
    let mut ctx = Context::new();
    let ty = ctx.make_tensor_type(
        vec![Dimension::Constant(4)], DataType::FP32, MemoryLayout::Contiguous,
    );
    let block = ctx.create_block();
    let x = ctx.create_block_arg(block, ty);

    let (op1, _) = ctx.create_op("tensor.relu", "tensor", vec![x], vec![ty], Attributes::new(), Location::unknown());
    ctx.add_op_to_block(block, op1);
    let (op2, _) = ctx.create_op("tensor.neg", "tensor", vec![x], vec![ty], Attributes::new(), Location::unknown());
    ctx.add_op_to_block(block, op2);

    let initial = ctx.ops.len();
    let mut cache = AnalysisCache::new();
    let result = lift_opt::DeadCodeElimination.run(&mut ctx, &mut cache);
    assert_eq!(result, PassResult::Changed);
    assert!(ctx.ops.len() < initial);
}

#[test]
fn test_dce_empty_context() {
    let mut ctx = Context::new();
    let mut cache = AnalysisCache::new();
    assert_eq!(lift_opt::DeadCodeElimination.run(&mut ctx, &mut cache), PassResult::Unchanged);
}

// ═══════════════════════════════════════════════════
//  CONSTANT FOLDING
// ═══════════════════════════════════════════════════

#[test]
fn test_constant_fold_add_int() {
    let mut ctx = Context::new();
    let ty = ctx.make_integer_type(64, true);
    let block = ctx.create_block();

    let mut a1 = Attributes::new(); a1.set("value", Attribute::Integer(10));
    let (c1, r1) = ctx.create_op("core.constant", "core", vec![], vec![ty], a1, Location::unknown());
    ctx.add_op_to_block(block, c1);

    let mut a2 = Attributes::new(); a2.set("value", Attribute::Integer(20));
    let (c2, r2) = ctx.create_op("core.constant", "core", vec![], vec![ty], a2, Location::unknown());
    ctx.add_op_to_block(block, c2);

    let (add_op, _) = ctx.create_op("tensor.add", "tensor", vec![r1[0], r2[0]], vec![ty], Attributes::new(), Location::unknown());
    ctx.add_op_to_block(block, add_op);

    let mut cache = AnalysisCache::new();
    assert_eq!(lift_opt::ConstantFolding.run(&mut ctx, &mut cache), PassResult::Changed);
    assert_eq!(ctx.get_op(add_op).unwrap().attrs.get_integer("value"), Some(30));
}

#[test]
fn test_constant_fold_mul_int() {
    let mut ctx = Context::new();
    let ty = ctx.make_integer_type(64, true);
    let block = ctx.create_block();

    let mut a1 = Attributes::new(); a1.set("value", Attribute::Integer(6));
    let (c1, r1) = ctx.create_op("core.constant", "core", vec![], vec![ty], a1, Location::unknown());
    ctx.add_op_to_block(block, c1);

    let mut a2 = Attributes::new(); a2.set("value", Attribute::Integer(7));
    let (c2, r2) = ctx.create_op("core.constant", "core", vec![], vec![ty], a2, Location::unknown());
    ctx.add_op_to_block(block, c2);

    let (mul_op, _) = ctx.create_op("tensor.mul", "tensor", vec![r1[0], r2[0]], vec![ty], Attributes::new(), Location::unknown());
    ctx.add_op_to_block(block, mul_op);

    let mut cache = AnalysisCache::new();
    lift_opt::ConstantFolding.run(&mut ctx, &mut cache);
    assert_eq!(ctx.get_op(mul_op).unwrap().attrs.get_integer("value"), Some(42));
}

#[test]
fn test_constant_fold_float() {
    let mut ctx = Context::new();
    let ty = ctx.make_float_type(64);
    let block = ctx.create_block();

    let mut a1 = Attributes::new(); a1.set("value", Attribute::Float(2.5));
    let (c1, r1) = ctx.create_op("core.constant", "core", vec![], vec![ty], a1, Location::unknown());
    ctx.add_op_to_block(block, c1);

    let mut a2 = Attributes::new(); a2.set("value", Attribute::Float(3.5));
    let (c2, r2) = ctx.create_op("core.constant", "core", vec![], vec![ty], a2, Location::unknown());
    ctx.add_op_to_block(block, c2);

    let (add_op, _) = ctx.create_op("tensor.add", "tensor", vec![r1[0], r2[0]], vec![ty], Attributes::new(), Location::unknown());
    ctx.add_op_to_block(block, add_op);

    let mut cache = AnalysisCache::new();
    lift_opt::ConstantFolding.run(&mut ctx, &mut cache);
    let val = ctx.get_op(add_op).unwrap().attrs.get_float("value").unwrap();
    assert!((val - 6.0).abs() < 1e-10);
}

#[test]
fn test_constant_fold_no_constants() {
    let mut ctx = Context::new();
    let ty = ctx.make_tensor_type(vec![Dimension::Constant(4)], DataType::FP32, MemoryLayout::Contiguous);
    let block = ctx.create_block();
    let x = ctx.create_block_arg(block, ty);
    let y = ctx.create_block_arg(block, ty);
    let (add_op, _) = ctx.create_op("tensor.add", "tensor", vec![x, y], vec![ty], Attributes::new(), Location::unknown());
    ctx.add_op_to_block(block, add_op);
    let mut cache = AnalysisCache::new();
    assert_eq!(lift_opt::ConstantFolding.run(&mut ctx, &mut cache), PassResult::Unchanged);
}

// ═══════════════════════════════════════════════════
//  TENSOR FUSION
// ═══════════════════════════════════════════════════

#[test]
fn test_tensor_fusion_matmul_bias_relu() {
    let mut ctx = Context::new();
    let ty = ctx.make_tensor_type(vec![Dimension::Constant(1), Dimension::Constant(256)], DataType::FP32, MemoryLayout::Contiguous);
    let wty = ctx.make_tensor_type(vec![Dimension::Constant(784), Dimension::Constant(256)], DataType::FP32, MemoryLayout::Contiguous);
    let bty = ctx.make_tensor_type(vec![Dimension::Constant(256)], DataType::FP32, MemoryLayout::Contiguous);
    let xty = ctx.make_tensor_type(vec![Dimension::Constant(1), Dimension::Constant(784)], DataType::FP32, MemoryLayout::Contiguous);

    let block = ctx.create_block();
    let x = ctx.create_block_arg(block, xty);
    let w = ctx.create_block_arg(block, wty);
    let b = ctx.create_block_arg(block, bty);

    let (mm, mm_r) = ctx.create_op("tensor.matmul", "tensor", vec![x, w], vec![ty], Attributes::new(), Location::unknown());
    ctx.add_op_to_block(block, mm);
    let (add, add_r) = ctx.create_op("tensor.add", "tensor", vec![mm_r[0], b], vec![ty], Attributes::new(), Location::unknown());
    ctx.add_op_to_block(block, add);
    let (relu, _) = ctx.create_op("tensor.relu", "tensor", vec![add_r[0]], vec![ty], Attributes::new(), Location::unknown());
    ctx.add_op_to_block(block, relu);

    let mut cache = AnalysisCache::new();
    assert_eq!(lift_opt::TensorFusion.run(&mut ctx, &mut cache), PassResult::Changed);
    let fused = ctx.get_op(relu).unwrap();
    assert_eq!(ctx.strings.resolve(fused.name), "tensor.fused_matmul_bias_relu");
    assert_eq!(fused.inputs.len(), 3);
}

#[test]
fn test_tensor_fusion_no_pattern() {
    let mut ctx = Context::new();
    let ty = ctx.make_tensor_type(vec![Dimension::Constant(4)], DataType::FP32, MemoryLayout::Contiguous);
    let block = ctx.create_block();
    let x = ctx.create_block_arg(block, ty);
    let (relu, _) = ctx.create_op("tensor.relu", "tensor", vec![x], vec![ty], Attributes::new(), Location::unknown());
    ctx.add_op_to_block(block, relu);
    let mut cache = AnalysisCache::new();
    assert_eq!(lift_opt::TensorFusion.run(&mut ctx, &mut cache), PassResult::Unchanged);
}

// ═══════════════════════════════════════════════════
//  GATE CANCELLATION
// ═══════════════════════════════════════════════════

#[test]
fn test_gate_cancel_h_h() {
    let mut ctx = Context::new();
    let q_ty = ctx.make_qubit_type();
    let block = ctx.create_block();
    let q = ctx.create_block_arg(block, q_ty);
    let (h1, h1r) = ctx.create_op("quantum.h", "quantum", vec![q], vec![q_ty], Attributes::new(), Location::unknown());
    ctx.add_op_to_block(block, h1);
    let (h2, _) = ctx.create_op("quantum.h", "quantum", vec![h1r[0]], vec![q_ty], Attributes::new(), Location::unknown());
    ctx.add_op_to_block(block, h2);

    let before = ctx.ops.len();
    let mut cache = AnalysisCache::new();
    assert_eq!(lift_opt::GateCancellation.run(&mut ctx, &mut cache), PassResult::Changed);
    assert!(ctx.ops.len() < before);
}

#[test]
fn test_gate_cancel_x_x() {
    let mut ctx = Context::new();
    let q_ty = ctx.make_qubit_type();
    let block = ctx.create_block();
    let q = ctx.create_block_arg(block, q_ty);
    let (x1, x1r) = ctx.create_op("quantum.x", "quantum", vec![q], vec![q_ty], Attributes::new(), Location::unknown());
    ctx.add_op_to_block(block, x1);
    let (x2, _) = ctx.create_op("quantum.x", "quantum", vec![x1r[0]], vec![q_ty], Attributes::new(), Location::unknown());
    ctx.add_op_to_block(block, x2);
    let mut cache = AnalysisCache::new();
    assert_eq!(lift_opt::GateCancellation.run(&mut ctx, &mut cache), PassResult::Changed);
}

#[test]
fn test_gate_cancel_s_sdg() {
    let mut ctx = Context::new();
    let q_ty = ctx.make_qubit_type();
    let block = ctx.create_block();
    let q = ctx.create_block_arg(block, q_ty);
    let (s, sr) = ctx.create_op("quantum.s", "quantum", vec![q], vec![q_ty], Attributes::new(), Location::unknown());
    ctx.add_op_to_block(block, s);
    let (sdg, _) = ctx.create_op("quantum.sdg", "quantum", vec![sr[0]], vec![q_ty], Attributes::new(), Location::unknown());
    ctx.add_op_to_block(block, sdg);
    let mut cache = AnalysisCache::new();
    assert_eq!(lift_opt::GateCancellation.run(&mut ctx, &mut cache), PassResult::Changed);
}

#[test]
fn test_gate_cancel_no_cancel() {
    let mut ctx = Context::new();
    let q_ty = ctx.make_qubit_type();
    let block = ctx.create_block();
    let q = ctx.create_block_arg(block, q_ty);
    let (h, hr) = ctx.create_op("quantum.h", "quantum", vec![q], vec![q_ty], Attributes::new(), Location::unknown());
    ctx.add_op_to_block(block, h);
    let (x, _) = ctx.create_op("quantum.x", "quantum", vec![hr[0]], vec![q_ty], Attributes::new(), Location::unknown());
    ctx.add_op_to_block(block, x);
    let mut cache = AnalysisCache::new();
    assert_eq!(lift_opt::GateCancellation.run(&mut ctx, &mut cache), PassResult::Unchanged);
}

// ═══════════════════════════════════════════════════
//  CANONICALIZE
// ═══════════════════════════════════════════════════

#[test]
fn test_canonicalize_add_zero() {
    let mut ctx = Context::new();
    let ty = ctx.make_tensor_type(vec![Dimension::Constant(4)], DataType::FP32, MemoryLayout::Contiguous);
    let block = ctx.create_block();
    let x = ctx.create_block_arg(block, ty);
    let mut za = Attributes::new(); za.set("value", Attribute::Integer(0));
    let (z, zr) = ctx.create_op("core.constant", "core", vec![], vec![ty], za, Location::unknown());
    ctx.add_op_to_block(block, z);
    let (add, _) = ctx.create_op("tensor.add", "tensor", vec![x, zr[0]], vec![ty], Attributes::new(), Location::unknown());
    ctx.add_op_to_block(block, add);
    let mut cache = AnalysisCache::new();
    assert_eq!(lift_opt::Canonicalize.run(&mut ctx, &mut cache), PassResult::Changed);
}

#[test]
fn test_canonicalize_mul_one() {
    let mut ctx = Context::new();
    let ty = ctx.make_tensor_type(vec![Dimension::Constant(4)], DataType::FP32, MemoryLayout::Contiguous);
    let block = ctx.create_block();
    let x = ctx.create_block_arg(block, ty);
    let mut oa = Attributes::new(); oa.set("value", Attribute::Integer(1));
    let (o, or_) = ctx.create_op("core.constant", "core", vec![], vec![ty], oa, Location::unknown());
    ctx.add_op_to_block(block, o);
    let (mul, _) = ctx.create_op("tensor.mul", "tensor", vec![x, or_[0]], vec![ty], Attributes::new(), Location::unknown());
    ctx.add_op_to_block(block, mul);
    let mut cache = AnalysisCache::new();
    assert_eq!(lift_opt::Canonicalize.run(&mut ctx, &mut cache), PassResult::Changed);
}

// ═══════════════════════════════════════════════════
//  FULL PIPELINE
// ═══════════════════════════════════════════════════

#[test]
fn test_full_optimization_pipeline_empty() {
    let mut pm = PassManager::new();
    pm.add_pass(Box::new(lift_opt::Canonicalize));
    pm.add_pass(Box::new(lift_opt::ConstantFolding));
    pm.add_pass(Box::new(lift_opt::TensorFusion));
    pm.add_pass(Box::new(lift_opt::GateCancellation));
    pm.add_pass(Box::new(lift_opt::DeadCodeElimination));

    let mut ctx = Context::new();
    let results = pm.run_all(&mut ctx);
    assert_eq!(results.len(), 5);
    for (_, r) in &results { assert_eq!(*r, PassResult::Unchanged); }
}
