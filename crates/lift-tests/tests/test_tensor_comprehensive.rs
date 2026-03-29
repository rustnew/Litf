/// Comprehensive tests for lift-tensor: all ops, shape inference, FLOP accuracy, memory
use lift_core::types::*;
use lift_tensor::ops::TensorOp;
use lift_tensor::shape::ShapeInference;

fn mk(shape: Vec<usize>, dtype: DataType) -> TensorTypeInfo {
    TensorTypeInfo {
        shape: shape.into_iter().map(Dimension::Constant).collect(),
        dtype,
        layout: MemoryLayout::Contiguous,
    }
}

// ═══════════════════════════════════════════════════
//  OP NAME ROUNDTRIP
// ═══════════════════════════════════════════════════

#[test]
fn test_every_op_name_roundtrip() {
    let ops = vec![
        TensorOp::MatMul, TensorOp::Conv2D, TensorOp::Add, TensorOp::Sub,
        TensorOp::Mul, TensorOp::Div, TensorOp::Neg, TensorOp::Linear,
        TensorOp::Embedding, TensorOp::ReLU, TensorOp::GeLU, TensorOp::SiLU,
        TensorOp::Sigmoid, TensorOp::Tanh, TensorOp::Softmax,
        TensorOp::LayerNorm, TensorOp::RMSNorm, TensorOp::BatchNorm,
        TensorOp::Reshape, TensorOp::Transpose, TensorOp::Concat, TensorOp::Split,
        TensorOp::Gather, TensorOp::Scatter,
        TensorOp::Constant, TensorOp::Zeros, TensorOp::Ones,
        TensorOp::Attention, TensorOp::PagedAttention,
        TensorOp::MoEDispatch, TensorOp::MoECombine,
        TensorOp::Quantize, TensorOp::Dequantize,
        TensorOp::Checkpoint, TensorOp::Offload, TensorOp::GradAccumulate,
        TensorOp::GradMatMul, TensorOp::GradReLU, TensorOp::GradSoftmax,
        TensorOp::GradLayerNorm, TensorOp::GradAttention,
        TensorOp::ParallelSplit, TensorOp::ParallelAllReduce,
        TensorOp::PipelineSend, TensorOp::PipelineReceive,
        TensorOp::FusedMatMulBiasReLU, TensorOp::FusedMatMulBias, TensorOp::FusedLinearGeLU,
    ];
    for op in &ops {
        let name = op.name();
        let recovered = TensorOp::from_name(name);
        assert!(recovered.is_some(), "op '{}' must roundtrip", name);
    }
}

#[test]
fn test_every_op_has_tensor_prefix() {
    let ops = vec![
        TensorOp::MatMul, TensorOp::Conv2D, TensorOp::Add, TensorOp::ReLU,
        TensorOp::Softmax, TensorOp::Attention, TensorOp::LayerNorm,
    ];
    for op in &ops {
        assert!(op.name().starts_with("tensor."), "{} missing tensor. prefix", op.name());
    }
}

// ═══════════════════════════════════════════════════
//  SHAPE INFERENCE
// ═══════════════════════════════════════════════════

#[test]
fn test_matmul_2d() {
    let a = mk(vec![4, 3], DataType::FP32);
    let b = mk(vec![3, 5], DataType::FP32);
    let out = ShapeInference::infer_output_shape(&TensorOp::MatMul, &[&a, &b]).unwrap();
    assert_eq!(out[0].shape[0].static_value(), Some(4));
    assert_eq!(out[0].shape[1].static_value(), Some(5));
}

#[test]
fn test_matmul_3d_batch() {
    let a = mk(vec![8, 4, 3], DataType::FP32);
    let b = mk(vec![8, 3, 5], DataType::FP32);
    let out = ShapeInference::infer_output_shape(&TensorOp::MatMul, &[&a, &b]).unwrap();
    assert_eq!(out[0].shape.len(), 3);
    assert_eq!(out[0].shape[0].static_value(), Some(8));
    assert_eq!(out[0].shape[1].static_value(), Some(4));
    assert_eq!(out[0].shape[2].static_value(), Some(5));
}

#[test]
fn test_matmul_dimension_mismatch() {
    let a = mk(vec![4, 3], DataType::FP32);
    let b = mk(vec![5, 6], DataType::FP32);
    assert!(ShapeInference::infer_output_shape(&TensorOp::MatMul, &[&a, &b]).is_err());
}

#[test]
fn test_elementwise_ops_shapes() {
    let a = mk(vec![2, 3], DataType::FP32);
    let b = mk(vec![2, 3], DataType::FP32);
    for op in &[TensorOp::Add, TensorOp::Sub, TensorOp::Mul, TensorOp::Div] {
        let out = ShapeInference::infer_output_shape(op, &[&a, &b]).unwrap();
        assert_eq!(out[0].shape[0].static_value(), Some(2));
        assert_eq!(out[0].shape[1].static_value(), Some(3));
    }
}

#[test]
fn test_unary_ops_preserve_shape() {
    let a = mk(vec![4, 8, 16], DataType::FP32);
    for op in &[TensorOp::ReLU, TensorOp::GeLU, TensorOp::Sigmoid, TensorOp::Tanh,
                TensorOp::Neg] {
        let out = ShapeInference::infer_output_shape(op, &[&a]).unwrap();
        assert_eq!(out[0].shape.len(), 3, "{:?} must preserve rank", op);
        assert_eq!(out[0].shape[0].static_value(), Some(4));
        assert_eq!(out[0].shape[1].static_value(), Some(8));
        assert_eq!(out[0].shape[2].static_value(), Some(16));
    }
}

#[test]
fn test_conv2d_shape() {
    let input = mk(vec![1, 3, 28, 28], DataType::FP32);
    let kernel = mk(vec![16, 3, 5, 5], DataType::FP32);
    let out = ShapeInference::infer_output_shape(&TensorOp::Conv2D, &[&input, &kernel]).unwrap();
    assert_eq!(out[0].shape[0].static_value(), Some(1));
    assert_eq!(out[0].shape[1].static_value(), Some(16));
    assert_eq!(out[0].shape[2].static_value(), Some(24));
    assert_eq!(out[0].shape[3].static_value(), Some(24));
}

#[test]
fn test_layernorm_shape() {
    let a = mk(vec![2, 128, 64], DataType::FP32);
    let w = mk(vec![64], DataType::FP32);
    let out = ShapeInference::infer_output_shape(&TensorOp::LayerNorm, &[&a, &w]).unwrap();
    assert_eq!(out[0].shape.len(), 3);
    assert_eq!(out[0].shape[2].static_value(), Some(64));
}

// ═══════════════════════════════════════════════════
//  FLOP COUNTING — KNOWN BENCHMARKS
// ═══════════════════════════════════════════════════

#[test]
fn test_matmul_flops_exact() {
    let a = mk(vec![128, 256], DataType::FP32);
    let b = mk(vec![256, 512], DataType::FP32);
    let flops = ShapeInference::compute_flops(&TensorOp::MatMul, &[&a, &b]).unwrap();
    assert_eq!(flops, 2 * 128 * 256 * 512);
}

#[test]
fn test_matmul_flops_batch() {
    let a = mk(vec![8, 64, 128], DataType::FP32);
    let b = mk(vec![8, 128, 256], DataType::FP32);
    let flops = ShapeInference::compute_flops(&TensorOp::MatMul, &[&a, &b]).unwrap();
    assert_eq!(flops, 8 * 2 * 64 * 128 * 256);
}

#[test]
fn test_conv2d_flops_exact() {
    let input = mk(vec![1, 3, 32, 32], DataType::FP32);
    let kernel = mk(vec![64, 3, 3, 3], DataType::FP32);
    let flops = ShapeInference::compute_flops(&TensorOp::Conv2D, &[&input, &kernel]).unwrap();
    assert_eq!(flops, 2 * 1 * 64 * 30 * 30 * 3 * 3 * 3);
}

#[test]
fn test_elementwise_flops() {
    let a = mk(vec![1024, 1024], DataType::FP32);
    let b = mk(vec![1024, 1024], DataType::FP32);
    let flops = ShapeInference::compute_flops(&TensorOp::Add, &[&a, &b]).unwrap();
    assert_eq!(flops, 1024 * 1024);
}

#[test]
fn test_relu_flops() {
    let a = mk(vec![256, 512], DataType::FP32);
    let flops = ShapeInference::compute_flops(&TensorOp::ReLU, &[&a]).unwrap();
    assert_eq!(flops, 256 * 512);
}

// ═══════════════════════════════════════════════════
//  MEMORY ESTIMATION
// ═══════════════════════════════════════════════════

#[test]
fn test_memory_matmul() {
    let a = mk(vec![64, 128], DataType::FP32);
    let b = mk(vec![128, 256], DataType::FP32);
    let mem = ShapeInference::compute_memory_bytes(&TensorOp::MatMul, &[&a, &b]).unwrap();
    let expected = (64*128 + 128*256 + 64*256) * 4;
    assert_eq!(mem, expected);
}

#[test]
fn test_memory_fp16_vs_fp32() {
    let fp32 = mk(vec![1024, 1024], DataType::FP32);
    let fp16 = mk(vec![1024, 1024], DataType::FP16);
    let mem32 = ShapeInference::compute_memory_bytes(&TensorOp::ReLU, &[&fp32]).unwrap();
    let mem16 = ShapeInference::compute_memory_bytes(&TensorOp::ReLU, &[&fp16]).unwrap();
    assert_eq!(mem32, 2 * mem16);
}

#[test]
fn test_memory_int8_vs_fp32() {
    let fp32 = mk(vec![1024, 1024], DataType::FP32);
    let int8 = mk(vec![1024, 1024], DataType::INT8);
    let mem32 = ShapeInference::compute_memory_bytes(&TensorOp::ReLU, &[&fp32]).unwrap();
    let mem8 = ShapeInference::compute_memory_bytes(&TensorOp::ReLU, &[&int8]).unwrap();
    assert_eq!(mem32, 4 * mem8);
}

// ═══════════════════════════════════════════════════
//  BENCHMARK: KNOWN MODEL FLOPS
// ═══════════════════════════════════════════════════

#[test]
fn test_benchmark_resnet50_first_conv() {
    let input = mk(vec![1, 3, 224, 224], DataType::FP32);
    let kernel = mk(vec![64, 3, 7, 7], DataType::FP32);
    let flops = ShapeInference::compute_flops(&TensorOp::Conv2D, &[&input, &kernel]).unwrap();
    assert!(flops > 100_000_000, "ResNet-50 first conv > 100M FLOPs: {}", flops);
}

#[test]
fn test_benchmark_gpt2_qk_matmul() {
    let q = mk(vec![1024, 64], DataType::FP32);
    let k = mk(vec![64, 1024], DataType::FP32);
    let flops = ShapeInference::compute_flops(&TensorOp::MatMul, &[&q, &k]).unwrap();
    assert_eq!(flops, 2 * 1024 * 64 * 1024);
    let total_heads = 12 * flops;
    assert!(total_heads > 1_500_000_000);
}

#[test]
fn test_benchmark_bert_base_matmul() {
    let x = mk(vec![512, 768], DataType::FP32);
    let w = mk(vec![768, 768], DataType::FP32);
    let flops = ShapeInference::compute_flops(&TensorOp::MatMul, &[&x, &w]).unwrap();
    assert_eq!(flops, 2 * 512 * 768 * 768);
    let total = 48 * flops;
    assert!(total > 20_000_000_000);
}

#[test]
fn test_benchmark_llama7b_layer() {
    let x = mk(vec![2048, 4096], DataType::FP16);
    let wq = mk(vec![4096, 4096], DataType::FP16);
    let q_flops = ShapeInference::compute_flops(&TensorOp::MatMul, &[&x, &wq]).unwrap();
    assert_eq!(q_flops, 2 * 2048 * 4096 * 4096);

    let wup = mk(vec![4096, 11008], DataType::FP16);
    let ffn_flops = ShapeInference::compute_flops(&TensorOp::MatMul, &[&x, &wup]).unwrap();
    assert_eq!(ffn_flops, 2 * 2048 * 4096 * 11008);

    let layer_flops = 4 * q_flops + 3 * ffn_flops;
    assert!(layer_flops > 300_000_000_000);
}
