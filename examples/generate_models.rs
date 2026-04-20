//! Generate .lif and .lith files from Rust code using ModelBuilder API.
//! Run: cargo run --example generate_models

use lift_core::model_builder::*;
use lift_core::types::DataType;

fn main() {
    println!("=== LIFT Model Generator ===\n");

    // 1. Generate Phi-3-mini
    generate_phi3();

    // 2. Generate DeepSeek MoE
    generate_deepseek();

    // 3. Generate config
    generate_config();

    println!("\nAll files generated successfully!");
}

fn generate_phi3() {
    let f = DataType::FP32;
    let model = ModelBuilder::new("phi3_mini_generated")
        .function("layer")
            .param("x", tensor(&[1, 128, 3072], f))
            .param("ln1_w", tensor_1d(3072, f))
            .param("wq", tensor_2d(3072, 3072, f))
            .param("wk", tensor_2d(3072, 3072, f))
            .param("wv", tensor_2d(3072, 3072, f))
            .param("wo", tensor_2d(3072, 3072, f))
            .param("ln2_w", tensor_1d(3072, f))
            .param("w_gate", tensor_2d(3072, 8192, f))
            .param("w_up", tensor_2d(3072, 8192, f))
            .param("w_down", tensor_2d(8192, 3072, f))
            // Attention block
            .op("tensor.rmsnorm", &["x", "ln1_w"], "n1", tensor(&[1, 128, 3072], f))
            .op("tensor.matmul", &["n1", "wq"], "q", tensor(&[1, 128, 3072], f))
            .op("tensor.matmul", &["n1", "wk"], "k", tensor(&[1, 128, 3072], f))
            .op("tensor.matmul", &["n1", "wv"], "v", tensor(&[1, 128, 3072], f))
            .op("tensor.grouped_query_attention", &["q", "k", "v"], "attn", tensor(&[1, 128, 3072], f))
            .op("tensor.matmul", &["attn", "wo"], "attn_proj", tensor(&[1, 128, 3072], f))
            .op("tensor.add", &["x", "attn_proj"], "r1", tensor(&[1, 128, 3072], f))
            // FFN block
            .op("tensor.rmsnorm", &["r1", "ln2_w"], "n2", tensor(&[1, 128, 3072], f))
            .op("tensor.matmul", &["n2", "w_gate"], "gate", tensor(&[1, 128, 8192], f))
            .op("tensor.silu", &["gate"], "gate_act", tensor(&[1, 128, 8192], f))
            .op("tensor.matmul", &["n2", "w_up"], "up", tensor(&[1, 128, 8192], f))
            .op("tensor.mul", &["gate_act", "up"], "gated", tensor(&[1, 128, 8192], f))
            .op("tensor.matmul", &["gated", "w_down"], "down", tensor(&[1, 128, 3072], f))
            .op("tensor.add", &["r1", "down"], "r2", tensor(&[1, 128, 3072], f))
            .returns("r2")
            .done()
        .build_lif();

    std::fs::write("phi3_generated.lif", &model).unwrap();
    println!("[OK] phi3_generated.lif ({} bytes)", model.len());
}

fn generate_deepseek() {
    let f = DataType::FP32;
    let model = ModelBuilder::new("deepseek_generated")
        .function("layer")
            .param("x", tensor(&[1, 128, 2048], f))
            .param("ln1_w", tensor_1d(2048, f))
            .param("wq", tensor_2d(2048, 2048, f))
            .param("wk", tensor_2d(2048, 2048, f))
            .param("wv", tensor_2d(2048, 2048, f))
            .param("wo", tensor_2d(2048, 2048, f))
            .param("ln2_w", tensor_1d(2048, f))
            .param("w_gate", tensor_2d(2048, 10944, f))
            .param("w_up", tensor_2d(2048, 10944, f))
            .param("w_down", tensor_2d(10944, 2048, f))
            .op("tensor.rmsnorm", &["x", "ln1_w"], "n1", tensor(&[1, 128, 2048], f))
            .op("tensor.matmul", &["n1", "wq"], "q", tensor(&[1, 128, 2048], f))
            .op("tensor.matmul", &["n1", "wk"], "k", tensor(&[1, 128, 2048], f))
            .op("tensor.matmul", &["n1", "wv"], "v", tensor(&[1, 128, 2048], f))
            .op("tensor.attention", &["q", "k", "v"], "attn", tensor(&[1, 128, 2048], f))
            .op("tensor.matmul", &["attn", "wo"], "proj", tensor(&[1, 128, 2048], f))
            .op("tensor.add", &["x", "proj"], "r1", tensor(&[1, 128, 2048], f))
            .op("tensor.rmsnorm", &["r1", "ln2_w"], "n2", tensor(&[1, 128, 2048], f))
            .op("tensor.matmul", &["n2", "w_gate"], "gate", tensor(&[1, 128, 10944], f))
            .op("tensor.silu", &["gate"], "ga", tensor(&[1, 128, 10944], f))
            .op("tensor.matmul", &["n2", "w_up"], "up", tensor(&[1, 128, 10944], f))
            .op("tensor.mul", &["ga", "up"], "gated", tensor(&[1, 128, 10944], f))
            .op("tensor.matmul", &["gated", "w_down"], "dn", tensor(&[1, 128, 2048], f))
            .op("tensor.add", &["r1", "dn"], "r2", tensor(&[1, 128, 2048], f))
            .returns("r2")
            .done()
        .build_lif();

    std::fs::write("deepseek_generated.lif", &model).unwrap();
    println!("[OK] deepseek_generated.lif ({} bytes)", model.len());
}

fn generate_config() {
    let config = build_lith_config(
        "llvm", "h100", "fp16",
        &["canonicalize", "constant-folding", "dce", "tensor-fusion", "flash-attention", "cse", "quantisation-pass"],
        Some(500_000_000_000),
        Some(80_000_000_000),
    );
    std::fs::write("optimize.lith", &config).unwrap();
    println!("[OK] optimize.lith ({} bytes)", config.len());
}
