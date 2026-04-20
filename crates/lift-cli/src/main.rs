use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "lift")]
#[command(version = "0.2.0")]
#[command(about = "LIFT — Language for Intelligent Frameworks and Technologies")]
#[command(long_about = "Unified IR for AI and Quantum Computing: Simulate → Predict → Optimise → Compile")]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Verbose output
    #[arg(short, long, global = true)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Verify a .lif file (SSA, types, linearity)
    Verify {
        /// Path to .lif source file
        #[arg(value_name = "FILE")]
        file: PathBuf,
    },
    /// Analyse a .lif file (FLOP count, memory, noise)
    Analyse {
        /// Path to .lif source file
        #[arg(value_name = "FILE")]
        file: PathBuf,
        /// Output format (text, json)
        #[arg(short, long, default_value = "text")]
        format: String,
    },
    /// Print the IR in human-readable form
    Print {
        /// Path to .lif source file
        #[arg(value_name = "FILE")]
        file: PathBuf,
    },
    /// Optimise a .lif file
    Optimise {
        /// Path to .lif source file
        #[arg(value_name = "FILE")]
        file: PathBuf,
        /// Configuration file (.lith)
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Output file
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Simulate and predict performance
    Predict {
        /// Path to .lif source file
        #[arg(value_name = "FILE")]
        file: PathBuf,
        /// Target device (a100, h100)
        #[arg(short, long, default_value = "a100")]
        device: String,
    },
    /// Export to target backend
    Export {
        /// Path to .lif source file
        #[arg(value_name = "FILE")]
        file: PathBuf,
        /// Target backend (llvm, qasm)
        #[arg(short, long)]
        backend: String,
        /// Output file
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
}

fn main() {
    let cli = Cli::parse();

    // Initialize tracing
    let filter = if cli.verbose { "debug" } else { "info" };
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .init();

    let result = match cli.command {
        Commands::Verify { file } => cmd_verify(&file),
        Commands::Analyse { file, format } => cmd_analyse(&file, &format),
        Commands::Print { file } => cmd_print(&file),
        Commands::Optimise { file, config, output } => cmd_optimise(&file, config.as_deref(), output.as_deref()),
        Commands::Predict { file, device } => cmd_predict(&file, &device),
        Commands::Export { file, backend, output } => cmd_export(&file, &backend, output.as_deref()),
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn load_and_parse(path: &std::path::Path) -> Result<lift_core::Context, String> {
    let source = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;

    let mut lexer = lift_ast::Lexer::new(&source);
    let tokens = lexer.tokenize().to_vec();
    if !lexer.errors().is_empty() {
        return Err(format!("Lexer errors: {:?}", lexer.errors()));
    }

    let mut parser = lift_ast::Parser::new(tokens);
    let program = parser.parse().map_err(|e| format!("Parse errors: {:?}", e))?;

    let mut ctx = lift_core::Context::new();
    let mut builder = lift_ast::IrBuilder::new();
    builder.build_program(&mut ctx, &program)?;

    Ok(ctx)
}

fn cmd_verify(path: &std::path::Path) -> Result<(), String> {
    let ctx = load_and_parse(path)?;

    match lift_core::verifier::verify(&ctx) {
        Ok(()) => {
            println!("Verification passed: {}", path.display());
            println!("  Values: {}", ctx.values.len());
            println!("  Operations: {}", ctx.ops.len());
            println!("  Blocks: {}", ctx.blocks.len());
            println!("  Regions: {}", ctx.regions.len());
            Ok(())
        }
        Err(errors) => {
            eprintln!("Verification failed with {} error(s):", errors.len());
            for err in &errors {
                eprintln!("  - {}", err);
            }
            Err(format!("{} verification error(s)", errors.len()))
        }
    }
}

fn cmd_analyse(path: &std::path::Path, format: &str) -> Result<(), String> {
    let ctx = load_and_parse(path)?;
    let report = lift_sim::analyze_module(&ctx);
    let quantum = lift_sim::analyze_quantum_ops(&ctx);

    match format {
        "json" => {
            let json = serde_json::to_string_pretty(&report)
                .map_err(|e| format!("JSON error: {}", e))?;
            println!("{}", json);
        }
        _ => {
            println!("=== LIFT Analysis Report ===");
            println!("File: {}", path.display());
            println!();
            println!("Operations: {}", report.num_ops);
            println!("  Tensor ops: {}", report.num_tensor_ops);
            println!("  Quantum ops: {}", report.num_quantum_ops);
            println!("  Hybrid ops: {}", report.num_hybrid_ops);
            println!();
            println!("Compute:");
            println!("  Total FLOPs: {}", format_flops(report.total_flops));
            println!("  Total memory: {}", format_bytes(report.total_memory_bytes));
            println!("  Peak memory: {}", format_bytes(report.peak_memory_bytes));

            if quantum.gate_count > 0 {
                println!();
                println!("Quantum:");
                println!("  Qubits: {}", quantum.num_qubits_used);
                println!("  Gate count: {}", quantum.gate_count);
                println!("  1Q gates: {}", quantum.one_qubit_gates);
                println!("  2Q gates: {}", quantum.two_qubit_gates);
                println!("  Measurements: {}", quantum.measurements);
                println!("  Estimated fidelity: {:.6}", quantum.estimated_fidelity);
            }

            if !report.op_breakdown.is_empty() {
                println!();
                println!("Op breakdown:");
                let mut ops: Vec<_> = report.op_breakdown.iter().collect();
                ops.sort_by(|a, b| b.1.cmp(a.1));
                for (name, count) in ops {
                    println!("  {}: {}", name, count);
                }
            }
        }
    }

    Ok(())
}

fn cmd_print(path: &std::path::Path) -> Result<(), String> {
    let ctx = load_and_parse(path)?;
    let output = lift_core::printer::print_ir(&ctx);
    println!("{}", output);
    Ok(())
}

fn cmd_optimise(path: &std::path::Path, config_path: Option<&std::path::Path>, output_path: Option<&std::path::Path>) -> Result<(), String> {
    let mut ctx = load_and_parse(path)?;

    let config = if let Some(cp) = config_path {
        let src = std::fs::read_to_string(cp)
            .map_err(|e| format!("Failed to read config: {}", e))?;
        lift_config::ConfigParser::new().parse(&src)
            .map_err(|e| format!("Config parse error: {}", e))?
    } else {
        lift_config::LithConfig::default()
    };

    let mut pm = lift_core::PassManager::new();

    // Add passes based on config
    for pass_name in &config.optimisation.passes {
        if config.optimisation.disabled_passes.contains(pass_name) {
            continue;
        }
        match pass_name.as_str() {
            "canonicalize" => pm.add_pass(Box::new(lift_opt::Canonicalize)),
            "constant-folding" => pm.add_pass(Box::new(lift_opt::ConstantFolding)),
            "dce" => pm.add_pass(Box::new(lift_opt::DeadCodeElimination)),
            "tensor-fusion" => pm.add_pass(Box::new(lift_opt::TensorFusion)),
            "gate-cancellation" => pm.add_pass(Box::new(lift_opt::GateCancellation)),
            "rotation-merge" => pm.add_pass(Box::new(lift_opt::RotationMerge)),
            "flash-attention" => pm.add_pass(Box::new(lift_opt::FlashAttentionPass::default())),
            "cse" => pm.add_pass(Box::new(lift_opt::CommonSubexprElimination)),
            "quantisation-pass" => pm.add_pass(Box::new(lift_opt::QuantisationPass::default())),
            "noise-aware-schedule" => pm.add_pass(Box::new(lift_opt::NoiseAwareSchedule)),
            "layout-mapping" => pm.add_pass(Box::new(lift_opt::LayoutMapping)),
            _ => {
                tracing::warn!("Unknown pass: {}", pass_name);
            }
        }
    }

    let results = pm.run_all(&mut ctx);

    println!("Optimisation results:");
    for (name, result) in &results {
        let status = match result {
            lift_core::PassResult::Changed => "changed",
            lift_core::PassResult::Unchanged => "unchanged",
            lift_core::PassResult::RolledBack => "rolled back",
            lift_core::PassResult::Error(e) => {
                eprintln!("  {} -> error: {}", name, e);
                "error"
            }
        };
        println!("  {} -> {}", name, status);
    }

    if let Some(out) = output_path {
        let ir = lift_core::printer::print_ir(&ctx);
        std::fs::write(out, ir)
            .map_err(|e| format!("Failed to write output: {}", e))?;
        println!("Output written to: {}", out.display());
    }

    Ok(())
}

fn cmd_predict(path: &std::path::Path, device: &str) -> Result<(), String> {
    let ctx = load_and_parse(path)?;
    let report = lift_sim::analyze_module(&ctx);

    let cost_model = match device {
        "a100" => lift_sim::cost::CostModel::a100(),
        "h100" => lift_sim::cost::CostModel::h100(),
        _ => return Err(format!("Unknown device: {}. Use 'a100' or 'h100'", device)),
    };

    let prediction = lift_predict::predict_performance(&report, &cost_model);

    println!("=== LIFT Performance Prediction ===");
    println!("Device: {}", device.to_uppercase());
    println!();
    println!("Compute time: {:.4} ms", prediction.compute_time_ms);
    println!("Memory time: {:.4} ms", prediction.memory_time_ms);
    println!("Predicted time: {:.4} ms", prediction.predicted_time_ms);
    println!("Arithmetic intensity: {:.2} FLOP/byte", prediction.arithmetic_intensity);
    println!("Bottleneck: {}", prediction.bottleneck);

    Ok(())
}

fn cmd_export(path: &std::path::Path, backend: &str, output_path: Option<&std::path::Path>) -> Result<(), String> {
    let ctx = load_and_parse(path)?;

    let output = match backend {
        "llvm" => {
            let exporter = lift_export::LlvmExporter::new();
            exporter.export(&ctx).map_err(|e| format!("{}", e))?
        }
        "qasm" => {
            let exporter = lift_export::QasmExporter::new();
            exporter.export(&ctx).map_err(|e| format!("{}", e))?
        }
        _ => return Err(format!("Unknown backend: {}. Use 'llvm' or 'qasm'", backend)),
    };

    if let Some(out) = output_path {
        std::fs::write(out, &output)
            .map_err(|e| format!("Failed to write output: {}", e))?;
        println!("Exported to: {}", out.display());
    } else {
        println!("{}", output);
    }

    Ok(())
}

fn format_flops(flops: u64) -> String {
    if flops >= 1_000_000_000_000 { format!("{:.2} TFLOP", flops as f64 / 1e12) }
    else if flops >= 1_000_000_000 { format!("{:.2} GFLOP", flops as f64 / 1e9) }
    else if flops >= 1_000_000 { format!("{:.2} MFLOP", flops as f64 / 1e6) }
    else if flops >= 1_000 { format!("{:.2} KFLOP", flops as f64 / 1e3) }
    else { format!("{} FLOP", flops) }
}

fn format_bytes(bytes: u64) -> String {
    if bytes >= 1_073_741_824 { format!("{:.2} GiB", bytes as f64 / 1_073_741_824.0) }
    else if bytes >= 1_048_576 { format!("{:.2} MiB", bytes as f64 / 1_048_576.0) }
    else if bytes >= 1_024 { format!("{:.2} KiB", bytes as f64 / 1_024.0) }
    else { format!("{} B", bytes) }
}
