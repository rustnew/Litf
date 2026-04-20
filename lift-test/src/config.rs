// ============================================================================
// config.rs — .lith configuration parsing and validation
// ============================================================================
//
// Validates that LIFT can parse `.lith` configuration files and apply their
// settings to the compilation pipeline.
//
// ============================================================================

use lift_config::ConfigParser;
use lift_config::types::LithConfig;
use crate::report::{TestReport, print_step};

/// Parse a `.lith` configuration file from an INI string.
pub fn parse_lith_config(source: &str, report: &mut TestReport) -> Option<LithConfig> {
    print_step(0, "Config Parse");
    let parser = ConfigParser::new();
    match parser.parse(source) {
        Ok(config) => {
            report.check("parse .lith config", true);
            println!("    Backend: {}", config.target.backend);
            println!("    Opt level: {:?}", config.optimisation.level);
            println!("    Passes: {:?}", config.optimisation.passes);
            if let Some(ref q) = config.quantum {
                println!("    Quantum: {} qubits, topology={}", q.num_qubits, q.topology);
            }
            Some(config)
        }
        Err(e) => {
            println!("    Error: {}", e);
            report.check("parse .lith config", false);
            None
        }
    }
}

/// Validate that a default configuration is sane.
pub fn validate_default_config(report: &mut TestReport) {
    print_step(0, "Config Validation");
    let config = LithConfig::default();

    report.check("default backend is llvm", config.target.backend == "llvm");
    report.check("default passes include canonicalize",
        config.optimisation.passes.contains(&"canonicalize".to_string()));
    report.check("FLOP counting enabled by default", config.simulation.enable_flop_counting);

    let qconfig = LithConfig::default().with_quantum("heavy_hex", 127);
    let q_ok = qconfig.quantum.as_ref()
        .map(|q| q.num_qubits == 127 && q.topology == "heavy_hex")
        .unwrap_or(false);
    report.check("with_quantum builder works", q_ok);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let mut report = TestReport::new();
        validate_default_config(&mut report);
        assert!(report.all_passed());
    }

    #[test]
    fn test_parse_config() {
        let lith_str = r#"
[target]
backend = "llvm"
precision = "fp16"

[budget]
max_flops = 1000000000

[optimisation]
level = O2
max_iterations = 10

[simulation]
shape_propagation = true
flop_counting = true
memory_analysis = true
noise_simulation = false
"#;
        let mut report = TestReport::new();
        let config = parse_lith_config(lith_str, &mut report);
        assert!(config.is_some());
        assert!(report.all_passed());
    }
}
