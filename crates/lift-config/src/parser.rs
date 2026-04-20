use crate::types::*;
use thiserror::Error;
use std::collections::HashMap;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("Missing required field: {0}")]
    MissingField(String),
    #[error("Invalid value for {field}: {value}")]
    InvalidValue { field: String, value: String },
    #[error("Parse error at line {line}: {message}")]
    ParseError { line: usize, message: String },
}

#[derive(Debug)]
pub struct ConfigParser;

impl ConfigParser {
    pub fn new() -> Self { Self }

    pub fn parse(&self, source: &str) -> Result<LithConfig, ConfigError> {
        let mut config = LithConfig::default();
        let mut current_section = String::new();
        let mut kv_map: HashMap<String, HashMap<String, String>> = HashMap::new();

        for (line_num, line) in source.lines().enumerate() {
            let trimmed = line.trim();

            if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with("//") {
                continue;
            }

            // Section header: [section]
            if trimmed.starts_with('[') && trimmed.ends_with(']') {
                current_section = trimmed[1..trimmed.len()-1].to_string();
                kv_map.entry(current_section.clone()).or_default();
                continue;
            }

            // Key = value
            if let Some((key, value)) = trimmed.split_once('=') {
                let key = key.trim().to_string();
                let value = value.trim().trim_matches('"').to_string();
                kv_map.entry(current_section.clone())
                    .or_default()
                    .insert(key, value);
            } else {
                return Err(ConfigError::ParseError {
                    line: line_num + 1,
                    message: format!("Expected key = value, got: {}", trimmed),
                });
            }
        }

        // Apply parsed values to config
        if let Some(target) = kv_map.get("target") {
            if let Some(backend) = target.get("backend") {
                config.target.backend = backend.clone();
            }
            if let Some(device) = target.get("device") {
                config.target.device = Some(device.clone());
            }
            if let Some(precision) = target.get("precision") {
                config.target.precision = Some(precision.clone());
            }
        }

        if let Some(budget) = kv_map.get("budget") {
            if let Some(v) = budget.get("max_flops") {
                config.budget.max_flops = v.parse().ok();
            }
            if let Some(v) = budget.get("max_memory_bytes") {
                config.budget.max_memory_bytes = v.parse().ok();
            }
            if let Some(v) = budget.get("max_time_ms") {
                config.budget.max_time_ms = v.parse().ok();
            }
            if let Some(v) = budget.get("min_fidelity") {
                config.budget.min_fidelity = v.parse().ok();
            }
            if let Some(v) = budget.get("max_circuit_depth") {
                config.budget.max_circuit_depth = v.parse().ok();
            }
        }

        if let Some(opt) = kv_map.get("optimisation") {
            if let Some(level) = opt.get("level") {
                config.optimisation.level = match level.as_str() {
                    "O0" | "0" => OptLevel::O0,
                    "O1" | "1" => OptLevel::O1,
                    "O2" | "2" => OptLevel::O2,
                    "O3" | "3" => OptLevel::O3,
                    _ => return Err(ConfigError::InvalidValue {
                        field: "optimisation.level".into(),
                        value: level.clone(),
                    }),
                };
            }
            if let Some(max_iter) = opt.get("max_iterations") {
                config.optimisation.max_iterations = max_iter.parse().unwrap_or(10);
            }
            if let Some(passes) = opt.get("passes") {
                config.optimisation.passes = passes
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
            }
            if let Some(disabled) = opt.get("disabled_passes") {
                config.optimisation.disabled_passes = disabled
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
            }
        }

        if let Some(sim) = kv_map.get("simulation") {
            if let Some(v) = sim.get("shape_propagation") {
                config.simulation.enable_shape_propagation = v == "true";
            }
            if let Some(v) = sim.get("flop_counting") {
                config.simulation.enable_flop_counting = v == "true";
            }
            if let Some(v) = sim.get("memory_analysis") {
                config.simulation.enable_memory_analysis = v == "true";
            }
            if let Some(v) = sim.get("noise_simulation") {
                config.simulation.enable_noise_simulation = v == "true";
            }
        }

        if let Some(quantum) = kv_map.get("quantum") {
            let qc = QuantumConfig {
                topology: quantum.get("topology").cloned().unwrap_or_else(|| "linear".into()),
                num_qubits: quantum.get("num_qubits").and_then(|v| v.parse().ok()).unwrap_or(5),
                error_mitigation: quantum.get("error_mitigation").cloned(),
                shots: quantum.get("shots").and_then(|v| v.parse().ok()),
            };
            config.quantum = Some(qc);
        }

        Ok(config)
    }

    pub fn parse_json(&self, json: &str) -> Result<LithConfig, ConfigError> {
        serde_json::from_str(json).map_err(|e| ConfigError::ParseError {
            line: 0,
            message: format!("JSON parse error: {}", e),
        })
    }
}

impl Default for ConfigParser {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_lith_config() {
        let src = r#"
[target]
backend = "cuda"
device = "A100"
precision = "fp16"

[budget]
max_flops = 1000000000
max_memory_bytes = 80000000000

[optimisation]
level = O3
max_iterations = 20

[simulation]
shape_propagation = true
flop_counting = true
"#;
        let parser = ConfigParser::new();
        let config = parser.parse(src).unwrap();
        assert_eq!(config.target.backend, "cuda");
        assert_eq!(config.optimisation.level, OptLevel::O3);
        assert_eq!(config.budget.max_flops, Some(1000000000));
    }

    #[test]
    fn test_parse_quantum_config() {
        let src = r#"
[target]
backend = "qasm"

[quantum]
topology = "grid"
num_qubits = 27
shots = 4096

[budget]
min_fidelity = 0.95
"#;
        let parser = ConfigParser::new();
        let config = parser.parse(src).unwrap();
        assert!(config.quantum.is_some());
        assert_eq!(config.quantum.as_ref().unwrap().num_qubits, 27);
        assert_eq!(config.budget.min_fidelity, Some(0.95));
    }
}
