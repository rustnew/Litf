/// Comprehensive tests for lift-quantum: gates, noise, topology, fidelity
use lift_quantum::gates::QuantumGate;
use lift_quantum::noise::*;
use lift_quantum::topology::DeviceTopology;

// ═══════════════════════════════════════════════════
//  GATE ROUNDTRIP
// ═══════════════════════════════════════════════════

#[test]
fn test_every_gate_name_roundtrip() {
    let gates = vec![
        QuantumGate::H, QuantumGate::X, QuantumGate::Y, QuantumGate::Z,
        QuantumGate::S, QuantumGate::Sdg, QuantumGate::T, QuantumGate::Tdg,
        QuantumGate::SX, QuantumGate::RX, QuantumGate::RY, QuantumGate::RZ,
        QuantumGate::U3, QuantumGate::CX, QuantumGate::CZ, QuantumGate::CY,
        QuantumGate::SWAP, QuantumGate::ISWAP, QuantumGate::ECR,
        QuantumGate::ZZ, QuantumGate::XX, QuantumGate::YY,
        QuantumGate::CCX, QuantumGate::CSWAP,
        QuantumGate::Measure, QuantumGate::MeasureAll,
        QuantumGate::Reset, QuantumGate::Barrier,
    ];
    for gate in &gates {
        let name = gate.op_name();
        let recovered = QuantumGate::from_name(name);
        assert!(recovered.is_some(), "gate '{}' must roundtrip", name);
        assert_eq!(recovered.unwrap().op_name(), name);
    }
}

// ═══════════════════════════════════════════════════
//  GATE CLASSIFICATION
// ═══════════════════════════════════════════════════

#[test]
fn test_single_qubit_gates() {
    for g in &[QuantumGate::H, QuantumGate::X, QuantumGate::Y, QuantumGate::Z,
               QuantumGate::S, QuantumGate::T, QuantumGate::SX,
               QuantumGate::RX, QuantumGate::RY, QuantumGate::RZ] {
        assert_eq!(g.num_qubits(), 1, "{:?} must be 1-qubit", g);
    }
}

#[test]
fn test_two_qubit_gates() {
    for g in &[QuantumGate::CX, QuantumGate::CZ, QuantumGate::CY,
               QuantumGate::SWAP, QuantumGate::ISWAP, QuantumGate::ECR,
               QuantumGate::ZZ, QuantumGate::XX, QuantumGate::YY] {
        assert_eq!(g.num_qubits(), 2, "{:?} must be 2-qubit", g);
    }
}

#[test]
fn test_three_qubit_gates() {
    assert_eq!(QuantumGate::CCX.num_qubits(), 3);
    assert_eq!(QuantumGate::CSWAP.num_qubits(), 3);
}

#[test]
fn test_parametric_gates() {
    assert!(QuantumGate::RX.is_parametric());
    assert!(QuantumGate::RY.is_parametric());
    assert!(QuantumGate::RZ.is_parametric());
    assert!(QuantumGate::ZZ.is_parametric());
    assert!(QuantumGate::U3.is_parametric());
    assert!(!QuantumGate::H.is_parametric());
    assert!(!QuantumGate::CX.is_parametric());
}

#[test]
fn test_self_inverse_gates() {
    assert!(QuantumGate::H.is_self_inverse());
    assert!(QuantumGate::X.is_self_inverse());
    assert!(QuantumGate::Y.is_self_inverse());
    assert!(QuantumGate::Z.is_self_inverse());
    assert!(QuantumGate::CX.is_self_inverse());
    assert!(QuantumGate::CZ.is_self_inverse());
    assert!(QuantumGate::SWAP.is_self_inverse());
    assert!(!QuantumGate::S.is_self_inverse());
    assert!(!QuantumGate::T.is_self_inverse());
}

#[test]
fn test_clifford_gates() {
    assert!(QuantumGate::H.is_clifford());
    assert!(QuantumGate::S.is_clifford());
    assert!(QuantumGate::CX.is_clifford());
    assert!(!QuantumGate::T.is_clifford());
    assert!(!QuantumGate::RZ.is_clifford());
}

// ═══════════════════════════════════════════════════
//  NOISE MODELS
// ═══════════════════════════════════════════════════

#[test]
fn test_gate_noise_depolarizing() {
    let noise = GateNoise::with_depolarizing(0.999, 0.02);
    assert!((noise.gate_fidelity - 0.999).abs() < 1e-10);
    assert!((noise.gate_time_us - 0.02).abs() < 1e-10);
}

#[test]
fn test_circuit_noise_accumulation() {
    let mut cn = CircuitNoise::new();
    assert!((cn.total_fidelity - 1.0).abs() < 1e-10);
    for _ in 0..10 {
        cn.add_gate(&GateNoise::with_depolarizing(0.999, 0.02), false);
    }
    let expected = 0.999f64.powi(10);
    assert!((cn.total_fidelity - expected).abs() < 1e-6);
    assert_eq!(cn.gate_count, 10);
}

#[test]
fn test_circuit_noise_two_qubit() {
    let mut cn = CircuitNoise::new();
    for _ in 0..5 {
        cn.add_gate(&GateNoise::with_depolarizing(0.99, 0.3), true);
    }
    let expected = 0.99f64.powi(5);
    assert!((cn.total_fidelity - expected).abs() < 1e-6);
    assert_eq!(cn.two_qubit_count, 5);
}

#[test]
fn test_circuit_noise_bell_state() {
    let mut cn = CircuitNoise::new();
    cn.add_gate(&GateNoise::with_depolarizing(0.999, 0.02), false);
    cn.add_gate(&GateNoise::with_depolarizing(0.99, 0.3), true);
    assert_eq!(cn.gate_count, 2);
    assert_eq!(cn.two_qubit_count, 1);
    let expected = 0.999 * 0.99;
    assert!((cn.total_fidelity - expected).abs() < 1e-6);
}

// ═══════════════════════════════════════════════════
//  TOPOLOGY
// ═══════════════════════════════════════════════════

#[test]
fn test_linear_topology() {
    let topo = DeviceTopology::linear(5);
    assert!(topo.are_connected(0, 1));
    assert!(topo.are_connected(3, 4));
    assert!(!topo.are_connected(0, 2));
    assert_eq!(topo.neighbors(0), vec![1]);
    assert_eq!(topo.neighbors(4), vec![3]);
    let path = topo.shortest_path(0, 4).unwrap();
    assert_eq!(path.len() - 1, 4);
}

#[test]
fn test_grid_topology() {
    let topo = DeviceTopology::grid(4, 4);
    assert_eq!(topo.neighbors(0).len(), 2);
    assert_eq!(topo.neighbors(5).len(), 4);
    assert!(topo.are_connected(0, 1));
    assert!(topo.are_connected(0, 4));
    assert!(!topo.are_connected(0, 5));
}

#[test]
fn test_grid_shortest_path() {
    let topo = DeviceTopology::grid(5, 5);
    let path = topo.shortest_path(0, 24).unwrap();
    assert_eq!(path.len() - 1, 8);
}

#[test]
fn test_swap_distance() {
    let topo = DeviceTopology::linear(5);
    let d = topo.swap_distance(0, 3);
    assert!(d.is_some());
    assert!(d.unwrap() >= 1);
}

// ═══════════════════════════════════════════════════
//  QUANTUM BENCHMARKS
// ═══════════════════════════════════════════════════

#[test]
fn test_benchmark_ghz_10_fidelity() {
    let mut cn = CircuitNoise::new();
    cn.add_gate(&GateNoise::with_depolarizing(0.999, 0.02), false);
    for _ in 0..9 {
        cn.add_gate(&GateNoise::with_depolarizing(0.99, 0.3), true);
    }
    let expected = 0.999 * 0.99f64.powi(9);
    assert!((cn.total_fidelity - expected).abs() < 1e-6);
    assert!(cn.total_fidelity > 0.90);
}

#[test]
fn test_benchmark_qft_8_gate_count() {
    let n = 8;
    let h_gates = n;
    let cr_gates = n * (n - 1) / 2;
    assert_eq!(h_gates + cr_gates, 36);

    let mut cn = CircuitNoise::new();
    for _ in 0..h_gates { cn.add_gate(&GateNoise::with_depolarizing(0.999, 0.02), false); }
    for _ in 0..cr_gates { cn.add_gate(&GateNoise::with_depolarizing(0.99, 0.3), true); }
    assert_eq!(cn.gate_count, 36);
    let expected = 0.999f64.powi(8) * 0.99f64.powi(28);
    assert!((cn.total_fidelity - expected).abs() < 1e-4);
}

#[test]
fn test_benchmark_vqe_4qubit() {
    let qubits = 4;
    let layers = 2;
    let mut cn = CircuitNoise::new();
    for _ in 0..layers {
        for _ in 0..qubits { cn.add_gate(&GateNoise::with_depolarizing(0.999, 0.02), false); }
        for _ in 0..(qubits-1) { cn.add_gate(&GateNoise::with_depolarizing(0.99, 0.3), true); }
    }
    assert_eq!(cn.gate_count, layers * (qubits + qubits - 1));
    assert!(cn.total_fidelity > 0.92);
}

#[test]
fn test_fidelity_scaling() {
    let sizes = vec![(5, 4), (10, 15), (20, 50), (50, 200), (100, 500)];
    let mut prev_fid = 1.0;
    for (n, cx) in &sizes {
        let mut cn = CircuitNoise::new();
        for _ in 0..*n { cn.add_gate(&GateNoise::with_depolarizing(0.999, 0.02), false); }
        for _ in 0..*cx { cn.add_gate(&GateNoise::with_depolarizing(0.99, 0.3), true); }
        assert!(cn.total_fidelity <= prev_fid);
        prev_fid = cn.total_fidelity;
    }
    assert!(prev_fid < 0.01);
}

#[test]
fn test_topology_scaling() {
    for n in &[5usize, 10, 20, 50] {
        let topo = DeviceTopology::linear(*n);
        let path = topo.shortest_path(0, n - 1).unwrap();
        assert_eq!(path.len() - 1, n - 1);
    }
    for side in &[3usize, 5, 7, 10] {
        let topo = DeviceTopology::grid(*side, *side);
        let path = topo.shortest_path(0, side * side - 1).unwrap();
        assert_eq!(path.len() - 1, 2 * (side - 1));
    }
}
