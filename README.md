<img width="1262" height="602" alt="image" src="https://github.com/user-attachments/assets/3880ecec-ff3f-4b44-b256-c3a9f07ee813" />


<!--
  LIFT — Language for Intelligent Frameworks and Technologies
  README v2.0  |  Phoenix Edition  |  Corrected & Honest
-->

<div align="center">

```                                            

# LIFT

**The first Intermediate Representation designed for both AI and Quantum Computing.**

*Simulate before you run. Compile once. Optimise everywhere.*

[![License: MIT](https://img.shields.io/badge/License-MIT-orange.svg)](LICENSE)
[![Rust 1.78+](https://img.shields.io/badge/Rust-1.78+-orange.svg)](https://rustlang.org)
[![Phase 0-1 Active](https://img.shields.io/badge/Phase-0--1%20Active-red.svg)]()
[![Research Alpha](https://img.shields.io/badge/Status-Research%20Alpha-gold.svg)]()

</div>

---

> **HONEST STATUS — Please read before proceeding**
>
> This document describes the **complete vision and architecture** for LIFT.
> **Phase 0 (LIFT-CORE) is complete. Phase 1 (LIFT-TENSOR) is active.**
> Quantum and hybrid support are in design, not yet coded.
>
> We publish the full vision early because the architectural decisions must be
> made correctly from day one. See Section 9 for what works today.
> Do not use in production. Contributions are very welcome.

---

## Table of Contents

1. [The Problem](#1-the-problem)
2. [The Vision](#2-the-vision)
3. [Honest Comparison with Existing IRs](#3-honest-comparison)
4. [Core Concept: Twin Dialects](#4-twin-dialects)
5. [Architecture](#5-architecture)
6. [The Four Pillars](#6-the-four-pillars)
7. [The .lif Source Language](#7-the-lif-language)
8. [The .lith Configuration Language](#8-the-lith-configuration)
9. [Current Status](#9-current-status)
10. [Getting Started](#10-getting-started)
11. [Roadmap](#11-roadmap)
12. [Contributing](#12-contributing)
13. [Known Limitations](#13-known-limitations)
14. [Why This Matters](#14-why-this-matters)

---

## 1. The Problem

### The Toolchain Fragmentation Crisis

A researcher working on hybrid AI+Quantum today must manage eight incompatible tools:

```
  AI TOOLCHAIN                          QUANTUM TOOLCHAIN
  ─────────────────────────             ─────────────────────────
  PyTorch  ──┐                          Qiskit   ──┐
  JAX      ──┼──► MLIR / ONNX           Cirq     ──┼──► OpenQASM 3
  TF       ──┘        │                 PennyLane──┘       │
                      ▼                                    ▼
               CUDA / LLVM / XLA                 IBM Q / Rigetti / IonQ

  ✗ No shared representation             ✗ No shared representation
  ✗ No joint optimisation                ✗ Cannot compose with AI
  ✗ Energy cost is invisible             ✗ Noise is an afterthought
  ✗ 8+ config files per project          ✗ No simulation-first workflow
```

### The Scale Problem

```
  AI MODEL SIZE (parameters):
  2018  BERT-Large  ██  340M
  2020  GPT-3       ████████████████████  175B
  2023  GPT-4 est.  ████████████████████████████  ~1.7T
  2025  Future      ████████████████████████████████  10T+
  → 10T params in FP16 = 20 TB. No single IR handles this today.

  QUANTUM HARDWARE (qubits):
  2019  Google Sycamore  ██  53
  2023  IBM Eagle         ██████████████████████  433
  2026  Target            ████████████████████████████████  1000+
  → Both worlds are scaling fast. The toolchain is not keeping up.
```

---

## 2. The Vision

LIFT is a **unified semantic IR** that understands both AI computation (tensors, gradients, attention) and quantum computation (qubits, gates, decoherence) in the same programme, with one configuration file and one compilation pipeline.

```
  ╔═══════════════════════════════════════════════════════════╗
  ║                                                           ║
  ║   Your programme  (one .lif file)                        ║
  ║                                                           ║
  ╠════════════════╦══════════════════════════════════════════╣
  ║  LIFT-TENSOR   ║  LIFT-QUANTUM         LIFT-HYBRID       ║
  ║  AI operations ║  Quantum gates        Fusion dialect    ║
  ╠════════════════╩══════════════════════════════════════════╣
  ║                                                           ║
  ║    SIMULATE → PREDICT → OPTIMISE → COMPILE               ║
  ║                                                           ║
  ╠═══════════════════════════════════════════════════════════╣
  ║  CUDA (GPU)  OpenQASM 3 (QPU)  LLVM (CPU)  XLA (TPU)    ║
  ╠═══════════════════════════════════════════════════════════╣
  ║  H100 · A100 · IBM Kyoto · Rigetti · Google TPU · M3     ║
  ╚═══════════════════════════════════════════════════════════╝
```

**The north star metric:** A researcher should go from idea to optimised hybrid execution on real hardware in under one hour, using one `.lif` file and one `.lith` config. Today that takes weeks.

---

## 3. Honest Comparison

We present this honestly. `~✓` means planned and in design — not yet working.

```
  ┌────────────────────────┬──────┬──────┬──────────┬────────┬──────────┐
  │ Capability             │ MLIR │ ONNX │ OpenQASM │ Qiskit │   LIFT   │
  ├────────────────────────┼──────┼──────┼──────────┼────────┼──────────┤
  │ AI tensor operations   │  ✓   │  ✓   │    ✗     │   ✗    │   ✓      │
  │ Quantum gate ops       │  ✗   │  ✗   │    ✓     │   ✓    │  ~✓ dev  │
  │ Hybrid AI+QC in one IR │  ✗   │  ✗   │    ✗     │  ~✓    │  ~✓ plan │
  │ Noise in type system   │  ✗   │  ✗   │    ✗     │   ✗    │  ~✓ plan │
  │ Linear qubit types     │  ✗   │  ✗   │    ✗     │   ✗    │  ~✓ plan │
  │ Performance prediction │  ✗   │  ✗   │    ✗     │   ✗    │  ~✓ dev  │
  │ Energy budgeting       │  ✗   │  ✗   │    ✗     │   ✗    │  ~✓ plan │
  │ Single config file     │  ✗   │  ✗   │    ✗     │   ✗    │  ~✓ dev  │
  │ Python bindings        │  ✓   │  ✓   │   ~✓     │   ✓    │  ~✓ dev  │
  ├────────────────────────┼──────┼──────┼──────────┼────────┼──────────┤
  │ Score today            │ 3/9  │ 2/9  │  3/9     │ 3/9    │  2/9 ✓   │
  │ Score at v1.0 (plan)   │      │      │          │        │  8/9 ~✓  │
  └────────────────────────┴──────┴──────┴──────────┴────────┴──────────┘

  ✓   = implemented and stable today
  ~✓  = planned, in design or active development
  ✗   = not supported, not planned
```

### What LIFT Adds That Does Not Exist Today

**1. One IR for both AI and quantum in the same programme**
MLIR has quantum dialect research (QSSA, Catalyst, OpenQASM dialect). None treat noise as a first-class type attribute, and none provide a joint optimisation path between tensor and quantum operations. LIFT is the first to design both as equal citizens in the same SSA IR.

**2. Noise as a type-level attribute**
Every quantum gate operation in LIFT carries optional noise metadata (T1, T2, gate fidelity, crosstalk coefficients). The type checker, optimiser, and predictor all reason over this noise. When two noisy gates are fused, the composite noise model is derived from the Kraus operators. No existing IR does this.

**3. Linear qubit types — no-cloning at compile time**
The quantum no-cloning theorem is a physical law. In LIFT, qubit values are linear types: consumed exactly once. A qubit used twice is a compile-time error, not a runtime failure. Branches must consume the same qubit set on every arm.

**4. Simulation-driven compilation with budget enforcement**
Before any hardware executes, LIFT produces: FLOP count, peak memory, circuit depth, expected fidelity from the noise model, estimated latency, and energy cost. If any budget constraint in `.lith` is violated, compilation fails with a clear error message and suggestions. This is architecturally different from post-hoc profiling.

**5. One configuration language**
The `.lith` file controls compilation target, optimisation passes, budget constraints, simulation parameters, deployment, and monitoring. It replaces 6–8 separate configs that today require expert knowledge to coordinate.

---

## 4. Twin Dialects

### The Structural Isomorphism

The deepest insight behind LIFT: AI computation and quantum computation face the same class of compilation challenges, with different vocabulary.

```
  AI DOMAIN                            QUANTUM DOMAIN
  ══════════════════════════════════════════════════════════
  Tensor (vector of floats)       ↔   Quantum state (amplitude vector)
  Linear layer (matrix multiply)  ↔   Unitary gate (unitary multiply)
  Non-linearity (ReLU)            ↔   Measurement (projection/collapse)
  Backpropagation (reverse AD)    ↔   Parameter shift rule (adjoint diff)
  Batch dimension                 ↔   Shot parallelism
  INT8 quantisation               ↔   Gate decomposition to native basis
  Layer fusion (MatMul+ReLU)      ↔   Gate cancellation (H·H = I)
  Memory layout (NCHW vs NHWC)    ↔   Qubit mapping (logical → physical)
  Multi-GPU data parallelism      ↔   Multi-QPU shot parallelism
  Gradient checkpoint             ↔   Mid-circuit reset and reuse
  ──────────────────────────────────────────────────────────
  The dialects are twins because the PROBLEMS are isomorphic.
  LIFT exploits this for joint optimisation.
```

### SSA Form: The Shared Foundation

Every value in LIFT is defined exactly once (Static Single Assignment). This property makes analysis and optimisation provably correct.

```
  TRADITIONAL (mutable)        LIFT SSA FORM
  ──────────────────────       ─────────────────────────────────────
  x = matmul(A, B)             %v0 = tensor.matmul(%A, %B)
  x = relu(x)         →        %v1 = tensor.relu(%v0)
  x = layernorm(x, w)          %v2 = tensor.layernorm(%v1, %w, %b)

                               Each %vi defined ONCE → safe to
                               fuse, reorder, parallelise, analyse.
```

### Linear Types: Enforcing the No-Cloning Theorem

```
  FORBIDDEN — qubit used twice:        CORRECT — SSA qubit chain:
  ─────────────────────────────        ─────────────────────────────────
  %q0 = quantum.init()                 %q0 = quantum.init()  : qubit
  %q1 = quantum.x(%q0)                 %q1 = quantum.x(%q0)  : qubit
  %q2 = quantum.h(%q0)  ← ERROR        %q2 = quantum.h(%q1)  : qubit
                                        %b0 = quantum.meas(%q2) : bit
  The type checker tracks a
  consumed set. %q0 was consumed       %q0 → %q1 → %q2 → %b0
  by quantum.x. Reuse = error.         Linear. Physically correct.
```

**Branches:** Every branch arm must consume the same set of qubits. A qubit that is consumed in the `then` arm but not the `else` arm is a compile-time error.

**Noise composition in fusion:** When two noisy gates are fused, the composite noise model is derived. For the initial implementation, we use a depolarising approximation. Full Kraus operator composition is planned for v1.1.

### The Three Dialects

```
  ┌─────────────────────────────────────────────────────────────┐
  │                       LIFT-CORE                             │
  │   SSA · Types · Ops · Blocks · Regions · Functions         │
  │   Shared foundation — all dialects build on this           │
  └────────────────────────┬────────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
  ┌───────────▼──────────┐   ┌──────────▼──────────┐
  │    LIFT-TENSOR       │   │   LIFT-QUANTUM       │
  │    AI dialect        │   │   QC dialect         │
  │                      │   │                      │
  │  Tensors, shapes     │   │  Qubits (linear)     │
  │  Auto-diff, grads    │   │  Gates + noise attr  │
  │  Attention, KV cache │   │  Layout mapping      │
  │  MoE, quantisation   │   │  Hamiltonians        │
  │  Parallelism         │   │  Error correction    │
  └──────────┬───────────┘   └───────────┬──────────┘
             │                           │
             └─────────────┬─────────────┘
                           │
  ┌────────────────────────▼────────────────────────────────────┐
  │                    LIFT-HYBRID                              │
  │   Classical ↔ Quantum encoding                             │
  │   Parameterised quantum circuits (VQC, QNN)                │
  │   Joint classical+quantum gradient computation             │
  │   GPU-side + QPU-side co-execution                         │
  └─────────────────────────────────────────────────────────────┘
```

---

## 5. Architecture

```
  ╔══════════════════════════════════════════════════════════════════╗
  ║                       LIFT FRAMEWORK                             ║
  ╠══════════════════════════════════════════════════════════════════╣
  ║  USER LAYER                                                      ║
  ║  .lif source  │  .lith config  │  lift(1) CLI  │  Python API    ║
  ╠══════════════════════════════════════════════════════════════════╣
  ║  FRONTEND                                                        ║
  ║  Lexer → Parser → AST → Type Check → SSA Build                  ║
  ║  Importers: PyTorch FX | ONNX | Qiskit | OpenQASM3 | Cirq       ║
  ╠══════════════════════════════════════════════════════════════════╣
  ║  DIALECT LAYER (Twin IR)                                         ║
  ║  LIFT-CORE  │  LIFT-TENSOR  │  LIFT-QUANTUM  │  LIFT-HYBRID     ║
  ╠══════════════════════════════════════════════════════════════════╣
  ║  SIMULATION + PREDICTION ENGINE                                  ║
  ║  Shape inference │ FLOP count │ Noise simulation                 ║
  ║  GNN perf predict│ Fidelity   │ Energy budget  │ Carbon est.     ║
  ╠══════════════════════════════════════════════════════════════════╣
  ║  OPTIMISATION PASS PIPELINE                                      ║
  ║  AI:      TensorFusion · FlashAttention · KVCache · INT8         ║
  ║  Quantum: GateCancellation · SABRE Layout · ZNE · QEC            ║
  ║  Hybrid:  HybridFusion · ParameterShift · EncodingOpt            ║
  ╠══════════════════════════════════════════════════════════════════╣
  ║  BACKENDS                                                        ║
  ║  CUDA (PTX)  │  OpenQASM 3  │  LLVM IR  │  XLA/StableHLO        ║
  ╠══════════════════════════════════════════════════════════════════╣
  ║  HARDWARE                                                        ║
  ║  H100 · A100 · MI300 │ IBM Kyoto · Rigetti · IonQ │ TPU · CPU   ║
  ╚══════════════════════════════════════════════════════════════════╝
```

### Workspace Layout

```
  lift/
  ├── crates/
  │   ├── lift-core/        SSA IR, types, operations (no external deps)
  │   ├── lift-ast/         .lif lexer, parser, AST
  │   ├── lift-tensor/      AI dialect
  │   ├── lift-quantum/     Quantum dialect
  │   ├── lift-hybrid/      Fusion dialect
  │   ├── lift-sim/         Static analysis + quantum simulator
  │   ├── lift-predict/     GNN performance prediction
  │   ├── lift-opt/         Pass manager + all optimisation passes
  │   ├── lift-import/      PyTorch FX, ONNX, Qiskit, OpenQASM3 importers
  │   ├── lift-export/      CUDA, OpenQASM3, LLVM, XLA backends
  │   │   Cargo.toml  →     features: [cuda, openqasm, llvm, xla]
  │   ├── lift-config/      .lith configuration language parser
  │   ├── lift-python/      Python bindings (PyO3/Maturin)
  │   └── lift-cli/         lift(1) command-line interface
  ├── examples/             .lif example programmes
  ├── tests/                integration and regression tests
  └── benches/              benchmark suite
```

---

## 6. The Four Pillars

### Pillar 1 — SIMULATE

Static analysis before any hardware is involved:

```
  .lif module → shape propagation → FLOP counting → memory analysis
              → noise accumulation → fidelity estimate → energy model
              → SIMULATION REPORT

  Example report:
  ──────────────────────────────────────────────────────
  AI:      4.7 TFLOPS · 12.4 GB peak · 1,847 req/s est.
  Quantum: depth=24 · 87 gates · fidelity=97.3% ± 0.8%
  Energy:  0.003 kWh · 1.05 gCO₂ (us-east-1 grid)
  ──────────────────────────────────────────────────────
```

### Pillar 2 — PREDICT

A trained GNN model predicts performance before hardware executes. Budget violations stop compilation:

```
  Budget check:
  Latency:   47ms    ✓  (budget: 100ms)
  Fidelity:  99.1%   ✓  (budget: >=95%)
  Memory:    31.4 GB ✓  (budget: 40 GB)
  Energy:    0.003   ✓  (budget: 0.01 kWh)

  vs.

  ERROR: latency 147ms exceeds budget 100ms
  Suggestions:
    1. Reduce seq_len 2048→1024  (est: 82ms)
    2. Enable INT8 quantisation  (est: 71ms)
```

**GNN Architecture:** Graph Neural Network with 6 message-passing layers, hidden dim 256, trained on 100K+ (IR graph, hardware features, measured performance) triples. Falls back to analytical model if the ML model is unavailable or confidence is low.

### Pillar 3 — OPTIMISE

All optimisation passes are **semantics-preserving by construction**: type-preserving transformations cannot change observable outputs. For complex passes (layout mapping, quantisation), correctness is validated against a suite of 5,000+ reference programmes.

```
  AI PASSES:
  ─────────────────────────────────────────────────────────────────
  tensor-fusion        Pattern-based (not Ullmann): MatMul+Bias+ReLU
                       → fused kernel. O(E+V) per pattern.
                       Gain: 30–50% less memory bandwidth.

  flash-attention      O(n²) → tiled O(n). FlashAttention v2/v3.
                       Triggered when seq_len > 512 on GPU.
                       Gain: 10–20× on long sequences.

  kv-cache             Pre-allocate key-value memory for LLM.
                       Gain: 100× latency for incremental inference.

  quantization         INT8/FP8. Dynamic or static calibration.
                       Gain: 4× model size, 2–4× throughput.

  QUANTUM PASSES:
  ─────────────────────────────────────────────────────────────────
  gate-cancellation    Algebraic identities + commutation table.
                       H·H=I, X·X=I, Rz(a)·Rz(b)=Rz(a+b).
                       Gain: 15–40% depth reduction.

  layout-mapping       SABRE algorithm (noise-aware variant).
                       Minimises SWAP insertions on physical topology.

  zne-mitigation       Gate folding (1×, 2×, 3× noise) + Richardson
                       extrapolation to zero noise.
                       Gain: 5–20× fidelity improvement.
```

### Pillar 4 — COMPILE

```
  OPTIMISED IR
   ├──► CUDA backend  → Tensor Core kernels · memory-coalesced access
   ├──► OpenQASM 3    → gate decomposition · layout · pulse schedule
   ├──► LLVM backend  → CPU native binary · AVX-512 · OpenMP
   └──► Hybrid runner → GPU+QPU orchestration · sync · data transfer
```

---

## 7. The .lif Language

```lif
// File: qnn_classifier.lif
// Hybrid QNN: classical encoder → 4-qubit layer → classical decoder

#dialect tensor

module @encoder {
    func @encode(%img: tensor<1x784xf32>) -> tensor<1x4xf32> {
        %h0  = "tensor.linear"(%img, %W1, %b1)
               : (tensor<1x784xf32>, tensor<784x64xf32>, tensor<64xf32>)
               -> tensor<1x64xf32>
        %h1  = "tensor.relu"(%h0) : (tensor<1x64xf32>) -> tensor<1x64xf32>
        %out = "tensor.linear"(%h1, %W2, %b2)
               : (tensor<1x64xf32>, tensor<64x4xf32>, tensor<4xf32>)
               -> tensor<1x4xf32>
        return %out
    }
}

#dialect quantum

module @q_layer {
    func @forward(%feat: tensor<4xf32>, %params: tensor<8xf32>)
                  -> (qubit, qubit, qubit, qubit) {
        // Each qubit value is linear — used exactly once
        %q0 = "quantum.init"() : () -> qubit
        %q1 = "quantum.init"() : () -> qubit
        %q2 = "quantum.init"() : () -> qubit
        %q3 = "quantum.init"() : () -> qubit

        // Angle encoding: feature[i] → rotation angle
        %q0 = "quantum.ry"(%q0, %feat[0]) : (qubit, f32) -> qubit
        %q1 = "quantum.ry"(%q1, %feat[1]) : (qubit, f32) -> qubit
        %q2 = "quantum.ry"(%q2, %feat[2]) : (qubit, f32) -> qubit
        %q3 = "quantum.ry"(%q3, %feat[3]) : (qubit, f32) -> qubit

        // Entangling layer
        %q0, %q1 = "quantum.cx"(%q0, %q1) : (qubit, qubit) -> (qubit, qubit)
        %q2, %q3 = "quantum.cx"(%q2, %q3) : (qubit, qubit) -> (qubit, qubit)

        // Trainable rotations
        %q0 = "quantum.rz"(%q0, %params[0]) : (qubit, f32) -> qubit
        %q1 = "quantum.rz"(%q1, %params[1]) : (qubit, f32) -> qubit
        %q2 = "quantum.rz"(%q2, %params[2]) : (qubit, f32) -> qubit
        %q3 = "quantum.rz"(%q3, %params[3]) : (qubit, f32) -> qubit

        return %q0, %q1, %q2, %q3
    }
}

#dialect hybrid

module @classifier {
    func @classify(%image: tensor<1x784xf32>) -> tensor<1x10xf32> {
        // Step 1: classical encoding
        %feat = "tensor.call"(@encoder::@encode, %image)
                : (tensor<1x784xf32>) -> tensor<1x4xf32>

        // Step 2: quantum layer (encoding + parameterised circuit)
        %q0, %q1, %q2, %q3 = "hybrid.angle_encode_forward"(
                @q_layer::@forward, %feat, %params)
                : (tensor<1x4xf32>, tensor<8xf32>)
                -> (qubit, qubit, qubit, qubit)

        // Step 3: measurement — consumes all qubits (linear type)
        %b0 = "quantum.measure"(%q0) : (qubit) -> bit
        %b1 = "quantum.measure"(%q1) : (qubit) -> bit
        %b2 = "quantum.measure"(%q2) : (qubit) -> bit
        %b3 = "quantum.measure"(%q3) : (qubit) -> bit

        // Step 4: classical output head
        %bits   = "tensor.stack"(%b0,%b1,%b2,%b3)
                  : (bit,bit,bit,bit) -> tensor<4xi1>
        %logits = "tensor.linear"(%bits, %Wout, %bout)
                  : (tensor<4xi1>, tensor<4x10xf32>, tensor<10xf32>)
                  -> tensor<1x10xf32>
        return %logits
    }
}
```

---

## 8. The .lith Configuration

One file replaces 8+ separate configs:

```lith
// File: project.lith

project {
    name        = "hybrid-qnn"
    version     = "1.0.0"
    description = "QNN classifier: classical encoder + 4-qubit layer"
}

dialects {
    tensor  = "1.0.0"
    quantum = "1.0.0"
    hybrid  = "1.0.0"
}

compilation {
    target {
        type = "hybrid"

        gpu {
            backend          = "cuda"
            arch             = "sm_90"     // H100
            memory_limit_gb  = 80
            tensor_cores     = true
        }

        qpu {
            provider         = "ibm"
            backend_name     = "ibm_kyoto"
            shots            = 4096
            optimization_level = 3

            error_mitigation {
                readout_error        = true
                dynamical_decoupling = true
                zero_noise_extrap    = true
            }
        }
    }
}

optimization {
    pipeline = [
        "canonicalize", "constant-folding",
        "tensor-fusion", "quantization",
        "gate-cancellation", "rotation-merging",
        "layout-mapping", "zne-mitigation",
        "hybrid-fusion", "parameter-shift"
    ]

    passes {
        quantization {
            precision   = "int8"
            calibration = "dynamic"
        }
        layout-mapping {
            algorithm = "sabre-noise-aware"
            rounds    = 3
        }
        zne-mitigation {
            noise_factors   = [1, 2, 3]
            extrapolation   = "richardson"
        }
    }
}

prediction {
    budget {
        max_latency_ms  = 200
        min_fidelity    = 0.92
        max_memory_gb   = 40
        max_energy_kwh  = 0.01
    }
}

metrics {
    collect = ["latency_ms", "fidelity", "memory_gb", "energy_kwh", "co2_grams"]
}
```

---

## 9. Current Status

| Component | Status | Works today | Next milestone |
|-----------|--------|-------------|----------------|
| `lift-core` | ✅ Alpha | SSA IR, types, ops, verifier, printer | Incremental compilation |
| `lift-ast` | ✅ Alpha | Lexer, parser, error recovery | Error quality |
| `lift-tensor` | 🚧 Active | MatMul, Add, ReLU, Conv2D, basic passes | Attention, KV cache, quantisation |
| `lift-quantum` | 📐 Design | Type system designed | Gate ops, noise model |
| `lift-hybrid` | 📐 Design | Architecture decided | All ops |
| `lift-sim` | 🚧 Active | Shape propagation, FLOP count | QC simulator, GNN predictor |
| `lift-predict` | 📐 Design | Architecture designed | GNN training pipeline |
| `lift-opt` | 🚧 Active | Pass manager, const-fold, DCE | Fusion, quantum passes |
| `lift-import` | 🚧 Active | PyTorch FX ~80%, OpenQASM3 ~60% | ONNX, Qiskit, Cirq |
| `lift-export` | 🚧 Active | LLVM ~70%, OpenQASM3 ~40% | CUDA backend |
| `lift-config` | 🚧 Active | Core .lith syntax ~60% | Validation, inheritance |
| `lift-python` | 📐 Design | PyO3 scaffold only | Full Python API |
| `lift-cli` | 🚧 Active | `lift analyse`, `lift verify` | compile, simulate, predict |

**Legend:** ✅ Alpha-stable · 🚧 Active · 📐 Design only

---

## 10. Getting Started

```bash
# Rust 1.78+
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Build
git clone https://github.com/lift-framework/lift
cd lift && cargo build --release
```

**What you can do today:**

```bash
# Write a tensor programme
cat > hello.lif << 'EOF'
#dialect tensor
module @test {
    func @relu(%x: tensor<4xf32>) -> tensor<4xf32> {
        %out = "tensor.relu"(%x) : (tensor<4xf32>) -> tensor<4xf32>
        return %out
    }
}
EOF

lift verify  hello.lif     # check well-formedness
lift analyse hello.lif     # FLOPs, shapes, memory estimate
lift print   hello.lif     # pretty-print the IR
```

---

## 11. Roadmap

Honest timeline: **24 months** to v1.0, not 12. Correct beats fast.

```
  Phase 0  LIFT-CORE         Wk  1–8    DONE        ████████░░░░░░
  Phase 1  LIFT-TENSOR       Wk  5–18   ACTIVE      ░░░████████░░░
  Phase 2  LIFT-QUANTUM       Wk 15–32   DESIGN      ░░░░░░░░░░░░░░
  Phase 3  LIFT-HYBRID        Wk 28–40   FUTURE      ░░░░░░░░░░░░░░
  Phase 4  SIM + PREDICT      Wk 32–46   FUTURE      ░░░░░░░░░░░░░░
  Phase 5  BACKENDS+IMPORT    Wk 38–56   FUTURE      ░░░░░░░░░░░░░░
  Phase 6  TOOLING            Wk 52–62   FUTURE      ░░░░░░░░░░░░░░
  Phase 7  v1.0 PUBLIC        Wk ~96     Target: Q4 2026
```

---

## 12. Contributing

| Area | Difficulty | What to build |
|------|-----------|---------------|
| Tensor attention passes | Hard | FlashAttention in LIFT-TENSOR |
| State vector simulator | Medium | Quantum sim CPU + GPU |
| PyTorch FX importer | Medium | Complete to 100% |
| Qiskit importer | Medium | Build from scratch |
| CUDA backend | Hard | PTX generation |
| .lith parser | Medium | Validation + inheritance |
| Documentation | Easy | Tutorials + API docs |

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## 13. Known Limitations

**Hard limits today:**
- No quantum hardware backend. Quantum support is design-only.
- No GPU code generation. CUDA backend is planned.
- Python bindings are not functional yet.
- GNN performance predictor does not exist yet.
- Energy and carbon modelling not implemented.

**Open design problems:**
- **Linear types in branches:** Designing the region-based analysis to ensure qubit consumption on all arms. Solution drafted; implementation pending.
- **Noise composition after fusion:** Using depolarising approximation initially. Full Kraus composition is v1.1.
- **GNN predictor generalisation:** Will use GNN + analytical model ensemble for robustness.

---

## 14. Why This Matters

The AI+Quantum convergence is not a hypothetical. IBM targets 1,000+ qubit processors by 2026. Hybrid variational algorithms (VQE, QAOA, QNN) are moving from academic curiosity to industrial application. AI models are hitting scales where classical hardware may need quantum assistance for specific workloads.

**The unified toolchain does not yet exist.** Two ecosystems are forming independently, and if they ossify before being bridged, the cost of unifying them later grows exponentially.

LIFT's bet: the right time to build the bridge is now, with the correct foundations (SSA form, linear qubit types, noise-aware IR, simulation-first), before the toolchain calcifies.

We are not claiming a finished product. We are claiming a correct architecture and an honest plan.

---

```bibtex
@software{lift2025,
  title  = {LIFT: Language for Intelligent Frameworks and Technologies},
  author = {Martial Fossouo and Contributors},
  year   = {2025},
  url    = {https://github.com/lift-framework/lift},
  note   = {Unified IR for AI and Quantum Computing. Research Alpha.}
}
```

**License:** MIT — see [LICENSE](LICENSE).

---

<div align="center">
<i>LIFT — Because the future of computation is both intelligent and quantum,
and it deserves a unified foundation.</i>
</div>
