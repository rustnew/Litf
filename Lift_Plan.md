# LIFT — Complete Implementation Plan

**Timeline:** 24 months (honest) — not 12
**Team:** 2 for MVP → 4 for Alpha → 8 for v1.0
**Start date:** Week 1

This plan incorporates all corrections from the implementation review:
realistic timelines, realistic team sizing, testing strategy, CI/CD,
security review, maintenance plan, and Python bindings from Phase 0.

---

## Timeline Overview

```
  Months  1–2    Phase 0: LIFT-CORE + Python scaffold
  Months  2–5    Phase 1: LIFT-TENSOR (AI dialect)
  Months  4–9    Phase 2: LIFT-QUANTUM (Quantum dialect, two sub-phases)
  Months  8–12   Phase 3: LIFT-HYBRID (Fusion dialect)
  Months  10–14  Phase 4: Simulation + Prediction engine
  Months  12–18  Phase 5: Backends + Importers
  Months  16–21  Phase 6: Tooling + Observability
  Months  20–24  Phase 7: Documentation + Public v1.0
```

Phases 1–7 **overlap** intentionally: backend work can begin as soon as the
dialect is alpha-stable, and prediction training data collection starts in
Phase 1 even though the model is not trained until Phase 4.

---

## Team Plan

```
  Months  1–6   (MVP Phase)   2 engineers
    Engineer A: compiler (Core, passes, AST, LLVM backend)
    Engineer B: AI+Quantum (Tensor dialect, Quantum dialect design)

  Months  6–14  (Alpha Phase)  4 engineers
    + Engineer C: ML / prediction engine + benchmarks
    + Engineer D: infrastructure (CI/CD, Python bindings, CLI)

  Months  14–24 (Release Phase) 8 engineers
    + Engineer E: quantum (noise models, QPU backends, ZNE)
    + Engineer F: hybrid dialect + joint gradient
    + Engineer G: documentation, tutorials, community
    + Engineer H: security review, performance, deployment
```

---

## Phase 0: LIFT-CORE (Weeks 1–8)

**Goal:** A correct, well-tested SSA IR that all dialects can build on.
**Team:** 2 engineers.

### Milestone 0.1 — IR Data Structures (Weeks 1–3)

```
[ ] Context (SlotMap-based arena for all IR objects)
[ ] ValueData (TypeId + name + DefSite)
[ ] OperationData (name, dialect, inputs, results, attrs, regions, location)
[ ] BlockData (operations list + block arguments)
[ ] RegionData (blocks list + entry block)
[ ] FunctionData (name + signature + region)
[ ] ModuleData (name + functions + globals + dialect registry)
[ ] TypeInterner (deduplication: structural equality = pointer equality)
[ ] StringInterner (deduplicated string storage)
[ ] Attributes (typed key-value map for compile-time constants)
[ ] Location (file + line + column, for error messages)
```

**Acceptance criteria:**
- Can construct a 3-operation module in memory
- All objects have O(1) lookup by key
- Valgrind / ASAN: no memory errors on 100 random programmes
- No panics on valid inputs (checked with cargo test + proptest)

---

### Milestone 0.2 — Parser and Printer (Weeks 2–5)

```
[ ] Lexer  (hand-written, not generated — simpler, better errors)
      Tokens: ident, integer, float, string, punctuation, keywords
      Error recovery: continue after bad token, collect all errors
[ ] Parser (recursive descent)
      module, func, block, operation, type, value, attribute
[ ] AST → IR Builder
[ ] IR Printer (human-readable .lif text from IR)
[ ] Round-trip test: parse → build → print → re-parse → compare
[ ] Error messages: file + line + column + suggestion
```

**Acceptance criteria:**
- 30 hand-written .lif files parse without errors
- Round-trip test passes for all 30 files
- Every error message has a line number

---

### Milestone 0.3 — Core Passes + Verifier (Weeks 4–7)

```
[ ] IR Verifier
      SSA property (every value defined once)
      Dominance (uses dominated by defs)
      Type consistency (inputs/outputs match declared types)
      Well-formedness (no dangling SlotMap keys)
[ ] ConstantFoldingPass
      Evaluate ops on constant inputs at compile time
[ ] DeadCodeEliminationPass
      Remove unreachable ops and functions
[ ] CanonicalisationPass
      Normalise IR patterns (e.g., add(x, 0) → x)
[ ] PassManager
      Sequential pass runner with analysis cache
      Budget-aware rollback (from design doc Section 7.3)
```

**Acceptance criteria:**
- Verifier catches all malformed IRs in test suite (20+ bad IRs)
- Each pass has ≥ 15 unit tests
- PassManager runs without panic on 100 random modules

---

### Milestone 0.4 — Python Bindings Scaffold (Week 6–8)

**This is done in Phase 0, not deferred. Researchers use Python.**

```
[ ] PyO3 crate setup (lift-python)
[ ] Maturin build configuration
[ ] Python stubs (.pyi) for type checkers
[ ] Expose: Context, Module, Function, Operation (read-only)
[ ] Expose: lift.parse(source: str) -> Module
[ ] Expose: lift.verify(module: Module) -> bool
[ ] Expose: lift.print(module: Module) -> str
[ ] Expose: lift.analyse(module: Module) -> AnalysisReport
[ ] pip install lift works (wheel published to TestPyPI)
```

**Acceptance criteria:**
- `import lift; m = lift.parse("..."); lift.verify(m)` works in Python
- Type stubs pass mypy --strict

---

### Milestone 0.5 — C API (Week 7–8)

```
[ ] lift.h public header
[ ] lift_context_new / lift_context_free
[ ] lift_parse(ctx, source, len) -> Module*
[ ] lift_module_free(module)
[ ] lift_verify(module) -> int
[ ] lift_print(module) -> char*   // caller frees
[ ] lift_analyse_json(module) -> char*  // caller frees
[ ] Valgrind test: no leaks through C API
```

---

## Phase 1: LIFT-TENSOR (Weeks 5–18)

**Goal:** A correct, optimisable AI dialect with a working LLVM CPU backend.
**Team:** 2 engineers.

### Milestone 1.1 — Tensor Type System (Weeks 5–8)

```
[ ] TensorType (shape: Vec<Dimension>, dtype: DataType, layout: MemoryLayout)
[ ] AttentionTensor (batch, seq_len, num_heads, head_dim, dtype)
[ ] KVCache (max_seq, num_heads, head_dim, dtype, is_paged)
[ ] SparseTensor (num_experts, capacity, dtype) — for MoE
[ ] Dimension enum (Constant(n), Symbolic(String), Product)
[ ] DataType (FP64, FP32, FP16, BF16, FP8_E4M3, FP8_E5M2, INT8, INT4)
[ ] MemoryLayout (Contiguous, NCHW, NHWC, Strided, Tiled)
[ ] Shape inference rules for every operation (mandatory)
[ ] Type printer: tensor<1x32x128xf16>
[ ] Type parser: parse type from .lif text
[ ] Unit tests: 20+ type system tests
```

---

### Milestone 1.2 — Core AI Operations (Weeks 7–11)

```
ARITHMETIC
[ ] tensor.add, tensor.mul, tensor.sub, tensor.div, tensor.neg
[ ] tensor.matmul {transpose_lhs, transpose_rhs}
[ ] tensor.linear (%x, %W, %b) — fused matmul+bias
[ ] tensor.conv2d {stride, padding, dilation, groups}
[ ] tensor.embedding — lookup table

ACTIVATIONS
[ ] tensor.relu, tensor.gelu, tensor.silu, tensor.sigmoid
[ ] tensor.softmax {dim}

NORMALISATION
[ ] tensor.layernorm {eps}
[ ] tensor.rmsnorm {eps}
[ ] tensor.batchnorm

SHAPE
[ ] tensor.reshape, tensor.transpose, tensor.concat, tensor.split
[ ] tensor.gather, tensor.scatter

CONSTANTS
[ ] tensor.constant (compile-time tensor values)
[ ] tensor.zeros, tensor.ones

For each operation, implement:
  (a) shape inference
  (b) type checking
  (c) FLOP count formula
  (d) memory footprint (bytes allocated)
  (e) pretty printer
  (f) parser
  (g) ≥ 10 unit tests
```

---

### Milestone 1.3 — Attention and LLM Operations (Weeks 10–14)

```
[ ] tensor.attention {implementation, causal, scale}
      Implementations: Standard, FlashAttentionV2, FlashAttentionV3, SDPA
[ ] tensor.paged_attention {block_tables, context_len, num_heads, head_dim}
[ ] tensor.moe_dispatch {num_experts, num_active, capacity}
[ ] tensor.moe_combine
[ ] tensor.quantize {quant_type, calibration, per_channel}
[ ] tensor.dequantize {original_type}
[ ] tensor.checkpoint {fn} — gradient recomputation boundary
[ ] tensor.offload {location, prefetch} — CPU/SSD offloading
[ ] Gradient operations for all above
```

---

### Milestone 1.4 — AI Optimisation Passes (Weeks 12–17)

```
[ ] TensorFusionPass
      Declarative pattern library (≥ 10 patterns)
      Topological matching — O(V+E×P) not Ullmann
      Profitability check: only fuse if single_use(intermediates)
      Unit tests: 15+ tests including negative cases (should not fuse)

[ ] FlashAttentionPass
      Detect tensor.attention {implementation=Standard}
      Applicability: seq_len > 512 AND GPU target
      Replace with FlashAttentionV2 or V3 based on arch

[ ] KVCachePass
      Detect attention in inference mode (no grad)
      Insert paged attention + KV cache allocation

[ ] QuantizationPass
      Dynamic INT8 (default)
      Static INT8 (requires calibration_dataset in .lith)
      FP8 (requires sm_90 or higher)
      Per-channel or per-tensor

[ ] ParallelismPass
      Data parallel: replicate model, split batch
      Tensor parallel: split weight matrices
      Pipeline parallel: partition layers across stages
      Insert explicit split/allreduce/send/receive operations

[ ] MemoryPlanningPass
      Liveness analysis
      Buffer reuse (non-overlapping lifetimes share allocation)
      Memory pool creation
```

**Correctness validation for every pass:**
Run all 5,000 reference programmes through the pass.
Compare outputs before/after within 1e-5 tolerance (FP32).

---

### Milestone 1.5 — LLVM CPU Backend (Weeks 14–18)

```
[ ] LLVM IR emitter for all tensor ops
[ ] AVX-512 SIMD hints for contiguous tensor ops
[ ] OpenMP pragmas for element-wise ops
[ ] cuBLAS calls for MatMul (when CUDA feature enabled)
[ ] Linker configuration
[ ] Shared library (.so) and executable output
[ ] Compile and run a complete ResNet-50 inference step
```

**Phase 1 acceptance test:**
LLaMA 7B single-token inference compiles (CPU backend) and produces
correct output (within 1e-4 of PyTorch baseline) in < 10 minutes of
compilation time.

---

## Phase 2: LIFT-QUANTUM (Weeks 15–36)

**Sub-phases due to complexity. Each builds on the previous.**

### Phase 2a: Basic Quantum (Weeks 15–24)

#### Milestone 2a.1 — Quantum Types (Weeks 15–18)

```
[ ] Qubit type (linear — consumed exactly once)
[ ] PhysicalQubit type (id, T1, T2, freq, fidelity)
[ ] ClassicalBit type
[ ] QuantumState type (StateVector, DensityMatrix, MPS, Stabiliser)
[ ] Hamiltonian type (Vec<PauliTerm>)
[ ] PauliTerm (Complex<f64> coeff + Vec<(qubit_id, Pauli)>)
[ ] Pauli enum (I, X, Y, Z)
[ ] NoiseModel (gate_errors, T1, T2, crosstalk, readout)

LINEARITY CHECKER (integrated into verifier)
[ ] Track consumed: HashSet<ValueKey> per block
[ ] Report error if qubit key appears twice as input
[ ] Report error if a branch arm does not consume the same qubits
    as the other arm
[ ] Unit tests: 20 linearity tests (10 valid, 10 invalid)
```

#### Milestone 2a.2 — Gate Operations (Weeks 17–21)

```
SINGLE-QUBIT GATES
[ ] quantum.h, quantum.x, quantum.y, quantum.z
[ ] quantum.s, quantum.sdg, quantum.t, quantum.tdg, quantum.sx
[ ] quantum.rx(θ), quantum.ry(θ), quantum.rz(θ)
[ ] quantum.u1(λ), quantum.u2(φ,λ), quantum.u3(θ,φ,λ)

TWO-QUBIT GATES
[ ] quantum.cx, quantum.cz, quantum.cy, quantum.swap
[ ] quantum.iswap, quantum.ecr
[ ] quantum.rzx(θ), quantum.xx(θ), quantum.yy(θ), quantum.zz(θ)

THREE-QUBIT GATES
[ ] quantum.ccx (Toffoli)
[ ] quantum.cswap (Fredkin)

MEASUREMENT AND CONTROL
[ ] quantum.measure {basis} (%q: qubit) -> bit
[ ] quantum.measure_all (%q0,...,%qn) -> tensor<n×bit>
[ ] quantum.reset (%q: qubit) -> qubit
[ ] quantum.barrier (no-optimise fence)
[ ] quantum.delay {duration, unit}
[ ] quantum.init () -> qubit

PARAMETRISED GATE
[ ] quantum.param_gate {type, qubits, params}
    For VQE/QAOA trainable circuits.
```

#### Milestone 2a.3 — State Vector Simulator (Weeks 20–24)

```
[ ] CPU state vector (up to 28 qubits, exact)
      Complex<f64> vector of size 2^n
      Gate application as sparse matrix multiply
      Measurement as projection + normalisation + sample
      
[ ] GPU state vector (up to 35 qubits, via cuStateVec if available)
      Fallback to CPU if CUDA not present
      
[ ] OpenQASM 3 backend (basic)
      Emit all gates as OpenQASM 3 statements
      IBM basis set decomposition {RZ, SX, X, CX}
      Rigetti basis set decomposition {RZ, RX, CZ}
```

### Phase 2b: Advanced Quantum (Weeks 22–36)

#### Milestone 2b.1 — Noise Models and Density Matrix (Weeks 22–27)

```
[ ] IBM device calibration loader (JSON from IBM Quantum API)
[ ] Rigetti device calibration loader
[ ] Depolarising channel representation
[ ] Amplitude damping channel
[ ] Pauli error channel
[ ] Crosstalk model (ZZ coupling between neighbours)
[ ] Readout error matrix
[ ] Noise propagation analysis (accumulated error through circuit)
[ ] Density matrix simulator (up to 20 qubits) — includes noise
[ ] MPS tensor network simulator (up to 100 qubits, low entanglement)
[ ] Monte Carlo noise simulation (trajectory sampling)
[ ] Fidelity estimation from noise model
```

#### Milestone 2b.2 — Layout Mapping and Quantum Passes (Weeks 26–33)

```
[ ] QuantumTopology (coupling map + gate fidelity + gate time per pair)
[ ] SABRE routing algorithm
      Standard SABRE (depth-minimising)
      Noise-aware SABRE (fidelity-weighted scoring)
[ ] A* exact routing (for small circuits, ≤ 10 qubits)
[ ] SWAP insertion verification (simulate before+after, compare)

[ ] GateCancellationPass
      Algebraic identities + commutation table
      Peephole window

[ ] RotationMergingPass
      Rz(a)·Rz(b) = Rz(a+b), with angle wrapping

[ ] GateDecompositionPass
      Decompose to hardware-native basis sets
      IBM, Rigetti, IonQ basis sets

[ ] TwoQubitWeylDecompositionPass
      Cartan (KAK) decomposition — any 2Q unitary → ≤ 3 CX gates
```

#### Milestone 2b.3 — Error Mitigation Passes (Weeks 30–36)

```
[ ] ZNE pass (Zero Noise Extrapolation)
      Gate folding: G → G G† G (3× noise factor)
      Richardson extrapolation (linear, quadratic, order-3)
      Auto-order selection based on circuit depth
      R² validation: warn if extrapolation fit quality < 0.95

[ ] Readout error mitigation pass
      Insert calibration circuits (all-zeros, all-ones)
      Compute correction matrix
      Apply matrix inversion to results

[ ] Dynamical Decoupling pass
      Detect idle periods on qubits
      Insert XY-4 sequences when idle > T2/10
      Respect hardware timing constraints

[ ] QEC code insertion (basic, surface code) — v1.1, not v1.0
```

---

## Phase 3: LIFT-HYBRID (Weeks 28–42)

### Milestone 3.1 — Encoding Operations (Weeks 28–33)

```
[ ] hybrid.amplitude_encode {normalize}
      tensor<N×f32> → log₂(N) qubits
      Decompose into gate sequence to initialise state

[ ] hybrid.angle_encode {gate}
      tensor<N×f32> → N qubits
      Domain mapping: arbitrary float → [0, 2π]

[ ] hybrid.basis_encode
      tensor<N×i32> → N qubits

[ ] hybrid.hamiltonian_encode
      tensor<K×f32> + Pauli structure → Hamiltonian value

[ ] hybrid.decode (measurement → classical tensor)
      Expectation value computation
      Sampling statistics
```

### Milestone 3.2 — Hybrid Operations (Weeks 32–39)

```
[ ] hybrid.angle_encode_forward
      Combined angle encode + parameterised circuit execution
      Most common QNN pattern

[ ] hybrid.measure_with_ml
      Quantum measurement → classical ML post-processing

[ ] hybrid.parameter_shift_gradient
      Compute dE/dθ for all circuit parameters using parameter shift
      Batch 2P circuit evaluations

[ ] hybrid.joint_optimisation
      Classical + quantum params, single optimizer step
      Supports Adam, COBYLA, SPSA

[ ] hybrid.cosimulation {interface}
      GPU-side + QPU-side co-execution
      Synchronisation and data transfer management
```

### Milestone 3.3 — Hybrid Optimisation Passes (Weeks 37–42)

```
[ ] HybridFusionPass
      Fuse classical post-processing with quantum measurement
      Eliminate GPU ↔ QPU round-trips

[ ] ParameterShiftPass
      Expand hybrid.parameter_shift_gradient into explicit
      2P forward evaluations

[ ] EncodingOptimisationPass
      Select encoding based on circuit depth budget
      Amplitude vs angle encoding tradeoff

[ ] ShotOptimisationPass
      Minimum shots for target statistical precision
      Reuse shots across repeated measurements
```

---

## Phase 4: Simulation + Prediction Engine (Weeks 32–46)

### Milestone 4.1 — Static Analysis Engine (Weeks 32–37)

```
[ ] Full shape propagation (all tensor ops)
[ ] Full FLOP counter (per-op formulae, per-module total)
[ ] Memory liveness analysis (peak memory computation)
[ ] Bandwidth pressure estimator
[ ] Circuit analyser: depth, gate counts by type, T1/T2 risk
[ ] Energy model (per-op energy table × count)
[ ] Carbon estimate (energy × grid intensity, configurable region)
[ ] HTML + JSON simulation report generator
```

### Milestone 4.2 — GNN Prediction Model (Weeks 35–44)

```
DATA COLLECTION (start in Phase 1, run continuously)
[ ] Benchmark runner: 200+ programmes × 5+ hardware configs
[ ] Trace format: (IR graph JSON, hw_spec JSON, latency_ms, memory_gb)
[ ] Initial dataset target: 50K examples before model training

MODEL (Weeks 38–42)
[ ] Computation graph extractor (IR → node/edge feature matrices)
[ ] GNN architecture:
      NodeEncoder Linear(node_feat_dim → 128)
      EdgeEncoder Linear(edge_feat_dim → 64)
      6 × GatedGraphConv(128)
      GlobalAttentionPool(128)
      HWEncoder Linear(hw_feat → 64)
      LatencyHead MLP(192 → 64 → 1)
      MemoryHead  MLP(192 → 64 → 1)
      FidelityHead MLP(192 → 64 → 1) — quantum only
[ ] Training pipeline (PyTorch, export to ONNX)
[ ] Rust inference engine (load ONNX, run prediction < 100ms)
[ ] Analytical fallback model (for new hardware / low confidence)
[ ] Confidence scoring (use GNN if confidence > 0.7, else analytical)

VALIDATION (Weeks 42–44)
[ ] Hold-out test set: 10K examples not seen during training
[ ] Acceptance: median error < 15% on held-out set
[ ] Out-of-distribution test: predict on hardware not in training set
```

### Milestone 4.3 — Budget Enforcement (Weeks 43–46)

```
[ ] Budget constraint language in .lith (max_latency_ms, min_fidelity, ...)
[ ] Budget checker: compare predictions vs constraints
[ ] Actionable error messages: what constraint is violated and by how much
[ ] Suggestion engine: top-3 passes that would bring budget into compliance
```

---

## Phase 5: Backends + Interoperability (Weeks 38–56)

### Milestone 5.1 — CUDA Backend (Weeks 38–44)

```
[ ] PTX code generation framework
[ ] tensor.matmul → cuBLAS GEMM (FP32, FP16, INT8)
[ ] tensor.flash_attention → custom FlashAttention v2/v3 template
[ ] tensor.conv2d → cuDNN convolution
[ ] tensor.quantize/dequantize → INT8 CUDA kernels
[ ] Tensor Core utilisation annotations (FP16, BF16, INT8, FP8)
[ ] Memory coalescing analysis
[ ] Kernel launch config optimisation (block size, grid size, smem)
[ ] NCCL integration for multi-GPU allreduce
[ ] CUDA graph capture for inference (amortise launch overhead)
[ ] Integration test: LLaMA 7B inference on H100, match PyTorch output
```

### Milestone 5.2 — OpenQASM 3 Full Backend (Weeks 42–47)

```
[ ] Complete OpenQASM 3.0 emitter (all gates, classicals, control flow)
[ ] IBM Qiskit Runtime submission
      Create job, poll status, retrieve results
[ ] AWS Braket submission (Rigetti, IonQ via Braket API)
[ ] Pulse-level lowering for IBM (drag pulse shaping)
[ ] Integration test: VQE H₂ on IBM Kyoto, compare energy to literature
```

### Milestone 5.3 — Importers (Weeks 44–52)

```
[ ] PyTorch FX importer — complete to 100% (from 80%)
      All aten:: ops handled or mapped to UnsupportedOp
      Dynamic shape annotation preserved

[ ] ONNX importer (opset 19)
      Complete type mapping
      All common ops

[ ] Qiskit QuantumCircuit importer
      All standard gates
      Parametrised gates
      Noise model import (from Qiskit NoiseModel)

[ ] Cirq Circuit importer
      All standard gates

[ ] OpenQASM 3 importer — complete to 100% (from 60%)
```

### Milestone 5.4 — .lith Parser (Weeks 48–54)

```
[ ] Complete grammar (all sections and fields)
[ ] Environment variable substitution ${VAR_NAME}
[ ] File inclusion: include "./base.lith"
[ ] Config inheritance: extends = "base.lith"
[ ] Conditional blocks: if target.type == "qpu" { ... }
[ ] Enum validation (all field values checked against allowed set)
[ ] Cross-section consistency checks
[ ] Helpful error messages (line + column + suggestion)
[ ] Default value resolution
[ ] Auto-generated reference documentation from struct annotations
```

---

## Phase 6: Tooling + Observability (Weeks 52–62)

### Milestone 6.1 — CLI (Weeks 52–56)

```
lift compile <file.lif> [--config <file.lith>] [--target cuda|qasm|llvm]
             [--passes <p1,p2,...>] [--output <dir>]

lift simulate <file.lif> [--config <file.lith>] [--report <html>]

lift predict  <file.lif> [--hardware h100|ibm_kyoto|...] [--noise <json>]

lift optimise <file.lif> [--passes <p1,p2,...>] [--output <file.lif>]

lift convert  [--from pytorch|onnx|qiskit|qasm] [--to lift] <input>

lift verify   <file.lif>     (IR well-formedness check)

lift analyse  <file.lif>     (FLOPs, shapes, circuit stats, energy)

lift info     <file.lif>     (dialect list, op counts, quick summary)
```

### Milestone 6.2 — Observability (Weeks 54–60)

```
[ ] Structured JSON logging (tracing crate)
      Log every pass: name, duration, IR delta, budget impact
      Log every backend step
      Log prediction results

[ ] Prometheus metrics endpoint (/metrics on compilation server)
      compilation_duration_seconds histogram per pass
      prediction_error_ratio (compare predicted vs actual)
      pass_improvement_ratio per pass type

[ ] Interactive web dashboard (port 8081)
      Compilation trace timeline
      Computation graph viewer (interactive, zoomable)
      Circuit diagram viewer for quantum modules
      Before/after IR diff viewer

[ ] Flamegraph profiler integration
      Profile compiler pass execution time
      Identify bottlenecks in compiler itself

[ ] Compilation replay
      Record all inputs + config
      Re-run exact compilation for debugging
      Compare IR at each checkpoint
```

### Milestone 6.3 — Auto-Tuning (Weeks 58–62)

```
[ ] Pass ordering search (Bayesian optimisation, 50-iteration budget)
      Search space: which passes, in what order
      Objective: minimise predicted latency

[ ] RL-based pass selector (lightweight GNN agent)
      State: current IR features
      Action: next pass to apply
      Reward: improvement in predicted performance

[ ] A/B testing infrastructure
      Run two compilation strategies in parallel (10% of traffic)
      Select winner based on real execution performance
```

---

## Phase 7: Documentation + Public Release (Weeks 60–96)

### Milestone 7.1 — Documentation (Weeks 60–70)

```
[ ] API documentation (rustdoc, all public items)
[ ] Language reference (.lif — all operations, all types, all dialects)
[ ] Configuration reference (.lith — all sections and fields)
[ ] Getting Started Guide (30 minutes: install → hello world → real model)

TUTORIALS
[ ] Tutorial 1: LLM Inference Optimisation
      LLaMA 7B → FlashAttention → KV cache → INT8 → benchmark vs PyTorch

[ ] Tutorial 2: VQE for Hydrogen Molecule
      H₂ VQE → LIFT-QUANTUM → noise model → ZNE → IBM Kyoto → compare energy

[ ] Tutorial 3: QNN Image Classifier
      MNIST → hybrid QNN → angle encoding → joint gradient → train end-to-end

[ ] Tutorial 4: Writing a Custom Optimisation Pass
      Implement simple fusion → register → use from .lith

[ ] Contributor Guide (add a dialect, pass, backend)

VIDEOS
[ ] Demo video: 10-minute overview of LIFT
[ ] Tutorial video series (one per tutorial above)
```

### Milestone 7.2 — Benchmark Suite (Weeks 68–80)

```
AI BENCHMARKS (compare vs PyTorch + torch.compile + TensorRT)
[ ] BERT-Large inference (batch=1, seq=512)
[ ] LLaMA 7B single-token inference
[ ] ResNet-50 training step (batch=256)
[ ] ViT-Large inference

QUANTUM BENCHMARKS (compare vs Qiskit transpile() + pytket)
[ ] QAOA MaxCut (n=20, p=3)
[ ] VQE H₂O (12 qubits)
[ ] Quantum Volume circuits (QV=128)
[ ] Random circuits (n=20, depth=100)

HYBRID BENCHMARKS (compare vs PennyLane)
[ ] QNN MNIST (4 qubits, 10 classes)
[ ] QAOA + classical post-processing

COMPILATION TIME BENCHMARKS
[ ] Time to compile each model above
[ ] Compare vs torch.compile, Qiskit transpile

[ ] 5-page benchmark paper (methodology + results)
```

### Milestone 7.3 — Security Review (Weeks 76–84)

```
[ ] Threat model document (see DESIGN.md Part 10)
[ ] Fuzzing campaign: 1M+ random .lif inputs to parser
[ ] Fuzzing campaign: 100K+ random .lith configs
[ ] cargo audit weekly (automated in CI)
[ ] cargo-deny for license + advisory policy
[ ] No unsafe in core crates audit (MIRI verification)
[ ] Sandboxed compilation implementation (seccomp or Docker)
[ ] Security disclosure policy published (security@lift-framework.org)
[ ] GPG signing setup for all releases
```

### Milestone 7.4 — Public Release v1.0 (Weeks 84–96)

```
[ ] GitHub repository: lift-framework/lift
      MIT license, README.md, CONTRIBUTING.md, CODE_OF_CONDUCT.md

[ ] crates.io publication
      lift-core, lift-tensor, lift-quantum, lift-hybrid, lift-cli, lift-python

[ ] PyPI package (via Maturin)
[ ] Docker image: lift:latest (all backends pre-installed)

[ ] arXiv preprint
      "LIFT: A Unified Intermediate Representation for AI and Quantum Computing"

[ ] Blog post (3000 words, technical introduction)
[ ] Announcement posts (HN, r/MachineLearning, r/QuantumComputing)

[ ] Community infrastructure
      Discord server (lift-framework)
      GitHub Discussions
      Issue templates (bug, feature, RFC)
```

---

## Continuous: CI/CD Pipeline

**Set up in Week 1. Never disabled.**

```yaml
# .github/workflows/ci.yml

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo test --all --all-features
      - run: cargo clippy --all -- -D warnings
      - run: cargo fmt --all -- --check
      - run: cargo audit

  coverage:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: cargo llvm-cov --all-features --lcov --output-path lcov.info
      - uses: codecov/codecov-action@v3
        with: { files: lcov.info }
      # Fail if coverage drops below 75%

  integration:
    runs-on: [self-hosted, gpu]
    steps:
      - run: cargo test --features cuda
      - run: python tests/run_quantum_sim.py

  bench:
    runs-on: [self-hosted, gpu]
    if: github.ref == 'refs/heads/main'
    steps:
      - run: cargo bench -- --output benchmarks.json
      - run: python scripts/check_regression.py benchmarks.json
      # Fail if any benchmark regresses > 10%

  publish:
    if: startsWith(github.ref, 'refs/tags/v')
    needs: [test, integration]
    steps:
      - run: cargo publish --package lift-core
      - run: cargo publish --package lift-tensor
      # etc.
      - run: docker build -t lift:${{ github.ref_name }} .
      - run: docker push ghcr.io/lift-framework/lift
```

**Nightly runs (additional):**
- Full benchmark suite (4 hours on GPU cluster)
- QPU regression tests (submit to IBM simulator, compare outputs)
- Retrain prediction model with accumulated traces

---

## Hardware Requirements

### Development (Ongoing)

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU | 1× A100 40GB | 2× H100 80GB |
| CPU | 16-core workstation | 32-core server |
| RAM | 128 GB | 256 GB |
| QPU access | 10 QPU-hours/month | 30 QPU-hours/month |

### CI/CD

| Resource | Purpose |
|----------|---------|
| 1× A100 runner | GPU integration tests |
| Quantum sim server (64-core, 256 GB) | Large circuit simulations |
| Cloud QPU credits | IBM Quantum, AWS Braket |

### Benchmark (Phase 7 only)

| Resource | Purpose |
|----------|---------|
| 8× H100 cluster | Distributed training benchmarks |
| 20 QPU-hours | Quantum benchmark suite |
| 2000 GPU-hours | GNN training data collection |

---

## Success Criteria

### At 6 Months (End of Phase 1)

```
[ ] 500+ passing tests across lift-core and lift-tensor
[ ] 10+ AI models compile to LLVM backend and produce correct output
[ ] Compilation time < 10 seconds for 7B-parameter model (LLVM target)
[ ] Python bindings functional: import lift; lift.analyse() works
[ ] Zero memory leaks (verified ASAN + Valgrind)
[ ] CI green on every commit
```

### At 12 Months (End of Phase 2b)

```
[ ] Quantum circuits of up to 100 qubits simulate correctly
[ ] Layout mapping (SABRE) produces correct results on IBM coupling maps
[ ] ZNE pass improves fidelity by > 5× on noisy simulations
[ ] Import Qiskit circuits and compile to OpenQASM 3
[ ] End-to-end: VQE H₂ runs on IBM Kyoto and returns correct energy
```

### At 18 Months (End of Phase 5)

```
[ ] LLaMA 7B inference on H100 within 10% of TensorRT performance
[ ] QNN MNIST trains end-to-end with joint gradients
[ ] .lith parser validates complete configs with helpful error messages
[ ] All 5 importers functional (PyTorch FX, ONNX, Qiskit, Cirq, OpenQASM3)
[ ] 86%+ test coverage
```

### At 24 Months (v1.0 Release)

```
[ ] arXiv preprint submitted
[ ] 100+ GitHub stars in first 30 days
[ ] 5+ external contributors (not on core team)
[ ] Benchmark results published (AI and quantum)
[ ] 3+ tutorials published
[ ] Used by at least 2 external research groups
```

---

## Risk Register

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Linear types in branches complex to implement | High | Medium | Prototype region-based analysis in Phase 0; descope to subset without branches if needed |
| GNN predictor does not generalise | Medium | High | Analytical fallback model; ensemble; more training data |
| QPU access limits quantum testing | High | Medium | Use simulators for 95% of tests; batch real QPU runs monthly |
| CUDA backend performance gap vs TensorRT | Medium | Medium | Use cuBLAS / cuDNN for critical kernels; custom only where needed |
| Timeline slips | High | Medium | Buffer built in (24 months not 12); descope features before delaying |
| Noise composition in fusion incorrect | Medium | High | Use depolarising approximation in v1.0; full Kraus in v1.1; flag in docs |
| Python bindings break on PyTorch upgrade | Low | Low | Pin tested PyTorch versions; use FX stable API |
| Community adoption too slow | Medium | High | Partner with 2 research groups early; co-author paper; active Discord |

---

