# LIFT Framework — Complete Technical Design Document

**Version:** 0.1.0-alpha
**Status:** Research / Pre-Alpha
**Authors:** Martial-Christian

---

## Executive Summary

LIFT (Language for Intelligent Frameworks and Technologies) is a unified Intermediate Representation (IR) designed from the ground up to handle both Artificial Intelligence (neural networks, transformers, LLMs) and Quantum Computing (gate circuits, noise models, variational algorithms) within a single coherent framework.

The core thesis is: **AI computation and Quantum computation are structurally isomorphic in their compilation challenges.** Both require graph-based IR, both need hardware-aware optimisation, both need simulation before hardware execution, and both are moving toward hybrid execution models. LIFT exploits this isomorphism through a *twin dialect* architecture — two specialised sub-languages that can be freely composed.

---

## Part 1: The Problem Space

### 1.1 State of AI Compilation (2024)

The modern AI compiler stack is fragmented:

```
User writes PyTorch → torch.compile → TorchDynamo/FX → MLIR → various backends
                                                       ↓
                                      TorchInductor → Triton → PTX → H100
                                      TensorRT      → CUDA kernels
                                      XLA           → HLO → TPU backend
```

Each arrow is a lossy transformation. Information about the *intent* of the computation (this is a causal attention layer, this tensor represents key-value cache state) is lost at every step. The result: sub-optimal code because the compiler never knows the semantic context.

**Key problems:**

| Problem | Impact |
|---------|--------|
| Intent loss across IR levels | Missed fusion opportunities |
| No semantic types for AI primitives | Cannot specialise attention patterns |
| Static shapes only (partially fixed) | Poor dynamic batching |
| No energy awareness | Training GPT-4 costs $100M+ |
| No multi-target unified path | 3 different toolchains for GPU/TPU/CPU |

### 1.2 State of Quantum Compilation (2024)

Quantum compilation is even more fragmented:

```
Qiskit circuit → transpile() → layout mapping → routing → basis decomposition → run on IBM
PennyLane QNode → device.execute() → OpenQASM → Qiskit → IBM (via bridge)
Cirq circuit → cirq.optimize_for_target_gateset() → cirq_google → Google Sycamore
```

**Key problems:**

| Problem | Impact |
|---------|--------|
| No unified IR across providers | Lock-in to single vendor |
| Noise as afterthought | Poor fidelity predictions |
| No semantic types (just qubits) | Cannot reason about qubit roles |
| Manual error mitigation | Expert knowledge required |
| No simulation-first workflow | Expensive hardware time wasted |

### 1.3 The Hybrid Gap

The most critical gap: **no framework handles hybrid AI+Quantum workloads natively.**

Hybrid quantum-classical algorithms (VQE, QAOA, QNN, quantum-enhanced ML) are the most promising near-term quantum applications. Yet to implement one today, a researcher must:

1. Write the classical part in PyTorch
2. Write the quantum part in Qiskit or PennyLane
3. Write glue code to pass tensors between them
4. Write separate optimisation loops for each
5. Debug failures across two completely different frameworks
6. Deploy on two completely different backends

LIFT eliminates steps 3-5 and unifies 1-2 into a single file.

---

## Part 2: The Twin Dialects Concept

### 2.1 What Is a Dialect in LIFT?

A LIFT dialect is a namespaced extension of the core IR that adds:
- **New types** (e.g., `qubit`, `kvcache<...>`)
- **New operations** (e.g., `quantum.cx`, `tensor.flash_attention`)
- **New attributes** (e.g., noise model, memory layout)
- **New verification rules** (e.g., qubits cannot be cloned)
- **Lowering passes** (e.g., quantum.cx → OpenQASM CX gate)

This follows the MLIR dialect philosophy but with three key differences:
1. Dialects are designed to be **composed by default**, not as an afterthought
2. Dialects carry **semantic metadata** that persists through lowering
3. Dialects support **cross-dialect joint optimisation**

### 2.2 The Structural Isomorphism

The deepest insight in LIFT's design:

```
AI CONCEPT                          QUANTUM CONCEPT
──────────                          ───────────────
Tensor (vector of values)           Quantum state (vector of amplitudes)
Linear layer (matrix multiply)      Unitary gate (unitary matrix multiply)
Non-linearity (ReLU, softmax)       Measurement (projection, collapse)
Backpropagation (reverse AD)        Adjoint differentiation (parameter shift)
Batch dimension                     Shot parallelism (repeated execution)
INT8 quantisation                   Gate decomposition to native basis
Layer fusion                        Gate cancellation and merging
Memory layout (NCHW vs NHWC)        Qubit mapping (logical to physical)
Data parallelism (N GPUs)           Multi-QPU execution
Checkpoint for memory reduction     Mid-circuit measurement and reset
```

This isomorphism means that the **same algorithmic ideas** apply to both domains, just with different semantics. LIFT's twin dialects are *twins* precisely because they solve the same class of problems with different vocabularies.

### 2.3 LIFT-CORE: The Shared Foundation

Every dialect builds on LIFT-CORE. LIFT-CORE provides:

**Values (SSA)**
Every value in LIFT is defined exactly once. A `Value` carries:
- A unique identifier (`%v42`)
- A `CoreType` (the type of the value)
- An optional debug name
- A pointer to its defining `Operation`

**Operations**
An `Operation` is the fundamental unit of computation:
- A dialect namespace + operation name (`tensor.matmul`, `quantum.cx`)
- Zero or more input `Value`s
- One or more output `Value`s
- A set of `Attribute`s (compile-time constants)
- A `Location` (source file, line, column)

**Blocks**
A `Block` is a basic block in the SSA sense: a sequence of `Operation`s where control flow only enters at the top. Blocks carry block arguments (the SSA equivalent of φ-functions).

**Regions**
A `Region` is a list of `Block`s. Regions are used to represent nested structure (function bodies, loop bodies, branch arms).

**Functions and Modules**
A `Function` = a name + signature + one `Region`.
A `Module` = a name + a list of `Function`s + a list of `Global`s + metadata.

```
Module
└── Function @attention_layer
    └── Region
        ├── Block ^entry(%query: tensor<...>, %key: tensor<...>)
        │   ├── %scores = tensor.matmul(%query, %key) : ...
        │   ├── %masked = tensor.add(%scores, %mask) : ...
        │   ├── %weights = tensor.softmax(%masked) {dim=1} : ...
        │   └── br ^exit(%weights)
        └── Block ^exit(%result: tensor<...>)
            └── return %result
```

### 2.4 LIFT-TENSOR: The AI Dialect

LIFT-TENSOR extends LIFT-CORE with semantically rich AI types and operations.

**Type System**

The type system in LIFT-TENSOR is more expressive than ONNX or StableHLO:

```
TensorType ::=
  | Tensor<Shape, DType, Layout>       -- standard n-dimensional tensor
  | AttentionTensor<B, S, H, D, DType> -- typed for attention kernels
  | KVCache<MaxSeq, Heads, Dim, DType> -- typed for LLM inference
  | SparseTensor<NumExperts, Cap, DType>-- typed for MoE routing

Shape ::= [Dimension, ...]
Dimension ::= Constant(n) | Symbolic("batch") | Product([Dimension, ...])
DType ::= FP32 | FP16 | BF16 | FP8_E4M3 | FP8_E5M2 | INT8 | INT4 | INT2
Layout ::= Contiguous | NCHW | NHWC | Strided(strides) | Tiled(size)
```

The key insight: **`AttentionTensor` carries its semantic role in its type.** The compiler knows this is an attention computation, not just a generic matrix multiply. This enables:
- Automatic FlashAttention substitution
- Correct memory layout selection
- Accurate FLOP counting (attention is O(n²), not O(n))
- Correct gradient computation (attention backward has specific structure)

**Critical Operations**

```
tensor.attention {implementation, causal, scale}
  → Standard, FlashAttention v2/v3, PagedAttention, SDPA

tensor.paged_attention {block_tables, context_len, num_heads, head_dim}
  → vLLM-style paged KV cache attention

tensor.moe_dispatch {num_experts, num_active, capacity}
  → MoE routing with load balancing

tensor.quantize {quant_type, calibration, per_channel}
tensor.dequantize {original_type}
  → INT8/FP8 quantisation with proper rounding semantics

tensor.fused_op {pattern}
  → Explicitly fused operation (MatMul+Bias+ReLU, etc.)
```

**Gradient Representation**

LIFT-TENSOR represents both forward and backward computation in the same IR:

```
// Forward
%y = tensor.matmul(%A, %B) : (tensor<M×K>, tensor<K×N>) → tensor<M×N>

// Backward (attached as a region attribute)
// ∂L/∂A = ∂L/∂y @ B^T
// ∂L/∂B = A^T @ ∂L/∂y
%dA = tensor.matmul(%dy, %B) {transpose_rhs = true} : ...
%dB = tensor.matmul(%A, %dy) {transpose_lhs = true} : ...
```

This dual representation enables:
- Gradient checkpointing (recompute forward during backward to save memory)
- Gradient fusion (fuse forward + backward when profitable)
- Correct memory accounting for training vs inference

### 2.5 LIFT-QUANTUM: The Quantum Dialect

LIFT-QUANTUM extends LIFT-CORE with quantum computation primitives. The key design choices:

**Qubit Linearity**

Qubits satisfy the quantum no-cloning theorem. In LIFT-QUANTUM, qubit values are **linear types**: a qubit value can only be used exactly once. The type checker enforces this.

```
// ERROR: qubit used twice
%q0 = quantum.init() : qubit
%q1 = quantum.x(%q0) : qubit    // consumes %q0
%q2 = quantum.h(%q0) : qubit    // ERROR: %q0 already consumed

// CORRECT: SSA chain
%q0 = quantum.init() : qubit
%q1 = quantum.x(%q0) : qubit    // %q0 → %q1
%q2 = quantum.h(%q1) : qubit    // %q1 → %q2
%b0 = quantum.measure(%q2) : bit // %q2 → %b0 (qubit consumed, bit produced)
```

This linearity is not just a type system nicety — it reflects a **physical truth**: quantum information cannot be duplicated. The type checker makes this physical law a compile-time guarantee.

**Noise as a First-Class Type Attribute**

In LIFT-QUANTUM, every gate operation carries optional noise metadata:

```
%q1 = quantum.cx(%q0, %q_target) {
    gate_fidelity = 0.995,
    coherent_error = false,
    depolarizing_prob = 0.005,
    crosstalk_pairs = [[1, 2], [2, 3]]
} : (qubit, qubit) → (qubit, qubit)
```

This noise metadata is:
- **Used by the simulator** to produce realistic output distributions
- **Used by the optimiser** to route computation away from noisy qubits
- **Used by the predictor** to estimate circuit fidelity before QPU submission
- **Used by error mitigation passes** to select the right mitigation strategy

**Physical Qubit Representation**

LIFT-QUANTUM has two qubit types: logical (abstract) and physical (hardware-bound):

```
qubit                             // logical qubit: abstract, no hardware assumptions
physical_qubit<id=5, T1=150µs,   // physical qubit: bound to a specific hardware qubit
               T2=100µs,          // with measured coherence properties
               freq=5.1GHz>
```

The layout mapping pass converts logical qubits to physical qubits, inserting SWAP gates as needed. After mapping, the IR carries full hardware fidelity information.

**Hamiltonian Representation**

For variational algorithms (VQE, QAOA, quantum chemistry), LIFT-QUANTUM supports Hamiltonian types:

```
hamiltonian {
    terms = [
        {coeff = -1.0531, paulis = [(0, X), (1, X)]},
        {coeff = +0.3979, paulis = [(0, Z), (1, Z)]},
        {coeff = +0.3979, paulis = [(0, I), (1, Z)]}
    ]
}
```

This representation enables:
- Automatic Trotterisation for time evolution
- Pauli grouping for simultaneous measurement
- Sparse Hamiltonian simulation

### 2.6 LIFT-HYBRID: The Fusion Dialect

LIFT-HYBRID is the glue that makes AI+Quantum computation natural.

**Encoding Operations**

The fundamental challenge in quantum ML is getting classical data into quantum form. LIFT-HYBRID provides explicit, semantically typed encoding operations:

```
// Amplitude encoding: N classical values → log₂(N) qubits
// State: Σᵢ xᵢ/‖x‖ |i⟩
%qubits = hybrid.amplitude_encode(%features) {normalize = true}
    : (tensor<4xf32>) → (qubit, qubit)   // 4 features → 2 qubits (log₂4=2)

// Angle encoding: N classical values → N qubits, each rotated by xᵢ
// State: ⊗ᵢ cos(xᵢ/2)|0⟩ + sin(xᵢ/2)|1⟩
%qubits = hybrid.angle_encode(%features) {gate = RY}
    : (tensor<4xf32>) → (qubit, qubit, qubit, qubit)   // 4 features → 4 qubits
```

**Joint Gradient Computation**

The parameter shift rule enables gradient computation through quantum circuits:

```
// For a parameterised gate: d⟨O⟩/dθ = [⟨O⟩(θ+π/2) - ⟨O⟩(θ-π/2)] / 2
%gradient = hybrid.parameter_shift_gradient(%circuit, %params, %observable)
    : (function, tensor<Pxf64>, hamiltonian) → tensor<Pxf64>
```

LIFT-HYBRID makes this a single operation, allowing the compiler to:
- Batch forward and backward evaluations efficiently
- Combine parameter shift with classical auto-diff
- Optimise the number of QPU evaluations required

---

## Part 3: The Compilation Pipeline

### 3.1 The Four Phases

Every LIFT programme goes through four phases:

**Phase 1: Simulation (Static)**
No hardware involved. The programme is analysed statically:
- Types are checked and inferred
- Shapes are propagated through the computation graph
- Resource usage is estimated (FLOPs, memory, gate count, circuit depth)
- Symbolic execution identifies potential errors (shape mismatches, invalid qubit usage)
- Energy and carbon footprint are estimated

**Phase 2: Prediction (ML-based)**
A trained machine learning model (GNN on computation graphs) predicts:
- Wall-clock latency on the target hardware
- Memory peak usage
- GPU/QPU utilisation
- Quantum circuit fidelity (using the noise model)
- Whether the budget constraints in `.lith` will be satisfied

If the prediction shows budget violation, the compiler **refuses to proceed** and reports which constraint is violated, why, and how to fix it.

**Phase 3: Optimisation (IR Transformation)**
The optimisation pass pipeline transforms the IR:
- Each pass is a function `Module → Module` that preserves semantics
- Passes are ordered by the `.lith` configuration or auto-selected
- The optimiser verifies that budget predictions are met after each pass
- Failed passes (that worsen predictions) are rolled back

**Phase 4: Compilation (Code Generation)**
The optimised IR is lowered to target-specific code:
- LIFT-TENSOR → CUDA PTX / LLVM IR / XLA HLO
- LIFT-QUANTUM → OpenQASM 3 / native gate sets / pulse schedules
- LIFT-HYBRID → orchestration code (Python runtime + CUDA + OpenQASM)

### 3.2 Pass Infrastructure

Passes in LIFT are strongly typed Rust traits:

```rust
pub trait Pass: Send + Sync {
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
    fn required_dialects(&self) -> &[&'static str];
    fn run(&self, module: &mut Module, ctx: &PassContext) -> Result<PassResult>;
    fn is_applicable(&self, module: &Module, ctx: &PassContext) -> bool;
}

pub struct PassResult {
    pub changed: bool,
    pub statistics: PassStatistics,
    pub budget_impact: BudgetImpact,
}
```

The `BudgetImpact` field is critical: every pass reports its effect on the performance budget. The pass manager uses this to decide whether to accept or roll back a transformation.

### 3.3 Key AI Passes: Design Notes

**TensorFusionPass**

Pattern-matching approach: the pass maintains a library of fusion patterns as graph templates. It uses a modified Ullmann subgraph isomorphism algorithm to find matches, then replaces matched subgraphs with fused operations.

Patterns library (built-in):
- MatMul + Bias + ReLU → fused_matmul_bias_relu
- Conv2D + BatchNorm + ReLU → fused_conv_bn_relu
- LayerNorm (as computed) → fused_layer_norm
- Attention (QK^T/√d softmax V) → flash_attention (when profitable)
- RMSNorm (as computed) → fused_rms_norm

**FlashAttentionPass**

Replaces `tensor.attention {implementation=Standard}` with `tensor.attention {implementation=FlashAttention}` when:
- Sequence length > 512 (below this, standard attention is faster due to overhead)
- GPU has HBM bandwidth as bottleneck (always true for A100/H100)
- Causal mask is present (enables causal FlashAttention)

The transformation changes the memory complexity from O(n²) to O(n) — this is a semantic-preserving transformation (same mathematical result, different computation order).

**QuantizationPass**

Selects quantisation scheme based on:
- `.lith` configuration (INT8, FP8, INT4)
- Sensitivity analysis (which layers can be quantised without accuracy loss)
- Hardware support (FP8 requires H100 or A100 with special config)
- Calibration data (if provided, use static calibration; otherwise dynamic)

The pass inserts explicit `tensor.quantize` and `tensor.dequantize` operations, which are then lowered to hardware-specific implementations.

### 3.4 Key Quantum Passes: Design Notes

**LayoutMappingPass (SABRE)**

The SABRE (SWAP-based BidiREctional heuristic search) algorithm:

1. Create a dependency DAG from the circuit
2. Build a "front layer" of gates that can execute immediately
3. For each gate in the front layer:
   - If the two logical qubits are adjacent in the coupling map: execute directly
   - If not: score candidate SWAP insertions by their effect on future gates
4. Insert the best SWAP, update the mapping, repeat

LIFT's implementation of SABRE adds:
- Noise-aware SABRE: prefers routing through high-fidelity qubit pairs
- Predictive SABRE: uses the ML model to estimate the effect of each SWAP on circuit fidelity

**GateCancellationPass**

Algebraic identity matching:
- H · H = I → remove both H gates
- X · X = I → remove both X gates
- CX · CX = I (same control/target) → remove both
- Rz(θ) · Rz(φ) = Rz(θ+φ) → merge into single rotation
- S · S = Z, T · T = S, S · S · S = S† → simplify S/T chains

The pass uses a commutation table to determine when gates can be freely reordered to expose more cancellation opportunities.

**ErrorMitigationPass**

Selects error mitigation strategy based on:
- Circuit depth vs T₂ coherence time ratio
- Gate error rates from the noise model
- Available shots (ZNE requires ~3× more shots)
- Required fidelity from the budget

Mitigation strategies implemented:
- **ZNE (Zero Noise Extrapolation)**: Scale noise up (by gate folding or pulse stretching), extrapolate to zero noise. Effective for shallow circuits.
- **PEC (Probabilistic Error Cancellation)**: Decompose noisy gates into virtual ideal gates with negative probabilities. Unbiased but high overhead.
- **Readout error mitigation**: Measure all-zeros and all-ones states to characterise readout, apply linear inversion to correct measurement results.
- **Dynamical Decoupling**: Insert sequences of X gates during idle periods to suppress T₂ dephasing.

---

## Part 4: The .lith Configuration Language

### 4.1 Design Philosophy

The `.lith` language was designed with three goals:
1. **Replace 8+ config files with one** — every aspect of a LIFT project in a single file
2. **Self-documenting** — reading a `.lith` file should explain the project to a newcomer
3. **Type-safe** — configuration errors are caught before any computation starts

### 4.2 Grammar Overview

`.lith` is a superset of TOML with:
- Nested sections (like TOML tables)
- First-class arrays of heterogeneous objects
- Environment variable substitution (`${VAR_NAME}`)
- File inclusion (`include "./base.lith"`)
- Conditional blocks (`if hardware.qpu.provider == "ibm" { ... }`)
- Enum validation (the parser knows the valid values for each field)

### 4.3 Section Reference

**`project`**: Metadata only. Name, version, description, authors, tags, repository URL.

**`dialects`**: Which LIFT dialects the project uses and their versions. Enables forward compatibility checking.

**`compilation`**: Target hardware definition. Separate sub-sections for `gpu`, `qpu`, `cpu`. Each sub-section carries hardware-specific parameters (arch, memory, connectivity).

**`optimization`**: The optimisation pipeline as an ordered list of pass names, plus per-pass configuration objects. The pipeline can be overridden from the CLI.

**`prediction`**: Budget constraints. If predicted metrics violate any budget, compilation fails with a clear error message.

**`simulation`**: How to run the simulator (simulator type, GPU/CPU, number of trajectories for Monte Carlo noise simulation, benchmark scenarios).

**`metrics`**: Which metrics to collect, where to store them, how to visualise them.

**`logging`**: Log level, destinations, structured format, tracing configuration.

**`deployment`**: Container configuration, cloud instance type, edge platform targets.

**`security`**: Sandboxing, encryption, audit logging.

---

## Part 5: The Simulation and Prediction Engines

### 5.1 Static Simulation

Static simulation analyses the IR without executing it. It answers questions like:
- "What are the shapes of all intermediate tensors?"
- "How many FLOPs does this function perform?"
- "What is the maximum memory usage at any point?"
- "How deep is this quantum circuit?"
- "Are there any shape mismatches that will cause runtime errors?"

These questions are answered through a single forward pass over the IR using abstract interpretation: each operation is evaluated with abstract values (shapes, ranges, symbolic expressions) rather than concrete values.

### 5.2 Quantum Simulation

For quantum programmes, LIFT provides three simulation backends:

**State Vector Simulator**: Maintains the full 2ⁿ-dimensional complex state vector. Exact, but exponential in qubit count. Practical up to ~30 qubits on CPU, ~35 on GPU.

**Density Matrix Simulator**: Maintains the 2ⁿ × 2ⁿ density matrix, enabling simulation of mixed states and noise. Practical up to ~20 qubits. Used for accurate noise simulation.

**MPS (Matrix Product State) Simulator**: Represents the state as a tensor network with bounded bond dimension. Approximate but scales to 50-100 qubits for circuits with limited entanglement.

The simulator is selected automatically based on circuit size and noise requirements.

### 5.3 ML Performance Prediction

The prediction engine uses a **Graph Neural Network (GNN)** trained on a large dataset of (IR, hardware, performance) triples collected from real executions.

**Input to GNN:**
- Computation graph (operations as nodes, data dependencies as edges)
- Node features: operation type, tensor shapes, dtype, implementation variant
- Edge features: tensor sizes, number of bytes transferred
- Hardware features: memory bandwidth, compute throughput, topology

**Output:**
- Latency distribution (mean + variance)
- Memory peak
- Hardware utilisation
- For quantum: fidelity prediction using noise-aware graph convolution

**Training:**
The model is continuously retrained as new hardware and new optimisations are added. Users can contribute their execution traces to improve predictions.

---

## Part 6: The Energy and Reliability Models

### 6.1 Energy Modelling

Every operation in LIFT has an associated energy model:

```
AI Energy:
  matmul(M, N, K) in FP16: energy_joules = M × N × K × 0.35e-12  (picojoules per FLOP)
  memory_read(bytes): energy_joules = bytes × 0.2 / 1e9          (joules per GB)
  memory_write(bytes): energy_joules = bytes × 0.4 / 1e9

Quantum Energy:
  single_qubit_gate: energy_joules = 100e-12  (picojoules)
  two_qubit_gate: energy_joules = 500e-12
  measurement: energy_joules = 1000e-12
  cryogenic_overhead: energy_joules_per_circuit = base_power / measurement_rate
```

Total programme energy is summed across all operations. The carbon footprint is computed as:

```
carbon_gCO2 = total_energy_kWh × grid_intensity_gCO2_per_kWh
```

The grid intensity is configurable per region. If the energy budget in `.lith` is violated, compilation fails.

### 6.2 Reliability Modelling

LIFT models hardware failures at the IR level:

**For GPU:**
- Soft errors (single-bit memory corruption): ~10⁻⁶ per hour, handled by ECC
- Hard errors (permanent failure): ~0.01 per GPU per year, handled by checkpointing

**For QPU:**
- Qubit decoherence: modelled as exponential decay with T₁ and T₂ times
- Gate errors: modelled as depolarising channels with measured error probabilities
- Calibration drift: modelled as time-varying gate frequencies

The reliability model is used to:
- Automatically insert checkpointing at profitable intervals
- Recommend error correction codes for circuits that exceed decoherence times
- Alert when circuit depth approaches the T₂ coherence limit

---

## Part 7: Comparison with Existing Systems

### 7.1 vs MLIR

MLIR is the most direct comparison. LIFT's relationship with MLIR:

**MLIR strengths that LIFT adopts:**
- SSA form as foundation
- Dialect extensibility
- Pass infrastructure
- Region/Block/Operation hierarchy

**What LIFT adds beyond MLIR:**
- Native quantum dialect (MLIR has no quantum support)
- Noise-aware IR (no IR has this)
- Semantic AI types (AttentionTensor, KVCache — MLIR has generic tensors)
- Energy budgets as first-class concept
- .lith configuration language (MLIR requires C++ for everything)
- Prediction engine integrated with the compiler
- Hybrid AI+QC composition (not possible in MLIR today)

**LIFT is not a replacement for MLIR.** LIFT's CUDA backend can lower to MLIR/LLVM for the final native code generation step.

### 7.2 vs OpenQASM 3

OpenQASM 3 is a good quantum gate description language. But:
- It has no types beyond qubits, bits, and scalars
- It has no noise modelling in the language itself
- It has no connection to AI computation
- It has no optimisation framework
- It has no prediction capability

LIFT uses OpenQASM 3 as an *output format* (backend), not as an IR.

### 7.3 vs PennyLane

PennyLane is the closest existing tool to LIFT-HYBRID. It supports:
- Quantum circuits with differentiable parameters
- Gradient computation via parameter shift
- Multiple hardware backends
- Some AI+QC integration

What LIFT adds beyond PennyLane:
- Compilation and optimisation (PennyLane is a runtime, not a compiler)
- Static analysis and simulation before execution
- Noise-aware compilation (not just execution)
- Energy budgets
- Unified representation for both AI and quantum in one IR
- Hardware-agnostic IR (PennyLane is tied to specific device APIs)

### 7.4 vs Qiskit

Qiskit provides excellent quantum compilation tools for IBM hardware. But:
- IBM-specific (limited portability to other QPU vendors)
- No AI integration
- No unified configuration language
- No ML-based prediction
- No energy modelling
- Compilation is Python-based (slower, harder to optimise)

LIFT uses Qiskit Runtime as one of its *backends* for submitting to IBM hardware.

---

## Part 8: Implementation Notes

### 8.1 Why Rust?

Rust was chosen for the following reasons:

1. **Memory safety without garbage collection**: A compiler must handle large IRs without GC pauses. Rust's ownership system provides safety guarantees without runtime cost.

2. **Performance**: IR passes over large models (1T+ parameter models) require efficient data structures and cache-friendly traversals. Rust gives C-level performance.

3. **Algebraic type system**: Rust's enums and pattern matching map naturally to IR hierarchies (Operation variants, Type variants, Pass variants).

4. **FFI**: Rust's C FFI is straightforward, enabling Python bindings via PyO3/Maturin, and C bindings for integration with existing C/C++ compiler infrastructure.

5. **Ecosystem**: The Rust ecosystem has excellent libraries for the components LIFT needs: `petgraph` (graph algorithms), `salsa` (incremental computation), `rayon` (parallel passes), `lalrpop` (parser generation).

### 8.2 Data Structure Design

The IR is stored as a **generational arena**: all Values, Operations, Blocks, and Regions are stored in a flat arena indexed by generational IDs. This provides:
- O(1) lookup by ID
- Cache-friendly traversal
- Safe ID invalidation when operations are deleted
- No pointer chasing through linked lists

```rust
pub struct Context {
    values: Arena<ValueData>,
    ops: Arena<OperationData>,
    blocks: Arena<BlockData>,
    regions: Arena<RegionData>,
    types: TypeInterner,   // deduplicated type storage
    strings: StringInterner, // deduplicated string storage
}
```

All handles (Value, Operation, Block, Region) are opaque ID types. The Context is the single source of truth.

### 8.3 Pass Manager Design

The pass manager uses a **declarative dependency specification**: each pass declares what it requires (analysis results) and what it invalidates. The pass manager schedules analyses lazily and caches results across passes.

```rust
pub struct PassManager {
    passes: Vec<(Box<dyn Pass>, PassConfig)>,
    analysis_cache: AnalysisCache,
    budget: Budget,
    predictor: Arc<dyn Predictor>,
}

impl PassManager {
    pub fn run(&mut self, module: &mut Module) -> Result<PassManagerReport> {
        for (pass, config) in &self.passes {
            if !pass.is_applicable(module, &self.analysis_cache) {
                continue;
            }

            // Snapshot for rollback
            let snapshot = module.snapshot();
            let prediction_before = self.predictor.predict(module)?;

            // Run pass
            let result = pass.run(module, &mut self.analysis_cache)?;

            if result.changed {
                // Predict again after transformation
                let prediction_after = self.predictor.predict(module)?;

                // Check if transformation improved things
                if self.budget.is_violated(&prediction_after)
                   && !self.budget.is_violated(&prediction_before) {
                    // Rollback: this transformation made things worse
                    module.restore(snapshot);
                    continue;
                }

                // Invalidate stale analyses
                self.analysis_cache.invalidate(pass.invalidates());
            }
        }
        Ok(PassManagerReport { ... })
    }
}
```

---

## Part 9: Long-Term Vision

### 9.1 The 5-Year Horizon

```
2025: LIFT v1.0
  • Full twin dialect implementation
  • CUDA + OpenQASM backends
  • PyTorch + Qiskit importers
  • ML prediction engine
  • .lith language
  • 10 research groups using it

2026: LIFT v2.0
  • Multi-chip GPU (NVLink clusters) support
  • Photonic quantum computing backend
  • Neutral atom QPU backend
  • Auto-tuning via reinforcement learning
  • 50 research groups, first industrial users

2027: LIFT v3.0
  • Real-time adaptive compilation (runtime feedback)
  • Hardware digital twins (simulate chips not yet fabricated)
  • Standardisation effort (propose to ISO/IEC JTC1 or similar)
  • Integration into major cloud platforms (AWS Braket, IBM Quantum)
  • 500+ users, de facto standard for hybrid AI+QC

2029: LIFT as infrastructure
  • The same role LLVM plays for classical computing
  • Every major AI framework targets LIFT as intermediate representation
  • Every major QPU vendor supports LIFT as input format
  • The IR that defines the hybrid computing era
```

### 9.2 The Standard-Setting Ambition

LLVM became indispensable not by being the best in any one dimension, but by being the right abstraction at the right level — high enough to reason about programs, low enough to generate efficient code, and open enough that everyone contributed to it.

LIFT's ambition is the same for the AI+Quantum era:
- High enough to reason about attention mechanisms and quantum circuits
- Low enough to generate PTX and OpenQASM
- Open enough that GPU vendors, QPU vendors, AI labs, and researchers all contribute

The twin dialect architecture is the core insight that makes this possible. By recognising the structural isomorphism between AI and quantum computation, LIFT provides a unification that is not arbitrary — it reflects a deep truth about the mathematical structure of computation in both domains.

---

## Conclusion

LIFT is not a framework that improves on existing tools by 10% or 20%. It is a framework that makes **previously impossible things possible**: hybrid AI+Quantum programmes that compile, optimise, and execute on heterogeneous hardware from a single unified representation.

The timing is right. The technology is ready. The problem is real and growing.

The question is not whether a unified IR for AI+Quantum will exist. It is whether LIFT will be the one that defines the standard.

---

*This document is a living specification. It will be updated as implementation progresses.*

*LIFT — Where Artificial Intelligence Meets Quantum Reality.*