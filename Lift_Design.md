# LIFT — Complete Technical Design Document

**Version:** 0.2.0
**Status:** Research Alpha — Phase 0 complete, Phase 1 active
**Incorporates:** All critique from design review (correctness, scalability, security, cost model)

---

## Executive Summary

LIFT is a unified Intermediate Representation for AI and Quantum Computing. Its core thesis: both domains face structurally isomorphic compilation challenges. LIFT exploits this with twin dialects sharing one SSA foundation, one configuration language, and one four-step pipeline: Simulate → Predict → Optimise → Compile.

This document is the complete technical reference. It covers every design decision, its justification, known limitations, and open problems.

---

## Part 1: Problem Statement

### 1.1 AI Compilation: What Is Missing

Current AI IRs (MLIR, ONNX, StableHLO) are good at expressing tensor computations. They lack:

| Missing capability | Consequence |
|-------------------|-------------|
| Semantic types for AI primitives | Compiler cannot identify attention patterns to apply FlashAttention |
| Energy-aware compilation | Training GPT-4 costs $100M+ with no tooling to optimise this |
| Simulation before execution | Performance surprises discovered only at runtime |
| Budget enforcement | No automated check that latency / memory targets are met |

### 1.2 Quantum Compilation: What Is Missing

Current quantum IRs (OpenQASM 3, QASM, Qiskit's internal DAG) are good at expressing gate circuits. They lack:

| Missing capability | Consequence |
|-------------------|-------------|
| Noise in the type system | Fidelity only known after QPU execution |
| No-cloning enforcement (linear types) | Invalid programmes discovered at runtime |
| Cross-provider portability | Code written for IBM does not run on Rigetti without manual porting |
| Simulation-first workflow | Expensive QPU time wasted on programmes that will fail |

### 1.3 The Hybrid Gap

No existing tool handles hybrid AI+Quantum workloads at the IR level. PennyLane is the closest but is a runtime library, not a compiler IR: it provides no static analysis, no simulation-driven compilation, no noise-aware IR, and no joint optimisation of classical and quantum parameters at the IR level.

---

## Part 2: Design Principles

These principles guided every decision. When two approaches conflict, these principles determine which wins.

**P1 — Correctness over performance.** A compiler that produces wrong results faster is worse than one that is slow and correct. Every pass must be semantics-preserving. Every type rule must be sound.

**P2 — Honesty in the IR.** If information is known at compile time (noise model, tensor shape, qubit connectivity), it must be representable in the IR. Compilers that lose information cannot recover it.

**P3 — Simulation before execution.** The compiler must answer "what will happen?" before the hardware is touched. Budget violations must be compiler errors, not runtime surprises.

**P4 — Openness.** LIFT must interoperate with existing frameworks (PyTorch, Qiskit, ONNX) via importers and exporters. It must not require users to abandon their existing investments.

**P5 — Incremental adoption.** A user can import one PyTorch model into LIFT-TENSOR without touching quantum at all. A user can import one Qiskit circuit into LIFT-QUANTUM without touching AI. The dialects are independent and composable.

---

## Part 3: LIFT-CORE

### 3.1 The SSA IR

LIFT-CORE provides the shared foundation. All dialects build on it.

**Key data structures:**

```rust
// Context: the single source of truth. All IR objects are stored here.
pub struct Context {
    values:  SlotMap<ValueKey, ValueData>,   // O(1) lookup by key
    ops:     SlotMap<OpKey,    OperationData>,
    blocks:  SlotMap<BlockKey, BlockData>,
    regions: SlotMap<RegionKey, RegionData>,
    strings: StringInterner,   // deduplicated string storage
    types:   TypeInterner,     // deduplicated type storage
}

// A Value is defined exactly once (SSA invariant)
pub struct ValueData {
    ty:   TypeId,           // interned type reference
    name: Option<StringId>, // optional debug name
    def:  DefSite,          // where this value is defined
}

pub enum DefSite {
    OpResult { op: OpKey, result_index: u32 },
    BlockArg { block: BlockKey, arg_index: u32 },
}

// An Operation is the unit of computation
pub struct OperationData {
    name:     StringId,          // "tensor.matmul", "quantum.cx", etc.
    dialect:  StringId,
    inputs:   Vec<ValueKey>,
    results:  Vec<ValueKey>,
    attrs:    Attributes,        // compile-time constants
    regions:  Vec<RegionKey>,    // nested regions (for control flow)
    location: Location,
}
```

**Why SlotMap?** Generational arena gives O(1) lookup and safe invalidation when operations are deleted, without pointer chasing through linked lists. The `SlotMap` crate provides this with well-tested Rust semantics.

### 3.2 Type System

```rust
pub enum CoreType {
    Integer  { bits: u32, signed: bool },  // i8, i16, i32, i64, u8, ...
    Float    { bits: u32 },                // f16, f32, f64
    Boolean,
    Tuple    (Vec<TypeId>),
    Function { params: Vec<TypeId>, returns: Vec<TypeId> },
    Opaque   { dialect: StringId, name: StringId, data: TypeData }, // dialect-specific
    Void,
}
```

Dialects extend the type system by registering `Opaque` types. For example:
- `lift-tensor` registers `TensorType`, `AttentionTensorType`, `KVCacheType`
- `lift-quantum` registers `QubitType` (linear), `PhysicalQubitType`, `HamiltonianType`

**Type interning:** All types are interned in the `Context`. Structural equality is pointer equality on the interned ID. This makes type checking O(1) in practice.

### 3.3 Verification

The IR verifier checks:
- SSA: every value is defined before use
- Dominance: uses dominated by definitions
- Type consistency: operation inputs/outputs match declared types
- Linearity: qubit values not used more than once (see Section 4.4)
- Well-formedness: all keys are valid (not dangling references)

Verification is run after every pass as a debug-mode assertion. In release mode, verification is run once before code generation.

---

## Part 4: LIFT-TENSOR

### 4.1 Semantic Types

The key innovation over generic tensor IRs: types carry semantic meaning.

```rust
pub enum TensorType {
    // Standard tensor: shapes, dtype, layout
    Tensor {
        shape:  Vec<Dimension>,
        dtype:  DataType,
        layout: MemoryLayout,
    },

    // Semantically typed for attention kernels
    // The compiler KNOWS this is an attention computation.
    AttentionTensor {
        batch:      Dimension,
        seq_len:    Dimension,
        num_heads:  usize,
        head_dim:   usize,
        dtype:      DataType,
    },

    // Semantically typed for LLM inference
    // The compiler knows this is a KV cache.
    KVCache {
        max_seq:    Dimension,
        num_heads:  usize,
        head_dim:   usize,
        dtype:      DataType,
        is_paged:   bool,
    },

    // Semantically typed for Mixture-of-Experts
    SparseTensor {
        num_experts: usize,
        capacity:    usize,
        dtype:       DataType,
    },
}

pub enum Dimension {
    Constant(usize),         // known at compile time
    Symbolic(String),        // e.g., "batch_size", "seq_len"
    Product(Vec<Dimension>), // composite
}

pub enum DataType {
    FP64, FP32, FP16, BF16,
    FP8_E4M3, FP8_E5M2,     // H100 native FP8
    INT8, INT4, INT2,
}
```

**Why semantic types matter:** When the compiler sees `AttentionTensor`, it can automatically apply FlashAttention without needing a fragile pattern-matching heuristic. The type is the intent. This is the insight that MLIR's generic `memref` types cannot provide.

### 4.2 Attention Operations

```
tensor.attention {implementation, causal, scale, mask}
  Implementations:
    Standard        — O(n²) baseline
    FlashAttentionV2 — tiled, O(n) memory (Dao et al. 2022)
    FlashAttentionV3 — H100-optimised, ping-pong warp specialisation
    PagedAttention   — vLLM-style paged KV cache
    SDPA             — PyTorch scaled dot product attention

tensor.paged_attention {block_tables, context_len, num_heads, head_dim}
  For KV-cache-backed inference. Models the paging interface exactly.

tensor.moe_dispatch {num_experts, num_active, capacity}
tensor.moe_combine
  Expert routing + combination for Mixture-of-Experts models.
```

### 4.3 Fusion Pass: Correct Algorithm

**We do not use Ullmann subgraph isomorphism.** Ullmann is O(n!) in the worst case and is impractical for large graphs. We use **declarative pattern matching with topological search**.

```rust
// A fusion pattern is a small DAG template
pattern! {
    name = "matmul_bias_relu",
    ops  = [
        %m = "tensor.matmul"(%a, %b),
        %t = "tensor.add"(%m, %bias),
        %r = "tensor.relu"(%t)
    ],
    // Condition: %m and %t have no other uses outside the pattern
    condition = single_use(%m) && single_use(%t),
    replace_with = "tensor.fused_matmul_bias_relu"(%a, %b, %bias)
}
```

The matcher runs a topological traversal over the computation DAG. For each node, it checks all patterns whose "anchor op" matches. Pattern matching is O(V + E × P) where P is the number of patterns. In practice P < 50, making this linear.

**Correctness guarantee:** A fusion is only applied if all intermediate values are single-use (no other consumers). This ensures the fused result is observationally equivalent to the original.

### 4.4 Gradient Representation

LIFT-TENSOR represents both forward and backward computation in the same IR:

```
// Forward
%y = tensor.matmul(%A, %B)

// Backward — stored as region attributes on the forward op
// ∂L/∂A = ∂L/∂y @ Bᵀ
// ∂L/∂B = Aᵀ @ ∂L/∂y
%dA = tensor.matmul(%dy, %B) {transpose_rhs = true}
%dB = tensor.matmul(%A,  %dy) {transpose_lhs = true}
```

The dual representation enables:
- Gradient checkpointing (recompute forward during backward)
- Gradient fusion (fuse forward+backward when profitable)
- Correct memory accounting for training vs inference modes

### 4.5 Memory Management for Giant Models

LIFT supports three strategies for models too large to fit in GPU memory:

**Gradient checkpointing:** LIFT inserts checkpoint boundaries at profitable intervals (determined by memory/compute tradeoff analysis). Activations before the boundary are discarded and recomputed during the backward pass.

**Gradient accumulation:** For models where even checkpointing is insufficient, LIFT partitions the batch and accumulates gradients. The pass inserts explicit `tensor.grad_accumulate` operations.

**CPU/SSD offloading:** For 10T+ parameter models, LIFT can emit `tensor.offload` operations that move weights and optimiser state to CPU RAM or NVMe SSD, prefetching them ahead of use.

```
%w = "tensor.offload"(%w_gpu) {
    location  = "cpu",
    prefetch  = true,
    lookahead = 2     // prefetch 2 steps ahead
} : (tensor<...xf32>) -> tensor<...xf32>
```

The compiler selects the strategy automatically based on the memory budget in `.lith`.

**Scale targets:**

| Scale | Parameters | Strategy |
|-------|-----------|---------|
| Small | < 1B | Single GPU, full fit |
| Medium | 1B–100B | Multi-GPU, tensor + pipeline parallel |
| Large | 100B–1T | Distributed + checkpointing |
| Extreme | > 1T | MoE + CPU offload + gradient accumulation |

---

## Part 5: LIFT-QUANTUM

### 5.1 Qubit Types and Linearity

```rust
pub enum QuantumType {
    // Abstract logical qubit. Linear type: consumed exactly once.
    Qubit,

    // Physical qubit with hardware properties.
    PhysicalQubit {
        id:        usize,
        t1_us:     f64,   // relaxation time (microseconds)
        t2_us:     f64,   // dephasing time (microseconds)
        freq_ghz:  f64,   // qubit frequency
        fidelity:  f64,   // average gate fidelity
    },

    // Classical bit (result of measurement)
    ClassicalBit,

    // Quantum state (for simulation only, not physical execution)
    QuantumState {
        dimension:      usize,
        representation: StateRepr,
    },

    // Hamiltonian (for VQE, QAOA)
    Hamiltonian { num_qubits: usize },
}

pub enum StateRepr {
    StateVector,   // 2^n amplitudes — exact, exponential
    DensityMatrix, // 2^n × 2^n matrix — includes mixed states
    MPS,           // matrix product state — approximate, polynomial
    Stabiliser,    // Clifford-only — polynomial, exact
}
```

**Linear type enforcement:** The verifier maintains a `consumed: HashSet<ValueKey>`. When a gate operation consumes a qubit input, it is added to `consumed`. If the same qubit key appears again as an input, the verifier reports a linearity violation. Gate operations produce new qubit values (they do not modify in place), naturally enforcing the SSA + linear property together.

**Branch linearity:** If a conditional branch arm consumes a qubit in the `then` region but not the `else` region, this is a type error. The verifier checks that both arms consume the same set of qubit values.

### 5.2 Gate Operations with Noise Attributes

```
quantum.h   {fidelity=0.9997}  (%q0: qubit) -> qubit
quantum.x   {fidelity=0.9998}  (%q1: qubit) -> qubit
quantum.cx  {fidelity=0.9950, crosstalk=[(2,0.001)]}
            (%q0: qubit, %q1: qubit) -> (qubit, qubit)
quantum.rz  {fidelity=0.9999}  (%q0: qubit, %theta: f64) -> qubit
quantum.measure                 (%q0: qubit) -> bit
quantum.reset                   (%q0: qubit) -> qubit  // recycles qubit
quantum.barrier                 (%q0: qubit, %q1: qubit) // no-optimise fence
```

**Noise propagation:** The static analyser computes the accumulated error through a circuit using the noise model. For a circuit of depth D with average gate error p per layer, the estimated fidelity is approximately (1-p)^D. The full analysis uses a Pauli error propagation model.

### 5.3 Noise Composition After Fusion

When two consecutive gates are fused into one operation, the noise model of the fused operation must be derived. LIFT uses a **two-step approach**:

**Step 1 (initial implementation):** Depolarising approximation. If gate G1 has error p1 and gate G2 has error p2, the fused gate has error p_fused ≈ p1 + p2 - p1·p2 (series composition of independent channels).

**Step 2 (v1.1):** Full Kraus operator composition. The composite channel is computed as the product of Kraus operators. This is more accurate but requires matrix operations at compile time.

### 5.4 Layout Mapping: SABRE

The SABRE (SWAP-based Bidirectional heuristic search) algorithm maps logical qubits to physical qubits on a constrained coupling map.

**Noise-aware SABRE:** The standard SABRE scores SWAP candidates by their effect on future gate depths. LIFT's noise-aware variant additionally weights by the fidelity of the qubit pairs involved:

```
score(swap(q1, q2)) = depth_improvement(swap) 
                    × fidelity(q1, q2)
                    / (1 + swap_overhead)
```

This biases routing through high-fidelity qubit pairs, at the cost of slightly longer paths.

**SWAP insertion verification:** After layout mapping, the verifier checks that the mapped circuit is semantically equivalent to the original by simulating both on a small test input and comparing measurement distributions within a tolerance of 1e-5.

### 5.5 Error Mitigation Passes

**ZNE (Zero Noise Extrapolation):** The pass triplicates each gate (G → G G† G for 3× noise factor) and also runs the original circuit (1× factor). It then extrapolates the expected value observable to zero noise using Richardson extrapolation. Order is selected automatically: for circuits of depth < 20, linear extrapolation; for depth 20–100, quadratic; above 100, Richardson order 3.

**Readout error mitigation:** The pass inserts calibration circuits that prepare all-zeros and all-ones states and measure them. The resulting error matrix M (where M[i,j] = P(measure i | prepared j)) is inverted and applied to all measurement results.

**Dynamical Decoupling:** For each idle period of duration t on a qubit, if t > T2/10, the pass inserts an XY-4 sequence (X, Y, X, Y with appropriate delays) to suppress dephasing. Hardware timing constraints are respected by querying the device calibration.

---

## Part 6: LIFT-HYBRID

### 6.1 Data Encoding

The fundamental challenge: getting classical data into quantum form. LIFT-HYBRID provides typed encoding operations with explicit semantics:

```
// Amplitude encoding: N floats → log₂(N) qubits
// |ψ⟩ = Σᵢ xᵢ/‖x‖ |i⟩
hybrid.amplitude_encode (%x: tensor<N×f32>) {normalize=true} 
    -> (qubit × log₂N)

// Angle encoding: N floats → N qubits, each rotated by xᵢ
// |ψ⟩ = ⊗ᵢ cos(xᵢ/2)|0⟩ + sin(xᵢ/2)|1⟩
hybrid.angle_encode (%x: tensor<N×f32>) {gate=RY} 
    -> (qubit × N)

// Basis encoding: integer → computational basis state |x⟩
hybrid.basis_encode (%x: tensor<N×i32>) 
    -> (qubit × N)
```

### 6.2 Joint Gradient Computation

The parameter shift rule enables gradients through quantum circuits:

```
d⟨O⟩/dθ = [⟨O⟩(θ+π/2) - ⟨O⟩(θ-π/2)] / 2
```

LIFT-HYBRID expresses this as a single operation:

```
hybrid.parameter_shift_gradient
    (%circuit: function,
     %params:  tensor<P×f64>,
     %obs:     hamiltonian)
    -> tensor<P×f64>
```

The compiler lowers this into 2P circuit evaluations (plus and minus shifts for each of the P parameters), batches them for efficiency, and combines the results. The gradient can then be passed directly into a classical `tensor.adam_update` operation for joint optimisation.

---

## Part 7: The Compilation Pipeline

### 7.1 Simulation Phase

Static simulation answers without running hardware:

1. **Type inference:** Fill in any missing types from context.
2. **Shape propagation:** Forward pass computing output shapes for every operation.
3. **FLOP counting:** Per-operation FLOP formulae. Attention is O(n²·d), MatMul is O(m·n·k), etc.
4. **Memory liveness:** Track allocation and deallocation of all tensors. Compute peak memory.
5. **Bandwidth pressure:** Sum of (tensor_size × access_count) per operation.
6. **Noise accumulation (quantum):** Propagate Pauli error through circuit. Estimate final fidelity.
7. **Energy model:** Sum energy per operation × count, add infrastructure overhead.
8. **Carbon estimate:** energy_kWh × grid_intensity_gCO2_per_kWh.

**Energy model detail:**

| Operation | Energy per unit |
|-----------|----------------|
| FP16 MatMul | 0.35 pJ/FLOP (H100) |
| INT8 MatMul | 0.12 pJ/FLOP |
| Memory read | 0.2 J/GB (HBM3) |
| Single-qubit gate | ~100 pJ |
| Two-qubit gate | ~500 pJ |
| Measurement | ~1000 pJ |
| Cryogenic base load | ~5000 W (dilution refrigerator) |

### 7.2 Prediction Phase

**GNN Architecture:**

```
Input graph:
  Nodes: (op_type_onehot, tensor_shapes_log, dtype_onehot, impl_variant)
  Edges: (tensor_size_log, direction_flag)

Model:
  NodeEncoder:  Linear(node_feat_dim → 128)
  EdgeEncoder:  Linear(edge_feat_dim → 64)
  MessagePass:  6 × GatedGraphConv(128, 128)
  GlobalPool:   GlobalAttentionPooling(128 → 128)
  HWEncoder:    Linear(hw_feat_dim → 64)
  LatencyHead:  MLP(192 → 64 → 1)   [predicts log(latency_ms)]
  MemoryHead:   MLP(192 → 64 → 1)   [predicts log(memory_gb)]
  FidelityHead: MLP(192 → 64 → 1)   [quantum circuits only]

Training: 100K+ (IR graph, hw_spec, measured_latency) triples
Fallback: analytical roofline model when confidence < 0.7
```

**Budget enforcement:**

If any predicted metric violates the corresponding budget in `.lith`, compilation fails:

```
ERROR: Budget violation
  Predicted latency: 147.3ms (±12ms, 95% CI)
  Budget:            100ms
  Violation:         47ms over budget

  Suggested actions (estimated gains):
    1. Enable flash-attention pass:     -62ms → 85ms  ✓ would satisfy
    2. Reduce sequence length 2048→512: -80ms → 67ms  ✓ would satisfy
    3. Enable INT8 quantisation:        -44ms → 103ms ✗ still over
```

### 7.3 Optimisation Phase

The pass manager uses **budget-aware rollback**: if a pass degrades the predicted budget metrics, it is rolled back automatically.

```rust
impl PassManager {
    fn run_pass(&mut self, pass: &dyn Pass, module: &mut Module) -> PassResult {
        let snapshot   = module.snapshot();
        let pred_before = self.predictor.predict(module);

        let result = pass.run(module, &mut self.analysis_cache);

        if result.changed {
            let pred_after = self.predictor.predict(module);

            // If this pass made a budget violation WORSE, roll back
            if self.budget.violation_worsened(&pred_before, &pred_after) {
                module.restore(snapshot);
                return PassResult::rolled_back();
            }

            self.analysis_cache.invalidate(pass.invalidates());
        }

        result
    }
}
```

**Pass ordering:** Passes are ordered in `.lith`. We recommend:
1. Canonicalise and constant-fold first (simplify the IR before expensive analyses)
2. Fusion passes second (reduce memory bandwidth)
3. Quantisation third (changes types, so must come before layout-sensitive passes)
4. Quantum passes fourth (gate cancellation, then layout)
5. Error mitigation last (adds overhead, do not cancel it with gate cancellation)
6. Hybrid passes at the end (depend on both AI and quantum passes being done)

### 7.4 Code Generation

**CUDA backend:** Generates PTX assembly. For tensor operations, uses cuBLAS for MatMul, custom FlashAttention kernels for attention, and cuDNN for convolution. The backend emits kernel launch configurations (block size, grid size, shared memory) derived from the hardware spec in `.lith`.

**OpenQASM 3 backend:** Emits standard OpenQASM 3.0. Gate decompositions are hardware-specific: IBM uses `{RZ, SX, X, CX}` basis, Rigetti uses `{RZ, RX, CZ}` basis. The backend queries the device coupling map to emit only native two-qubit gates between physically connected pairs.

**LLVM backend:** For CPU targets. Uses LLVM 17+ IR. The backend annotates with SIMD target features (AVX-512) and emits OpenMP parallel regions for multi-core tensor operations.

---

## Part 8: Semantic Correctness

**This is the most important section for academic credibility.**

### 8.1 What "Semantics-Preserving" Means

A transformation T is semantics-preserving if for all inputs I:

```
  execute(original_IR, I) ≈ execute(T(IR), I)
```

where ≈ means: same output values within floating-point tolerance (for AI) or same probability distribution within statistical tolerance (for quantum).

### 8.2 Correctness Mechanisms

**Type system soundness:** Type-preserving transformations cannot introduce type errors. The type system is defined formally as a set of inference rules. We prove (by structural induction on the operation set) that every pass produces a well-typed IR if its input is well-typed.

**Invariant preservation:** Each pass declares the invariants it preserves and those it may invalidate. The pass manager checks invariants after each pass in debug mode. Current invariants tracked:
- SSA property (every value defined once)
- Qubit linearity (every qubit consumed once)
- Shape consistency (all shapes resolved)
- Dominance (all uses dominated by definitions)

**Regression testing:** 5,000+ reference programmes. Before releasing any pass, we run all reference programmes through the pass and compare outputs (within tolerance) against a trusted reference implementation (PyTorch for AI, Qiskit StateVector simulator for quantum).

**Formal verification (roadmap):** For the most critical passes (tensor fusion, layout mapping), we plan to add proof-carrying passes: each transformed IR includes a certificate that the verifier can check in O(|IR|) time. This is a v2.0 research goal, not v1.0.

### 8.3 Known Unsound Cases (Documented)

**Quantisation:** INT8 quantisation is inherently lossy. The output of a quantised model will differ numerically from the original. This is by design and is flagged explicitly in the compilation report:

```
Warning: quantisation pass applied. Outputs will differ from FP32 baseline.
  Expected accuracy impact: < 0.5% on typical classification tasks.
  Validate on your task before deploying.
```

**ZNE extrapolation:** Zero Noise Extrapolation assumes the observable varies smoothly with noise factor. This assumption fails for highly non-linear observables. LIFT flags when the R² of the extrapolation fit is below 0.95.

---

## Part 9: Interoperability

### 9.1 Import/Export Matrix

| Framework | Import → LIFT | Export ← LIFT | Status |
|-----------|--------------|--------------|--------|
| PyTorch FX | `torch.fx.Graph` → LIFT-TENSOR | — | 80% done |
| ONNX | opset 19 → LIFT-TENSOR | LIFT-TENSOR → ONNX opset 19 | 40% done |
| TensorFlow | `tf.function` → LIFT-TENSOR | — | Planned |
| Qiskit | `QuantumCircuit` → LIFT-QUANTUM | LIFT-QUANTUM → OpenQASM 3 | Design |
| Cirq | `Circuit` → LIFT-QUANTUM | OpenQASM 3 | Planned |
| PennyLane | `QNode` → LIFT-HYBRID | — | Planned |
| OpenQASM 3 | QASM parser → LIFT-QUANTUM | LIFT-QUANTUM → OpenQASM 3 | 60% done |

### 9.2 Python Bindings (PyO3)

The Python API is the primary user-facing interface for researchers:

```python
import lift

# Import a PyTorch model
model = MyModel()
lift_ir = lift.from_pytorch(model, example_inputs=(torch.zeros(1, 784),))

# Compile
result = lift.compile(lift_ir, config="project.lith")

# Import a Qiskit circuit
from qiskit import QuantumCircuit
qc = QuantumCircuit(4)
qc.h(0); qc.cx(0, 1)
lift_ir = lift.from_qiskit(qc)

# Analyse
report = lift.analyse(lift_ir)
print(report.circuit_depth, report.gate_count)
```

**Python bindings from Phase 0:** Python bindings are not an afterthought — they are scaffolded in Phase 0 so that researchers can use LIFT from Python as soon as Phase 1 features are available. The initial bindings expose `analyse`, `verify`, `print`, and basic LLVM compilation.

### 9.3 C API

For integration with C/C++ tools (PyTorch C++ API, TensorFlow C API, custom hardware SDKs):

```c
// lift.h
typedef struct LiftContext LiftContext;
typedef struct LiftModule  LiftModule;

LiftContext* lift_context_new(void);
void         lift_context_free(LiftContext*);

LiftModule*  lift_parse(LiftContext*, const char* source, size_t len);
void         lift_module_free(LiftModule*);

int          lift_verify(LiftModule*);       // 0 = ok, 1 = error
char*        lift_print(LiftModule*);        // caller frees
char*        lift_analyse_json(LiftModule*); // caller frees
```

---

## Part 10: Security Model

### 10.1 Threat Model

| Threat | Risk | Mitigation |
|--------|------|------------|
| Malicious .lif file causing parser crash | Medium | Sandboxed parsing with memory limits and timeouts |
| Integer overflow in shape arithmetic | Low | All shape computations use checked arithmetic |
| Generated CUDA code with buffer overflows | Low | All tensor accesses are bounds-checked in IR; backend enforces bounds |
| Supply chain compromise | Medium | `cargo audit` weekly, dependency pinning, signed releases |
| Arbitrary code in .lith config | Low | .lith is data, not code; no `eval` semantics |

### 10.2 Sandboxed Compilation

For untrusted inputs (e.g., a cloud API accepting user-submitted .lif files), the compiler runs in a sandbox:

```toml
# Sandbox configuration
[sandbox]
type           = "seccomp"     # or "docker", "nsjail"
memory_limit   = "4GB"
time_limit_s   = 60
allowed_syscalls = ["read", "write", "mmap", "munmap", "exit"]
no_network     = true
```

### 10.3 Generated Code Safety

- All tensor memory accesses in the generated CUDA are bounds-checked in debug mode.
- All qubit indices in generated OpenQASM are validated against the device qubit count before submission.
- No `unsafe` code in `lift-core`, `lift-tensor`, `lift-quantum`, or `lift-hybrid`. `unsafe` is allowed only in `lift-export` (CUDA FFI) with explicit safety documentation.

### 10.4 Audit Trail

Every compilation is logged with:
- SHA256 hash of the input .lif source
- SHA256 hash of the .lith config
- Compiler version
- Predicted and actual performance metrics (if available)
- Pass pipeline applied and results

---

## Part 11: Cost Model

### 11.1 Compilation Cost

| Task | Hardware | Estimated time |
|------|---------|---------------|
| Compile 7B-parameter model | 16-core CPU | 5–10 minutes |
| Compile 100-qubit circuit | 8-core CPU | 30–60 seconds |
| GNN prediction query | CPU (ONNX runtime) | < 100ms |
| Full pass pipeline on 1B model | 16-core CPU | 2–3 minutes |

### 11.2 Predicted Execution Cost

LIFT reports the predicted economic cost of execution:

```
$ lift predict model.lif --target aws --hardware ibm_kyoto
──────────────────────────────────────────────────────────
COST PREDICTION
  GPU (p4d.24xlarge spot):  $0.47 per 1M tokens
  QPU (IBM Kyoto):          $12.40 per 1000 shots
  Total per inference:      $12.87

BUDGET CHECK
  Budget: $10.00
  Predicted: $12.87 — OVER by $2.87 (28.7%)

  Suggested: reduce shots 4096→500
    New cost: $8.20 (fidelity drops 99.1%→91.8%)
──────────────────────────────────────────────────────────
```

### 11.3 Carbon Footprint

```
CARBON ESTIMATE
  Compute: 0.003 kWh × 350 gCO₂/kWh = 1.05 gCO₂ (us-east-1)
  Cryogenic overhead (QPU): 0.012 kWh × 350 = 4.20 gCO₂
  Total per inference: 5.25 gCO₂

  Equivalent: 0.013 km car driving
  To stay under 1 gCO₂ per inference:
    → Use us-west-1 (150 gCO₂/kWh): 2.25 gCO₂ per inference
    → Reduce QPU shots to 100: 1.05 gCO₂ per inference
```

---

## Part 12: Scalability

### 12.1 Compiler Scalability

For very large models (1T+ parameters), the compiler itself must scale:

**Incremental compilation:** Only recompile layers that changed. Unchanged layers are retrieved from the compilation cache (keyed by layer content hash). For a 1T parameter model with 128 layers, modifying one layer requires recompiling 1/128 of the model.

**Distributed compilation:** The pass pipeline can be parallelised using `rayon`. Independent passes (those with no data dependency) run in parallel across CPU cores. For a 32-core machine, compilation of a large model is roughly 8–12× faster than single-threaded.

**Lazy analysis:** Analyses are computed on demand and cached. If a pass does not need shape information, shapes are not propagated.

### 12.2 Distributed Execution

For models that must run across multiple GPUs or multiple QPUs:

```lith
compilation {
    target {
        type = "distributed"

        classical {
            nodes         = 16
            gpu_per_node  = 8
            strategy {
                tensor_parallel   = 8
                pipeline_stages   = 4
                data_parallel     = 4
            }
        }
    }
}
```

LIFT inserts explicit `tensor.parallel_split`, `tensor.parallel_allreduce`, and `tensor.pipeline_send`/`receive` operations in the IR for distributed execution, making the communication pattern explicit and optimisable.

---

## Part 13: Testing Strategy

### 13.1 Unit Tests (per crate)

Each operation: type checking, shape inference, FLOP counting, printer, parser round-trip.
Each pass: at least 15 test cases (apply, do not apply, edge cases, rejects incorrectly).
Target: 80% code coverage on all crates.

### 13.2 Integration Tests

End-to-end: .lif source → parse → verify → passes → backend → compare against reference.
Cross-dialect: tensor + quantum + hybrid in the same file.
Regression: 5,000+ programmes that must compile and produce correct results.

### 13.3 Quantum Tests

Compare LIFT simulator output against Qiskit StateVector simulator.
Tolerance: 1e-5 for noiseless circuits, 1e-2 for noisy circuits.
Weekly run on real QPU to detect calibration drift.

### 13.4 Performance Tests

Compilation time: no pass is allowed to increase compilation time by more than 10%.
Generated code quality: no pass is allowed to decrease execution performance by more than 5%.
Memory: peak compiler memory < 2GB for 7B-parameter models.

---

## Part 14: Maintenance Plan

### 14.1 Release Cadence

| Release type | Frequency | Supported for |
|-------------|-----------|--------------|
| Nightly | Daily | Not supported |
| Alpha | Weekly | 2 weeks |
| Beta | Monthly | 3 months |
| Stable | Quarterly | 18 months |
| LTS | Every 2 years | 3 years |

### 14.2 Deprecation Policy

1. Announce deprecation in release notes with migration guide.
2. Emit compiler warning when deprecated feature is used.
3. Remove after one full stable release cycle (minimum 3 months notice).

### 14.3 Security Policy

- Security contact: security@lift-framework.org
- 90-day coordinated disclosure.
- All releases signed with GPG.
- `cargo audit` runs on every CI build.

---

## Conclusion

LIFT's design is guided by correctness, honesty, and incremental adoption. The twin dialect architecture is not an aesthetic choice — it is a recognition that the same class of problems appears in AI and quantum compilation, and that solving them once in a unified IR is better than solving them twice in separate tools.

The open design problems (linear types in branches, noise composition after fusion, GNN predictor generalisation) are documented honestly. They are solvable; they are not blockers for Phase 1. They will be addressed in sequence as the implementation matures.

The architecture is ready. The foundations are correct. The implementation begins.

---

