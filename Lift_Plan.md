# LIFT — Complete Implementation Plan

**Document Type:** Engineering Roadmap
**Status:** Living Document

---

## Overview: 48-Week Plan to First Public Release

```
WEEK  1─────4   PHASE 0: LIFT-CORE
WEEK  5────10   PHASE 1: LIFT-TENSOR (AI Dialect)
WEEK 11────18   PHASE 2: LIFT-QUANTUM (Quantum Dialect)
WEEK 19────24   PHASE 3: LIFT-HYBRID (Fusion Dialect)
WEEK 25────30   PHASE 4: Simulation + Prediction Engine
WEEK 31────38   PHASE 5: Backends + Interoperability
WEEK 39────44   PHASE 6: Tooling + Observability
WEEK 45────48   PHASE 7: Documentation + Public Release
```

---

## Phase 0: LIFT-CORE (Weeks 1–4)

### Milestone 0.1 — IR Data Structures (Week 1–2)

**Goal:** All core data structures compile and basic IR can be constructed in memory.

**Tasks:**
```
[ ] Context struct (generational arena for all IR objects)
[ ] Value struct (ID + type + name)
[ ] CoreType enum (Integer, Float, Boolean, Tuple, Function, Opaque, Void)
[ ] Attribute struct (key-value store for compile-time constants)
[ ] Location struct (file, line, column for debug info)
[ ] Operation struct (name, dialect, inputs, outputs, attributes, location)
[ ] Block struct (ID, operations list, block arguments)
[ ] Region struct (blocks list, entry block pointer)
[ ] Function struct (name, signature, region)
[ ] Module struct (name, functions, globals, dialect registry)
```

**Test criteria:**
- Can construct a simple module with a function containing 3 operations
- Can print the module to a string in human-readable form
- Valgrind / ASAN shows no memory errors
- No panics on valid inputs

**Estimated time:** 2 developers × 2 weeks

---

### Milestone 0.2 — Parser and Printer (Week 2–3)

**Goal:** Round-trip: `.lif` text → IR → `.lif` text produces identical output.

**Tasks:**
```
[ ] Lexer (tokenise .lif source files)
      Tokens: identifiers, numbers, strings, punctuation, keywords
[ ] Parser (recursive descent, produce AST)
      module, func, block, operation, value, type, attribute
[ ] AST → IR builder (convert parse tree to LIFT IR)
[ ] IR Printer (emit human-readable .lif text from IR)
[ ] Round-trip test: parse → build → print → re-parse → compare
[ ] Error recovery: parser continues after errors, collects all errors
```

**Grammar (subset):**
```
module   ::= 'module' '@' ident '{' function* '}'
function ::= 'func' '@' ident '(' arg-list ')' '->' type-list '{' block+ '}'
block    ::= '^' ident '(' arg-list ')' ':' operation*
operation::= ('%' ident (',' '%' ident)* '=')? string '(' value-list ')' 
             ('{' attr-list '}')? ':' type-list '->' type-list
```

**Test criteria:**
- 20 hand-written .lif files parse without errors
- Round-trip test passes for all 20 files
- Error messages include file, line, column

---

### Milestone 0.3 — Core Passes (Week 3–4)

**Goal:** A pass pipeline can be constructed, run, and verified.

**Tasks:**
```
[ ] Pass trait definition (name, run, is_applicable, invalidates)
[ ] PassManager (sequential pass runner with analysis cache)
[ ] ConstantFoldingPass (eval ops on constant inputs at compile time)
[ ] DeadCodeEliminationPass (remove unreachable ops and functions)
[ ] CanonicalisationPass (normalise IR patterns to canonical form)
[ ] IR Verifier (check well-formedness: SSA, types, block dominance)
[ ] Unit test framework (compare IR before/after each pass)
```

**Test criteria:**
- Each pass has ≥ 10 unit tests
- IR verifier catches all malformed IRs in the test suite
- PassManager runs without panics on all test modules

---

## Phase 1: LIFT-TENSOR (Weeks 5–10)

### Milestone 1.1 — Tensor Type System (Week 5–6)

**Goal:** All tensor types can be expressed and type-checked.

**Tasks:**
```
[ ] TensorType (shape: Vec<Dimension>, dtype: DataType, layout: MemoryLayout)
[ ] AttentionTensor type (batch, seq_len, num_heads, head_dim, dtype)
[ ] KVCache type (max_seq, num_heads, head_dim, dtype, is_paged)
[ ] SparseTensor type (num_experts, capacity, dtype)
[ ] Dimension enum (Constant, Symbolic, Product)
[ ] DataType enum (FP32, FP16, BF16, FP8_E4M3, FP8_E5M2, INT8, INT4, INT2)
[ ] MemoryLayout enum (Contiguous, NCHW, NHWC, Strided, Tiled, Blocked)
[ ] Type printer (e.g., "tensor<1x32x128xf16>")
[ ] Type parser (parse type expressions from .lif text)
[ ] Shape inference rules (per-operation shape propagation)
```

---

### Milestone 1.2 — Core AI Operations (Week 6–7)

**Tasks:**
```
[ ] tensor.add, tensor.mul, tensor.sub, tensor.div
[ ] tensor.matmul (with transpose flags)
[ ] tensor.conv2d (stride, padding, dilation, groups)
[ ] tensor.relu, tensor.gelu, tensor.silu, tensor.sigmoid
[ ] tensor.softmax (with dim attribute)
[ ] tensor.layer_norm, tensor.rms_norm, tensor.batch_norm
[ ] tensor.embedding (lookup table)
[ ] tensor.reshape, tensor.transpose, tensor.concat, tensor.split
[ ] tensor.constant (compile-time tensor values)
[ ] tensor.linear (matmul + bias, fused)
```

Each operation implements:
- Shape inference (what is the output shape given input shapes?)
- Type checking (are input types valid?)
- FLOP counting (how many FLOPs does this operation perform?)
- Memory footprint (how much memory does this operation require?)

---

### Milestone 1.3 — Attention and LLM Operations (Week 7–8)

**Tasks:**
```
[ ] tensor.attention {implementation, causal, scale, mask}
      Implementation variants: Standard, FlashAttentionV2, FlashAttentionV3, SDPA
[ ] tensor.paged_attention {block_tables, context_len, ...}
      For vLLM-style inference
[ ] tensor.moe_dispatch {num_experts, num_active, capacity}
[ ] tensor.moe_combine (reverse of dispatch)
[ ] tensor.speculative_decode {draft_model, target_model, k}
[ ] tensor.quantize {quant_type, per_channel, symmetric}
[ ] tensor.dequantize {original_type}
[ ] tensor.fused_op {pattern, inputs, outputs}
[ ] Gradient operations (tensor.grad_matmul, tensor.grad_relu, ...)
[ ] tensor.checkpoint {fn, inputs} (recomputation checkpoint)
```

---

### Milestone 1.4 — AI Optimisation Passes (Week 8–10)

**Tasks:**
```
[ ] TensorFusionPass
      - Pattern library: MatMul+Bias+Act, Conv+BN+ReLU, etc.
      - Subgraph isomorphism matching (Ullmann algorithm)
      - Profitability analysis (only fuse if it reduces memory or latency)

[ ] FlashAttentionPass
      - Detect tensor.attention {implementation=Standard}
      - Check applicability conditions (seq_len threshold, hardware)
      - Replace with flash_attention variant

[ ] KVCachePass
      - Detect attention patterns in inference mode
      - Insert KVCache type annotations
      - Transform to tensor.paged_attention when profitable

[ ] QuantizationPass
      - Dynamic INT8 (calibration at runtime)
      - Static INT8 (calibration from provided dataset)
      - FP8 (H100 / A100 with appropriate flags)
      - Per-channel vs per-tensor

[ ] ParallelismPass
      - Detect data-parallelism opportunities
      - Insert tensor.parallel_split / tensor.parallel_reduce
      - Annotate with parallelism strategy (DP/TP/PP)

[ ] MemoryPlanningPass
      - Liveness analysis
      - Buffer reuse (assign non-overlapping lifetimes to same memory)
      - Memory pooling
```

**Test criteria for each pass:**
- At least 15 unit tests
- At least 3 end-to-end tests (real-world model fragment)
- Performance regression tests (pass must not degrade performance on reference models)

---

## Phase 2: LIFT-QUANTUM (Weeks 11–18)

### Milestone 2.1 — Quantum Type System (Week 11–12)

**Tasks:**
```
[ ] Qubit type (logical, linear — can only be used once)
[ ] PhysicalQubit type (id, T1, T2, frequency, gate_fidelity)
[ ] ClassicalBit type
[ ] QuantumState type (dimension, representation: SV / DM / MPS / Stabiliser)
[ ] Hamiltonian type (Vec<PauliTerm>)
[ ] PauliTerm struct (coefficient: Complex<f64>, paulis: Vec<(qubit_id, Pauli)>)
[ ] Pauli enum (I, X, Y, Z)
[ ] NoiseModel struct (gate_errors, T1, T2, crosstalk, readout_error)
[ ] GateError struct (probability, type, coherent)
[ ] QuantumTopology struct (coupling_map, gate_fidelity, gate_time)
[ ] Qubit linearity checker (verifier pass: no qubit used twice)
```

---

### Milestone 2.2 — Gate Operations (Week 12–13)

**Tasks:**
```
[ ] Single-qubit gates: H, X, Y, Z, S, Sdg, T, Tdg, SX, SXdg
[ ] Rotation gates: RX(θ), RY(θ), RZ(θ), P(λ), U1(λ), U2(φ,λ), U3(θ,φ,λ)
[ ] Two-qubit gates: CX, CZ, CY, SWAP, iSWAP, ECR, RZX(θ), XX(θ), YY(θ), ZZ(θ)
[ ] Three-qubit gates: Toffoli (CCX), Fredkin (CSWAP)
[ ] Parametrised gate (gate_type, qubits, parameters: Vec<Value>)
[ ] quantum.measure {basis} → ClassicalBit
[ ] quantum.measure_all → tensor<n×bit>
[ ] quantum.reset
[ ] quantum.barrier {qubits} (prevent optimisation across barrier)
[ ] quantum.delay {duration, unit} (explicit idle time)
[ ] quantum.init {num_qubits} → qubit×n
```

---

### Milestone 2.3 — Noise Modelling (Week 13–14)

**Tasks:**
```
[ ] NoiseModel parser (from JSON calibration files)
[ ] IBM device calibration loader (from IBM Quantum API)
[ ] Rigetti device calibration loader
[ ] Depolarising noise channel representation
[ ] Amplitude damping channel representation
[ ] Phase damping channel representation
[ ] Pauli error channel representation
[ ] Crosstalk model (ZZ coupling between neighbouring qubits)
[ ] Readout error matrix
[ ] Noise-annotated gate operations (gate + noise channel combined)
[ ] Noise propagation analysis (how noise accumulates through circuit)
```

---

### Milestone 2.4 — Error Mitigation Passes (Week 14–15)

**Tasks:**
```
[ ] ZNE (Zero Noise Extrapolation) pass
      - Gate folding (replace G → G G† G for 3× noise scaling)
      - Pulse stretching (scale gate duration, requires pulse-level access)
      - Richardson extrapolation to zero noise
      - Automatic order selection based on circuit depth / noise

[ ] PEC (Probabilistic Error Cancellation) pass
      - Compute quasi-probability decomposition of noisy gates
      - Insert sampling overhead estimation
      - Annotate with overhead factor

[ ] ReadoutErrorMitigation pass
      - Insert all-zeros / all-ones calibration circuits
      - Compute correction matrix
      - Apply matrix inversion to measurement results

[ ] DynamicalDecoupling pass
      - Identify idle periods in the circuit
      - Insert XY-4, CPMG, or UR sequences
      - Respect hardware timing constraints
```

---

### Milestone 2.5 — Layout Mapping (Week 15–17)

**Tasks:**
```
[ ] QuantumTopology loader (from device specification)
[ ] Trivial layout (identity mapping, no SWAP)
[ ] SABRE layout mapping
      - Front layer computation
      - SWAP scoring heuristic
      - Bidirectional search
      - Noise-aware variant (prefer high-fidelity qubit pairs)

[ ] A* layout search (for small circuits, exact minimum SWAP)
[ ] SWAP insertion verification (circuit preserves semantics after mapping)
[ ] Layout mapping statistics (SWAP count, circuit depth increase, fidelity estimate)
```

---

### Milestone 2.6 — Gate Optimisation Passes (Week 17–18)

**Tasks:**
```
[ ] GateCancellationPass
      - Algebraic identities: H·H=I, X·X=I, CX·CX=I
      - Commutation table for reordering
      - Peephole window optimisation

[ ] RotationMergingPass
      - Merge consecutive same-axis rotations: Rz(a)·Rz(b) = Rz(a+b)
      - Handle phase wrapping (angles modulo 2π)

[ ] GateDecompositionPass
      - Decompose non-native gates to hardware basis
      - IBM basis: {U, CX}, {RZ, SX, X, CX}, {RZ, X, SX, CX, CZ, ECR}
      - Rigetti basis: {RZ, RX, CZ}
      - Solovay-Kitaev approximation for arbitrary unitaries

[ ] TwoQubitWeylDecompositionPass
      - Exact decomposition of arbitrary 2-qubit unitaries
      - Cartan (KAK) decomposition
      - Reduce to at most 3 CNOT gates

[ ] TemplateMatchingPass
      - Library of circuit templates with known optimised equivalents
      - Subgraph matching and replacement
```

---

## Phase 3: LIFT-HYBRID (Weeks 19–24)

### Milestone 3.1 — Encoding Operations (Week 19–20)

**Tasks:**
```
[ ] hybrid.amplitude_encode
      - Input: tensor<N×f32>, Output: log₂(N) qubits
      - Normalisation (automatic if normalize=true)
      - Gate decomposition to initialise state vector

[ ] hybrid.angle_encode
      - Input: tensor<N×f32>, Output: N qubits
      - Rotation gate selection (RX, RY, RZ)
      - Domain mapping (arbitrary float → [0, 2π])

[ ] hybrid.basis_encode
      - Input: tensor<N×i32>, Output: N qubits
      - Binary encoding of integers

[ ] hybrid.hamiltonian_encode
      - Input: tensor<K×f32> (coefficients), Output: Hamiltonian
      - Assign coefficients to Pauli terms

[ ] hybrid.decode (measurement → classical tensor)
      - Expectation value computation
      - Sampling statistics
      - Quantum state tomography (for small circuits)
```

---

### Milestone 3.2 — Hybrid Operations (Week 20–22)

**Tasks:**
```
[ ] hybrid.parameterized_circuit
      - Classical network generates gate parameters
      - Circuit is a LIFT-QUANTUM function
      - Gradient flows through parameter shift rule

[ ] hybrid.measure_with_ml
      - Quantum measurement results fed into classical network
      - Useful for classification (quantum feature map → classical classifier)

[ ] hybrid.joint_optimisation
      - Combined classical + quantum parameter optimisation
      - Supports Adam, COBYLA, SPSA optimisers
      - Handles parameter shift for quantum gradients

[ ] hybrid.cosimulation
      - Explicit split: classical part runs on GPU, quantum part on QPU (or simulator)
      - Interface management (data transfer, synchronisation)
      - Pipelining (overlapping classical and quantum execution)
```

---

### Milestone 3.3 — Hybrid Optimisation Passes (Week 22–24)

**Tasks:**
```
[ ] HybridFusionPass
      - Fuse consecutive classical operations with quantum encoding
      - Fuse quantum measurement with classical post-processing
      - Eliminate redundant encoding/decoding pairs

[ ] ParameterShiftPass
      - Transform hybrid.joint_optimisation into explicit parameter shift evaluations
      - Batch the 2P circuit evaluations required for P parameters

[ ] EncodingOptimisationPass
      - Select encoding strategy based on circuit depth budget
      - Trade qubit count for circuit depth (amplitude vs angle)

[ ] ShotOptimisationPass
      - Compute minimum shots needed for given statistical precision
      - Reuse shots across repeated measurements on same circuit
```

---

## Phase 4: Simulation + Prediction Engine (Weeks 25–30)

### Milestone 4.1 — Static Analysis Engine (Week 25–26)

**Tasks:**
```
[ ] Shape propagation (propagate shapes through all operations)
[ ] Type inference (fill in missing types from context)
[ ] FLOP counter (per-operation FLOP formulae)
[ ] Memory footprint analyser
      - Tensor allocation sizes
      - Peak memory (liveness-based)
      - Memory reuse opportunities

[ ] Bandwidth pressure estimator
      - Memory reads + writes per operation
      - Cache locality analysis

[ ] Circuit analyser (quantum)
      - Gate count by type
      - Circuit depth (critical path)
      - Two-qubit gate count (most expensive)
      - Expected T1/T2 decoherence contribution

[ ] Static report generator (HTML + JSON output)
```

---

### Milestone 4.2 — Quantum Simulator (Week 26–28)

**Tasks:**
```
[ ] State vector simulator (CPU, single-threaded)
      - Complex vector of size 2^n
      - Gate application as sparse matrix multiply
      - Measurement as projection + normalisation

[ ] State vector simulator (GPU, via CUDA/cuStateVec)
      - GPU-accelerated gate operations
      - Scales to ~35 qubits

[ ] Density matrix simulator
      - 2^n × 2^n complex matrix
      - Noise channel application (Kraus operators)
      - Scales to ~20 qubits

[ ] MPS simulator (Matrix Product States)
      - Tensor train representation
      - Bond dimension control (accuracy vs speed tradeoff)
      - Scales to ~100 qubits for shallow circuits

[ ] Monte Carlo noise simulation
      - Sample from noise channel distributions
      - Aggregate statistics over N trajectories
      - Fidelity estimation from trajectory ensemble

[ ] Simulator auto-selection (choose backend based on qubit count + noise)
```

---

### Milestone 4.3 — ML Performance Predictor (Week 28–30)

**Tasks:**
```
[ ] Computation graph extractor (IR → graph features)
      - Node features: op type, shapes, dtype, impl variant
      - Edge features: tensor sizes, transfer bytes

[ ] GNN model definition (PyTorch, exported to ONNX for runtime)
      - Gated Graph Neural Network architecture
      - 6 message passing layers
      - Readout: graph-level MLP

[ ] Training data collection pipeline
      - Benchmark suite: 100+ micro-kernels and full models
      - Run on H100, A100, RTX 4090, IBM Kyoto
      - Store (graph, hardware_features, measurements)

[ ] Predictor Rust inference engine
      - Load ONNX model via candle or tract
      - Feature extraction from LIFT IR
      - Prediction: latency (ms), memory (GB), utilisation (%)

[ ] Fidelity predictor (quantum-specific)
      - Input: circuit graph + noise model parameters
      - Output: expected fidelity distribution
      - Uses noise propagation analysis from static engine

[ ] Budget checker
      - Compare predictions vs .lith budget constraints
      - Generate actionable error messages on violation
```

---

## Phase 5: Backends + Interoperability (Weeks 31–38)

### Milestone 5.1 — CUDA Backend (Week 31–33)

**Tasks:**
```
[ ] PTX code generation framework
[ ] tensor.matmul → cuBLAS SGEMM / DGEMM / HGEMM
[ ] tensor.attention → cuDNN attention / custom FlashAttention kernel
[ ] tensor.conv2d → cuDNN convolution
[ ] tensor.flash_attention → custom FlashAttention v2/v3 template
[ ] Tensor Cores utilisation (for f16/bf16/int8)
[ ] Memory coalescing analysis and enforcement
[ ] Kernel launch parameter optimisation (block size, grid size, shared memory)
[ ] NCCL integration for multi-GPU operations
[ ] CUDA graph capture (reduce kernel launch overhead)
```

---

### Milestone 5.2 — OpenQASM 3 Backend (Week 33–34)

**Tasks:**
```
[ ] OpenQASM 3.0 emitter
      - Gate definitions
      - Qubit declarations
      - Classical bit declarations
      - Gate applications (with parameters)
      - Measurements
      - Classical control flow

[ ] IBM Qiskit Runtime integration
      - Submit OpenQASM circuits to IBM backends
      - Poll for results
      - Return measurement statistics

[ ] AWS Braket integration
      - Submit circuits to Rigetti, IonQ, OQC via Braket API
      - Parse Braket result format

[ ] Pulse-level lowering (IBM pulse backend)
      - Decompose gates to microwave pulses
      - Schedule on IBM pulse channels
      - DRAG pulse shaping for reduced leakage
```

---

### Milestone 5.3 — LLVM and XLA Backends (Week 34–35)

**Tasks:**
```
[ ] LLVM IR emitter (for CPU targets)
      - Scalar operations → LLVM instructions
      - SIMD vectorisation (AVX-512 for tensor ops on CPU)
      - OpenMP pragmas for multi-core

[ ] XLA/StableHLO bridge (for TPU)
      - LIFT-TENSOR → StableHLO dialect
      - Leverage XLA's TPU compiler
```

---

### Milestone 5.4 — Importers (Week 35–37)

**Tasks:**
```
[ ] PyTorch FX graph importer
      - Walk PyTorch FX IR
      - Map aten:: ops to tensor.* ops
      - Infer shapes from PyTorch shape propagation
      - Handle dynamic shapes (mark as Symbolic)

[ ] ONNX importer
      - Parse ONNX protobuf
      - Map ONNX ops to tensor.* ops
      - Handle opset differences

[ ] Qiskit QuantumCircuit importer
      - Walk Qiskit DAGCircuit
      - Map Qiskit gates to quantum.* ops
      - Import noise model from Qiskit NoiseModel

[ ] OpenQASM 3 importer
      - Parse OpenQASM 3 source
      - Map gate statements to quantum.* ops
      - Import classical control flow

[ ] Exporters (reverse direction)
      - LIFT-TENSOR → ONNX
      - LIFT-QUANTUM → OpenQASM 3
      - LIFT-QUANTUM → Qiskit QuantumCircuit
```

---

### Milestone 5.5 — .lith Parser (Week 37–38)

**Tasks:**
```
[ ] Lexer (tokenise .lith source)
[ ] Parser (recursive descent, produce typed config tree)
[ ] Environment variable substitution (${VAR} → actual value)
[ ] File inclusion (include "base.lith")
[ ] Config struct population (parse tree → LiftConfig)
[ ] Validation engine
      - Check all enum values are valid
      - Check cross-section consistency (e.g., hybrid target requires both gpu and qpu)
      - Check budget values are positive
[ ] Helpful error messages (point to offending line + suggest fix)
[ ] Default value resolution
[ ] Config documentation generator (auto-generate reference docs from struct annotations)
```

---

## Phase 6: Tooling + Observability (Weeks 39–44)

### Milestone 6.1 — CLI (Week 39–40)

**Tasks:**
```
lift compile <file.lif>
  --config <project.lith>   (configuration file)
  --target <cuda|qasm|llvm> (override target)
  --passes <pass1,pass2>    (override pass pipeline)
  --output <path>           (output directory)
  --verbose                 (verbose output)

lift simulate <file.lif>
  --config <project.lith>
  --report <report.html>    (generate HTML simulation report)
  --json <metrics.json>     (JSON metrics output)

lift predict <file.lif>
  --hardware <h100|ibm_kyoto|...>
  --noise-model <file.json>

lift optimise <file.lif>
  --passes <pass1,pass2,...>
  --output <optimised.lif>

lift convert
  --from <pytorch|onnx|qiskit|qasm>
  --to <lift>
  <input_file>
  --output <output.lif>

lift verify <file.lif>
  (check IR well-formedness and type correctness)

lift info <file.lif>
  (print statistics: op counts, shapes, estimated FLOPs)
```

---

### Milestone 6.2 — Observability (Week 40–42)

**Tasks:**
```
[ ] Structured logging (JSON, using tracing crate)
      - log every pass: name, duration, IR changes
      - log every backend lowering step
      - log prediction results

[ ] Prometheus metrics endpoint (/metrics)
      - compilation_duration_seconds (histogram by pass)
      - prediction_accuracy (compare predicted vs actual for tests)
      - pass_improvement (latency reduction per pass)

[ ] Interactive web dashboard
      - Show compilation trace (passes, durations)
      - Show computation graph (interactive, zoomable)
      - Show circuit diagram (for quantum modules)
      - Show performance predictions vs actuals

[ ] Flamegraph profiler integration
      - Profile pass execution time
      - Identify bottlenecks in the compiler itself

[ ] Compilation replay
      - Record all inputs + config
      - Re-run exact compilation for debugging
      - Compare IR at each pass checkpoint
```

---

### Milestone 6.3 — Auto-Tuning (Week 42–44)

**Tasks:**
```
[ ] Pass ordering search (Bayesian optimisation over pass sequences)
      - Define search space (which passes, in what order)
      - Objective: minimise predicted latency
      - Constraints: budget from .lith

[ ] GNN-based pass selector (RL agent trained on compilation traces)
      - State: current IR features
      - Action: which pass to apply next
      - Reward: improvement in predicted performance

[ ] Runtime feedback loop
      - Collect actual performance after execution
      - Feed back to prediction model (online learning)
      - Adjust pass selection based on observed vs predicted gap

[ ] A/B testing infrastructure
      - Run two compilation strategies in parallel
      - Select winner based on real performance
      - Gradual rollout of new optimisations
```

---

## Phase 7: Documentation + Public Release (Weeks 45–48)

### Milestone 7.1 — Documentation (Week 45–46)

**Deliverables:**
```
[ ] API Documentation (rustdoc for all public types and functions)
[ ] Language Reference (.lif syntax, all operations, all types)
[ ] Configuration Reference (.lith all sections and fields)
[ ] Getting Started Guide (hello world → real model in 30 min)

[ ] Tutorials:
      Tutorial 1: LLM Inference Optimisation
        - Start with LLaMA 7B in naive LIFT-TENSOR
        - Apply FlashAttention, KV cache, INT8 quantisation
        - Compare before/after performance

      Tutorial 2: VQE for Quantum Chemistry
        - Hydrogen molecule ground state energy
        - LIFT-QUANTUM, noise model, ZNE error mitigation
        - Compare simulated vs real IBM results

      Tutorial 3: Quantum Neural Network Classifier
        - MNIST classification with 4-qubit QNN
        - LIFT-HYBRID: classical encoder + quantum layer
        - End-to-end training with joint gradients

      Tutorial 4: Writing a Custom Pass
        - Implement a simple tensor fusion pass
        - Register it with the pass manager
        - Use it from .lith

[ ] Contributor Guide (how to add a dialect, pass, backend)
[ ] Architecture Document (this document, updated)
```

---

### Milestone 7.2 — Benchmarks (Week 46–47)

**Benchmark suite:**
```
[ ] AI Benchmarks
      - BERT-Large forward pass (batch=1, seq=512)
      - LLaMA 7B single-token inference
      - ResNet-50 training step (batch=256)
      - Vision Transformer (ViT-Large) inference

    Compare vs: PyTorch eager, torch.compile, TensorRT, ONNX Runtime

[ ] Quantum Benchmarks
      - QAOA MaxCut (n=20 variables, depth p=3)
      - VQE H2O molecule (12 qubits)
      - Quantum Volume circuits (QV=128)
      - Random circuits (n=20 qubits, depth=100)

    Compare vs: Qiskit transpile(), Pytket, tket2

[ ] Hybrid Benchmarks
      - QNN MNIST (4 qubits, 10 classes)
      - QAOA + classical post-processing
      - VQE + gradient computation

    Compare vs: PennyLane, Qiskit ML

[ ] Compilation Time Benchmarks
      - Time to compile each model
      Compare vs: torch.compile, Qiskit transpile

[ ] Report: 5-page benchmark paper with methodology + results
```

---

### Milestone 7.3 — Public Release (Week 47–48)

**Deliverables:**
```
[ ] GitHub repository (lift-framework/lift)
      - MIT license
      - README.md (this document, rendered)
      - CONTRIBUTING.md
      - CODE_OF_CONDUCT.md
      - SECURITY.md (responsible disclosure policy)

[ ] crates.io publication (lift-core, lift-tensor, lift-quantum, lift-hybrid, lift-cli)
[ ] PyPI package (Python bindings via PyO3/Maturin)
[ ] Docker image (lift:latest with all backends pre-installed)

[ ] arXiv preprint
      Title: "LIFT: A Unified Intermediate Representation for AI and Quantum Computing"
      Sections: Introduction, Related Work, Design, Implementation, Evaluation, Conclusion

[ ] Blog post (technical introduction, 3000 words)
[ ] HN / r/MachineLearning / r/QuantumComputing announcement

[ ] Community infrastructure
      - Discord server (lift-framework)
      - GitHub Discussions enabled
      - Issue templates (bug report, feature request, RFC)
```

---

## Resource Requirements

### Team

```
MINIMUM VIABLE TEAM (4 people):
  1 × Compiler Engineer (IR design, pass infrastructure, LLVM backend)
  1 × AI Systems Engineer (TENSOR dialect, CUDA backend, AI passes)
  1 × Quantum Engineer (QUANTUM dialect, noise models, QC passes, QASM backend)
  1 × ML Engineer (prediction engine, benchmarks, evaluation)

IDEAL TEAM (7 people):
  + 1 × Systems Engineer (performance, tooling, CI/CD)
  + 1 × Developer Advocate (documentation, tutorials, community)
  + 1 × Research Scientist (algorithm design, publications)
```

### Hardware

```
Development:
  - 2× NVIDIA A100 or H100 (AI backend development and testing)
  - IBM Quantum access (5-10 QPU hours/month for quantum testing)
  - AWS Braket credits (for multi-vendor QPU testing)

CI/CD:
  - 1× A100 instance for GPU tests
  - Quantum simulator server (32-core, 256 GB RAM, for large quantum simulations)

Benchmarking:
  - Access to full H100 cluster (8×) for distributed training benchmarks
  - 20 QPU hours on IBM Kyoto or similar for quantum benchmarks
```

### Compute Budget (Estimated)

```
Training the GNN prediction model:
  - Dataset collection: ~500 GPU-hours on various hardware
  - Training: ~50 GPU-hours
  - Validation: ~20 GPU-hours

Quantum calibration data:
  - ~100 QPU hours to build calibration database for major backends

Ongoing CI:
  - ~10 GPU-hours/week for full benchmark suite
  - ~5 QPU hours/month for quantum regression tests
```

---

## Risk Register

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| QPU access limits slow quantum testing | High | Medium | Use simulators for most tests; batch real QPU runs |
| GNN predictor accuracy insufficient | Medium | High | Maintain analytical fallback; improve with more data |
| MLIR API changes break CUDA lowering | Low | Medium | Abstract MLIR dependency; version pin |
| Quantum hardware calibration drift | High | Low | Re-fetch calibration daily; build drift model |
| Python binding performance | Medium | Low | PyO3 is fast; benchmark and optimise critical paths |
| Community adoption slower than expected | Medium | High | Partner with 2-3 research groups early; co-author papers |

---

## Success Metrics

### At 6 Months (Phase 3 Complete)

```
[ ] 500+ passing tests across all three dialects
[ ] 10 AI models can be expressed and compiled to CUDA
[ ] 5 quantum circuits can be expressed and compiled to OpenQASM 3
[ ] 2 hybrid examples working end-to-end
[ ] Compilation time < 10 seconds for all test cases
[ ] Zero memory leaks (verified by ASAN/Valgrind)
```

### At 12 Months (Full Release)

```
[ ] Benchmark results competitive with PyTorch+torch.compile for AI
[ ] Benchmark results competitive with Qiskit for quantum
[ ] 100+ GitHub stars in first month
[ ] 5+ external contributors (not on core team)
[ ] 2 research papers citing LIFT
[ ] 3 tutorial videos published
[ ] arXiv preprint submitted
```

### Long-Term (2 Years)

```
[ ] 1000+ GitHub stars
[ ] 20+ external contributors
[ ] Used by at least 2 industrial research labs
[ ] Standard for hybrid AI+QC at 1 major conference (NeurIPS, ICML, QIP)
[ ] Adoption by at least 1 QPU vendor as preferred input format
```

---

*This implementation plan is a living document. Update it as priorities shift and new information arrives.*

*LIFT — Built by engineers who believe the future of computing is unified.*