#!/bin/bash
# ============================================================================
# LIFT End-to-End Validation Script
# Tests: verify → analyse → optimise → predict → export for ALL models
# ============================================================================

set -e

LIFT="cargo run --bin lift-cli --"
PASS=0
FAIL=0
TOTAL=0

check() {
    local name="$1"
    local cmd="$2"
    TOTAL=$((TOTAL + 1))
    echo -n "  [$name] "
    if eval "$cmd" > /dev/null 2>&1; then
        echo "PASS"
        PASS=$((PASS + 1))
    else
        echo "FAIL"
        FAIL=$((FAIL + 1))
    fi
}

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     LIFT — Full Pipeline Validation                         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── Models to test ──
MODELS=(
    "examples/phi3_mini.lif"
    "examples/deepseek_v2_lite.lif"
    "examples/llama2_7b.lif"
    "examples/mistral_7b.lif"
    "examples/bert_base.lif"
    "examples/tensor_mlp.lif"
    "examples/attention.lif"
    "examples/quantum_bell.lif"
)

for model in "${MODELS[@]}"; do
    name=$(basename "$model" .lif)
    echo "── Model: $name ──"

    # Step 1: Verify
    check "verify" "$LIFT verify $model"

    # Step 2: Analyse
    check "analyse" "$LIFT analyse $model"

    # Step 3: Analyse (JSON)
    check "analyse-json" "$LIFT analyse $model --format json"

    # Step 4: Print IR
    check "print" "$LIFT print $model"

    # Step 5: Optimise (default passes)
    check "optimise" "$LIFT optimise $model"

    # Step 6: Predict (A100)
    check "predict-a100" "$LIFT predict $model --device a100"

    # Step 7: Predict (H100)
    check "predict-h100" "$LIFT predict $model --device h100"

    # Step 8: Export LLVM
    check "export-llvm" "$LIFT export $model --backend llvm"

    # Step 9: Export QASM (quantum models only)
    check "export-qasm" "$LIFT export $model --backend qasm"

    echo ""
done

# ── Test with config file ──
echo "── Config-driven optimisation ──"
check "phi3+config" "$LIFT optimise examples/phi3_mini.lif --config examples/phi3_optimize.lith"
check "phi3+output" "$LIFT optimise examples/phi3_mini.lif --output /tmp/phi3_opt.lif"
echo ""

# ── Test all 11 passes via config ──
echo "── All 11 passes (config) ──"
cat > /tmp/all_passes.lith << 'EOF'
[target]
backend = "llvm"

[optimisation]
level = O3
passes = canonicalize, constant-folding, dce, tensor-fusion, gate-cancellation, rotation-merge, flash-attention, cse, quantisation-pass, noise-aware-schedule, layout-mapping
max_iterations = 5

[simulation]
shape_propagation = true
flop_counting = true
memory_analysis = true
noise_simulation = true
EOF
check "all-passes-phi3" "$LIFT optimise examples/phi3_mini.lif --config /tmp/all_passes.lith"
check "all-passes-bell" "$LIFT optimise examples/quantum_bell.lif --config /tmp/all_passes.lith"
echo ""

# ── Summary ──
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  VALIDATION SUMMARY                                         ║"
echo "╠══════════════════════════════════════════════════════════════╣"
printf "║  Passed: %3d                                               ║\n" $PASS
printf "║  Failed: %3d                                               ║\n" $FAIL
printf "║  Total:  %3d                                               ║\n" $TOTAL
echo "╚══════════════════════════════════════════════════════════════╝"

if [ $FAIL -eq 0 ]; then
    echo ""
    echo "ALL TESTS PASSED"
    exit 0
else
    echo ""
    echo "SOME TESTS FAILED"
    exit 1
fi
