# Signal wavelets CUDA binding tests
# Run with:
#   LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda julia tests/cuda/julia/test_signal_wavelets.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "..", "bindings", "julia"))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "..", "build", "bindings", "julia"))

using Insight

# Try GPU
try
    Insight.load_backend("gpu")
    Insight.set_device(Insight.GPUPlace(0))
catch e
    println("SKIP: GPU backend not available: $e")
    exit(0)
end

passed = 0
failed = 0

function check(name, cond)
    global passed, failed
    if cond
        passed += 1
    else
        failed += 1
        println("FAIL: $name")
    end
end

# ============================================================================
# Wavelet Functions
# ============================================================================
println("=== Signal Wavelets ===")

# morlet
w = Insight.morlet(32, w=5.0, s=1.0, complete=true)
check("morlet", Insight.numel(w) == 32)

# morlet short
w = Insight.morlet(16, w=5.0, s=1.0, complete=true)
check("morlet_short", Insight.numel(w) == 16)

# morlet incomplete
w = Insight.morlet(32, w=5.0, s=1.0, complete=false)
check("morlet_incomplete", Insight.numel(w) == 32)

# ricker
w = Insight.ricker(100, 4.0)
check("ricker", Insight.numel(w) == 100)

# ricker narrow
w = Insight.ricker(50, 2.0)
check("ricker_narrow", Insight.numel(w) == 50)

# ricker wide
w = Insight.ricker(200, 10.0)
check("ricker_wide", Insight.numel(w) == 200)

# morlet2
w = Insight.morlet2(32, 1.0, w=5.0)
check("morlet2", Insight.numel(w) == 32)

# morlet2 scaled
w = Insight.morlet2(64, 2.0, w=5.0)
check("morlet2_scaled", Insight.numel(w) == 64)

# ============================================================================
# Results
# ============================================================================
println("\n" * "="^40)
println("Results: $passed passed, $failed failed")
if failed > 0
    exit(1)
end
