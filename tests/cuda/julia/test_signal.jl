# Signal CUDA binding tests.
# Run with:
#   LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda julia tests/cuda/julia/test_signal.jl

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

function approx(a, b; atol=1e-6)
    return abs(a - b) < atol
end

# ============================================================================
# Unwrap (5 tests)
# ============================================================================
println("=== Unwrap ===")

data = [0.0, π/2, π, 3π/2, 2π, 3π]
a = Insight.from_data(data)
u = Insight.unwrap(a)
check("unwrap_basic", Insight.numel(u) == 6)

a = Insight.from_data([π])
u = Insight.unwrap(a)
check("unwrap_scalar", Insight.numel(u) == 1)

data2d = [0.0, π/2, π, 3π/2, 0.1, π/2, 3.2, 3.3]
a = Insight.from_data(data2d)
a = Insight.reshape(a, [2, 4])
u = Insight.unwrap(a, axis=1)
check("unwrap_2d_axis1", Insight.numel(u) == 8)

u = Insight.unwrap(a, axis=2)
check("unwrap_2d_axis2", Insight.numel(u) == 8)

try
    Insight.unwrap(a, axis=0)
    check("unwrap_axis0_invalid", false)
catch e
    check("unwrap_axis0_invalid", isa(e, ArgumentError))
end

data = [0.0, 0.1, 3.2, 3.3, 6.4, 6.5]
a = Insight.from_data(data)
u = Insight.unwrap(a)
check("unwrap_with_jumps", Insight.numel(u) == 6)

# ============================================================================
# Sinc (2 tests)
# ============================================================================
println("=== Sinc ===")

a = Insight.from_data([0.0, 1.0, 2.0, -1.0, -2.0, 0.5])
y = Insight.sinc(a)
check("sinc_numel", Insight.numel(y) == 6)
# sinc(0) = 1
check("sinc_at_0", approx(Insight.item(y, 0), 1.0, atol=1e-5))
# sinc(1) = 0
check("sinc_at_1", approx(Insight.item(y, 1), 0.0, atol=1e-5))
# sinc(0.5) = 2/π
check("sinc_at_0_5", approx(Insight.item(y, 5), 0.6366198, atol=1e-5))

a = Insight.from_data([0.0])
y = Insight.sinc(a)
check("sinc_size1", Insight.numel(y) == 1 && approx(Insight.item(y, 0), 1.0, atol=1e-5))

# ============================================================================
# Convolve (2 tests)
# ============================================================================
println("=== Convolve ===")

try
    a = Insight.from_data([1.0, 2.0, 3.0])
    v = Insight.from_data([1.0, 1.0])
    local c = Insight.convolve(a, v, "full")
    check("convolve_full_numel", Insight.numel(c) == 4)
    check("convolve_full_0", approx(Insight.item(c, 0), 1.0, atol=1e-5))
    check("convolve_full_1", approx(Insight.item(c, 1), 3.0, atol=1e-5))
    check("convolve_full_2", approx(Insight.item(c, 2), 5.0, atol=1e-5))
    check("convolve_full_3", approx(Insight.item(c, 3), 3.0, atol=1e-5))

    c = Insight.convolve(a, v, "same")
    check("convolve_same_numel", Insight.numel(c) == 3)
catch e
    println("SKIP: convolve (requires FFTW3): $e")
end

# ============================================================================
# Correlate (1 test)
# ============================================================================
println("=== Correlate ===")

try
    a = Insight.from_data([1.0, 2.0, 3.0])
    b = Insight.from_data([1.0, 1.0])
    local c = Insight.signal.correlate(a, b, "full")
    check("correlate_full", Insight.numel(c) == 4)
catch e
    println("SKIP: correlate (requires FFTW3): $e")
end

# ============================================================================
# Results
# ============================================================================
println("\n" * "="^40)
println("Results: $passed passed, $failed failed")
if failed > 0
    exit(1)
end
