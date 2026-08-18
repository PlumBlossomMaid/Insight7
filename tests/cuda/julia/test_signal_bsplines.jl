# Signal Bsplines CUDA binding tests.
# Run with:
#   LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda julia tests/cuda/julia/test_signal_bsplines.jl

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

function approx(a, b; atol=1e-10)
    return abs(a - b) < atol
end

# ============================================================================
# Gauss Spline (3 tests)
# ============================================================================
println("=== Gauss Spline ===")

x = Insight.from_data([0.0])
y = Insight.signal.gauss_spline(x, 3)
sigma_sq = 4.0 / 12.0
expected = 1.0 / sqrt(2π * sigma_sq)
check("gauss_spline_basic", Insight.numel(y) == 1 && approx(Insight.item(y, 0), expected))

x = Insight.from_data([-2.0, -1.0, 0.0, 1.0, 2.0])
y = Insight.signal.gauss_spline(x, 2)
check("gauss_spline_sym_numel", Insight.numel(y) == 5)
check("gauss_spline_sym_04", approx(Insight.item(y, 0), Insight.item(y, 4)))
check("gauss_spline_sym_13", approx(Insight.item(y, 1), Insight.item(y, 3)))
check("gauss_spline_peak", Insight.item(y, 2) > Insight.item(y, 0))

x = Insight.from_data([0.0, 1.0, 2.0, 3.0, 5.0])
y = Insight.signal.gauss_spline(x, 3)
check("gauss_spline_decay", Insight.item(y, 0) > Insight.item(y, 1) > Insight.item(y, 2) > Insight.item(y, 3))

# ============================================================================
# Cubic (5 tests)
# ============================================================================
println("=== Cubic ===")

x = Insight.from_data([0.0])
y = Insight.signal.cubic(x)
check("cubic_basic", Insight.numel(y) == 1 && approx(Insight.item(y, 0), 2.0/3.0))

x = Insight.from_data([-1.5, -0.5, 0.0, 0.5, 1.5])
y = Insight.signal.cubic(x)
check("cubic_sym_numel", Insight.numel(y) == 5)
check("cubic_sym_04", approx(Insight.item(y, 0), Insight.item(y, 4)))
check("cubic_sym_13", approx(Insight.item(y, 1), Insight.item(y, 3)))

x = Insight.from_data([-2.5, -2.0, 2.0, 2.5])
y = Insight.signal.cubic(x)
check("cubic_zero_outside", approx(Insight.item(y, 0), 0.0) &&
                             approx(Insight.item(y, 1), 0.0) &&
                             approx(Insight.item(y, 2), 0.0) &&
                             approx(Insight.item(y, 3), 0.0))

x = Insight.from_data([0.5])
y = Insight.signal.cubic(x)
expected_r1 = 2.0/3.0 - 0.5 * 0.25 * 1.5
check("cubic_region1", approx(Insight.item(y, 0), expected_r1))

x = Insight.from_data([1.5])
y = Insight.signal.cubic(x)
expected_r2 = (1.0/6.0) * 0.125
check("cubic_region2", approx(Insight.item(y, 0), expected_r2))

# ============================================================================
# Quadratic (3 tests)
# ============================================================================
println("=== Quadratic ===")

x = Insight.from_data([0.0])
y = Insight.signal.quadratic(x)
check("quadratic_basic", Insight.numel(y) == 1 && approx(Insight.item(y, 0), 0.75))

x = Insight.from_data([-1.0, -0.25, 0.0, 0.25, 1.0])
y = Insight.signal.quadratic(x)
check("quadratic_sym_numel", Insight.numel(y) == 5)
check("quadratic_sym_04", approx(Insight.item(y, 0), Insight.item(y, 4)))
check("quadratic_sym_13", approx(Insight.item(y, 1), Insight.item(y, 3)))

x = Insight.from_data([-2.0, -1.5, 1.5, 2.0])
y = Insight.signal.quadratic(x)
check("quadratic_zero_outside", approx(Insight.item(y, 0), 0.0) &&
                                  approx(Insight.item(y, 1), 0.0) &&
                                  approx(Insight.item(y, 2), 0.0) &&
                                  approx(Insight.item(y, 3), 0.0))

# ============================================================================
# Results
# ============================================================================
println("\n" * "="^40)
println("Results: $passed passed, $failed failed")
if failed > 0
    exit(1)
end
