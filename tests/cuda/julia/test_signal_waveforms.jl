# Signal Waveforms CUDA binding tests.
# Run with:
#   LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda julia tests/cuda/julia/test_signal_waveforms.jl

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
# Sawtooth (3 tests)
# ============================================================================
println("=== Sawtooth ===")

t = Insight.from_data([0.0, π, 2π - 1e-10])
y = Insight.signal.sawtooth(t, width=1.0)
check("sawtooth_numel", Insight.numel(y) == 3)
check("sawtooth_0", approx(Insight.item(y, 0), -1.0))
check("sawtooth_1", approx(Insight.item(y, 1), 0.0))
check("sawtooth_2", approx(Insight.item(y, 2), 1.0))

t = Insight.from_data([0.0, π/2, π])
y = Insight.signal.sawtooth(t, width=0.5)
check("sawtooth_triangle_numel", Insight.numel(y) == 3)
check("sawtooth_triangle_0", approx(Insight.item(y, 0), -1.0))
check("sawtooth_triangle_2", approx(Insight.item(y, 2), 1.0))

t = Insight.from_data([0.5, 0.5 + 2π, 0.5 + 4π])
y = Insight.signal.sawtooth(t, width=1.0)
check("sawtooth_periodic", approx(Insight.item(y, 0), Insight.item(y, 1)) &&
                           approx(Insight.item(y, 0), Insight.item(y, 2)))

# ============================================================================
# Square (2 tests)
# ============================================================================
println("=== Square ===")

t = Insight.from_data([0.0, π, 2π - 1e-10])
y = Insight.signal.square_wf(t, duty=0.5)
check("square_numel", Insight.numel(y) == 3)
check("square_0", approx(Insight.item(y, 0), 1.0))
check("square_1", approx(Insight.item(y, 1), -1.0))
check("square_2", approx(Insight.item(y, 2), -1.0))

t = Insight.from_data([1.0, 1.0 + 2π, 1.0 + 4π])
y = Insight.signal.square_wf(t, duty=0.5)
check("square_periodic", approx(Insight.item(y, 0), Insight.item(y, 1)) &&
                          approx(Insight.item(y, 0), Insight.item(y, 2)))

# ============================================================================
# Chirp (2 tests)
# ============================================================================
println("=== Chirp ===")

t = Insight.from_data([0.0])
y = Insight.signal.chirp(t, 6.0, 1.0, 10.0, method=Int32(0), phi=0.0)
check("chirp_linear", Insight.numel(y) == 1 && approx(Insight.item(y, 0), 1.0))

y = Insight.signal.chirp(t, 6.0, 1.0, 10.0, method=Int32(0), phi=π/2)
check("chirp_with_phase", Insight.numel(y) == 1 && approx(Insight.item(y, 0), 0.0))

# ============================================================================
# Unit Impulse (1 test)
# ============================================================================
println("=== Unit Impulse ===")

y = Insight.signal.unit_impulse([11])
check("unit_impulse_numel", Insight.numel(y) == 11)
check("unit_impulse_center", approx(Insight.item(y, 5), 1.0, atol=1e-10))
check("unit_impulse_zero", approx(Insight.item(y, 0), 0.0, atol=1e-10))

y = Insight.signal.unit_impulse([7], idx=3)
check("unit_impulse_at_idx", approx(Insight.item(y, 3), 1.0, atol=1e-10) &&
                              approx(Insight.item(y, 0), 0.0, atol=1e-10))

# ============================================================================
# Results
# ============================================================================
println("\n" * "="^40)
println("Results: $passed passed, $failed failed")
if failed > 0
    exit(1)
end
