# Signal Convolution CUDA binding tests.
# Run with:
#   LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda julia tests/cuda/julia/test_signal_convolution.jl

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
# FFT Convolve (5 tests)
# ============================================================================
println("=== FFT Convolve ===")

a = Insight.from_data([1.0, 2.0, 3.0])
b = Insight.from_data([1.0, 1.0])
c = Insight.signal.fftconvolve(a, b, mode="full")
check("fftconvolve_full_numel", Insight.numel(c) == 4)
check("fftconvolve_full_0", approx(Insight.item(c, 0), 1.0))
check("fftconvolve_full_1", approx(Insight.item(c, 1), 3.0))
check("fftconvolve_full_2", approx(Insight.item(c, 2), 5.0))
check("fftconvolve_full_3", approx(Insight.item(c, 3), 3.0))

a = Insight.from_data([1.0, 2.0, 3.0, 4.0, 5.0])
b = Insight.from_data([1.0, 1.0, 1.0])
c = Insight.signal.fftconvolve(a, b, mode="same")
check("fftconvolve_same_numel", Insight.numel(c) == 5)
check("fftconvolve_same_0", approx(Insight.item(c, 0), 3.0))
check("fftconvolve_same_3", approx(Insight.item(c, 3), 12.0))

c = Insight.signal.fftconvolve(a, b, mode="valid")
check("fftconvolve_valid_numel", Insight.numel(c) == 3)
check("fftconvolve_valid_0", approx(Insight.item(c, 0), 6.0))
check("fftconvolve_valid_2", approx(Insight.item(c, 2), 12.0))

a = Insight.from_data([1.0, 2.0, 3.0])
b = Insight.from_data([4.0, 5.0])
c1 = Insight.signal.fftconvolve(a, b, mode="full")
c2 = Insight.signal.fftconvolve(b, a, mode="full")
comm_ok = Insight.numel(c1) == Insight.numel(c2)
for i in 0:Insight.numel(c1)-1
    if !approx(Insight.item(c1, i), Insight.item(c2, i))
        comm_ok = false; break
    end
end
check("fftconvolve_commutative", comm_ok)

a = Insight.from_data([1.0, 2.0, 3.0, 4.0, 5.0])
b = Insight.from_data([1.0])
c = Insight.signal.fftconvolve(a, b, mode="full")
imp_ok = Insight.numel(c) == 5
for i in 0:4
    if !approx(Insight.item(c, i), Float64(i + 1))
        imp_ok = false; break
    end
end
check("fftconvolve_impulse", imp_ok)

# ============================================================================
# Correlate (3 tests)
# ============================================================================
println("=== Correlate ===")

a = Insight.from_data([1.0, 2.0, 3.0])
b = Insight.from_data([1.0, 1.0])
c = Insight.signal.correlate(a, b, mode="full")
check("correlate_full_numel", Insight.numel(c) == 4)
check("correlate_full_0", approx(Insight.item(c, 0), 1.0))
check("correlate_full_1", approx(Insight.item(c, 1), 3.0))
check("correlate_full_2", approx(Insight.item(c, 2), 5.0))
check("correlate_full_3", approx(Insight.item(c, 3), 3.0))

a = Insight.from_data([1.0, 2.0, 3.0, 4.0, 5.0])
b = Insight.from_data([1.0, 1.0, 1.0])
c = Insight.signal.correlate(a, b, mode="same")
check("correlate_same_numel", Insight.numel(c) == 5)

a = Insight.from_data([1.0, 0.0, -1.0])
c = Insight.signal.correlate(a, a, mode="full")
check("autocorr_numel", Insight.numel(c) == 5)
check("autocorr_0", approx(Insight.item(c, 0), -1.0))
check("autocorr_2", approx(Insight.item(c, 2), 2.0))
check("autocorr_4", approx(Insight.item(c, 4), -1.0))

# ============================================================================
# Convolve2d (2 tests)
# ============================================================================
println("=== Convolve2d ===")

a = Insight.from_data([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
a = Insight.reshape(a, [3, 3])
b = Insight.from_data([1.0, 0.0, 0.0, 1.0])
b = Insight.reshape(b, [2, 2])
c = Insight.signal.convolve2d(a, b, mode="full")
check("convolve2d_full_numel", Insight.numel(c) == 16)

c = Insight.signal.convolve2d(a, b, mode="same")
check("convolve2d_same_numel", Insight.numel(c) == 9)

# ============================================================================
# Correlate2d (1 test)
# ============================================================================
println("=== Correlate2d ===")

a = Insight.from_data([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
a = Insight.reshape(a, [3, 3])
b = Insight.from_data([1.0, 1.0, 1.0, 1.0])
b = Insight.reshape(b, [2, 2])
c = Insight.signal.correlate2d(a, b, mode="valid")
check("correlate2d_valid_numel", Insight.numel(c) == 4)

# ============================================================================
# Choose Conv Method (1 test)
# ============================================================================
println("=== Choose Conv Method ===")

a = Insight.from_data([1.0, 2.0, 3.0])
b = Insight.from_data([1.0, 1.0])
method = Insight.signal.choose_conv_method(a, b, mode="full")
check("choose_conv_method_small", method == "direct")

# ============================================================================
# Correlation Lags (2 tests)
# ============================================================================
println("=== Correlation Lags ===")

lags = Insight.signal.correlation_lags(5, 3, mode="full")
check("corr_lags_full_numel", Insight.numel(lags) == 7)
check("corr_lags_full_start", approx(Insight.item(lags, 0), -2.0))
check("corr_lags_full_end", approx(Insight.item(lags, 6), 4.0))

lags = Insight.signal.correlation_lags(5, 3, mode="same")
check("corr_lags_same_numel", Insight.numel(lags) == 5)
check("corr_lags_same_start", approx(Insight.item(lags, 0), -1.0))
check("corr_lags_same_end", approx(Insight.item(lags, 4), 3.0))

# ============================================================================
# Results
# ============================================================================
println("\n" * "="^40)
println("Results: $passed passed, $failed failed")
if failed > 0
    exit(1)
end
