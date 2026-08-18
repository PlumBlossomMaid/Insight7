# Signal Filtering CUDA binding tests.
# Run with:
#   LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda julia tests/cuda/julia/test_signal_filtering.jl

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
# Hilbert (3 tests)
# ============================================================================
println("=== Hilbert ===")

N = 64
data = [cos(2π * i / N) for i in 0:N-1]
x = Insight.from_data(data)
y = Insight.signal.hilbert(x, N=N)
check("hilbert_numel", Insight.numel(y) == N)

data = ones(8)
x = Insight.from_data(data)
y = Insight.signal.hilbert(x, N=8)
check("hilbert_dc_numel", Insight.numel(y) == 8)

x = Insight.from_data([1.0, 2.0, 3.0, 4.0])
y = Insight.signal.hilbert(x)
check("hilbert_default_n", Insight.numel(y) == 4)

# ============================================================================
# Detrend (3 tests)
# ============================================================================
println("=== Detrend ===")

x = Insight.from_data([1.0, 2.0, 3.0, 4.0, 5.0])
y = Insight.signal.detrend(x, axis=-1, type="constant")
check("detrend_constant_numel", Insight.numel(y) == 5)
check("detrend_constant_0", approx(Insight.item(y, 0), -2.0, atol=1e-10))
check("detrend_constant_2", approx(Insight.item(y, 2), 0.0, atol=1e-10))
check("detrend_constant_4", approx(Insight.item(y, 4), 2.0, atol=1e-10))

x = Insight.from_data([0.0, 3.0, 6.0, 9.0, 12.0])
y = Insight.signal.detrend(x, axis=-1, type="linear")
check("detrend_linear_numel", Insight.numel(y) == 5)
linear_ok = true
for i in 0:4
    if !approx(Insight.item(y, i), 0.0, atol=1e-10)
        linear_ok = false; break
    end
end
check("detrend_linear_zero", linear_ok)

x = Insight.from_data([0.1, 3.0, 5.9, 9.1, 12.0])
y = Insight.signal.detrend(x, axis=-1, type="linear")
noisy_ok = true
for i in 0:4
    if !approx(Insight.item(y, i), 0.0, atol=0.2)
        noisy_ok = false; break
    end
end
check("detrend_linear_noisy", noisy_ok)

# ============================================================================
# Lfilter (3 tests)
# ============================================================================
println("=== Lfilter ===")

b = Insight.from_data([1.0, 1.0])
a = Insight.from_data([1.0])
x = Insight.from_data([1.0, 0.0, 0.0, 0.0, 0.0])
y = Insight.signal.lfilter(b, a, x)
check("lfilter_fir_numel", Insight.numel(y) == 6)
check("lfilter_fir_0", approx(Insight.item(y, 0), 1.0, atol=1e-10))
check("lfilter_fir_1", approx(Insight.item(y, 1), 1.0, atol=1e-10))
check("lfilter_fir_2", approx(Insight.item(y, 2), 0.0, atol=1e-10))

b = Insight.from_data([1.0/3, 1.0/3, 1.0/3])
a = Insight.from_data([1.0])
x = Insight.from_data([1.0, 2.0, 3.0, 4.0, 5.0])
y = Insight.signal.lfilter(b, a, x)
check("lfilter_ma_1", approx(Insight.item(y, 1), 1.0, atol=1e-10))
check("lfilter_ma_2", approx(Insight.item(y, 2), 2.0, atol=1e-10))
check("lfilter_ma_3", approx(Insight.item(y, 3), 3.0, atol=1e-10))
check("lfilter_ma_4", approx(Insight.item(y, 4), 4.0, atol=1e-10))

b = Insight.from_data([1.0])
a = Insight.from_data([1.0, -0.5])
x = Insight.from_data([1.0, 0.0, 0.0, 0.0, 0.0])
y = Insight.signal.lfilter(b, a, x)
check("lfilter_iir_0", approx(Insight.item(y, 0), 1.0, atol=1e-10))
check("lfilter_iir_1", approx(Insight.item(y, 1), 0.5, atol=1e-10))
check("lfilter_iir_2", approx(Insight.item(y, 2), 0.25, atol=1e-10))
check("lfilter_iir_3", approx(Insight.item(y, 3), 0.125, atol=1e-10))
check("lfilter_iir_4", approx(Insight.item(y, 4), 0.0625, atol=1e-10))

# ============================================================================
# Filtfilt (1 test)
# ============================================================================
println("=== Filtfilt ===")

b = Insight.from_data([1.0, 1.0])
a = Insight.from_data([1.0])
x = Insight.from_data([1.0, 2.0, 3.0, 4.0, 5.0])
y = Insight.signal.filtfilt(b, a, x)
check("filtfilt_numel", Insight.numel(y) == 5)

# ============================================================================
# Freq Shift (2 tests)
# ============================================================================
println("=== Freq Shift ===")

N = 64
x = Insight.from_data(ones(N))
y = Insight.signal.freq_shift(x, 0.0, 1.0)
check("freq_shift_zero", Insight.numel(y) == N)

N = 16
x = Insight.from_data(ones(N))
y = Insight.signal.freq_shift(x, 0.25, 1.0)
check("freq_shift_positive", Insight.numel(y) == N)

# ============================================================================
# Decimate (2 tests)
# ============================================================================
println("=== Decimate ===")

N = 100
data = [sin(2π * i / N) for i in 0:N-1]
x = Insight.from_data(data)
y = Insight.signal.decimate(x, 2)
check("decimate_basic", Insight.numel(y) == 50)

y = Insight.signal.decimate(x, 4)
check("decimate_zero_phase", Insight.numel(y) == 25)

# ============================================================================
# Resample (2 tests)
# ============================================================================
println("=== Resample ===")

x = Insight.from_data([1.0, 2.0, 3.0, 4.0, 5.0])
y = Insight.signal.resample(x, 5)
check("resample_identity_numel", Insight.numel(y) == 5)
check("resample_identity_0", approx(Insight.item(y, 0), 1.0, atol=1e-6))
check("resample_identity_4", approx(Insight.item(y, 4), 5.0, atol=1e-6))

x = Insight.from_data([1.0, 2.0, 3.0, 4.0])
y = Insight.signal.resample(x, 8)
check("resample_upsample", Insight.numel(y) == 8)

# ============================================================================
# Resample Poly (2 tests)
# ============================================================================
println("=== Resample Poly ===")

x = Insight.from_data([1.0, 2.0, 3.0, 4.0, 5.0])
y = Insight.signal.resample_poly(x, 1, 1)
check("resample_poly_identity_numel", Insight.numel(y) == 5)
check("resample_poly_identity_0", approx(Insight.item(y, 0), 1.0, atol=1e-10))

x = Insight.from_data([1.0, 0.0, 1.0, 0.0])
y = Insight.signal.resample_poly(x, 2, 1)
check("resample_poly_upsample", Insight.numel(y) >= 7 && Insight.numel(y) <= 9)

# ============================================================================
# Wiener (1 test)
# ============================================================================
println("=== Wiener ===")

x = Insight.from_data([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
x = Insight.reshape(x, [3, 3])
y = Insight.signal.wiener(x)
check("wiener_numel", Insight.numel(y) == 9)

# ============================================================================
# Firfilter (1 test)
# ============================================================================
println("=== Firfilter ===")

b = Insight.from_data([1.0, 2.0, 1.0])
x = Insight.from_data([1.0, 0.0, 0.0, 0.0, 0.0])
y = Insight.signal.firfilter(b, x)
check("firfilter_numel", Insight.numel(y) == 7)
check("firfilter_0", approx(Insight.item(y, 0), 1.0, atol=1e-10))
check("firfilter_1", approx(Insight.item(y, 1), 2.0, atol=1e-10))
check("firfilter_2", approx(Insight.item(y, 2), 1.0, atol=1e-10))

# ============================================================================
# Results
# ============================================================================
println("\n" * "="^40)
println("Results: $passed passed, $failed failed")
if failed > 0
    exit(1)
end
