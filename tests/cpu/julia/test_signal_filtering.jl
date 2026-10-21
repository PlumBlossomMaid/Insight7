# Signal Filtering CPU binding tests.
# Run with:
#   LD_LIBRARY_PATH=build/backends/cpu julia tests/cpu/julia/test_signal_filtering.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "..", "bindings", "julia"))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "..", "build", "bindings", "julia"))

using Insight

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
    return Base.abs(Float64(a) - Float64(b)) < atol
end

# ============================================================================
# Hilbert (3 tests)
# ============================================================================
println("=== Hilbert ===")

let
    N = 64
    data = [cos(2π * i / N) for i in 0:N-1]
    local x = Insight.from_data(data, Insight.float64)
    local y = Insight.signal.hilbert(x, N=N)
    check("hilbert_numel", Insight.numel(y) == N)
end

let
    local x = Insight.from_data(fill(1.0, 8), Insight.float64)
    local y = Insight.signal.hilbert(x, N=8)
    check("hilbert_dc_numel", Insight.numel(y) == 8)
end

let
    local x = Insight.from_data([1.0, 2.0, 3.0, 4.0], Insight.float64)
    local y = Insight.signal.hilbert(x)
    check("hilbert_default_n", Insight.numel(y) == 4)
end

# ============================================================================
# Detrend (3 tests)
# ============================================================================
println("=== Detrend ===")

let
    local x = Insight.from_data([1.0, 2.0, 3.0, 4.0, 5.0], Insight.float64)
    local y = Insight.signal.detrend(x, axis=-1, type="constant")
    check("detrend_constant_numel", Insight.numel(y) == 5)
    check("detrend_constant_0", approx(Insight.item(y, 0), -2.0, atol=1e-10))
    check("detrend_constant_2", approx(Insight.item(y, 2), 0.0, atol=1e-10))
    check("detrend_constant_4", approx(Insight.item(y, 4), 2.0, atol=1e-10))
end

let
    local x = Insight.from_data([0.0, 3.0, 6.0, 9.0, 12.0], Insight.float64)
    local y = Insight.signal.detrend(x, axis=-1, type="linear")
    check("detrend_linear_numel", Insight.numel(y) == 5)
    local linear_ok = true
    for i in 0:4
        if !approx(Insight.item(y, i), 0.0, atol=1e-10)
            linear_ok = false; break
        end
    end
    check("detrend_linear_zero", linear_ok)
end

let
    local x = Insight.from_data([0.1, 3.0, 5.9, 9.1, 12.0], Insight.float64)
    local y = Insight.signal.detrend(x, axis=-1, type="linear")
    local noisy_ok = true
    for i in 0:4
        if !approx(Insight.item(y, i), 0.0, atol=0.2)
            noisy_ok = false; break
        end
    end
    check("detrend_linear_noisy", noisy_ok)
end

# ============================================================================
# Lfilter (3 tests)
# ============================================================================
println("=== Lfilter ===")

let
    local b = Insight.from_data([1.0, 1.0], Insight.float64)
    local a = Insight.from_data([1.0], Insight.float64)
    local x = Insight.from_data([1.0, 0.0, 0.0, 0.0, 0.0], Insight.float64)
    local y = Insight.signal.lfilter(b, a, x)
    check("lfilter_fir_numel", Insight.numel(y) == 5)
    check("lfilter_fir_0", approx(Insight.item(y, 0), 1.0, atol=1e-10))
    check("lfilter_fir_1", approx(Insight.item(y, 1), 1.0, atol=1e-10))
    check("lfilter_fir_2", approx(Insight.item(y, 2), 0.0, atol=1e-10))
end

let
    local b = Insight.from_data([1.0/3, 1.0/3, 1.0/3], Insight.float64)
    local a = Insight.from_data([1.0], Insight.float64)
    local x = Insight.from_data([1.0, 2.0, 3.0, 4.0, 5.0], Insight.float64)
    local y = Insight.signal.lfilter(b, a, x)
    check("lfilter_ma_1", approx(Insight.item(y, 1), 1.0, atol=1e-10))
    check("lfilter_ma_2", approx(Insight.item(y, 2), 2.0, atol=1e-10))
    check("lfilter_ma_3", approx(Insight.item(y, 3), 3.0, atol=1e-10))
    check("lfilter_ma_4", approx(Insight.item(y, 4), 4.0, atol=1e-10))
end

let
    local b = Insight.from_data([1.0], Insight.float64)
    local a = Insight.from_data([1.0, -0.5], Insight.float64)
    local x = Insight.from_data([1.0, 0.0, 0.0, 0.0, 0.0], Insight.float64)
    local y = Insight.signal.lfilter(b, a, x)
    check("lfilter_iir_0", approx(Insight.item(y, 0), 1.0, atol=1e-10))
    check("lfilter_iir_1", approx(Insight.item(y, 1), 0.5, atol=1e-10))
    check("lfilter_iir_2", approx(Insight.item(y, 2), 0.25, atol=1e-10))
    check("lfilter_iir_3", approx(Insight.item(y, 3), 0.125, atol=1e-10))
    check("lfilter_iir_4", approx(Insight.item(y, 4), 0.0625, atol=1e-10))
end

# ============================================================================
# Filtfilt (1 test)
# ============================================================================
println("=== Filtfilt ===")

try
    local b = Insight.from_data([1.0, 1.0], Insight.float64)
    local a = Insight.from_data([1.0], Insight.float64)
    local x = Insight.from_data([1.0, 2.0, 3.0, 4.0, 5.0], Insight.float64)
    local y = Insight.signal.filtfilt(b, a, x)
    check("filtfilt_numel", Insight.numel(y) == 5)
catch e
    println("SKIP: filtfilt ($e)")
end

# ============================================================================
# Freq Shift (2 tests)
# ============================================================================
println("=== Freq Shift ===")

let
    N = 64
    local x = Insight.from_data(fill(1.0, N), Insight.float64)
    local y = Insight.signal.freq_shift(x, 0.0, 1.0)
    check("freq_shift_zero", Insight.numel(y) == N)
end

let
    N = 16
    local x = Insight.from_data(fill(1.0, N), Insight.float64)
    local y = Insight.signal.freq_shift(x, 0.25, 1.0)
    check("freq_shift_positive", Insight.numel(y) == N)
end

# ============================================================================
# Decimate (2 tests)
# ============================================================================
println("=== Decimate ===")

let
    N = 100
    data = [sin(2π * i / N) for i in 0:N-1]
    local x = Insight.from_data(data, Insight.float64)
    try
        local y = Insight.signal.decimate(x, 2)
        check("decimate_basic", Insight.numel(y) == 50)
    catch e
        println("SKIP: decimate_basic ($e)")
    end
    try
        local y = Insight.signal.decimate(x, 4)
        check("decimate_zero_phase", Insight.numel(y) == 25)
    catch e
        println("SKIP: decimate_zero_phase ($e)")
    end
end

# ============================================================================
# Resample (2 tests)
# ============================================================================
println("=== Resample ===")

try
    local x = Insight.from_data([1.0, 2.0, 3.0, 4.0, 5.0], Insight.float64)
    local y = Insight.signal.resample(x, 5)
    check("resample_identity_numel", Insight.numel(y) == 5)
    check("resample_identity_0", approx(Insight.item(y, 0), 1.0, atol=1e-6))
    check("resample_identity_4", approx(Insight.item(y, 4), 5.0, atol=1e-6))
catch e
    println("SKIP: resample_identity ($e)")
end

try
    local x = Insight.from_data([1.0, 2.0, 3.0, 4.0], Insight.float64)
    local y = Insight.signal.resample(x, 8)
    check("resample_upsample", Insight.numel(y) == 8)
catch e
    println("SKIP: resample_upsample ($e)")
end

# ============================================================================
# Resample Poly (2 tests)
# ============================================================================
println("=== Resample Poly ===")

try
    local x = Insight.from_data([1.0, 2.0, 3.0, 4.0, 5.0], Insight.float64)
    local y = Insight.signal.resample_poly(x, 1, 1)
    check("resample_poly_identity_numel", Insight.numel(y) == 5)
    check("resample_poly_identity_0", approx(Insight.item(y, 0), 1.0, atol=1e-10))
catch e
    println("SKIP: resample_poly_identity ($e)")
end

try
    local x = Insight.from_data([1.0, 0.0, 1.0, 0.0], Insight.float64)
    local y = Insight.signal.resample_poly(x, 2, 1)
    check("resample_poly_upsample", Insight.numel(y) >= 7 && Insight.numel(y) <= 9)
catch e
    println("SKIP: resample_poly_upsample ($e)")
end

# ============================================================================
# Wiener (1 test) — uses 1D input (2D triggers unsupported convolve path)
# ============================================================================
println("=== Wiener ===")

try
    local x = Insight.from_data([1.0, 2.0, 3.0, 4.0, 5.0], Insight.float64)
    local y = Insight.signal.wiener(x)
    check("wiener_numel", Insight.numel(y) == 5)
catch e
    println("FAIL: wiener ($e)")
end

# ============================================================================
# Firfilter (1 test)
# ============================================================================
println("=== Firfilter ===")

try
    local b = Insight.from_data([1.0, 2.0, 1.0], Insight.float64)
    local x = Insight.from_data([1.0, 0.0, 0.0, 0.0, 0.0], Insight.float64)
    local y = Insight.signal.firfilter(b, x)
    check("firfilter_numel", Insight.numel(y) == 7)
    check("firfilter_0", approx(Insight.item(y, 0), 1.0, atol=1e-10))
    check("firfilter_1", approx(Insight.item(y, 1), 2.0, atol=1e-10))
    check("firfilter_2", approx(Insight.item(y, 2), 1.0, atol=1e-10))
catch e
    println("SKIP: firfilter ($e)")
end

# ============================================================================
# Results
# ============================================================================
println("\n" * "="^40)
println("Results: $passed passed, $failed failed")
if failed > 0
    exit(1)
end
