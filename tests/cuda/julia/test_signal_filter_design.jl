# Signal Filter Design CUDA binding tests.
# Run with:
#   LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda julia tests/cuda/julia/test_signal_filter_design.jl

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
# Kaiser Beta (4 tests)
# ============================================================================
println("=== Kaiser Beta ===")

check("kaiser_beta_low_0", approx(Insight.signal.kaiser_beta(0.0), 0.0))
check("kaiser_beta_low_21", approx(Insight.signal.kaiser_beta(21.0), 0.0))

a = 40.0
expected = 0.5842 * (a - 21.0)^0.4 + 0.07886 * (a - 21.0)
check("kaiser_beta_mid", approx(Insight.signal.kaiser_beta(a), expected))

a = 60.0
expected = 0.1102 * (a - 8.7)
check("kaiser_beta_high", approx(Insight.signal.kaiser_beta(a), expected))

beta = Insight.signal.kaiser_beta(50.0)
check("kaiser_beta_boundary", beta > 4.0 && beta < 6.0)

# ============================================================================
# Kaiser Atten (1 test)
# ============================================================================
println("=== Kaiser Atten ===")

atten = Insight.signal.kaiser_atten(101, 0.1)
expected = 2.285 * 100.0 * π * 0.1 + 7.95
check("kaiser_atten_basic", approx(atten, expected))

# ============================================================================
# Firwin (5 tests)
# ============================================================================
println("=== Firwin ===")

h = Insight.signal.firwin(11, [0.3], window="hamming", pass_zero="lowpass", scale=true)
check("firwin_lowpass_numel", Insight.numel(h) == 11)
# Symmetric
sym_ok = true
for i in 0:4
    if !approx(Insight.item(h, i), Insight.item(h, 10 - i))
        sym_ok = false; break
    end
end
check("firwin_lowpass_sym", sym_ok)

h = Insight.signal.firwin(21, [0.25], window="hamming", pass_zero="lowpass", scale=true)
dc = sum([Insight.item(h, i) for i in 0:20])
check("firwin_lowpass_dc_gain", approx(dc, 1.0, atol=1e-6))

h = Insight.signal.firwin(11, [0.3], window="hamming", pass_zero="highpass", scale=true)
check("firwin_highpass_numel", Insight.numel(h) == 11)

h = Insight.signal.firwin(21, [0.2, 0.5], window="hamming", pass_zero="bandpass", scale=false)
check("firwin_bandpass_numel", Insight.numel(h) == 21)

h = Insight.signal.firwin(21, [0.2, 0.5], window="hamming", pass_zero="bandstop", scale=false)
check("firwin_bandstop_numel", Insight.numel(h) == 21)

# ============================================================================
# Firwin2 (2 tests)
# ============================================================================
println("=== Firwin2 ===")

h = Insight.signal.firwin2(15, [0.0, 0.5, 0.5, 1.0], [1.0, 1.0, 0.0, 0.0])
check("firwin2_lowpass", Insight.numel(h) == 15)

h_short = Insight.signal.firwin2(7, [0.0, 0.5, 1.0], [1.0, 1.0, 0.0])
h_long = Insight.signal.firwin2(31, [0.0, 0.5, 1.0], [1.0, 1.0, 0.0])
check("firwin2_diff_lengths", Insight.numel(h_short) == 7 && Insight.numel(h_long) == 31)

# ============================================================================
# Cmplx Sort (2 tests)
# ============================================================================
println("=== Cmplx Sort ===")

a = Insight.from_data([3.0, -1.0, 2.0, -4.0])
sorted = Insight.signal.cmplx_sort(a)
check("cmplx_sort_numel", Insight.numel(sorted) == 4)
check("cmplx_sort_0", approx(Insight.item(sorted, 0), -1.0))
check("cmplx_sort_1", approx(Insight.item(sorted, 1), 2.0))
check("cmplx_sort_2", approx(Insight.item(sorted, 2), 3.0))
check("cmplx_sort_3", approx(Insight.item(sorted, 3), -4.0))

a = Insight.from_data([42.0])
sorted = Insight.signal.cmplx_sort(a)
check("cmplx_sort_single", Insight.numel(sorted) == 1 && approx(Insight.item(sorted, 0), 42.0))

# ============================================================================
# Results
# ============================================================================
println("\n" * "="^40)
println("Results: $passed passed, $failed failed")
if failed > 0
    exit(1)
end
