# Signal Windows CUDA binding tests.
# Run with:
#   LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda julia tests/cuda/julia/test_signal_windows.jl

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

function all_approx(arr, expected; atol=1e-10)
    for i in eachindex(expected)
        if !approx(Insight.item(arr, i - 1), expected[i], atol=atol)
            return false
        end
    end
    return true
end

# ============================================================================
# Boxcar (2 tests)
# ============================================================================
println("=== Boxcar ===")

w = Insight.signal.boxcar(5)
check("boxcar_numel", Insight.numel(w) == 5)
check("boxcar_values", all_approx(w, [1.0, 1.0, 1.0, 1.0, 1.0]))

w = Insight.signal.boxcar(1)
check("boxcar_size1", Insight.numel(w) == 1 && approx(Insight.item(w, 0), 1.0))

# ============================================================================
# Triang (2 tests)
# ============================================================================
println("=== Triang ===")

w = Insight.signal.triang(5)
check("triang_odd_numel", Insight.numel(w) == 5)
check("triang_odd_values", all_approx(w, [1/3, 2/3, 1.0, 2/3, 1/3]))

w = Insight.signal.triang(4)
check("triang_even_numel", Insight.numel(w) == 4)
check("triang_even_values", all_approx(w, [0.25, 0.75, 0.75, 0.25]))

# ============================================================================
# Parzen (1 test)
# ============================================================================
println("=== Parzen ===")

w = Insight.signal.parzen_win(8)
check("parzen_numel", Insight.numel(w) == 8)
check("parzen_0", approx(Insight.item(w, 0), 0.00390625, atol=1e-6))
check("parzen_1", approx(Insight.item(w, 1), 0.10546875, atol=1e-6))
check("parzen_2", approx(Insight.item(w, 2), 0.47265625, atol=1e-6))
check("parzen_3", approx(Insight.item(w, 3), 0.91796875, atol=1e-6))

# ============================================================================
# Bohman (1 test)
# ============================================================================
println("=== Bohman ===")

w = Insight.signal.bohman_win(11)
check("bohman_numel", Insight.numel(w) == 11)
check("bohman_start", approx(Insight.item(w, 0), 0.0, atol=1e-10))
check("bohman_end", approx(Insight.item(w, 10), 0.0, atol=1e-10))
check("bohman_center", approx(Insight.item(w, 5), 1.0, atol=1e-6))

# ============================================================================
# Bartlett (1 test)
# ============================================================================
println("=== Bartlett ===")

w = Insight.signal.bartlett(5)
check("bartlett_numel", Insight.numel(w) == 5)
check("bartlett_values", all_approx(w, [0.0, 0.5, 1.0, 0.5, 0.0]))

# ============================================================================
# Cosine (1 test)
# ============================================================================
println("=== Cosine ===")

w = Insight.signal.cosine_win(4)
check("cosine_numel", Insight.numel(w) == 4)
check("cosine_0", approx(Insight.item(w, 0), sin(π/8), atol=1e-10))
check("cosine_1", approx(Insight.item(w, 1), sin(3π/8), atol=1e-10))

# ============================================================================
# Exponential (1 test)
# ============================================================================
println("=== Exponential ===")

w = Insight.signal.exponential_win(5, center=2.0, tau=1.0)
check("exponential_numel", Insight.numel(w) == 5)
check("exponential_center", approx(Insight.item(w, 2), 1.0, atol=1e-10))
check("exponential_sym", approx(Insight.item(w, 0), Insight.item(w, 4), atol=1e-10))
check("exponential_decay", approx(Insight.item(w, 0), exp(-2.0), atol=1e-10))

# ============================================================================
# Blackman (1 test)
# ============================================================================
println("=== Blackman ===")

w = Insight.signal.blackman(5)
check("blackman_numel", Insight.numel(w) == 5)
check("blackman_start", approx(Insight.item(w, 0), 0.0, atol=1e-10))
check("blackman_end", approx(Insight.item(w, 4), 0.0, atol=1e-10))
check("blackman_center", approx(Insight.item(w, 2), 1.0, atol=1e-6))

# ============================================================================
# Nuttall (1 test)
# ============================================================================
println("=== Nuttall ===")

w = Insight.signal.nuttall(5)
check("nuttall_numel", Insight.numel(w) == 5)
check("nuttall_0", approx(Insight.item(w, 0), 0.0003628, atol=1e-4))
check("nuttall_center", approx(Insight.item(w, 2), 1.0, atol=1e-6))

# ============================================================================
# Blackmanharris (1 test)
# ============================================================================
println("=== Blackmanharris ===")

w = Insight.signal.blackmanharris(5)
check("blackmanharris_numel", Insight.numel(w) == 5)
check("blackmanharris_0", approx(Insight.item(w, 0), 6.0e-5, atol=1e-4))
check("blackmanharris_center", approx(Insight.item(w, 2), 1.0, atol=1e-6))

# ============================================================================
# Flattop (1 test)
# ============================================================================
println("=== Flattop ===")

w = Insight.signal.flattop(5)
check("flattop_numel", Insight.numel(w) == 5)
check("flattop_sym_0", approx(Insight.item(w, 0), Insight.item(w, 4), atol=1e-10))
check("flattop_sym_1", approx(Insight.item(w, 1), Insight.item(w, 3), atol=1e-10))

# ============================================================================
# Hann (2 tests)
# ============================================================================
println("=== Hann ===")

w = Insight.signal.hann(5)
check("hann_numel", Insight.numel(w) == 5)
check("hann_start", approx(Insight.item(w, 0), 0.0, atol=1e-10))
check("hann_end", approx(Insight.item(w, 4), 0.0, atol=1e-10))
check("hann_center", approx(Insight.item(w, 2), 1.0, atol=1e-10))
check("hann_1", approx(Insight.item(w, 1), 0.5, atol=1e-10))

w = Insight.signal.hann(16)
check("hann_size16", Insight.numel(w) == 16)

# ============================================================================
# Hamming (1 test)
# ============================================================================
println("=== Hamming ===")

w = Insight.signal.hamming(5)
check("hamming_numel", Insight.numel(w) == 5)
check("hamming_start", approx(Insight.item(w, 0), 0.08, atol=1e-10))
check("hamming_end", approx(Insight.item(w, 4), 0.08, atol=1e-10))
check("hamming_center", approx(Insight.item(w, 2), 1.0, atol=1e-10))

# ============================================================================
# Tukey (2 tests)
# ============================================================================
println("=== Tukey ===")

w = Insight.signal.tukey(10, alpha=0.0)
check("tukey_alpha0_numel", Insight.numel(w) == 10)
check("tukey_alpha0_values", all_approx(w, ones(10)))

w_tukey = Insight.signal.tukey(10, alpha=1.0)
w_hann = Insight.signal.hann(10)
check("tukey_alpha1_match", all_approx(w_tukey, [Insight.item(w_hann, i) for i in 0:9]))

# ============================================================================
# Barthann (1 test)
# ============================================================================
println("=== Barthann ===")

w = Insight.signal.barthann_win(11)
check("barthann_numel", Insight.numel(w) == 11)
check("barthann_center", approx(Insight.item(w, 5), 1.0, atol=0.02))

# ============================================================================
# Kaiser (1 test)
# ============================================================================
println("=== Kaiser ===")

w = Insight.signal.kaiser(5, 0.0)
check("kaiser_beta0_numel", Insight.numel(w) == 5)
check("kaiser_beta0_values", all_approx(w, ones(5)))

# ============================================================================
# Gaussian (1 test)
# ============================================================================
println("=== Gaussian ===")

w = Insight.signal.gaussian(5, 1.0)
check("gaussian_numel", Insight.numel(w) == 5)
check("gaussian_center", approx(Insight.item(w, 2), 1.0, atol=1e-10))
check("gaussian_sym", approx(Insight.item(w, 0), Insight.item(w, 4), atol=1e-10))
check("gaussian_decay", approx(Insight.item(w, 0), exp(-2.0), atol=1e-10))

# ============================================================================
# General Gaussian (1 test)
# ============================================================================
println("=== General Gaussian ===")

w = Insight.signal.general_gaussian_win(5, 1.0, 1.0)
check("gen_gaussian_numel", Insight.numel(w) == 5)
check("gen_gaussian_center", approx(Insight.item(w, 2), 1.0, atol=1e-10))
check("gen_gaussian_decay", approx(Insight.item(w, 0), exp(-2.0), atol=1e-10))

# ============================================================================
# Chebwin (1 test)
# ============================================================================
println("=== Chebwin ===")

w = Insight.signal.chebwin(11, 50.0)
check("chebwin_numel", Insight.numel(w) == 11)
check("chebwin_sym", approx(Insight.item(w, 0), Insight.item(w, 10), atol=1e-6))

# ============================================================================
# Taylor (1 test)
# ============================================================================
println("=== Taylor ===")

w = Insight.signal.taylor(32, nbar=4, sll=-30.0, norm=true)
check("taylor_numel", Insight.numel(w) == 32)
check("taylor_sym", approx(Insight.item(w, 0), Insight.item(w, 31), atol=1e-6))

# ============================================================================
# Get Window (2 tests)
# ============================================================================
println("=== Get Window ===")

w = Insight.signal.get_window("boxcar", 8)
check("get_window_boxcar_numel", Insight.numel(w) == 8)
check("get_window_boxcar_values", all_approx(w, ones(8)))

w = Insight.signal.get_window("hann", 5)
ref = Insight.signal.hann(5)
check("get_window_hann", all_approx(w, [Insight.item(ref, i) for i in 0:4]))

# ============================================================================
# Results
# ============================================================================
println("\n" * "="^40)
println("Results: $passed passed, $failed failed")
if failed > 0
    exit(1)
end
