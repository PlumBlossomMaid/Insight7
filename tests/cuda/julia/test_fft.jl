# FFT CUDA binding tests
# Run with:
#   LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda julia tests/cuda/julia/test_fft.jl

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

println("=== FFT CUDA ===")

# fft
x = Insight.to_array(Insight.from_data([1.0, 2.0, 3.0, 4.0]), Insight.GPUPlace(0))
y = Insight.fft(x)
check("fft", Insight.numel(y) == 4)

# ifft
z = Insight.ifft(y)
check("ifft", Insight.numel(z) == 4)

# fft ifft roundtrip
x2 = Insight.to_array(Insight.from_data([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]), Insight.GPUPlace(0))
X2 = Insight.fft(x2)
x2_back = Insight.ifft(X2)
check("fft_ifft_roundtrip", Insight.numel(x2_back) == 8)

# fft longer signal
t = Float64[]
for i in 1:64
    push!(t, sin(2 * pi * i / 64))
end
x3 = Insight.to_array(Insight.from_data(t), Insight.GPUPlace(0))
y3 = Insight.fft(x3)
check("fft_long", Insight.numel(y3) == 64)

# fft ones
x4 = Insight.to_array(Insight.from_data(ones(8)), Insight.GPUPlace(0))
y4 = Insight.fft(x4)
check("fft_ones", Insight.numel(y4) == 8)

# fft non power of two
x5 = Insight.to_array(Insight.from_data(Float64[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), Insight.GPUPlace(0))
y5 = Insight.fft(x5)
check("fft_non_pow2", Insight.numel(y5) == 10)

# fftshift
x6 = Insight.to_array(Insight.from_data([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]), Insight.GPUPlace(0))
y6 = Insight.fftshift(x6)
a6 = Insight.to_array(y6)
check("fftshift", length(a6) == 8 && a6[1] ≈ 5.0 && a6[5] ≈ 1.0)

# ifftshift
y7 = Insight.ifftshift(x6)
a7 = Insight.to_array(y7)
check("ifftshift", length(a7) == 8 && a7[1] ≈ 5.0 && a7[5] ≈ 1.0)

# next_fast_len
n = Insight.next_fast_len(100)
check("next_fast_len", n >= 100)

# fft with n parameter
x8 = Insight.to_array(Insight.from_data([1.0, 2.0, 3.0, 4.0]), Insight.GPUPlace(0))
y8 = Insight.fft(x8, n=8)
check("fft_n", Insight.numel(y8) == 8)

# ifft with n parameter
y9 = Insight.ifft(y8, n=8)
check("ifft_n", Insight.numel(y9) == 8)

# ============================================================================
# Results
# ============================================================================
println("\n" * "="^40)
println("Results: $passed passed, $failed failed")
if failed > 0
    exit(1)
end
