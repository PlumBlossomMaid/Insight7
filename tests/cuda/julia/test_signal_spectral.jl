# Signal spectral CUDA binding tests
# Run with:
#   LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda julia tests/cuda/julia/test_signal_spectral.jl

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

function make_sine(freq::Float64, fs::Float64, n::Int)
    t = [sin(2 * pi * freq * i / fs) for i in 0:n-1]
    cpu_arr = Insight.from_data(t)
    return Insight.to_array(cpu_arr, Insight.GPUPlace(0))
end

# ============================================================================
# Spectral Analysis
# ============================================================================
println("=== Signal Spectral ===")

# welch
x = make_sine(10.0, 256.0, 1024)
f, pxx = Insight.welch_jl(x, fs=256.0, nperseg=256)
check("welch", Insight.numel(f) > 0 && Insight.numel(pxx) > 0)

# periodogram
x = make_sine(10.0, 256.0, 512)
f, pxx = Insight.periodogram_jl(x, fs=256.0)
check("periodogram", Insight.numel(f) > 0 && Insight.numel(pxx) > 0)

# csd
x = make_sine(10.0, 256.0, 1024)
y = make_sine(10.0, 256.0, 1024)
result = Insight.csd(x, y, fs=256.0, nperseg=256)
check("csd", Insight.numel(result.f) > 0 && Insight.numel(result.Pxx) > 0)

# coherence
x = make_sine(10.0, 256.0, 1024)
y = make_sine(10.0, 256.0, 1024)
result = Insight.coherence(x, y, fs=256.0, nperseg=256)
check("coherence", Insight.numel(result.f) > 0 && Insight.numel(result.Pxx) > 0)

# spectrogram
x = make_sine(10.0, 256.0, 1024)
result = Insight.spectrogram(x, fs=256.0, nperseg=256)
check("spectrogram", Insight.numel(result.f) > 0 && Insight.numel(result.t) > 0 && Insight.numel(result.Sxx) > 0)

# stft
x = make_sine(10.0, 256.0, 1024)
result = Insight.stft(x, fs=256.0, nperseg=256)
check("stft", Insight.numel(result.f) > 0 && Insight.numel(result.t) > 0 && Insight.numel(result.Sxx) > 0)

# vectorstrength
events = Insight.from_data([0.0, 1.0, 2.0, 3.0])
events_gpu = Insight.to_array(events, Insight.GPUPlace(0))
result = Insight.vectorstrength(events_gpu, 1.0)
check("vectorstrength", result.strength >= 0.0 && result.phase >= 0.0)

# welch with custom params
x = make_sine(10.0, 256.0, 2048)
f, pxx = Insight.welch_jl(x, fs=256.0, nperseg=512, noverlap=256, scaling="spectrum")
check("welch_custom", Insight.numel(f) > 0 && Insight.numel(pxx) > 0)

# ============================================================================
# Results
# ============================================================================
println("\n" * "="^40)
println("Results: $passed passed, $failed failed")
if failed > 0
    exit(1)
end
