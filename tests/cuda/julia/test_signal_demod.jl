# Signal demod CUDA binding tests
# Run with:
#   LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda julia tests/cuda/julia/test_signal_demod.jl

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

function make_fm_signal(n::Int, fs::Float64, fc::Float64, fm::Float64, dev::Float64)
    t = [i / fs for i in 0:n-1]
    message = [sin(2 * pi * fm * ti) for ti in t]
    phase = zeros(n)
    dt = 1.0 / fs
    for i in 2:n
        phase[i] = phase[i-1] + 2 * pi * (fc + dev * message[i-1]) * dt
    end
    signal = [cos(p) for p in phase]
    cpu_arr = Insight.from_data(signal)
    return Insight.to_array(cpu_arr, Insight.GPUPlace(0))
end

# ============================================================================
# FM Demodulation
# ============================================================================
println("=== Signal Demod ===")

# fm_demod
x = make_fm_signal(512, 256.0, 20.0, 2.0, 5.0)
result = Insight.fm_demod(x)
check("fm_demod", Insight.numel(result) > 0)

# fm_demod output length
x = make_fm_signal(256, 256.0, 20.0, 2.0, 5.0)
result = Insight.fm_demod(x)
check("fm_demod_length", Insight.numel(result) == 256)

# fm_demod short signal
x = make_fm_signal(128, 256.0, 10.0, 1.0, 3.0)
result = Insight.fm_demod(x)
check("fm_demod_short", Insight.numel(result) == 128)

# fm_demod longer signal
x = make_fm_signal(1024, 256.0, 20.0, 2.0, 10.0)
result = Insight.fm_demod(x)
check("fm_demod_long", Insight.numel(result) == 1024)

# ============================================================================
# Results
# ============================================================================
println("\n" * "="^40)
println("Results: $passed passed, $failed failed")
if failed > 0
    exit(1)
end
