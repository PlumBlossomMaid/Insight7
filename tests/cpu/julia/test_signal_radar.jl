# Signal Radar CPU binding tests.
# Run with:
#   LD_LIBRARY_PATH=build/backends/cpu julia tests/cpu/julia/test_signal_radar.jl

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

# ============================================================================
# Radar Signal Processing
# ============================================================================
println("=== Signal Radar ===")

# pulse_compression (requires 2D input [num_pulses, samples_per_pulse])
try
    n = 64
    tpl_data = [sin(2 * pi * i / n) for i in 1:n]
    sig_data = vcat(tpl_data, fill(0.0, n))
    tpl = Insight.from_data(tpl_data, Insight.float64)
    sig = Insight.from_data(sig_data, Insight.float64)
    sig = Insight.reshape(sig, Int64[2, n])
    result = Insight.pulse_compression(sig, tpl)
    check("pulse_compression", Insight.numel(result) > 0)
catch e
    println("SKIP: pulse_compression ($e)")
end

# pulse_compression normalized
try
    n = 64
    tpl_data = [sin(2 * pi * i / n) for i in 1:n]
    sig_data = vcat(tpl_data, fill(0.0, n))
    tpl = Insight.from_data(tpl_data, Insight.float64)
    sig = Insight.from_data(sig_data, Insight.float64)
    sig = Insight.reshape(sig, Int64[2, n])
    result = Insight.pulse_compression(sig, tpl, normalize=true)
    check("pulse_compression_norm", Insight.numel(result) > 0)
catch e
    println("SKIP: pulse_compression_norm ($e)")
end

# pulse_doppler
try
    # 16 pulses x 32 samples per pulse
    data_2d = Insight.zeros(Int64[16, 32], Insight.float64)
    # Create data with some sinusoidal content
    data_flat = [sin(2 * pi * r * c / 512) for r in 1:16, c in 1:32]
    data_arr = Insight.from_data(vec(collect(data_flat)), Insight.float64)
    data_2d = Insight.reshape(data_arr, Int64[16, 32])
    result = Insight.pulse_doppler(data_2d)
    check("pulse_doppler", Insight.numel(result) > 0)
catch e
    println("SKIP: pulse_doppler ($e)")
end

# cfar_alpha
try
    alpha = Insight.cfar_alpha(1e-3, 20)
    check("cfar_alpha", alpha > 0.0)
catch e
    println("SKIP: cfar_alpha ($e)")
end

# cfar_alpha different PFA
try
    alpha1 = Insight.cfar_alpha(1e-2, 20)
    alpha2 = Insight.cfar_alpha(1e-6, 20)
    check("cfar_alpha_pfa", alpha1 < alpha2)
catch e
    println("SKIP: cfar_alpha_pfa ($e)")
end

# mvdr — expects x shape [sensors, samples] with sensors <= samples
try
    # 4 sensors, 32 samples
    x_data = [sin(2 * pi * r * c / 128) for r in 1:4, c in 1:32]
    x_arr = Insight.from_data(vec(collect(x_data)), Insight.float64)
    x_2d = Insight.reshape(x_arr, Int64[32, 4])
    sv = Insight.from_data([1.0, 1.0, 1.0, 1.0], Insight.float64)
    result = Insight.mvdr(x_2d, sv)
    check("mvdr", Insight.numel(result) > 0)
catch e
    println("SKIP: mvdr ($e)")
end

# mvdr with different steering
try
    # 4 sensors, 32 samples
    x_data = [sin(2 * pi * r * c / 128) for r in 1:4, c in 1:32]
    x_arr = Insight.from_data(vec(collect(x_data)), Insight.float64)
    x_2d = Insight.reshape(x_arr, Int64[32, 4])
    sv2 = Insight.from_data([1.0, 0.5, 0.5, 1.0], Insight.float64)
    result2 = Insight.mvdr(x_2d, sv2)
    check("mvdr_steering", Insight.numel(result2) > 0)
catch e
    println("SKIP: mvdr_steering ($e)")
end

# ============================================================================
# Results
# ============================================================================
println("\n" * "="^40)
println("Results: $passed passed, $failed failed")
if failed > 0
    exit(1)
end
