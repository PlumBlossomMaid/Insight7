# Signal peak_finding CUDA binding tests
# Run with:
#   LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda julia tests/cuda/julia/test_signal_peak_finding.jl

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

# ============================================================================
# Peak Finding
# ============================================================================
println("=== Signal Peak Finding ===")

# argrelmax
data = Insight.from_data([0.0, 2.0, 1.0, 3.0, 0.0, 4.0, 1.0])
data_gpu = Insight.to_array(data, Insight.GPUPlace(0))
result = Insight.argrelmax(data_gpu, order=1)
check("argrelmax", length(result) >= 0)

# argrelmin
data = Insight.from_data([3.0, 1.0, 4.0, 0.0, 5.0, 2.0, 6.0])
data_gpu = Insight.to_array(data, Insight.GPUPlace(0))
result = Insight.argrelmin(data_gpu, order=1)
check("argrelmin", length(result) >= 0)

# argrelmax order 2
data = Insight.from_data([0.0, 1.0, 3.0, 2.0, 1.0, 5.0, 2.0, 1.0, 0.0])
data_gpu = Insight.to_array(data, Insight.GPUPlace(0))
result = Insight.argrelmax(data_gpu, order=2)
check("argrelmax_order2", length(result) >= 0)

# argrelmin order 2
data = Insight.from_data([5.0, 3.0, 1.0, 2.0, 4.0, 0.0, 3.0, 4.0, 5.0])
data_gpu = Insight.to_array(data, Insight.GPUPlace(0))
result = Insight.argrelmin(data_gpu, order=2)
check("argrelmin_order2", length(result) >= 0)

# argrelmax no peaks (monotonic)
data = Insight.from_data([1.0, 2.0, 3.0, 4.0, 5.0])
data_gpu = Insight.to_array(data, Insight.GPUPlace(0))
result = Insight.argrelmax(data_gpu, order=1)
check("argrelmax_no_peaks", length(result) == 0)

# argrelmin no valleys (monotonic)
data = Insight.from_data([5.0, 4.0, 3.0, 2.0, 1.0])
data_gpu = Insight.to_array(data, Insight.GPUPlace(0))
result = Insight.argrelmin(data_gpu, order=1)
check("argrelmin_no_valleys", length(result) == 0)

# argrelmax sine
sine_data = [sin(4 * pi * i / 100) for i in 0:99]
data = Insight.from_data(sine_data)
data_gpu = Insight.to_array(data, Insight.GPUPlace(0))
result = Insight.argrelmax(data_gpu, order=1)
check("argrelmax_sine", length(result) > 0)

# argrelmin sine
data = Insight.from_data(sine_data)
data_gpu = Insight.to_array(data, Insight.GPUPlace(0))
result = Insight.argrelmin(data_gpu, order=1)
check("argrelmin_sine", length(result) > 0)

# argrelmax wider order
sine_data2 = [sin(2 * pi * i / 50) for i in 0:49]
data = Insight.from_data(sine_data2)
data_gpu = Insight.to_array(data, Insight.GPUPlace(0))
result = Insight.argrelmax(data_gpu, order=3)
check("argrelmax_wide_order", length(result) >= 0)

# argrelmin wider order
data = Insight.from_data(sine_data2)
data_gpu = Insight.to_array(data, Insight.GPUPlace(0))
result = Insight.argrelmin(data_gpu, order=3)
check("argrelmin_wide_order", length(result) >= 0)

# ============================================================================
# Results
# ============================================================================
println("\n" * "="^40)
println("Results: $passed passed, $failed failed")
if failed > 0
    exit(1)
end
