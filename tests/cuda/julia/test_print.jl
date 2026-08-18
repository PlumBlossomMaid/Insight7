# Print CUDA tests — aligned with C++ test_print.cpp
# Run with:
#   LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda julia tests/cuda/julia/test_print.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "..", "bindings", "julia"))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "..", "build", "bindings", "julia"))

using Insight

try
    Insight.load_backend("gpu")
    Insight.set_device(Insight.GPUPlace(0))
catch e
    println("SKIP: GPU backend not available: $e")
    exit(0)
end

passed = 0; failed = 0
function check(name, cond)
    global passed, failed
    if cond; passed += 1; else; failed += 1; println("FAIL: $name"); end
end

println("=== Print CUDA ===")

# show_1d
a = Insight.to_array(Insight.zeros(Int64[5], Insight.float32), Insight.GPUPlace(0))
s = sprint(show, a)
check("show_1d", length(s) > 0)

# show_2d
a = Insight.to_array(Insight.ones(Int64[3, 4], Insight.float64), Insight.GPUPlace(0))
s = sprint(show, a)
check("show_2d", length(s) > 0)

# show_int
a = Insight.to_array(Insight.zeros(Int64[3], Insight.int32), Insight.GPUPlace(0))
s = sprint(show, a)
check("show_int", length(s) > 0)

# show_from_data
a = Insight.to_array(Insight.from_data([1.0, 2.0, 3.0]), Insight.GPUPlace(0))
s = sprint(show, a)
check("show_from_data", length(s) > 0)

# show_2d_data
a = Insight.to_array(Insight.from_data(Float64[1 2; 3 4]), Insight.GPUPlace(0))
s = sprint(show, a)
check("show_2d_data", length(s) > 0)

# show_does_not_crash_large
a = Insight.to_array(Insight.ones(Int64[100, 100], Insight.float32), Insight.GPUPlace(0))
s = sprint(show, a)
check("show_large", length(s) > 0)

# show_empty
a = Insight.to_array(Insight.zeros(Int64[0], Insight.float32), Insight.GPUPlace(0))
s = sprint(show, a)
check("show_empty", length(s) > 0)

# show_after_cast
a = Insight.to_array(Insight.ones(Int64[3], Insight.float32), Insight.GPUPlace(0))
b = Insight.cast(a, Insight.int32)
s = sprint(show, b)
check("show_after_cast", length(s) > 0)

println("\n" * "="^40)
println("Results: $passed passed, $failed failed")
if failed > 0; exit(1); end
