# CSV CUDA tests — aligned with C++ test_csv.cpp
# Note: CSV read/write is not exposed in the Julia bindings. This test verifies
# GPU data roundtrip via from_data/to_array which is the binding-level equivalent.
# Run with:
#   LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda julia tests/cuda/julia/test_csv.jl

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

println("=== CSV CUDA ===")

# roundtrip_1d
a = Insight.to_array(Insight.from_data([1.0, 2.0, 3.0, 4.0]), Insight.GPUPlace(0))
arr = Insight.to_array(a)
check("roundtrip_1d", length(arr) == 4)

# roundtrip_2d
a = Insight.to_array(Insight.from_data(Float64[1 2 3; 4 5 6]), Insight.GPUPlace(0))
arr = Insight.to_array(a)
check("roundtrip_2d", length(arr) == 6)

# roundtrip_int
a = Insight.to_array(Insight.from_data(Int32[10, 20, 30], Insight.int32), Insight.GPUPlace(0))
arr = Insight.to_array(a)
check("roundtrip_int", length(arr) == 3)

# roundtrip_float64
a = Insight.to_array(Insight.from_data([1.1, 2.2, 3.3]), Insight.GPUPlace(0))
arr = Insight.to_array(a)
check("roundtrip_float64", length(arr) == 3)

# roundtrip_after_ops
a = Insight.to_array(Insight.from_data([1.0, 2.0, 3.0]), Insight.GPUPlace(0))
b = Insight.add(a, a)
arr = Insight.to_array(b)
check("roundtrip_after_ops", length(arr) == 3)

println("\n" * "="^40)
println("Results: $passed passed, $failed failed")
if failed > 0; exit(1); end
