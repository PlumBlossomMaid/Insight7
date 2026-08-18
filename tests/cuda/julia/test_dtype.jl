# DType CUDA tests — aligned with C++ test_dtype.cpp
# Run with:
#   LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda julia tests/cuda/julia/test_dtype.jl

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

println("=== DType CUDA ===")

# dtype_shortcuts
check("float32 exists", Insight.float32 isa Int32)
check("float64 exists", Insight.float64 isa Int32)
check("int32 exists", Insight.int32 isa Int32)
check("int64 exists", Insight.int64 isa Int32)
check("bool exists", Insight.bool isa Int32)

# cast_f32_to_f64
a = Insight.to_array(Insight.ones(Int64[3], Insight.float32), Insight.GPUPlace(0))
b = Insight.cast(a, Insight.float64)
check("cast_f32_to_f64 dtype", Insight.dtype(b) == Insight.float64)
check("cast_f32_to_f64 numel", Insight.numel(b) == 3)

# cast_f64_to_i32
a = Insight.to_array(Insight.from_data([1.9, 2.5, 3.1]), Insight.GPUPlace(0))
b = Insight.cast(a, Insight.int32)
check("cast_f64_to_i32", Insight.dtype(b) == Insight.int32)

# cast_i32_to_f32
a = Insight.to_array(Insight.from_data(Int32[1, 2, 3], Insight.int32), Insight.GPUPlace(0))
b = Insight.cast(a, Insight.float32)
check("cast_i32_to_f32", Insight.dtype(b) == Insight.float32)

# cast_to_bool
a = Insight.to_array(Insight.from_data([0.0, 1.0, 2.0]), Insight.GPUPlace(0))
b = Insight.cast(a, Insight.bool)
check("cast_to_bool", Insight.dtype(b) == Insight.bool)

# cast_i64_to_f64
a = Insight.to_array(Insight.from_data(Int64[100, 200, 300], Insight.int64), Insight.GPUPlace(0))
b = Insight.cast(a, Insight.float64)
check("cast_i64_to_f64", Insight.dtype(b) == Insight.float64)

# cast_preserves_values
a = Insight.to_array(Insight.from_data([1.0, 2.0, 3.0]), Insight.GPUPlace(0))
b = Insight.cast(a, Insight.float32)
c = Insight.cast(b, Insight.float64)
check("cast_preserves_values", Insight.numel(c) == 3)

# array_dtype_property
a = Insight.to_array(Insight.zeros(Int64[3], Insight.float32), Insight.GPUPlace(0))
check("dtype_property_f32", Insight.dtype(a) == Insight.float32)
b = Insight.to_array(Insight.zeros(Int64[3], Insight.int64), Insight.GPUPlace(0))
check("dtype_property_i64", Insight.dtype(b) == Insight.int64)

println("\n" * "="^40)
println("Results: $passed passed, $failed failed")
if failed > 0; exit(1); end
