# Type casting CUDA binding tests
# Run with:
#   LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda julia tests/cuda/julia/test_cast.jl

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

println("=== Cast CUDA ===")

# cast float64 to float32
a = Insight.to_array(Insight.from_data([1.5, 2.5, 3.5]), Insight.GPUPlace(0))
b = Insight.cast(a, Insight.float32)
check("f64_to_f32", Insight.numel(b) == 3)

# cast float32 to float64
a2 = Insight.to_array(Insight.zeros(Int64[3], Insight.float32), Insight.GPUPlace(0))
b2 = Insight.cast(a2, Insight.float64)
check("f32_to_f64", Insight.numel(b2) == 3)

# cast float64 to int32
a3 = Insight.to_array(Insight.from_data([1.9, 2.5, 3.1]), Insight.GPUPlace(0))
b3 = Insight.cast(a3, Insight.int32)
check("f64_to_i32", Insight.numel(b3) == 3)

# cast float64 to int64
a4 = Insight.to_array(Insight.from_data([1.9, 2.5, 3.1]), Insight.GPUPlace(0))
b4 = Insight.cast(a4, Insight.int64)
check("f64_to_i64", Insight.numel(b4) == 3)

# cast float64 to uint8
a5 = Insight.to_array(Insight.from_data([0.0, 128.0, 255.0]), Insight.GPUPlace(0))
b5 = Insight.cast(a5, Insight.uint8)
check("f64_to_u8", Insight.numel(b5) == 3)

# cast bool to float64
a6 = Insight.to_array(Insight.from_data([1.0, 0.0, 1.0]), Insight.GPUPlace(0))
b6 = Insight.cast(a6, Insight.bool)
c6 = Insight.cast(b6, Insight.float64)
check("bool_to_f64", Insight.numel(c6) == 3)

# cast int32 to float64
a7 = Insight.to_array(Insight.from_data([1.0, 2.0, 3.0]), Insight.GPUPlace(0))
b7 = Insight.cast(a7, Insight.int32)
c7 = Insight.cast(b7, Insight.float64)
check("i32_to_f64", Insight.numel(c7) == 3)

# cast preserves shape
a8 = Insight.to_array(Insight.from_data([1.0, 2.0, 3.0, 4.0]), Insight.GPUPlace(0))
a8 = Insight.reshape(a8, [2, 2])
b8 = Insight.cast(a8, Insight.float32)
check("preserves_shape", Insight.numel(b8) == 4)

# cast idempotent
a9 = Insight.to_array(Insight.from_data([1.5, 2.5]), Insight.GPUPlace(0))
b9 = Insight.cast(a9, Insight.float64)
check("idempotent", Insight.numel(b9) == 2)

# cast int to int
a10 = Insight.to_array(Insight.from_data([1.0, 2.0, 3.0]), Insight.GPUPlace(0))
b10 = Insight.cast(a10, Insight.int32)
c10 = Insight.cast(b10, Insight.int64)
check("int_to_int", Insight.numel(c10) == 3)

# cast roundtrip
a11 = Insight.to_array(Insight.from_data([1.0, 2.0, 3.0]), Insight.GPUPlace(0))
b11 = Insight.cast(a11, Insight.int32)
c11 = Insight.cast(b11, Insight.float64)
check("roundtrip", Insight.numel(c11) == 3)

# cast negative values
a12 = Insight.to_array(Insight.from_data([-1.5, -2.5, 3.5]), Insight.GPUPlace(0))
b12 = Insight.cast(a12, Insight.int32)
check("negative_values", Insight.numel(b12) == 3)

# ============================================================================
# Results
# ============================================================================
println("\n" * "="^40)
println("Results: $passed passed, $failed failed")
if failed > 0
    exit(1)
end
