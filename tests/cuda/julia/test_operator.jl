# Operator CUDA tests — aligned with C++ test_operator.cpp
# Run with:
#   LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda julia tests/cuda/julia/test_operator.jl

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

println("=== Operators CUDA ===")

a = Insight.to_array(Insight.from_data([1.0, 2.0, 3.0]), Insight.GPUPlace(0))
b = Insight.to_array(Insight.from_data([4.0, 5.0, 6.0]), Insight.GPUPlace(0))

# add operator
c = a + b
check("add_op", Insight.numel(c) == 3)

# sub operator
c = a - b
check("sub_op", Insight.numel(c) == 3)

# mul operator
c = a * b
check("mul_op", Insight.numel(c) == 3)

# div operator
c = a / b
check("div_op", Insight.numel(c) == 3)

# negation
c = -a
check("neg", Insight.numel(c) == 3)

# add function
c = Insight.add(a, b)
check("add_fn", Insight.numel(c) == 3)

# sub function
c = Insight.sub(a, b)
check("sub_fn", Insight.numel(c) == 3)

# mul function
c = Insight.mul(a, b)
check("mul_fn", Insight.numel(c) == 3)

# div function
c = Insight.div(a, b)
check("div_fn", Insight.numel(c) == 3)

# equal
a3 = Insight.to_array(Insight.from_data([1.0, 2.0, 3.0]), Insight.GPUPlace(0))
b3 = Insight.to_array(Insight.from_data([1.0, 0.0, 3.0]), Insight.GPUPlace(0))
c = Insight.equal(a3, b3)
check("equal", Insight.numel(c) == 3)

# greater
a4 = Insight.to_array(Insight.from_data([3.0, 1.0, 5.0]), Insight.GPUPlace(0))
b4 = Insight.to_array(Insight.from_data([2.0, 4.0, 5.0]), Insight.GPUPlace(0))
c = Insight.greater(a4, b4)
check("greater", Insight.numel(c) == 3)

# less
c = Insight.less(a4, b4)
check("less", Insight.numel(c) == 3)

# negative function
c = Insight.negative(a)
check("negative_fn", Insight.numel(c) == 3)

# indexing data
a5 = Insight.to_array(Insight.from_data([10.0, 20.0, 30.0, 40.0]), Insight.GPUPlace(0))
check("indexing_data", Insight.numel(a5) == 4)

println("\n" * "="^40)
println("Results: $passed passed, $failed failed")
if failed > 0; exit(1); end
