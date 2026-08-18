# Elementwise CUDA tests — aligned with C++ test_elementwise.cpp
# Run with:
#   LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda julia tests/cuda/julia/test_elementwise.jl

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

println("=== Elementwise CUDA ===")

a = Insight.to_array(Insight.from_data([1.0, 2.0, 3.0, 4.0, 5.0]), Insight.GPUPlace(0))
b = Insight.to_array(Insight.from_data([5.0, 4.0, 3.0, 2.0, 1.0]), Insight.GPUPlace(0))

# add
c = Insight.add(a, b)
check("add", Insight.numel(c) == 5)

# sub
c = Insight.sub(a, b)
check("sub", Insight.numel(c) == 5)

# mul
c = Insight.mul(a, b)
check("mul", Insight.numel(c) == 5)

# div
c = Insight.div(a, b)
check("div", Insight.numel(c) == 5)

# pow
a2 = Insight.to_array(Insight.from_data([2.0, 3.0, 4.0]), Insight.GPUPlace(0))
b2 = Insight.to_array(Insight.from_data([3.0, 2.0, 0.5]), Insight.GPUPlace(0))
c = Insight.pow(a2, b2)
check("pow", Insight.numel(c) == 3)

# add operator
c = a + b
check("add_op", Insight.numel(c) == 5)

# sub operator
c = a - b
check("sub_op", Insight.numel(c) == 5)

# mul operator
c = a * b
check("mul_op", Insight.numel(c) == 5)

# div operator
c = a / b
check("div_op", Insight.numel(c) == 5)

# negation
c = -a
check("neg", Insight.numel(c) == 5)

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

# logical_and
a_bool = Insight.to_array(Insight.from_data([1, 1, 0], Insight.bool), Insight.GPUPlace(0))
b_bool = Insight.to_array(Insight.from_data([1, 0, 0], Insight.bool), Insight.GPUPlace(0))
c = Insight.logical_and(a_bool, b_bool)
check("logical_and", Insight.numel(c) == 3)

# logical_or
c = Insight.logical_or(a_bool, b_bool)
check("logical_or", Insight.numel(c) == 3)

# logical_not
c = Insight.logical_not(a_bool)
check("logical_not", Insight.numel(c) == 3)

println("\n" * "="^40)
println("Results: $passed passed, $failed failed")
if failed > 0; exit(1); end
