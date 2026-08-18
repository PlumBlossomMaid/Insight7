# Reduction CUDA tests — aligned with C++ test_reduction.cpp
# Run with:
#   LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda julia tests/cuda/julia/test_reduction.jl

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

println("=== Reduction CUDA ===")

# sum
a = Insight.to_array(Insight.from_data([1.0, 2.0, 3.0, 4.0]), Insight.GPUPlace(0))
s = Insight.sum(a)
check("sum", Insight.numel(s) == 1)

# sum_axis
b = Insight.to_array(Insight.from_data(Float64[1 2 3; 4 5 6]), Insight.GPUPlace(0))
s = Insight.sum(b, axis=1)
check("sum_axis", Insight.numel(s) == 3)

# mean
m = Insight.mean(a)
check("mean", Insight.numel(m) == 1)

# mean_axis
m = Insight.mean(b, axis=1)
check("mean_axis", Insight.numel(m) == 2)

# max
c = Insight.to_array(Insight.from_data([3.0, 1.0, 4.0, 1.0, 5.0]), Insight.GPUPlace(0))
m = Insight.max(c)
check("max", Insight.numel(m) == 1)

# min
m = Insight.min(c)
check("min", Insight.numel(m) == 1)

# prod
p = Insight.prod(a)
check("prod", Insight.numel(p) == 1)

# argmax
d = Insight.to_array(Insight.from_data([1.0, 5.0, 3.0, 2.0]), Insight.GPUPlace(0))
m = Insight.argmax(d)
check("argmax", Insight.numel(m) == 1)

# argmin
m = Insight.argmin(d)
check("argmin", Insight.numel(m) == 1)

# cummax
m2 = Insight.to_array(Insight.from_data(Float32[1 2 3; 4 5 6]), Insight.GPUPlace(0))
check("cummax", Insight.cummax(m2, 1).ptr != C_NULL)

# cummin
check("cummin", Insight.cummin(m2, 1).ptr != C_NULL)

# count_nonzero
check("count_nonzero", Insight.count_nonzero(m2).ptr != C_NULL)

# median
check("median", Insight.median(m2).ptr != C_NULL)

# quantile
check("quantile", Insight.quantile(m2, 0.5).ptr != C_NULL)

# nansum
check("nansum", Insight.nansum(m2).ptr != C_NULL)

# nanmean
check("nanmean", Insight.nanmean(m2).ptr != C_NULL)

# nanstd
check("nanstd", Insight.nanstd(m2).ptr != C_NULL)

# sem
check("sem", Insight.sem(m2).ptr != C_NULL)

println("\n" * "="^40)
println("Results: $passed passed, $failed failed")
if failed > 0; exit(1); end
