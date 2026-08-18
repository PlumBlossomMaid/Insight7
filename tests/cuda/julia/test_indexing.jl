# Indexing and sorting CUDA binding tests
# Run with:
#   LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda julia tests/cuda/julia/test_indexing.jl

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

println("=== Indexing CUDA ===")

# unique
a = Insight.to_array(Insight.from_data([3.0, 1.0, 2.0, 1.0, 3.0, 2.0]), Insight.GPUPlace(0))
result = Insight.unique_ins(a)
check("unique", Insight.numel(result.unique) >= 1)

# topk
a2 = Insight.to_array(Insight.from_data([1.0, 5.0, 3.0, 2.0, 4.0]), Insight.GPUPlace(0))
vals, idx = Insight.topk(a2, 3)
check("topk", Insight.numel(vals) == 3)

# topk smallest
vals2, idx2 = Insight.topk(a2, 3, largest=false)
check("topk_smallest", Insight.numel(vals2) == 3)

# gather
a3 = Insight.to_array(Insight.from_data([10.0, 20.0, 30.0, 40.0]), Insight.GPUPlace(0))
a3 = Insight.reshape(a3, [2, 2])
idx3 = Insight.to_array(Insight.from_data([0, 1, 1, 0]), Insight.GPUPlace(0))
idx3 = Insight.reshape(idx3, [2, 2])
g = Insight.gather(a3, 1, idx3)
check("gather", Insight.numel(g) == 4)

try
    Insight.gather(a3, 0, idx3)
    check("gather_axis0_invalid", false)
catch e
    check("gather_axis0_invalid", isa(e, ArgumentError))
end

# scatter
a4 = Insight.to_array(Insight.zeros(Int64[4], Insight.float64), Insight.GPUPlace(0))
idx4 = Insight.to_array(Insight.cast(Insight.from_data([0.0, 2.0]), Insight.int32), Insight.GPUPlace(0))
src = Insight.to_array(Insight.from_data([10.0, 20.0]), Insight.GPUPlace(0))
s = Insight.scatter(a4, 1, idx4, src)
check("scatter", Insight.numel(s) == 4)

# scatter_add
a5 = Insight.to_array(Insight.zeros(Int64[4], Insight.float64), Insight.GPUPlace(0))
idx5 = Insight.to_array(Insight.cast(Insight.from_data([0.0, 0.0, 2.0]), Insight.int32), Insight.GPUPlace(0))
src5 = Insight.to_array(Insight.from_data([1.0, 2.0, 3.0]), Insight.GPUPlace(0))
sa = Insight.scatter_add(a5, 1, idx5, src5)
check("scatter_add", Insight.numel(sa) == 4)

# interp
xp = Insight.to_array(Insight.from_data([0.0, 1.0, 2.0, 3.0]), Insight.GPUPlace(0))
fp = Insight.to_array(Insight.from_data([0.0, 2.0, 4.0, 6.0]), Insight.GPUPlace(0))
x = Insight.to_array(Insight.from_data([0.5, 1.5, 2.5]), Insight.GPUPlace(0))
result = Insight.interp(x, xp, fp)
check("interp", Insight.numel(result) == 3)

# unique with counts
a6 = Insight.to_array(Insight.from_data([5.0, 3.0, 1.0, 3.0, 5.0]), Insight.GPUPlace(0))
result6 = Insight.unique_ins(a6, return_counts=true)
check("unique_counts", Insight.numel(result6.counts) >= 1)

# scatter_reduce
a7 = Insight.to_array(Insight.zeros(Int64[4], Insight.float64), Insight.GPUPlace(0))
idx7 = Insight.to_array(Insight.cast(Insight.from_data([0.0, 0.0, 2.0]), Insight.int32), Insight.GPUPlace(0))
src7 = Insight.to_array(Insight.from_data([1.0, 2.0, 3.0]), Insight.GPUPlace(0))
sr = Insight.scatter_reduce(a7, 1, idx7, src7, reduce="add")
check("scatter_reduce", Insight.numel(sr) == 4)

# unique sorted
a8 = Insight.to_array(Insight.from_data([5.0, 3.0, 1.0, 3.0, 5.0]), Insight.GPUPlace(0))
result8 = Insight.unique_ins(a8)
check("unique_sorted", Insight.numel(result8.unique) >= 1)

# ============================================================================
# Results
# ============================================================================
println("\n" * "="^40)
println("Results: $passed passed, $failed failed")
if failed > 0
    exit(1)
end
