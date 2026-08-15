# Indexing and sorting CPU binding tests
# Run with:
#   LD_LIBRARY_PATH=build/backends/cpu julia tests/cpu/julia/test_indexing.jl

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

println("=== Indexing ===")

# unique
a = Insight.from_data([3.0, 1.0, 2.0, 1.0, 3.0, 2.0])
result = Insight.unique_ins(a)
check("unique", Insight.numel(result.unique) >= 1)

# topk
a2 = Insight.from_data([1.0, 5.0, 3.0, 2.0, 4.0])
vals, idx = Insight.topk(a2, 3)
check("topk", Insight.numel(vals) == 3)

# topk smallest
vals2, idx2 = Insight.topk(a2, 3, largest=false)
check("topk_smallest", Insight.numel(vals2) == 3)

# gather
a3 = Insight.from_data([10.0, 20.0, 30.0, 40.0])
a3 = Insight.reshape(a3, [2, 2])
idx3 = Insight.from_data([0, 1, 1, 0])
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
a4 = Insight.zeros(Int64[4], Insight.float64)
idx4 = Insight.from_data([0.0, 2.0])
idx4 = Insight.cast(idx4, Insight.int32)
src = Insight.from_data([10.0, 20.0])
s = Insight.scatter(a4, 1, idx4, src)
check("scatter", Insight.numel(s) == 4)

# scatter_add
a5 = Insight.zeros(Int64[4], Insight.float64)
idx5 = Insight.from_data([0.0, 0.0, 2.0])
idx5 = Insight.cast(idx5, Insight.int32)
src5 = Insight.from_data([1.0, 2.0, 3.0])
sa = Insight.scatter_add(a5, 1, idx5, src5)
check("scatter_add", Insight.numel(sa) == 4)

# interp
xp = Insight.from_data([0.0, 1.0, 2.0, 3.0])
fp = Insight.from_data([0.0, 2.0, 4.0, 6.0])
x = Insight.from_data([0.5, 1.5, 2.5])
result = Insight.interp(x, xp, fp)
check("interp", Insight.numel(result) == 3)

# unique with return values
a6 = Insight.from_data([5.0, 3.0, 1.0, 3.0, 5.0])
result6 = Insight.unique_ins(a6, return_counts=true)
check("unique_counts", Insight.numel(result6.counts) >= 1)

# scatter_reduce
a7 = Insight.zeros(Int64[4], Insight.float64)
idx7 = Insight.from_data([0.0, 0.0, 2.0])
idx7 = Insight.cast(idx7, Insight.int32)
src7 = Insight.from_data([1.0, 2.0, 3.0])
sr = Insight.scatter_reduce(a7, 1, idx7, src7, reduce="add")
check("scatter_reduce", Insight.numel(sr) == 4)

# unique sorted output
a8 = Insight.from_data([5.0, 3.0, 1.0, 3.0, 5.0])
result8 = Insight.unique_ins(a8)
d = Insight.to_array(result8.unique)
check("unique_sorted", d[1] <= d[end])

# ============================================================================
# Results
# ============================================================================
println("\n" * "="^40)
println("Results: $passed passed, $failed failed")
if failed > 0
    exit(1)
end
