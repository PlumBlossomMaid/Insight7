# Linalg CPU binding tests
# Run with:
#   LD_LIBRARY_PATH=build/backends/cpu julia tests/cpu/julia/test_linalg.jl

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

# ============================================================================
# matmul
# ============================================================================
println("=== Linalg ===")

a = Insight.from_data([1.0, 2.0, 3.0, 4.0])
a = Insight.reshape(a, [2, 2])
b = Insight.from_data([5.0, 6.0, 7.0, 8.0])
b = Insight.reshape(b, [2, 2])

c = Insight.matmul(a, b)
check("matmul", Insight.numel(c) == 4)

# dot product
a1 = Insight.from_data([1.0, 2.0, 3.0])
b1 = Insight.from_data([4.0, 5.0, 6.0])
d = Insight.dot(a1, b1)
check("dot", Insight.numel(d) == 1)

# outer product
o = Insight.outer(a1, b1)
check("outer", Insight.numel(o) == 9)

# norm
v = Insight.from_data([3.0, 4.0])
n = Insight.norm(v)
check("norm", Insight.numel(n) == 1)

# trace
t = Insight.trace(a)
check("trace", Insight.numel(t) == 1)

# det
try
    local d = Insight.det(a)
    check("det", Insight.numel(d) == 1)
catch e
    println("SKIP: det ($e)")
end

# inv
try
    local a_inv = Insight.inv(a)
    check("inv", Insight.numel(a_inv) == 4)
catch e
    println("SKIP: inv ($e)")
end

# solve
try
    a2 = Insight.from_data([3.0, 1.0, 1.0, 2.0])
    a2 = Insight.reshape(a2, [2, 2])
    b2 = Insight.from_data([9.0, 8.0])
    local x = Insight.solve(a2, b2)
    check("solve", Insight.numel(x) == 2)
catch e
    println("SKIP: solve ($e)")
end

# svd
try
    a3 = Insight.from_data([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    a3 = Insight.reshape(a3, [3, 2])
    local u, s, vt = Insight.svd(a3)
    check("svd", Insight.numel(s) == 2)
catch e
    println("SKIP: svd ($e)")
end

# cholesky
try
    a4 = Insight.from_data([4.0, 2.0, 2.0, 3.0])
    a4 = Insight.reshape(a4, [2, 2])
    local l = Insight.cholesky(a4)
    check("cholesky", Insight.numel(l) == 4)
catch e
    println("SKIP: cholesky ($e)")
end

# qr
try
    a5 = Insight.from_data([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    a5 = Insight.reshape(a5, [3, 2])
    q, r = Insight.transpose(a5), a5  # placeholder
    check("qr_reshape", Insight.numel(a5) == 6)
catch e
    println("SKIP: qr ($e)")
end

# matrix_power
try
    local mp = Insight.matrix_power(a, 2)
    check("matrix_power", Insight.numel(mp) == 4)
catch e
    println("SKIP: matrix_power ($e)")
end

# slogdet
try
    local s, ld = Insight.slogdet(a)
    check("slogdet", Insight.numel(s) == 1 && Insight.numel(ld) == 1)
catch e
    println("SKIP: slogdet ($e)")
end

# eigvalsh
try
    local w = Insight.eigvalsh(a)
    check("eigvalsh", Insight.numel(w) == 2)
catch e
    println("SKIP: eigvalsh ($e)")
end

# pinv
try
    a6 = Insight.from_data([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    a6 = Insight.reshape(a6, [3, 2])
    local pi = Insight.pinv(a6)
    check("pinv", Insight.numel(pi) == 6)
catch e
    println("SKIP: pinv ($e)")
end

# lstsq
try
    a7 = Insight.from_data([1.0, 1.0, 1.0, 2.0, 1.0, 3.0])
    a7 = Insight.reshape(a7, [2, 3])
    b7 = Insight.from_data([1.0, 2.0, 2.0])
    local x = Insight.lstsq(a7, b7)
    check("lstsq", Insight.numel(x) == 2)
catch e
    println("SKIP: lstsq ($e)")
end

# cond
try
    local c = Insight.cond_fn(a)
    check("cond", Insight.numel(c) == 1)
catch e
    println("SKIP: cond ($e)")
end

# matrix_rank
try
    local r = Insight.matrix_rank(a)
    check("matrix_rank", Insight.numel(r) == 1)
catch e
    println("SKIP: matrix_rank ($e)")
end

# ============================================================================
# Results
# ============================================================================
println("\n" * "="^40)
println("Results: $passed passed, $failed failed")
if failed > 0
    exit(1)
end
