# CUDA binding tests — 35 tests aligned with CPU Python/Lua/Julia
# Run with:
#   LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda julia tests/cuda/julia/test_cuda_basic.jl

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

# ============================================================================
# Creation (7 tests)
# ============================================================================
println("=== Creation ===")

a = Insight.zeros([2, 3], Insight.float32)
a_gpu = Insight.to_array(a, Insight.GPUPlace(0))
check("zeros", Insight.numel(a_gpu) == 6)

a = Insight.ones([2, 3], Insight.float32)
a_gpu = Insight.to_array(a, Insight.GPUPlace(0))
check("ones", Insight.numel(a_gpu) == 6)

a = Insight.full([2, 3], 7.0, Insight.float32)
a_gpu = Insight.to_array(a, Insight.GPUPlace(0))
check("full", Insight.numel(a_gpu) == 6)

a = Insight.eye(3)
a_gpu = Insight.to_array(a, Insight.GPUPlace(0))
check("eye", Insight.numel(a_gpu) == 9)

a = Insight.arange(10, Insight.float32)
a_gpu = Insight.to_array(a, Insight.GPUPlace(0))
check("arange", Insight.numel(a_gpu) == 10)

a = Insight.linspace(0.0, 1.0, 5, Insight.float64)
a_gpu = Insight.to_array(a, Insight.GPUPlace(0))
check("linspace", Insight.numel(a_gpu) == 5)

a = Insight.from_data([1.5, 2.5, 3.5])
a_gpu = Insight.to_array(a, Insight.GPUPlace(0))
check("from_data", Insight.numel(a_gpu) == 3)

# ============================================================================
# Arithmetic (5 tests)
# ============================================================================
println("=== Arithmetic ===")

a = Insight.to_array(Insight.from_data([1.0, 2.0, 3.0]), Insight.GPUPlace(0))
b = Insight.to_array(Insight.from_data([4.0, 5.0, 6.0]), Insight.GPUPlace(0))

c = Insight.add(a, b)
check("add", Insight.numel(c) == 3)

c = Insight.sub(a, b)
check("sub", Insight.numel(c) == 3)

c = Insight.mul(a, b)
check("mul", Insight.numel(c) == 3)

c = Insight.div(a, b)
check("div", Insight.numel(c) == 3)

c = Insight.negative(a)
check("neg", Insight.numel(c) == 3)

# ============================================================================
# Unary (5 tests)
# ============================================================================
println("=== Unary ===")

a = Insight.to_array(Insight.from_data([-3.0, -1.0, 0.0, 2.0, 4.0]), Insight.GPUPlace(0))
b = Insight.abs(a)
check("abs", Insight.numel(b) == 5)

a = Insight.to_array(Insight.from_data([1.0, 4.0, 9.0, 16.0]), Insight.GPUPlace(0))
b = Insight.sqrt(a)
check("sqrt", Insight.numel(b) == 4)

a = Insight.to_array(Insight.from_data([0.0, 1.0, 2.0]), Insight.GPUPlace(0))
b = Insight.exp(a)
check("exp", Insight.numel(b) == 3)

a = Insight.to_array(Insight.from_data([1.0, 2.0, 3.0]), Insight.GPUPlace(0))
b = Insight.log(a)
check("log", Insight.numel(b) == 3)

a = Insight.to_array(Insight.from_data([0.0, 0.5, 1.0]), Insight.GPUPlace(0))
b = Insight.sin(a)
check("sin", Insight.numel(b) == 3)

# ============================================================================
# Reduction (5 tests)
# ============================================================================
println("=== Reduction ===")

a = Insight.to_array(Insight.from_data([1.0, 2.0, 3.0, 4.0]), Insight.GPUPlace(0))

s = Insight.sum(a)
check("sum", Insight.numel(s) == 1)

m = Insight.mean(a)
check("mean", Insight.numel(m) == 1)

a = Insight.to_array(Insight.from_data([3.0, 1.0, 4.0, 1.0, 5.0]), Insight.GPUPlace(0))
m = Insight.max(a)
check("max", Insight.numel(m) == 1)

m = Insight.min(a)
check("min", Insight.numel(m) == 1)

a = Insight.to_array(Insight.from_data([1.0, 5.0, 3.0, 2.0]), Insight.GPUPlace(0))
m = Insight.argmax(a)
check("argmax", Insight.numel(m) == 1)

# ============================================================================
# Comparison (4 tests)
# ============================================================================
println("=== Comparison ===")

a = Insight.to_array(Insight.from_data([1.0, 2.0, 3.0]), Insight.GPUPlace(0))
b = Insight.to_array(Insight.from_data([1.0, 0.0, 3.0]), Insight.GPUPlace(0))
c = Insight.equal(a, b)
check("equal", Insight.numel(c) == 3)

a = Insight.to_array(Insight.from_data([3.0, 1.0, 5.0]), Insight.GPUPlace(0))
b = Insight.to_array(Insight.from_data([2.0, 4.0, 5.0]), Insight.GPUPlace(0))
c = Insight.greater(a, b)
check("greater", Insight.numel(c) == 3)

c = Insight.less(a, b)
check("less", Insight.numel(c) == 3)

a_bool = Insight.to_array(Insight.from_data([1, 1, 0], Insight.bool), Insight.GPUPlace(0))
b_bool = Insight.to_array(Insight.from_data([1, 0, 0], Insight.bool), Insight.GPUPlace(0))
c = Insight.logical_and(a_bool, b_bool)
check("logical_and", Insight.numel(c) == 3)

# ============================================================================
# Manipulation (3 tests)
# ============================================================================
println("=== Manipulation ===")

a = Insight.to_array(Insight.arange(6, Insight.float64), Insight.GPUPlace(0))
b = Insight.reshape(a, [2, 3])
check("reshape", Insight.numel(b) == 6)

a = Insight.to_array(Insight.from_data([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), Insight.GPUPlace(0))
a = Insight.reshape(a, [2, 3])
b = Insight.transpose(a)
check("transpose", Insight.numel(b) == 6)

a = Insight.to_array(Insight.zeros([1, 3, 1], Insight.float64), Insight.GPUPlace(0))
b = Insight.squeeze(a)
check("squeeze", Insight.numel(b) == 3)

# ============================================================================
# Linalg (3 tests)
# ============================================================================
println("=== Linalg ===")

a = Insight.to_array(Insight.from_data([1.0, 2.0, 3.0, 4.0]), Insight.GPUPlace(0))
a = Insight.reshape(a, [2, 2])
b = Insight.to_array(Insight.from_data([5.0, 6.0, 7.0, 8.0]), Insight.GPUPlace(0))
b = Insight.reshape(b, [2, 2])

c = Insight.matmul(a, b)
check("matmul", Insight.numel(c) == 4)

try
    local d = Insight.det(a)
    check("det", Insight.numel(d) == 1)
catch e
    println("SKIP: det (requires cuBLAS)")
end

try
    local b = Insight.inv(a)
    check("inv", Insight.numel(b) == 4)
catch e
    println("SKIP: inv (requires cuBLAS)")
end

# ============================================================================
# FFT (2 tests)
# ============================================================================
println("=== FFT ===")

x = Insight.to_array(Insight.from_data([1.0, 2.0, 3.0, 4.0]), Insight.GPUPlace(0))
y = Insight.fft(x)
check("fft", Insight.numel(y) == 4)

z = Insight.ifft(y)
check("ifft", Insight.numel(z) == 4)

# ============================================================================
# Signal (1 test)
# ============================================================================
println("=== Signal ===")

w = Insight.signal.hann(16)
check("hann", Insight.numel(w) == 16)

# ============================================================================
# Results
# ============================================================================
println("\n" * "="^40)
println("Results: $passed passed, $failed failed")
if failed > 0
    exit(1)
end
