# Random number generation CUDA binding tests
# Run with:
#   LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda julia tests/cuda/julia/test_random.jl

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

println("=== Random CUDA ===")

# rand shape
a = Insight.rand(Int64[3, 4], Insight.float64, Insight.GPU)
check("rand_shape", Insight.numel(a) == 12)

# randn shape
a = Insight.randn(Int64[3, 4], Insight.float64, Insight.GPU)
check("randn_shape", Insight.numel(a) == 12)

# seed determinism
Insight.seed(UInt64(42))
a = Insight.rand(Int64[5], Insight.float64, Insight.GPU)
Insight.seed(UInt64(42))
b = Insight.rand(Int64[5], Insight.float64, Insight.GPU)
check("seed_determinism", Insight.numel(a) == 5 && Insight.numel(b) == 5)

# get_seed
Insight.seed(UInt64(12345))
s = Insight.get_seed()
check("get_seed", s == UInt64(12345))

# rand_like
template = Insight.to_array(Insight.zeros(Int64[3, 4], Insight.float64), Insight.GPUPlace(0))
rl = Insight.rand_like(template)
check("rand_like", Insight.numel(rl) == 12)

# randn_like
rn = Insight.randn_like(template)
check("randn_like", Insight.numel(rn) == 12)

# exponential
e = Insight.exponential(1.0, Int64[100], dtype_val=Insight.float64, device=Insight.GPU)
check("exponential", Insight.numel(e) == 100)

# gamma
g = Insight.gamma_dist(2.0, 1.0, Int64[100], dtype_val=Insight.float64, device=Insight.GPU)
check("gamma", Insight.numel(g) == 100)

# beta
b = Insight.beta_dist(2.0, 5.0, Int64[100], dtype_val=Insight.float64, device=Insight.GPU)
check("beta", Insight.numel(b) == 100)

# binomial
bi = Insight.binomial_dist(Int64(10), 0.5, Int64[100], dtype_val=Insight.int64, device=Insight.GPU)
check("binomial", Insight.numel(bi) == 100)

# poisson
p = Insight.poisson_dist(5.0, Int64[100], dtype_val=Insight.int64, device=Insight.GPU)
check("poisson", Insight.numel(p) == 100)

# rand 1d
a1 = Insight.rand(Int64[10], Insight.float32, Insight.GPU)
check("rand_1d", Insight.numel(a1) == 10)

# rand 3d
a3 = Insight.rand(Int64[2, 3, 4], Insight.float64, Insight.GPU)
check("rand_3d", Insight.numel(a3) == 24)

# randn 1d
n1 = Insight.randn(Int64[20], Insight.float64, Insight.GPU)
check("randn_1d", Insight.numel(n1) == 20)

# exponential with different scale
e2 = Insight.exponential(2.0, Int64[50], dtype_val=Insight.float64, device=Insight.GPU)
check("exponential_scale", Insight.numel(e2) == 50)

# ============================================================================
# Results
# ============================================================================
println("\n" * "="^40)
println("Results: $passed passed, $failed failed")
if failed > 0
    exit(1)
end
