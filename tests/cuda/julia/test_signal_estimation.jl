# Signal estimation CUDA binding tests
# Run with:
#   LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda julia tests/cuda/julia/test_signal_estimation.jl

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
# Kalman Filter
# ============================================================================
println("=== Signal Estimation ===")

# constructor
kf = Insight.KalmanFilter(2, 1)
check("constructor", kf.dim_x == 2 && kf.dim_z == 1)

# state shape
x = kf.x
check("state_shape", Insight.numel(x) == 2)

# covariance shape
P = kf.P
check("covariance_shape", Insight.numel(P) == 4)

# set F
F = Insight.from_data([1.0, 1.0, 0.0, 1.0])
F = Insight.reshape(F, [2, 2])
kf.F = F
check("set_F", Insight.numel(kf.F) == 4)

# set H
H = Insight.from_data([1.0, 0.0])
H = Insight.reshape(H, [1, 2])
kf.H = H
check("set_H", Insight.numel(kf.H) == 2)

# predict
Insight.predict(kf)
x_after = kf.x
check("predict", Insight.numel(x_after) == 2)

# update
z = Insight.from_data([1.0])
Insight.update(kf, z)
x_after = kf.x
check("update", Insight.numel(x_after) == 2)

# predict-update cycle
kf2 = Insight.KalmanFilter(2, 1)
F2 = Insight.from_data([1.0, 1.0, 0.0, 1.0])
F2 = Insight.reshape(F2, [2, 2])
H2 = Insight.from_data([1.0, 0.0])
H2 = Insight.reshape(H2, [1, 2])
kf2.F = F2
kf2.H = H2
for _ in 1:5
    Insight.predict(kf2)
    z2 = Insight.from_data([1.0])
    Insight.update(kf2, z2)
end
check("predict_update_cycle", Insight.numel(kf2.x) == 2)

# ============================================================================
# Results
# ============================================================================
println("\n" * "="^40)
println("Results: $passed passed, $failed failed")
if failed > 0
    exit(1)
end
