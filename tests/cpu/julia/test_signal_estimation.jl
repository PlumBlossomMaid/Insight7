# Signal estimation CPU binding tests
# Run with:
#   LD_LIBRARY_PATH=build/backends/cpu julia tests/cpu/julia/test_signal_estimation.jl

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
F = Insight.from_data([1.0, 1.0, 0.0, 1.0], Insight.float64)
F = Insight.reshape(F, Int64[2, 2])
kf.F = F
check("set_F", Insight.numel(kf.F) == 4)

# set H
H = Insight.from_data([1.0, 0.0], Insight.float64)
H = Insight.reshape(H, Int64[2, 1])
kf.H = H
check("set_H", Insight.numel(kf.H) == 2)

# predict
kf2 = Insight.KalmanFilter(2, 1)
F = Insight.from_data([1.0, 1.0, 0.0, 1.0], Insight.float64)
F = Insight.reshape(F, Int64[2, 2])
kf2.F = F
H = Insight.from_data([1.0, 0.0], Insight.float64)
H = Insight.reshape(H, Int64[2, 1])
kf2.H = H
# Set initial covariance and noise matrices
P = Insight.from_data([1.0, 0.0, 0.0, 1.0], Insight.float64)
P = Insight.reshape(P, Int64[2, 2])
kf2.P = P
R = Insight.from_data([1.0], Insight.float64)
R = Insight.reshape(R, Int64[1, 1])
kf2.R = R
Q = Insight.from_data([0.1, 0.0, 0.0, 0.1], Insight.float64)
Q = Insight.reshape(Q, Int64[2, 2])
kf2.Q = Q
Insight.predict(kf2)
x_after = kf2.x
check("predict", Insight.numel(x_after) == 2)

# update
kf3 = Insight.KalmanFilter(2, 1)
kf3.F = F
kf3.H = H
kf3.P = P
kf3.R = R
kf3.Q = Q
Insight.predict(kf3)
z = Insight.from_data([1.0], Insight.float64)
z = Insight.reshape(z, Int64[1, 1])
Insight.update(kf3, z)
x_after_u = kf3.x
check("update", Insight.numel(x_after_u) == 2)

# predict-update cycle
kf4 = Insight.KalmanFilter(2, 1)
kf4.F = F
kf4.H = H
kf4.P = P
kf4.R = R
kf4.Q = Q
for i in 1:5
    Insight.predict(kf4)
    zi = Insight.from_data([1.0], Insight.float64)
    zi = Insight.reshape(zi, Int64[1, 1])
    Insight.update(kf4, zi)
end
check("predict_update_cycle", Insight.numel(kf4.x) == 2)

# ============================================================================
# Results
# ============================================================================
println("\n" * "="^40)
println("Results: $passed passed, $failed failed")
if failed > 0
    exit(1)
end
