# modules/signal/estimation.jl
# Kalman Filter wrapper for Julia bindings.

"""
    KalmanFilter

Multi-point Kalman Filter for state estimation.

Tracks multiple independent Kalman filters simultaneously. All matrices are
stacked on the `points` dimension for batched processing.

# Fields (read-write)
- `x` — State estimate `[points, dim_x, 1]`
- `P` — State covariance `[points, dim_x, dim_x]`
- `z` — Last measurement `[points, dim_z, 1]`
- `R` — Measurement noise `[points, dim_z, dim_z]`
- `Q` — Process noise `[points, dim_x, dim_x]`
- `F` — State transition `[points, dim_x, dim_x]`
- `H` — Measurement function `[points, dim_z, dim_x]`

# Read-only properties
- `dim_x` — Number of state variables
- `dim_z` — Number of measurement inputs
- `dim_u` — Number of control inputs
- `points` — Number of filter points

# Example
```julia
kf = KalmanFilter(2, 1)
# Set initial state and covariance
kf.F = from_list([1.0, 0.0, 1.0, 1.0], [2, 2])
kf.H = from_list([1.0, 0.0], [1, 2])
predict(kf)
update(kf, measurement)
```
"""
mutable struct KalmanFilter
    ptr::Ptr{Cvoid}

    function KalmanFilter(ptr::Ptr{Cvoid})
        kf = new(ptr)
        finalizer(kf) do k
            p = getfield(k, :ptr)
            if p != C_NULL
                ccall((:insight_jl_kalman_filter_free, LIB_INSIGHT),
                      Cvoid, (Ptr{Cvoid},), p)
                setfield!(k, :ptr, C_NULL)
            end
        end
        return kf
    end
end

"""
    KalmanFilter(dim_x::Int, dim_z::Int; dim_u::Int=0, points::Int=1, dtype::DType=float64) -> KalmanFilter

Construct a Kalman Filter.

# Arguments
- `dim_x` — Number of state variables
- `dim_z` — Number of measurement inputs
- `dim_u` — Number of control inputs (default: 0)
- `points` — Number of filter points (default: 1)
- `dtype` — Data type, `float64` or `float32` (default: `float64`)
"""
function KalmanFilter(dim_x::Int, dim_z::Int; dim_u::Int=0, points::Int=1,
                      dtype=DType(float64))::KalmanFilter
    dt_val = dtype isa DType ? Int32(dtype.val) : Int32(dtype)
    ptr = ccall((:insight_jl_kalman_filter_new, LIB_INSIGHT), Ptr{Cvoid},
                (Int32, Int32, Int32, Int32, Int32),
                Int32(dim_x), Int32(dim_z), Int32(dim_u), Int32(points), dt_val)
    return KalmanFilter(ptr)
end

# --- Property accessors ---

function Base.getproperty(kf::KalmanFilter, sym::Symbol)
    if sym === :x
        ptr = ccall((:insight_jl_kf_get_x, LIB_INSIGHT), Ptr{Cvoid},
                    (Ptr{Cvoid},), kf.ptr)
        return _wrap_ptr(ptr, "KalmanFilter.$sym")
    elseif sym === :P
        ptr = ccall((:insight_jl_kf_get_P, LIB_INSIGHT), Ptr{Cvoid},
                    (Ptr{Cvoid},), kf.ptr)
        return _wrap_ptr(ptr, "KalmanFilter.$sym")
    elseif sym === :z
        ptr = ccall((:insight_jl_kf_get_z, LIB_INSIGHT), Ptr{Cvoid},
                    (Ptr{Cvoid},), kf.ptr)
        return _wrap_ptr(ptr, "KalmanFilter.$sym")
    elseif sym === :R
        ptr = ccall((:insight_jl_kf_get_R, LIB_INSIGHT), Ptr{Cvoid},
                    (Ptr{Cvoid},), kf.ptr)
        return _wrap_ptr(ptr, "KalmanFilter.$sym")
    elseif sym === :Q
        ptr = ccall((:insight_jl_kf_get_Q, LIB_INSIGHT), Ptr{Cvoid},
                    (Ptr{Cvoid},), kf.ptr)
        return _wrap_ptr(ptr, "KalmanFilter.$sym")
    elseif sym === :F
        ptr = ccall((:insight_jl_kf_get_F, LIB_INSIGHT), Ptr{Cvoid},
                    (Ptr{Cvoid},), kf.ptr)
        return _wrap_ptr(ptr, "KalmanFilter.$sym")
    elseif sym === :H
        ptr = ccall((:insight_jl_kf_get_H, LIB_INSIGHT), Ptr{Cvoid},
                    (Ptr{Cvoid},), kf.ptr)
        return _wrap_ptr(ptr, "KalmanFilter.$sym")
    elseif sym === :dim_x
        return Int(ccall((:insight_jl_kf_get_dim_x, LIB_INSIGHT), Int32,
                         (Ptr{Cvoid},), kf.ptr))
    elseif sym === :dim_z
        return Int(ccall((:insight_jl_kf_get_dim_z, LIB_INSIGHT), Int32,
                         (Ptr{Cvoid},), kf.ptr))
    elseif sym === :dim_u
        return Int(ccall((:insight_jl_kf_get_dim_u, LIB_INSIGHT), Int32,
                         (Ptr{Cvoid},), kf.ptr))
    elseif sym === :points
        return Int(ccall((:insight_jl_kf_get_points, LIB_INSIGHT), Int32,
                         (Ptr{Cvoid},), kf.ptr))
    elseif sym === :ptr
        return getfield(kf, :ptr)
    else
        error("KalmanFilter has no property named :$sym")
    end
end

function Base.setproperty!(kf::KalmanFilter, sym::Symbol, v)
    arr = v isa InsightArray ? v : error("expected InsightArray for :$sym")
    if sym === :x
        ccall((:insight_jl_kf_set_x, LIB_INSIGHT), Cvoid,
              (Ptr{Cvoid}, Ptr{Cvoid}), kf.ptr, arr.ptr)
    elseif sym === :P
        ccall((:insight_jl_kf_set_P, LIB_INSIGHT), Cvoid,
              (Ptr{Cvoid}, Ptr{Cvoid}), kf.ptr, arr.ptr)
    elseif sym === :R
        ccall((:insight_jl_kf_set_R, LIB_INSIGHT), Cvoid,
              (Ptr{Cvoid}, Ptr{Cvoid}), kf.ptr, arr.ptr)
    elseif sym === :Q
        ccall((:insight_jl_kf_set_Q, LIB_INSIGHT), Cvoid,
              (Ptr{Cvoid}, Ptr{Cvoid}), kf.ptr, arr.ptr)
    elseif sym === :F
        ccall((:insight_jl_kf_set_F, LIB_INSIGHT), Cvoid,
              (Ptr{Cvoid}, Ptr{Cvoid}), kf.ptr, arr.ptr)
    elseif sym === :H
        ccall((:insight_jl_kf_set_H, LIB_INSIGHT), Cvoid,
              (Ptr{Cvoid}, Ptr{Cvoid}), kf.ptr, arr.ptr)
    else
        error("cannot set property :$sym on KalmanFilter")
    end
    return v
end

"""
    predict(kf::KalmanFilter)

Predict step: `x = F * x`, `P = F * P * F' + Q`.
"""
function predict(kf::KalmanFilter)::Nothing
    ccall((:insight_jl_kalman_filter_predict, LIB_INSIGHT), Cvoid,
          (Ptr{Cvoid},), kf.ptr)
    return nothing
end

"""
    update(kf::KalmanFilter, z::InsightArray)

Update step with measurement `z` of shape `[points, dim_z, 1]`.

Computes the Kalman gain, updates the state estimate and covariance.
"""
function update(kf::KalmanFilter, z::InsightArray)::Nothing
    ccall((:insight_jl_kalman_filter_update, LIB_INSIGHT), Cvoid,
          (Ptr{Cvoid}, Ptr{Cvoid}), kf.ptr, z.ptr)
    return nothing
end
