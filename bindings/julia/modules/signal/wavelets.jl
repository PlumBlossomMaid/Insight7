# modules/signal/wavelets.jl
# Wavelet functions.

"""
    morlet(M::Int; w=5.0, s=1.0, complete=true) -> InsightArray

Return a Morlet wavelet of length `M`.
"""
function morlet(M::Int; w::Float64=5.0, s::Float64=1.0,
                complete::Bool=true)::InsightArray
    ptr = ccall((:insight_jl_morlet, LIB_INSIGHT), Ptr{Cvoid},
                (Int64, Float64, Float64, Int32),
                Int64(M), w, s, complete ? Int32(1) : Int32(0))
    return _wrap_ptr(ptr, "morlet")
end

"""
    ricker(points::Int, a::Float64) -> InsightArray

Return a Ricker wavelet (Mexican hat) with `points` samples and width `a`.
"""
function ricker(points::Int, a::Float64)::InsightArray
    ptr = ccall((:insight_jl_ricker, LIB_INSIGHT), Ptr{Cvoid},
                (Int64, Float64), Int64(points), a)
    return _wrap_ptr(ptr, "ricker")
end

"""
    morlet2(M::Int, s::Float64; w=5.0) -> InsightArray

Return a Morlet wavelet parameterized by scale `s`.
"""
function morlet2(M::Int, s::Float64; w::Float64=5.0)::InsightArray
    ptr = ccall((:insight_jl_morlet2, LIB_INSIGHT), Ptr{Cvoid},
                (Int64, Float64, Float64), Int64(M), s, w)
    return _wrap_ptr(ptr, "morlet2")
end
