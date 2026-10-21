# modules/signal/filter_design.jl
# Filter design functions.

"""
    kaiser_beta(a::Float64) -> Float64

Compute the Kaiser beta parameter for a given stopband attenuation.
"""
function kaiser_beta(a::Float64)::Float64
    ccall((:insight_jl_kaiser_beta, LIB_INSIGHT), Float64, (Float64,), a)
end

"""
    kaiser_atten(numtaps::Int, width::Float64) -> Float64

Compute the Kaiser window attenuation for given taps and transition width.
"""
function kaiser_atten(numtaps::Int, width::Float64)::Float64
    ccall((:insight_jl_kaiser_atten, LIB_INSIGHT), Float64,
          (Int64, Float64), Int64(numtaps), width)
end

"""
    firwin(numtaps::Int, cutoff::Vector{Float64}; ...) -> InsightArray

Design a FIR filter using the window method.
"""
function firwin(numtaps::Int, cutoff::Vector{Float64};
                window::String="hamming", pass_zero::String="lowpass",
                scale::Bool=true)::InsightArray
    ptr = ccall((:insight_jl_firwin, LIB_INSIGHT), Ptr{Cvoid},
                (Int64, Ptr{Float64}, Int32, Cstring, Cstring, Int32),
                Int64(numtaps), cutoff, Int32(length(cutoff)),
                window, pass_zero, scale ? Int32(1) : Int32(0))
    return _wrap_ptr(ptr, "firwin")
end

"""
    firwin2(numtaps::Int, freq, gain; window="hamming") -> InsightArray

Design a FIR filter with arbitrary frequency response.
"""
function firwin2(numtaps::Int, freq::Vector{Float64}, gain::Vector{Float64};
                 window::String="hamming")::InsightArray
    ptr = ccall((:insight_jl_firwin2, LIB_INSIGHT), Ptr{Cvoid},
                (Int64, Ptr{Float64}, Int32, Ptr{Float64}, Int32, Cstring),
                Int64(numtaps), freq, Int32(length(freq)), gain, Int32(length(gain)), window)
    return _wrap_ptr(ptr, "firwin2")
end

"""
    cmplx_sort(p::InsightArray) -> InsightArray

Sort complex roots by magnitude and conjugate pairs.
"""
function cmplx_sort(p::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_cmplx_sort, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid},), p)
    return _wrap_ptr(ptr, "cmplx_sort")
end
