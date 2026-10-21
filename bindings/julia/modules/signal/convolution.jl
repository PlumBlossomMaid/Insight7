# modules/signal/convolution.jl
# Convolution and correlation functions.

"""
    fftconvolve(in1::InsightArray, in2::InsightArray; mode="full") -> InsightArray

N-D convolution using the FFT method.
"""
function fftconvolve(in1::InsightArray, in2::InsightArray;
                     mode::String="full")::InsightArray
    ptr = ccall((:insight_jl_fftconvolve, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Ptr{Cvoid}, Cstring), in1, in2, mode)
    return _wrap_ptr(ptr, "fftconvolve")
end

"""
    correlate(in1::InsightArray, in2::InsightArray; mode="full") -> InsightArray

N-D cross-correlation.
"""
function correlate(in1::InsightArray, in2::InsightArray;
                   mode::String="full")::InsightArray
    ptr = ccall((:insight_jl_correlate, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Ptr{Cvoid}, Cstring), in1, in2, mode)
    return _wrap_ptr(ptr, "correlate")
end

"""
    convolve2d(in1::InsightArray, in2::InsightArray; mode="full") -> InsightArray

2-D convolution.
"""
function convolve2d(in1::InsightArray, in2::InsightArray;
                    mode::String="full")::InsightArray
    ptr = ccall((:insight_jl_convolve2d, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Ptr{Cvoid}, Cstring), in1, in2, mode)
    return _wrap_ptr(ptr, "convolve2d")
end

"""
    correlate2d(in1::InsightArray, in2::InsightArray; mode="full") -> InsightArray

2-D cross-correlation.
"""
function correlate2d(in1::InsightArray, in2::InsightArray;
                     mode::String="full")::InsightArray
    ptr = ccall((:insight_jl_correlate2d, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Ptr{Cvoid}, Cstring), in1, in2, mode)
    return _wrap_ptr(ptr, "correlate2d")
end

"""
    correlation_lags(in1_len::Int, in2_len::Int; mode="full") -> InsightArray

Return lag values for a cross-correlation.
"""
function correlation_lags(in1_len::Int, in2_len::Int;
                          mode::String="full")::InsightArray
    ptr = ccall((:insight_jl_correlation_lags, LIB_INSIGHT), Ptr{Cvoid},
                (Int64, Int64, Cstring), Int64(in1_len), Int64(in2_len), mode)
    return _wrap_ptr(ptr, "correlation_lags")
end

"""
    choose_conv_method(in1::InsightArray, in2::InsightArray; mode="full") -> String

Choose the faster convolution method (direct or FFT).
"""
function choose_conv_method(in1::InsightArray, in2::InsightArray;
                            mode::String="full")::String
    ptr = ccall((:insight_jl_choose_conv_method, LIB_INSIGHT), Ptr{UInt8},
                (Ptr{Cvoid}, Ptr{Cvoid}, Cstring), in1, in2, mode)
    ptr == C_NULL && _fail("choose_conv_method")
    return unsafe_string(ptr)
end
