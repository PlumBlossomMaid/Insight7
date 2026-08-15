# modules/signal/filtering.jl
# Digital filtering functions.

"""
    hilbert(x::InsightArray; N=-1) -> InsightArray

Compute the analytic signal using the Hilbert transform.
"""
function hilbert(x::InsightArray; N::Int=-1)::InsightArray
    ptr = ccall((:insight_jl_hilbert, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int64), x, Int64(N))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

"""
    hilbert2(x::InsightArray; N=-1) -> InsightArray

Compute the 2-D analytic signal using the Hilbert transform.
"""
function hilbert2(x::InsightArray; N::Int=-1)::InsightArray
    ptr = ccall((:insight_jl_hilbert2, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int64), x, Int64(N))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

"""
    detrend(data::InsightArray; axis=-1, type="linear") -> InsightArray

Remove a trend from the data along the given axis.
"""
function detrend(data::InsightArray; axis::Int=-1,
                 type::String="linear")::InsightArray
    ptr = ccall((:insight_jl_detrend, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32, Cstring), data,
                Int32(_julia_axis(data, axis)), type)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

"""
    wiener(im::InsightArray; noise=-1.0) -> InsightArray

Apply a Wiener filter to an N-dimensional array.
"""
function wiener(im::InsightArray; noise::Float64=-1.0)::InsightArray
    ptr = ccall((:insight_jl_wiener, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Float64), im, noise)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

"""
    firfilter(b::InsightArray, x::InsightArray; axis=-1) -> InsightArray

Apply an FIR filter to a signal.
"""
function firfilter(b::InsightArray, x::InsightArray;
                   axis::Int=-1)::InsightArray
    ptr = ccall((:insight_jl_firfilter, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Ptr{Cvoid}, Int32), b, x,
                Int32(_julia_axis(x, axis)))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

"""
    lfilter(b::InsightArray, a::InsightArray, x::InsightArray; axis=-1) -> InsightArray

Apply an IIR or FIR filter using direct-form II transposed.
"""
function lfilter(b::InsightArray, a::InsightArray, x::InsightArray;
                 axis::Int=-1)::InsightArray
    ptr = ccall((:insight_jl_lfilter, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Int32),
                b, a, x, Int32(_julia_axis(x, axis)))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

"""
    lfilter_zi(b::InsightArray, a::InsightArray) -> InsightArray

Compute the initial state for lfilter for step-response steady-state.
"""
function lfilter_zi(b::InsightArray, a::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_lfilter_zi, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Ptr{Cvoid}), b, a)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

"""
    filtfilt(b::InsightArray, a::InsightArray, x::InsightArray; axis=-1) -> InsightArray

Apply a forward-backward digital filter (zero-phase filtering).
"""
function filtfilt(b::InsightArray, a::InsightArray, x::InsightArray;
                  axis::Int=-1)::InsightArray
    ptr = ccall((:insight_jl_filtfilt, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Int32),
                b, a, x, Int32(_julia_axis(x, axis)))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

"""
    decimate(x::InsightArray, q::Int; axis=-1, zero_phase=true) -> InsightArray

Downsample a signal after applying an anti-aliasing filter.
"""
function decimate(x::InsightArray, q::Int; axis::Int=-1,
                  zero_phase::Bool=true)::InsightArray
    ptr = ccall((:insight_jl_decimate, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int64, Int32, Int32),
                x, Int64(q), Int32(_julia_axis(x, axis)),
                zero_phase ? Int32(1) : Int32(0))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

"""
    resample(x::InsightArray, num::Int; axis=-1) -> InsightArray

Resample a signal to a given number of samples using FFT.
"""
function resample(x::InsightArray, num::Int; axis::Int=-1)::InsightArray
    ptr = ccall((:insight_jl_resample, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int64, Int32), x, Int64(num),
                Int32(_julia_axis(x, axis)))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

"""
    resample_poly(x::InsightArray, up::Int, down::Int; axis=-1) -> InsightArray

Resample a signal using polyphase filtering.
"""
function resample_poly(x::InsightArray, up::Int, down::Int;
                       axis::Int=-1)::InsightArray
    ptr = ccall((:insight_jl_resample_poly, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int64, Int64, Int32), x, Int64(up),
                Int64(down), Int32(_julia_axis(x, axis)))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

"""
    freq_shift(x::InsightArray, freq::Float64, fs::Float64) -> InsightArray

Shift the frequency content of a signal.
"""
function freq_shift(x::InsightArray, freq::Float64,
                    fs::Float64)::InsightArray
    ptr = ccall((:insight_jl_freq_shift, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Float64, Float64), x, freq, fs)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

"""
    firfilter_zi_state(b, x, zi; axis=-1) -> (y, zf)

Apply FIR filter with initial/final state tracking.
"""
function firfilter_zi_state(b::InsightArray, x::InsightArray, zi::InsightArray;
                            axis::Int=-1)
    y_ref = Ref{Ptr{Cvoid}}(C_NULL)
    zf_ref = Ref{Ptr{Cvoid}}(C_NULL)
    ccall((:insight_jl_firfilter_zi_state, LIB_INSIGHT), Cvoid,
          (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Int32,
           Ptr{Ptr{Cvoid}}, Ptr{Ptr{Cvoid}}),
          b, x, zi, Int32(_julia_axis(x, axis)), y_ref, zf_ref)
    y_arr = InsightArray(y_ref[]); finalizer(_free, y_arr)
    zf_arr = InsightArray(zf_ref[]); finalizer(_free, zf_arr)
    return (y=y_arr, zf=zf_arr)
end
