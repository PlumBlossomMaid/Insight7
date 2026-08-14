# modules/signal/demod.jl
# FM demodulation.

"""
    fm_demod(x::InsightArray; axis=-1) -> InsightArray

Demodulate an FM signal.
"""
function fm_demod(x::InsightArray; axis::Int=-1)::InsightArray
    ptr = ccall((:insight_jl_fm_demod, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32), x, Int32(_julia_axis(x, axis)))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end
