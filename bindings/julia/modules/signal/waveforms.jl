# modules/signal/waveforms.jl
# Waveform generators.

"""
    sawtooth(t::InsightArray; width::Float64=1.0) -> InsightArray

Return a periodic sawtooth or triangle waveform.
"""
function sawtooth(t::InsightArray; width::Float64=1.0)::InsightArray
    ptr = ccall((:insight_jl_sawtooth, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Float64), t, width)
    return _wrap_ptr(ptr, "sawtooth")
end

"""
    square_wf(t::InsightArray; duty::Float64=0.5) -> InsightArray

Return a periodic square-wave signal.
"""
function square_wf(t::InsightArray; duty::Float64=0.5)::InsightArray
    ptr = ccall((:insight_jl_square_wf, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Float64), t, duty)
    return _wrap_ptr(ptr, "square_wf")
end

"""
    chirp(t::InsightArray, f0::Float64, t1::Float64, f1::Float64; ...) -> InsightArray

Return a frequency-swept cosine signal (chirp).
"""
function chirp(t::InsightArray, f0::Float64, t1::Float64, f1::Float64;
               method::Int32=Int32(0), phi::Float64=0.0,
               vertex_zero::Bool=true)::InsightArray
    ptr = ccall((:insight_jl_chirp, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Float64, Float64, Float64, Int32, Float64, Int32),
                t, f0, t1, f1, method, phi, vertex_zero ? Int32(1) : Int32(0))
    return _wrap_ptr(ptr, "chirp")
end

"""
    unit_impulse(dims::Vector{Int64}; idx::Int64=-1) -> InsightArray

Return a unit impulse (Dirac delta) signal.
"""
function unit_impulse(dims::Vector{Int64}; idx::Int64=Int64(-1))::InsightArray
    ptr = ccall((:insight_jl_unit_impulse, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Int64}, Int32, Int64), dims, Int32(length(dims)), idx)
    return _wrap_ptr(ptr, "unit_impulse")
end
