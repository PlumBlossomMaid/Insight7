# modules/signal/acoustics.jl
# Acoustic scale conversions.

"""
    mel2hz(mels::InsightArray) -> InsightArray

Convert mel frequencies to Hz.
"""
function mel2hz(mels::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_mel2hz, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid},), mels)
    return _wrap_ptr(ptr, "mel2hz")
end

"""
    hz2mel(hz::InsightArray) -> InsightArray

Convert Hz frequencies to mel scale.
"""
function hz2mel(hz::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_hz2mel, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid},), hz)
    return _wrap_ptr(ptr, "hz2mel")
end

"""
    mel_frequencies(n_mels::Int; f_low=0.0, f_high=11000.0) -> InsightArray

Return an array of `n_mels` equally spaced frequencies in mel scale.
"""
function mel_frequencies(n_mels::Int; f_low::Float64=0.0,
                         f_high::Float64=11000.0)::InsightArray
    ptr = ccall((:insight_jl_mel_frequencies, LIB_INSIGHT), Ptr{Cvoid},
                (Int64, Float64, Float64), Int64(n_mels), f_low, f_high)
    return _wrap_ptr(ptr, "mel_frequencies")
end

"""
    hz2bark(hz::InsightArray) -> InsightArray

Convert Hz frequencies to Bark scale.
"""
function hz2bark(hz::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_hz2bark, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid},), hz)
    return _wrap_ptr(ptr, "hz2bark")
end

"""
    bark2hz(bark::InsightArray) -> InsightArray

Convert Bark scale frequencies to Hz.
"""
function bark2hz(bark::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_bark2hz, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid},), bark)
    return _wrap_ptr(ptr, "bark2hz")
end
