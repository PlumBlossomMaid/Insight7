# modules/signal/spectral.jl
# Spectral analysis functions.

"""
    welch_jl(x::InsightArray; ...) -> (f, Pxx)

Estimate power spectral density using Welch's method.
"""
function welch_jl(x::InsightArray; fs::Float64=1.0, window::String="hann",
                  nperseg::Int=256, noverlap::Int=0, nfft::Int=0,
                  detrend::String="constant", return_onesided::Bool=true,
                  scaling::String="density")
    f_ref = Ref{Ptr{Cvoid}}(C_NULL)
    p_ref = Ref{Ptr{Cvoid}}(C_NULL)
    ccall((:insight_jl_welch, LIB_INSIGHT), Cvoid,
          (Ptr{Cvoid}, Float64, Cstring, Int64, Int64, Int64, Cstring, Int32,
           Cstring, Ptr{Ptr{Cvoid}}, Ptr{Ptr{Cvoid}}),
          x, fs, window, Int64(nperseg), Int64(noverlap), Int64(nfft),
          detrend, return_onesided ? Int32(1) : Int32(0), scaling,
          f_ref, p_ref)
    f = _wrap_ptr(f_ref[], "welch_jl")
    pxx = _wrap_ptr(p_ref[], "welch_jl")
    return (f, pxx)
end

"""
    periodogram_jl(x::InsightArray; ...) -> (f, Pxx)

Estimate power spectral density using a periodogram.
"""
function periodogram_jl(x::InsightArray; fs::Float64=1.0,
                        window::String="boxcar", nfft::Int=0,
                        detrend::String="constant",
                        return_onesided::Bool=true,
                        scaling::String="density")
    f_ref = Ref{Ptr{Cvoid}}(C_NULL)
    p_ref = Ref{Ptr{Cvoid}}(C_NULL)
    ccall((:insight_jl_periodogram, LIB_INSIGHT), Cvoid,
          (Ptr{Cvoid}, Float64, Cstring, Int64, Cstring, Int32, Cstring,
           Ptr{Ptr{Cvoid}}, Ptr{Ptr{Cvoid}}),
          x, fs, window, Int64(nfft), detrend,
          return_onesided ? Int32(1) : Int32(0), scaling, f_ref, p_ref)
    f = _wrap_ptr(f_ref[], "periodogram_jl")
    pxx = _wrap_ptr(p_ref[], "periodogram_jl")
    return (f, pxx)
end

"""
    csd(x::InsightArray, y::InsightArray; ...) -> (f, Pxx)

Estimate cross-spectral density.
"""
function csd(x::InsightArray, y::InsightArray; fs::Float64=1.0,
             window::String="hann", nperseg::Int=256,
             noverlap::Int=0, nfft::Int=0)
    f_ref = Ref{Ptr{Cvoid}}(C_NULL)
    Pxx_ref = Ref{Ptr{Cvoid}}(C_NULL)
    ccall((:insight_jl_csd, LIB_INSIGHT), Cvoid,
          (Ptr{Cvoid}, Ptr{Cvoid}, Float64, Cstring, Int64, Int64, Int64,
           Ptr{Ptr{Cvoid}}, Ptr{Ptr{Cvoid}}),
          x, y, fs, window, Int64(nperseg), Int64(noverlap), Int64(nfft),
          f_ref, Pxx_ref)
    f_arr = _wrap_ptr(f_ref[], "csd")
    Pxx_arr = _wrap_ptr(Pxx_ref[], "csd")
    return (f=f_arr, Pxx=Pxx_arr)
end

"""
    coherence(x::InsightArray, y::InsightArray; ...) -> (f, Pxx)

Estimate magnitude-squared coherence.
"""
function coherence(x::InsightArray, y::InsightArray; fs::Float64=1.0,
                   window::String="hann", nperseg::Int=256,
                   noverlap::Int=0, nfft::Int=0)
    f_ref = Ref{Ptr{Cvoid}}(C_NULL)
    Pxx_ref = Ref{Ptr{Cvoid}}(C_NULL)
    ccall((:insight_jl_coherence, LIB_INSIGHT), Cvoid,
          (Ptr{Cvoid}, Ptr{Cvoid}, Float64, Cstring, Int64, Int64, Int64,
           Ptr{Ptr{Cvoid}}, Ptr{Ptr{Cvoid}}),
          x, y, fs, window, Int64(nperseg), Int64(noverlap), Int64(nfft),
          f_ref, Pxx_ref)
    f_arr = _wrap_ptr(f_ref[], "coherence")
    Pxx_arr = _wrap_ptr(Pxx_ref[], "coherence")
    return (f=f_arr, Pxx=Pxx_arr)
end

"""
    spectrogram(x::InsightArray; ...) -> (f, t, Sxx)

Compute a spectrogram using STFT.
"""
function spectrogram(x::InsightArray; fs::Float64=1.0,
                     window::String="hann", nperseg::Int=256,
                     noverlap::Int=0, nfft::Int=0)
    f_ref = Ref{Ptr{Cvoid}}(C_NULL)
    t_ref = Ref{Ptr{Cvoid}}(C_NULL)
    Sxx_ref = Ref{Ptr{Cvoid}}(C_NULL)
    ccall((:insight_jl_spectrogram, LIB_INSIGHT), Cvoid,
          (Ptr{Cvoid}, Float64, Cstring, Int64, Int64, Int64,
           Ptr{Ptr{Cvoid}}, Ptr{Ptr{Cvoid}}, Ptr{Ptr{Cvoid}}),
          x, fs, window, Int64(nperseg), Int64(noverlap), Int64(nfft),
          f_ref, t_ref, Sxx_ref)
    f_arr = _wrap_ptr(f_ref[], "spectrogram")
    t_arr = _wrap_ptr(t_ref[], "spectrogram")
    Sxx_arr = _wrap_ptr(Sxx_ref[], "spectrogram")
    return (f=f_arr, t=t_arr, Sxx=Sxx_arr)
end

"""
    stft(x::InsightArray; ...) -> (f, t, Sxx)

Compute the Short-Time Fourier Transform.
"""
function stft(x::InsightArray; fs::Float64=1.0,
              window::String="hann", nperseg::Int=256,
              noverlap::Int=0, nfft::Int=0)
    f_ref = Ref{Ptr{Cvoid}}(C_NULL)
    t_ref = Ref{Ptr{Cvoid}}(C_NULL)
    Sxx_ref = Ref{Ptr{Cvoid}}(C_NULL)
    ccall((:insight_jl_stft, LIB_INSIGHT), Cvoid,
          (Ptr{Cvoid}, Float64, Cstring, Int64, Int64, Int64,
           Ptr{Ptr{Cvoid}}, Ptr{Ptr{Cvoid}}, Ptr{Ptr{Cvoid}}),
          x, fs, window, Int64(nperseg), Int64(noverlap), Int64(nfft),
          f_ref, t_ref, Sxx_ref)
    f_arr = _wrap_ptr(f_ref[], "stft")
    t_arr = _wrap_ptr(t_ref[], "stft")
    Sxx_arr = _wrap_ptr(Sxx_ref[], "stft")
    return (f=f_arr, t=t_arr, Sxx=Sxx_arr)
end

"""
    vectorstrength(events::InsightArray, period::Float64) -> (strength, phase)

Compute the vector strength of events relative to a period.
"""
function vectorstrength(events::InsightArray, period::Float64)
    strength_ref = Ref{Float64}(0.0)
    phase_ref = Ref{Float64}(0.0)
    ccall((:insight_jl_vectorstrength, LIB_INSIGHT), Cvoid,
          (Ptr{Cvoid}, Float64, Ptr{Float64}, Ptr{Float64}),
          events, period, strength_ref, phase_ref)
    return (strength=strength_ref[], phase=phase_ref[])
end

"""
    lombscargle(x::InsightArray, y::InsightArray, freqs::InsightArray) -> InsightArray

Compute the Lomb-Scargle periodogram. Returns power spectral density at each frequency.
"""
function lombscargle(x::InsightArray, y::InsightArray, freqs::InsightArray)
    ptr = ccall((:insight_jl_lombscargle, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}),
                x, y, freqs)
    return _wrap_ptr(ptr, "lombscargle")
end
