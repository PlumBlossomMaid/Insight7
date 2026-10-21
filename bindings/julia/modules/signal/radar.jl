# modules/signal/radar.jl
# Radar signal processing functions.

"""
    cfar_alpha(pfa::Float64, N::Int) -> Float64

Compute the CA-CFAR threshold multiplier for given PFA and window size.
"""
function cfar_alpha(pfa::Float64, N::Int)::Float64
    return ccall((:insight_jl_cfar_alpha, LIB_INSIGHT), Float64,
                 (Float64, Int32), pfa, Int32(N))
end

"""
    ca_cfar(data::InsightArray, guard_cells::Vector{Int32},
            reference_cells::Vector{Int32}, pfa::Float64=1e-3)
            -> (InsightArray, InsightArray)

Cell-Averaging CFAR detector. Returns (threshold, detections).
"""
function ca_cfar(data::InsightArray, guard_cells::Vector{Int32},
                 reference_cells::Vector{Int32},
                 pfa::Float64=1e-3)
    thresh_ref = Ref{Ptr{Cvoid}}(C_NULL)
    det_ref = Ref{Ptr{Cvoid}}(C_NULL)
    ccall((:insight_jl_ca_cfar, LIB_INSIGHT), Cvoid,
          (Ptr{Cvoid}, Ptr{Int32}, Int32, Ptr{Int32}, Int32, Float64,
           Ptr{Ptr{Cvoid}}, Ptr{Ptr{Cvoid}}),
          data, guard_cells, Int32(length(guard_cells)),
          reference_cells, Int32(length(reference_cells)), pfa,
          thresh_ref, det_ref)
    thresh = _wrap_ptr(thresh_ref[], "ca_cfar")
    det = _wrap_ptr(det_ref[], "ca_cfar")
    return (thresh, det)
end

"""
    pulse_compression(x::InsightArray, tpl::InsightArray; normalize=false) -> InsightArray

Perform pulse compression (matched filtering) using FFT correlation.
"""
function pulse_compression(x::InsightArray, tpl::InsightArray;
                           normalize::Bool=false)::InsightArray
    ptr = ccall((:insight_jl_pulse_compression, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Ptr{Cvoid}, Int32), x, tpl, Int32(normalize ? 1 : 0))
    return _wrap_ptr(ptr, "pulse_compression")
end

"""
    pulse_doppler(x::InsightArray) -> InsightArray

Compute the pulse-Doppler processing (FFT along slow-time axis).
"""
function pulse_doppler(x::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_pulse_doppler, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid},), x)
    return _wrap_ptr(ptr, "pulse_doppler")
end

"""
    pulse_doppler(x::InsightArray, window::String, nfft::Integer) -> InsightArray

Compute the pulse-Doppler processing with windowing (FFT along slow-time axis).
"""
function pulse_doppler(x::InsightArray, window::String, nfft::Integer)::InsightArray
    ptr = ccall((:insight_jl_pulse_doppler_window, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Cstring, Int64), x, window, Int64(nfft))
    return _wrap_ptr(ptr, "pulse_doppler")
end

"""
    mvdr(x::InsightArray, sv::InsightArray) -> InsightArray

Apply MVDR (Capon) beamforming.
"""
function mvdr(x::InsightArray, sv::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_mvdr, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Ptr{Cvoid}), x, sv)
    return _wrap_ptr(ptr, "mvdr")
end

"""
    ambgfun(x::InsightArray, fs::Float64, prf::Float64) -> InsightArray

Compute the ambiguity function of a signal.
"""
function ambgfun(x::InsightArray, fs::Float64, prf::Float64)::InsightArray
    ptr = ccall((:insight_jl_ambgfun, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Float64, Float64), x, fs, prf)
    return _wrap_ptr(ptr, "ambgfun")
end
