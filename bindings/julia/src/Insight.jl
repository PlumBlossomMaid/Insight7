# bindings/julia/Insight.jl
#
# Julia bindings for the Insight7 scientific computing framework.
# Uses ccall() to call into libinsight_julia.so (C ABI wrappers).
# API style: PaddlePaddle (Insight.float32, Insight.CPUPlace(), etc.)
#
# Usage:
#   push!(LOAD_PATH, "/path/to/bindings/julia")
#   using Insight
#
#   a = Insight.zeros([2, 3], Insight.float32)
#   b = Insight.ones([2, 3], Insight.float32)
#   c = a + b

module Insight

using Libdl

export Array, zeros, ones, full, arange, linspace, eye,
       CPUPlace, GPUPlace,
       add, sub, mul, div, pow, matmul, dot, det, inv, solve, svd,
       fft, ifft, rand, randn, cast,
       # Convenience
       item, to, to_data, negative, abs, equal, greater, less,
       max, min, argmax, argmin, prod, outer, norm, trace, DType,
       # Complex
       is_complex, has_complex_shape, to_complex, as_complex, as_real,
       real_part, imag_part,
       # Additional unary
       exp2_fn, expm1_fn, log1p_fn, cbrt_fn, reciprocal_fn,
       asinh_fn, acosh_fn, atanh_fn, trunc_fn, deg2rad_fn, rad2deg_fn,
       conj_fn, angle_fn,
       # Logical binary
       logical_and, logical_or, logical_xor, logical_not,
       # Manipulation
       squeeze,
       # Additional reduction
       cummax, cummin, sem, count_nonzero, median, quantile, percentile,
       nansum, nanmean, nanmax, nanmin, nanstd, nanvar,
       # Additional manipulation
       permute, swapaxes, moveaxis, fliplr, flipud, rot90,
       diag_fn, diagonal, tril, triu, diff_fn,
       # Additional indexing
       unique_ins, topk, gather, scatter, scatter_add, scatter_reduce,
       interp, indices_fn, ix_fn,
       # Additional random
       seed, get_seed, rand_like, randn_like,
       exponential, gamma_dist, beta_dist, binomial_dist, poisson_dist,
       # Additional FFT
       fftshift, ifftshift, fftfreq, next_fast_len, hfft, ihfft,
       rfft, irfft, rfft2, irfft2, rfftn, irfftn,
       # Additional linalg
       lstsq, cond_fn, matrix_rank, matrix_power, slogdet, eigvalsh, pinv,
       # Signal
       convolve, unwrap, sinc,
       hann, hamming, blackman, kaiser, gaussian, boxcar, triang,
       bartlett, flattop, nuttall, blackmanharris, tukey, chebwin, taylor,
       get_window, sawtooth, square_wf, chirp, unit_impulse,
       gauss_spline, cubic, quadratic,
       kaiser_beta, kaiser_atten, firwin, cmplx_sort,
       fftconvolve, correlate, correlation_lags,
       hilbert, detrend, lfilter, filtfilt, decimate, resample, freq_shift,
       welch_jl, periodogram_jl,
       morlet, ricker,
       mel2hz, hz2mel, mel_frequencies, hz2bark, bark2hz,
       fm_demod, argrelmax, argrelmin, cfar_alpha, ca_cfar,
       pulse_compression, pulse_doppler, mvdr, ambgfun,
       read_bin, write_bin, pack_bin, unpack_bin, read_sigmf, write_sigmf,
       cosine_win, general_hamming, parzen_win, bohman_win, barthann_win,
       exponential_win, general_gaussian_win,
       firwin2, convolve2d, correlate2d,
       hilbert2, wiener, firfilter, lfilter_zi, resample_poly,
       morlet2,
       csd, coherence, spectrogram, stft, vectorstrength, lombscargle,
       choose_conv_method, firfilter_zi_state,
       # Device info
       device_name, active_gpu_backend_name, active_gpu_backend_version,
       gpu_version, driver_version, compute_capability,
       device_memory, device_memory_info, gpu_count, load_backend, has_device,
       get_device, set_device,
       # Profiler / Timer
       Timer, timer_start, timer_stop, timer_elapsed_ms, timer_destroy,
       Profiler, profiler_start, profiler_stop, profiler_destroy,
       profiler_reset, profiler_begin_event, profiler_end_event,
       profiler_get_events, profiler_report,
       # Signal submodule
       signal


# ============================================================================
# Configuration
# ============================================================================

const LIB_INSIGHT = let
    _suffix = Sys.iswindows() ? ".dll" : ".so"
    _prefix = Sys.iswindows() ? "" : "lib"
    _basename = _prefix * "insight_julia" * _suffix
    _local = joinpath(@__DIR__, _basename)
    _parent = joinpath(@__DIR__, "..", _basename)
    if isfile(_local)
        _local
    elseif isfile(_parent)
        _parent
    else
        _basename
    end
end

# Auto-initialize CPU backend on module load
function __init__()
    # Pre-load ALL backend .so/.dll files (CPU + GPU) so that C++ dlopen finds them
    _dir = @__DIR__
    _parent = joinpath(_dir, "..")
    _backend_suffix = Sys.iswindows() ? ".dll" : ".so"
    for _d in (_dir, _parent)
        if isdir(_d)
            for _f in readdir(_d; join=true)
                if occursin("insight_", _f) && endswith(_f, _backend_suffix) && occursin("_backend", _f)
                    try
                        Libdl.dlopen(_f, Libdl.RTLD_GLOBAL)
                    catch
                    end
                end
            end
        end
        if !(_d in Libdl.DL_LOAD_PATH)
            push!(Libdl.DL_LOAD_PATH, _d)
        end
    end
    ccall((:insight_jl_init, LIB_INSIGHT), Cvoid, ())
end

"""
    load_backend(name::String)::Bool

Load an additional backend. Use "gpu" for the selected GPU backend.
"""
function load_backend(name::String)::Bool
    ccall((:insight_jl_load_backend, LIB_INSIGHT), Int32, (Cstring,), name) == 1
end

"""
    has_device(device_type::Int) -> Bool

Check if a device kind is available. 0=CPU, 1=GPU.
"""
function has_device(device_type::Int)::Bool
    ccall((:insight_jl_has_device, LIB_INSIGHT), Int32, (Int32,), Int32(device_type)) == 1
end

"""
    get_device() -> (device_type::Int, device_id::Int)

Get the current default device. Returns (0, 0) for CPU, (1, id) for GPU.
"""
function get_device()::Tuple{Int,Int}
    dtype = Int(ccall((:insight_jl_get_device_type, LIB_INSIGHT), Int32, ()))
    did = Int(ccall((:insight_jl_get_device_id, LIB_INSIGHT), Int32, ()))
    return (dtype, did)
end

"""
    CPUPlace() -> Place
    GPUPlace(id=0) -> Place

Create a Place object for device specification.
"""
const CPUPlace = () -> (Int32(0), Int32(0))
const GPUPlace = (id::Int=0) -> (Int32(1), Int32(id))

_place_type(p::Tuple{Int32,Int32}) = p[1]
_place_id(p::Tuple{Int32,Int32}) = p[2]

_place_type(::Nothing) = Int32(-1)  # "unspecified"

# Get current device type (0=CPU, 1=GPU) as Int32
function _current_device()::Int32
    return ccall((:insight_jl_get_device_type, LIB_INSIGHT), Int32, ())
end

"""
    set_device(device_type::Int, device_id::Int=0)
    set_device(p::Tuple{Int32,Int32})  e.g. CPUPlace() or GPUPlace(0)
    set_device(s::String)              "cpu" or "gpu"

Set the current default device. Throws if the requested device is not available.
"""
function set_device(device_type::Int, device_id::Int=0)
    ok = ccall((:insight_jl_set_device, LIB_INSIGHT), Int32,
               (Int32, Int32), Int32(device_type), Int32(device_id))
    if ok == 0
        error("Insight: device not available (type=$device_type, id=$device_id)")
    end
end
function set_device(p::Tuple{Int32,Int32})
    set_device(Int(p[1]), Int(p[2]))
end
function set_device(s::String)
    if s == "cpu"
        set_device(Int32(0), Int32(0))
    elseif s == "gpu"
        set_device(Int32(1), Int32(0))
    else
        error("Insight: unknown device \"$s\", use \"cpu\" or \"gpu\"")
    end
end

# DType enum mapping (matches InsightDType in c_api/dtype.h)
function _dtype_code(dt::DataType)::Int32
    dt === Bool && return Int32(1)      # INSIGHT_DTYPE_BOOL
    dt === UInt8 && return Int32(2)     # INSIGHT_DTYPE_U8
    dt === Int8 && return Int32(3)      # INSIGHT_DTYPE_I8
    dt === Int16 && return Int32(4)     # INSIGHT_DTYPE_I16
    dt === Int32 && return Int32(5)     # INSIGHT_DTYPE_I32
    dt === Int64 && return Int32(6)     # INSIGHT_DTYPE_I64
    dt === Float32 && return Int32(9)   # INSIGHT_DTYPE_F32
    dt === Float64 && return Int32(10)  # INSIGHT_DTYPE_F64
    dt === UInt16 && return Int32(15)   # INSIGHT_DTYPE_U16
    dt === UInt32 && return Int32(16)   # INSIGHT_DTYPE_U32
    dt === UInt64 && return Int32(17)   # INSIGHT_DTYPE_U64
    error("Unsupported dtype: $dt")
end

# ============================================================================
# Device information
# ============================================================================

"""
    device_name(device_kind::Int=0, device_id::Int=0)::String

Return the public device name for CPU (`device_kind=0`) or GPU
(`device_kind=1`). Concrete backend details are available through
`active_gpu_backend_name()` and `active_gpu_backend_version()`.
"""
function device_name(device_kind::Int=0, device_id::Int=0)::String
    buf = Vector{UInt8}(undef, 256)
    ccall((:insight_jl_device_name, LIB_INSIGHT), Cvoid,
          (Int32, Int32, Ptr{UInt8}, Csize_t), Int32(device_kind),
          Int32(device_id), buf, 256)
    return String(buf[1:findfirst(==(0x00), buf)-1])
end

function active_gpu_backend_name()::String
    buf = Vector{UInt8}(undef, 256)
    ccall((:insight_jl_active_gpu_backend_name, LIB_INSIGHT), Cvoid,
          (Ptr{UInt8}, Csize_t), buf, 256)
    return String(buf[1:findfirst(==(0x00), buf)-1])
end

function active_gpu_backend_version()::String
    buf = Vector{UInt8}(undef, 256)
    ccall((:insight_jl_active_gpu_backend_version, LIB_INSIGHT), Cvoid,
          (Ptr{UInt8}, Csize_t), buf, 256)
    return String(buf[1:findfirst(==(0x00), buf)-1])
end

"""Get the GPU runtime version (major*1000+minor*10, 0 if not available)."""
function gpu_version()::Int
    Int(ccall((:insight_jl_gpu_runtime_version, LIB_INSIGHT), Int32, ()))
end

function driver_version()::Int
    Int(ccall((:insight_jl_driver_version, LIB_INSIGHT), Int32, ()))
end

function compute_capability(device_id::Int=0)::Int
    Int(ccall((:insight_jl_compute_capability, LIB_INSIGHT), Int32,
              (Int32,), Int32(device_id)))
end

function device_memory(device_id::Int=0)
    total = Ref{Csize_t}(0)
    free = Ref{Csize_t}(0)
    ccall((:insight_jl_device_memory, LIB_INSIGHT), Cvoid,
          (Int32, Ptr{Csize_t}, Ptr{Csize_t}), Int32(device_id), total, free)
    return (total=total[], free=free[])
end

function device_memory_info(device_kind::Int, device_id::Int=0)
    total = Ref{Csize_t}(0)
    free = Ref{Csize_t}(0)
    ccall((:insight_jl_device_memory_info, LIB_INSIGHT), Cvoid,
          (Int32, Int32, Ptr{Csize_t}, Ptr{Csize_t}),
          Int32(device_kind), Int32(device_id), total, free)
    return (total=total[], free=free[])
end

function gpu_count()::Int
    Int(ccall((:insight_jl_gpu_count, LIB_INSIGHT), Int32, ()))
end

# ============================================================================
# DType enum (mirrors InsightDType in C)
# ============================================================================

module DTypeValues
const UNKNOWN  = Int32(0)
const BOOL     = Int32(1)
const U8       = Int32(2)
const I8       = Int32(3)
const I16      = Int32(4)
const I32      = Int32(5)
const I64      = Int32(6)
const F16      = Int32(7)
const BF16     = Int32(8)
const F32      = Int32(9)
const F64      = Int32(10)
const C32      = Int32(11)
const C64      = Int32(12)
const F8_E4M3  = Int32(13)
const F8_E5M2  = Int32(14)
const U16      = Int32(15)
const U32      = Int32(16)
const U64      = Int32(17)
end

# Module-level dtype shortcuts
const bool     = DTypeValues.BOOL
const uint8    = DTypeValues.U8
const int8     = DTypeValues.I8
const int16    = DTypeValues.I16
const int32    = DTypeValues.I32
const int64    = DTypeValues.I64

# DType wrapper struct for API compatibility
struct DType
    val::Int32
end
const uint16   = DTypeValues.U16
const uint32   = DTypeValues.U32
const uint64   = DTypeValues.U64
const float16  = DTypeValues.F16
const bfloat16 = DTypeValues.BF16
const float32  = DTypeValues.F32
const float64  = DTypeValues.F64
const complex64  = DTypeValues.C32
const complex128 = DTypeValues.C64

# Place shortcuts
const CPU = Int32(0)
const GPU = Int32(1)

# ============================================================================
# Array wrapper
# ============================================================================

mutable struct InsightArray
    ptr::Ptr{Cvoid}  # Pointer to C++ Array object
end

function _free(arr::InsightArray)
    if arr.ptr != C_NULL
        ccall((:insight_jl_array_free, LIB_INSIGHT), Cvoid, (Ptr{Cvoid},),
              arr.ptr)
        arr.ptr = C_NULL
    end
end

Base.unsafe_convert(::Type{Ptr{Cvoid}}, arr::InsightArray) = arr.ptr

# ============================================================================
# In-place mutation
# ============================================================================

"""Fill all elements with a scalar value (in-place)."""
function fill_!(arr::InsightArray, value::Float64)
    ccall((:insight_jl_fill, LIB_INSIGHT), Cvoid, (Ptr{Cvoid}, Float64),
          arr, value)
    arr
end

"""Copy data from src into dst (in-place). Shapes must match."""
function copy_from_!(dst::InsightArray, src::InsightArray)
    ccall((:insight_jl_copy_from, LIB_INSIGHT), Cvoid,
          (Ptr{Cvoid}, Ptr{Cvoid}), dst, src)
    dst
end

"""
    slice(arr::InsightArray, dim::Int, start::Int, stop::Int) -> InsightArray

Create a slice view of arr along dimension dim (1-based).
start is inclusive, stop is exclusive. Returns a view (no data copy).
"""
function slice(arr::InsightArray, dim::Int, start::Int, stop::Int)::InsightArray
    ptr = ccall((:insight_jl_slice, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32, Int64, Int64),
                arr, Int32(dim - 1), Int64(start - 1), Int64(stop - 1))
    if ptr == C_NULL
        error("Insight: slice failed")
    end
    result = InsightArray(ptr)
    finalizer(_free, result)
    return result
end

# ============================================================================
# Metadata
# ============================================================================

function ndim(arr::InsightArray)::Int
    Int(ccall((:insight_jl_ndim, LIB_INSIGHT), Int32, (Ptr{Cvoid},), arr))
end

function numel(arr::InsightArray)::Int
    Int(ccall((:insight_jl_numel, LIB_INSIGHT), Int64, (Ptr{Cvoid},), arr))
end

function dtype(arr::InsightArray)::Int32
    ccall((:insight_jl_dtype, LIB_INSIGHT), Int32, (Ptr{Cvoid},), arr)
end

function shape(arr::InsightArray)::Vector{Int64}
    n = ndim(arr)
    dims = Vector{Int64}(undef, n)
    ccall((:insight_jl_shape_reversed, LIB_INSIGHT), Cvoid,
          (Ptr{Cvoid}, Ptr{Int64}, Int32), arr, dims, Int32(n))
    return dims
end

function device_type(arr::InsightArray)::Int32
    ccall((:insight_jl_device_type, LIB_INSIGHT), Int32, (Ptr{Cvoid},), arr)
end

# ============================================================================
# Array creation
# ============================================================================

"""
    zeros(dims::Vector{Int64}, dtype_val::Int32=float32, device::Int32=_current_device()) -> InsightArray

Create an array filled with zeros.

# Arguments
- `dims`: Shape of the array, e.g. `[2, 3]`.
- `dtype_val`: Data type (default `float32`).
- `device`: Device placement (default `CPU`).

# Returns
- `InsightArray`: Array of zeros with the given shape and dtype.

# Example
```julia
a = Insight.zeros([2, 3], Insight.float32)
Insight.numel(a)  # 6
```
"""
function zeros(dims::Vector{Int64}, dtype_val::Int32=float32,
               device::Int32=_current_device())::InsightArray

    ptr = ccall((:insight_jl_zeros, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Int64}, Int32, Int32, Int32),
                dims, Int32(length(dims)), dtype_val, device)
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

"""
    ones(dims::Vector{Int64}, dtype_val::Int32=float32, device::Int32=_current_device()) -> InsightArray

Create an array filled with ones.

# Arguments
- `dims`: Shape of the array, e.g. `[2, 3]`.
- `dtype_val`: Data type (default `float32`).
- `device`: Device placement (default `CPU`).

# Returns
- `InsightArray`: Array of ones with the given shape and dtype.
"""
function ones(dims::Vector{Int64}, dtype_val::Int32=float32,
              device::Int32=_current_device())::InsightArray

    ptr = ccall((:insight_jl_ones, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Int64}, Int32, Int32, Int32),
                dims, Int32(length(dims)), dtype_val, device)
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

function full(dims::Vector{Int64}, fill_value::Float64,
              dtype_val::Int32=float32, device::Int32=_current_device())::InsightArray

    ptr = ccall((:insight_jl_full, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Int64}, Int32, Float64, Int32, Int32),
                dims, Int32(length(dims)), fill_value, dtype_val, device)
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

function eye(n::Int64, m::Int64=Int64(-1), dtype_val::Int32=float32,
             device::Int32=_current_device())::InsightArray
    ptr = ccall((:insight_jl_eye, LIB_INSIGHT), Ptr{Cvoid},
                (Int64, Int64, Int32, Int32), n, m, dtype_val, device)
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

function arange(start::Float64, stop::Float64, step::Float64=1.0,
                dtype_val::Int32=int64, device::Int32=_current_device())::InsightArray
    ptr = ccall((:insight_jl_arange, LIB_INSIGHT), Ptr{Cvoid},
                (Float64, Float64, Float64, Int32, Int32),
                start, stop, step, dtype_val, device)
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

function linspace(start::Float64, stop::Float64, num::Int64,
                  dtype_val::Int32=float32, device::Int32=_current_device())::InsightArray
    ptr = ccall((:insight_jl_linspace, LIB_INSIGHT), Ptr{Cvoid},
                (Float64, Float64, Int64, Int32, Int32),
                start, stop, num, dtype_val, device)
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

# Auto-detect Insight dtype from Julia element type
_auto_dtype(::Type{Float64}) = DTypeValues.F64
_auto_dtype(::Type{Float32}) = DTypeValues.F32
_auto_dtype(::Type{Int64}) = DTypeValues.I64
_auto_dtype(::Type{Int32}) = DTypeValues.I32
_auto_dtype(::Type{Int16}) = DTypeValues.I16
_auto_dtype(::Type{Int8}) = DTypeValues.I8
_auto_dtype(::Type{UInt64}) = DTypeValues.U64
_auto_dtype(::Type{UInt32}) = DTypeValues.U32
_auto_dtype(::Type{UInt16}) = DTypeValues.U16
_auto_dtype(::Type{UInt8}) = DTypeValues.U8
_auto_dtype(::Type{Bool}) = DTypeValues.BOOL
_auto_dtype(::Type{Complex{Float32}}) = DTypeValues.C32
_auto_dtype(::Type{Complex{Float64}}) = DTypeValues.C64
_auto_dtype(::Type) = DTypeValues.F32  # fallback

function from_data(data::AbstractArray{T}, dtype_val::Int32=Int32(-1),
                   device::Int32=_current_device()) where T
    # Auto-detect dtype from Julia element type when not explicitly specified
    actual_dtype = dtype_val == Int32(-1) ? _auto_dtype(T) : dtype_val

    # Convert data to match the requested dtype before passing to C
    _julia_type(dt::Int32) = dt == DTypeValues.F32 ? Float32 :
                             dt == DTypeValues.F64 ? Float64 :
                             dt == DTypeValues.I32 ? Int32 :
                             dt == DTypeValues.I64 ? Int64 :
                             dt == DTypeValues.I8  ? Int8  :
                             dt == DTypeValues.U8  ? UInt8 :
                             dt == DTypeValues.I16 ? Int16 :
                             dt == DTypeValues.U16 ? UInt16 :
                             dt == DTypeValues.U32 ? UInt32 :
                             dt == DTypeValues.U64 ? UInt64 :
                             dt == DTypeValues.BOOL ? Bool :
                             dt == DTypeValues.C32 ? Complex{Float32} :
                             dt == DTypeValues.C64 ? Complex{Float64} : Float32
    target_type = _julia_type(actual_dtype)
    converted = T === target_type ? data : convert(Array{target_type}, data)
    dims = collect(Int64, size(converted))
    ptr = ccall((:insight_jl_from_data, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Ptr{Int64}, Int32, Int32, Int32),
                pointer(converted), dims, Int32(length(dims)), actual_dtype, device)
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

function to_array(arr::InsightArray)::Array
    n = numel(arr)
    dt = dtype(arr)
    if dt == DTypeValues.F32
        dst = Vector{Float32}(undef, n)
    elseif dt == DTypeValues.F64
        dst = Vector{Float64}(undef, n)
    elseif dt == DTypeValues.I32
        dst = Vector{Int32}(undef, n)
    elseif dt == DTypeValues.I64
        dst = Vector{Int64}(undef, n)
    elseif dt == DTypeValues.I8
        dst = Vector{Int8}(undef, n)
    elseif dt == DTypeValues.U8
        dst = Vector{UInt8}(undef, n)
    elseif dt == DTypeValues.I16
        dst = Vector{Int16}(undef, n)
    elseif dt == DTypeValues.U16
        dst = Vector{UInt16}(undef, n)
    elseif dt == DTypeValues.U32
        dst = Vector{UInt32}(undef, n)
    elseif dt == DTypeValues.U64
        dst = Vector{UInt64}(undef, n)
    elseif dt == DTypeValues.BOOL
        dst = Vector{Bool}(undef, n)
    elseif dt == DTypeValues.C32
        dst = Vector{Complex{Float32}}(undef, n)
    elseif dt == DTypeValues.C64
        dst = Vector{Complex{Float64}}(undef, n)
    else
        dst = Vector{Float64}(undef, n)
    end
    ccall((:insight_jl_to_data, LIB_INSIGHT), Cvoid,
          (Ptr{Cvoid}, Ptr{Cvoid}), arr, dst)
    return Base.reshape(dst, Tuple(shape(arr)))
end

# ============================================================================
# Arithmetic operators
# ============================================================================

macro jl_binary(name, cfunc)
    quote
        function $(esc(name))(a::InsightArray, b::InsightArray)::InsightArray
            ptr = ccall(($cfunc, LIB_INSIGHT), Ptr{Cvoid},
                        (Ptr{Cvoid}, Ptr{Cvoid}), a, b)
            arr = InsightArray(ptr)
            finalizer(_free, arr)
            return arr
        end
    end
end

macro jl_unary(name, cfunc)
    quote
        function $(esc(name))(x::InsightArray)::InsightArray
            ptr = ccall(($cfunc, LIB_INSIGHT), Ptr{Cvoid},
                        (Ptr{Cvoid},), x)
            arr = InsightArray(ptr)
            finalizer(_free, arr)
            return arr
        end
    end
end

@jl_binary add  :insight_jl_add
@jl_binary sub  :insight_jl_sub
@jl_binary mul  :insight_jl_mul
@jl_binary div  :insight_jl_div
@jl_binary pow  :insight_jl_pow

# Operator overloading
Base.:+(a::InsightArray, b::InsightArray) = add(a, b)
Base.:-(a::InsightArray, b::InsightArray) = sub(a, b)
Base.:*(a::InsightArray, b::InsightArray) = mul(a, b)
Base.:/(a::InsightArray, b::InsightArray) = div(a, b)
# Scalar promotion: Array * Number and Number * Array
Base.:*(a::InsightArray, b::Number) = mul(a, from_data([b], float64))
Base.:*(a::Number, b::InsightArray) = mul(from_data([a], float64), b)
Base.:+(a::InsightArray, b::Number) = add(a, from_data([b], float64))
Base.:+(a::Number, b::InsightArray) = add(from_data([a], float64), b)
Base.:-(a::InsightArray, b::Number) = sub(a, from_data([b], float64))
Base.:-(a::Number, b::InsightArray) = sub(from_data([a], float64), b)
Base.:/(a::InsightArray, b::Number) = div(a, from_data([b], float64))
Base.:/(a::Number, b::InsightArray) = div(from_data([a], float64), b)
Base.:-(x::InsightArray) = begin
    ptr = ccall((:insight_jl_negative, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid},), x)
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

# ============================================================================
# Unary math
# ============================================================================

@jl_unary abs_fn  :insight_jl_abs
@jl_unary sqrt_fn :insight_jl_sqrt
@jl_unary exp_fn  :insight_jl_exp
@jl_unary log_fn  :insight_jl_log
@jl_unary sin_fn  :insight_jl_sin
@jl_unary cos_fn  :insight_jl_cos
@jl_unary tan_fn  :insight_jl_tan
@jl_unary floor_fn :insight_jl_floor
@jl_unary ceil_fn  :insight_jl_ceil
@jl_unary round_fn :insight_jl_round

# ============================================================================
# Additional Unary (Phase D)
# ============================================================================

@jl_unary exp2_fn      :insight_jl_exp2
@jl_unary expm1_fn     :insight_jl_expm1
@jl_unary log1p_fn     :insight_jl_log1p
@jl_unary cbrt_fn      :insight_jl_cbrt
@jl_unary reciprocal_fn :insight_jl_reciprocal
@jl_unary asinh_fn     :insight_jl_asinh
@jl_unary acosh_fn     :insight_jl_acosh
@jl_unary atanh_fn     :insight_jl_atanh
@jl_unary trunc_fn     :insight_jl_trunc_unary
@jl_unary deg2rad_fn   :insight_jl_deg2rad
@jl_unary rad2deg_fn   :insight_jl_rad2deg

# Complex unary
@jl_unary conj_fn      :insight_jl_conj
@jl_unary angle_fn     :insight_jl_angle

# ============================================================================
# Complex
# ============================================================================

function is_complex(x::InsightArray)::Bool
    ccall((:insight_jl_is_complex, LIB_INSIGHT), Int32, (Ptr{Cvoid},), x) != 0
end

function has_complex_shape(x::InsightArray)::Bool
    ccall((:insight_jl_has_complex_shape, LIB_INSIGHT), Int32,
          (Ptr{Cvoid},), x) != 0
end

function to_complex(real::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_to_complex, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid},), real)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function to_complex(real::InsightArray, imag_arr::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_to_complex2, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Ptr{Cvoid}), real, imag_arr)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function as_complex(x::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_as_complex, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid},), x)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function as_real(x::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_as_real, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid},), x)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function real_part(z::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_real_part, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid},), z)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function imag_part(z::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_imag_part, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid},), z)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

# ============================================================================
# Additional Reduction (Phase D)
# ============================================================================

# Helper: convert Julia 1-based axes to Insight row-major axes.
# With dim reversal, Julia dimension k maps to Insight axis ndim-k.
function _julia_axis(arr::InsightArray, axis::Int)
    n = ndim(arr)
    axis == 0 && throw(ArgumentError("axis 0 is invalid for Julia bindings"))
    julia_dim = axis > 0 ? axis : n + axis + 1
    1 <= julia_dim <= n || throw(ArgumentError("axis $axis out of bounds for array with $n dimensions"))
    return n - julia_dim
end

_julia_axes(arr::InsightArray, axes) = Int32[_julia_axis(arr, Int(axis)) for axis in axes]

function cummax(x::InsightArray, axis::Int)::InsightArray
    ptr = ccall((:insight_jl_cummax, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32), x, Int32(_julia_axis(x, axis)))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function cummin(x::InsightArray, axis::Int)::InsightArray
    ptr = ccall((:insight_jl_cummin, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32), x, Int32(_julia_axis(x, axis)))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function sem(x::InsightArray; axis::Union{Int,Nothing}=nothing,
             keepdims::Bool=false, ddof::Int=0)::InsightArray
    has_axis = axis !== nothing ? Int32(1) : Int32(0)
    ax = axis !== nothing ? Int32(_julia_axis(x, axis)) : Int32(0)
    kd = keepdims ? Int32(1) : Int32(0)
    ptr = ccall((:insight_jl_sem, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32, Int32, Int32, Int32),
                x, has_axis, ax, kd, Int32(ddof))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function count_nonzero(x::InsightArray; axis::Union{Int,Nothing}=nothing,
                       keepdims::Bool=false)::InsightArray
    has_axis = axis !== nothing ? Int32(1) : Int32(0)
    ax = axis !== nothing ? Int32(_julia_axis(x, axis)) : Int32(0)
    kd = keepdims ? Int32(1) : Int32(0)
    ptr = ccall((:insight_jl_count_nonzero, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32, Int32, Int32), x, has_axis, ax, kd)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function median(x::InsightArray; axis::Union{Int,Nothing}=nothing,
                keepdims::Bool=false)::InsightArray
    has_axis = axis !== nothing ? Int32(1) : Int32(0)
    ax = axis !== nothing ? Int32(_julia_axis(x, axis)) : Int32(0)
    kd = keepdims ? Int32(1) : Int32(0)
    ptr = ccall((:insight_jl_median, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32, Int32, Int32), x, has_axis, ax, kd)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function quantile(x::InsightArray, q::Float64;
                  axis::Union{Int,Nothing}=nothing,
                  keepdims::Bool=false)::InsightArray
    has_axis = axis !== nothing ? Int32(1) : Int32(0)
    ax = axis !== nothing ? Int32(_julia_axis(x, axis)) : Int32(0)
    kd = keepdims ? Int32(1) : Int32(0)
    ptr = ccall((:insight_jl_quantile, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Float64, Int32, Int32, Int32),
                x, q, has_axis, ax, kd)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function percentile(x::InsightArray, q::Float64;
                    axis::Union{Int,Nothing}=nothing,
                    keepdims::Bool=false)::InsightArray
    has_axis = axis !== nothing ? Int32(1) : Int32(0)
    ax = axis !== nothing ? Int32(_julia_axis(x, axis)) : Int32(0)
    kd = keepdims ? Int32(1) : Int32(0)
    ptr = ccall((:insight_jl_percentile, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Float64, Int32, Int32, Int32),
                x, q, has_axis, ax, kd)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function nansum(x::InsightArray; axis::Union{Int,Nothing}=nothing,
                keepdims::Bool=false)::InsightArray
    has_axis = axis !== nothing ? Int32(1) : Int32(0)
    ax = axis !== nothing ? Int32(_julia_axis(x, axis)) : Int32(0)
    kd = keepdims ? Int32(1) : Int32(0)
    ptr = ccall((:insight_jl_nansum, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32, Int32, Int32), x, has_axis, ax, kd)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function nanmean(x::InsightArray; axis::Union{Int,Nothing}=nothing,
                 keepdims::Bool=false)::InsightArray
    has_axis = axis !== nothing ? Int32(1) : Int32(0)
    ax = axis !== nothing ? Int32(_julia_axis(x, axis)) : Int32(0)
    kd = keepdims ? Int32(1) : Int32(0)
    ptr = ccall((:insight_jl_nanmean, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32, Int32, Int32), x, has_axis, ax, kd)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function nanmax(x::InsightArray; axis::Union{Int,Nothing}=nothing,
                keepdims::Bool=false)::InsightArray
    has_axis = axis !== nothing ? Int32(1) : Int32(0)
    ax = axis !== nothing ? Int32(_julia_axis(x, axis)) : Int32(0)
    kd = keepdims ? Int32(1) : Int32(0)
    ptr = ccall((:insight_jl_nanmax, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32, Int32, Int32), x, has_axis, ax, kd)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function nanmin(x::InsightArray; axis::Union{Int,Nothing}=nothing,
                keepdims::Bool=false)::InsightArray
    has_axis = axis !== nothing ? Int32(1) : Int32(0)
    ax = axis !== nothing ? Int32(_julia_axis(x, axis)) : Int32(0)
    kd = keepdims ? Int32(1) : Int32(0)
    ptr = ccall((:insight_jl_nanmin, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32, Int32, Int32), x, has_axis, ax, kd)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function nanstd(x::InsightArray; axis::Union{Int,Nothing}=nothing,
                keepdims::Bool=false, ddof::Int=0)::InsightArray
    has_axis = axis !== nothing ? Int32(1) : Int32(0)
    ax = axis !== nothing ? Int32(_julia_axis(x, axis)) : Int32(0)
    kd = keepdims ? Int32(1) : Int32(0)
    ptr = ccall((:insight_jl_nanstd, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32, Int32, Int32, Int32),
                x, has_axis, ax, kd, Int32(ddof))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function nanvar(x::InsightArray; axis::Union{Int,Nothing}=nothing,
                keepdims::Bool=false, ddof::Int=0)::InsightArray
    has_axis = axis !== nothing ? Int32(1) : Int32(0)
    ax = axis !== nothing ? Int32(_julia_axis(x, axis)) : Int32(0)
    kd = keepdims ? Int32(1) : Int32(0)
    ptr = ccall((:insight_jl_nanvar, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32, Int32, Int32, Int32),
                x, has_axis, ax, kd, Int32(ddof))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

# ============================================================================
# Reduction
# ============================================================================

"""
    sum(x::InsightArray; axis::Union{Int,Nothing}=nothing, keepdims::Bool=false) -> InsightArray

Sum of array elements over a given axis.

# Arguments
- `x`: Input array.
- `axis`: Axis along which to sum (default: sum over all elements).
- `keepdims`: If `true`, retains reduced axes with size 1.

# Returns
- `InsightArray`: Sum result.
"""
function sum(x::InsightArray; axis::Union{Int,Nothing}=nothing,
             keepdims::Bool=false)::InsightArray
    has_axis = axis !== nothing ? Int32(1) : Int32(0)
    ax = axis !== nothing ? Int32(_julia_axis(x, axis)) : Int32(0)
    kd = keepdims ? Int32(1) : Int32(0)
    ptr = ccall((:insight_jl_sum, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32, Int32, Int32), x, has_axis, ax, kd)
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

"""
    mean(x::InsightArray; axis::Union{Int,Nothing}=nothing, keepdims::Bool=false) -> InsightArray

Mean of array elements over a given axis.

# Arguments
- `x`: Input array.
- `axis`: Axis along which to compute mean (default: all elements).
- `keepdims`: If `true`, retains reduced axes with size 1.

# Returns
- `InsightArray`: Mean result.
"""
function mean(x::InsightArray; axis::Union{Int,Nothing}=nothing,
              keepdims::Bool=false)::InsightArray
    has_axis = axis !== nothing ? Int32(1) : Int32(0)
    ax = axis !== nothing ? Int32(_julia_axis(x, axis)) : Int32(0)
    kd = keepdims ? Int32(1) : Int32(0)
    ptr = ccall((:insight_jl_mean, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32, Int32, Int32), x, has_axis, ax, kd)
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

# ============================================================================
# Linear Algebra
# ============================================================================

"""
    matmul(a::InsightArray, b::InsightArray) -> InsightArray

Matrix multiplication of two 2-D arrays.

# Arguments
- `a`: Left matrix (M x K).
- `b`: Right matrix (K x N).

# Returns
- `InsightArray`: Result matrix (M x N).
"""
function matmul(a::InsightArray, b::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_matmul, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Ptr{Cvoid}), a, b)
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

function det(x::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_det, LIB_INSIGHT), Ptr{Cvoid}, (Ptr{Cvoid},), x)
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

function inv(x::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_inv, LIB_INSIGHT), Ptr{Cvoid}, (Ptr{Cvoid},), x)
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

function solve(a::InsightArray, b::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_solve, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Ptr{Cvoid}), a, b)
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

function svd(x::InsightArray)
    u_ref = Ref{Ptr{Cvoid}}(C_NULL)
    s_ref = Ref{Ptr{Cvoid}}(C_NULL)
    vt_ref = Ref{Ptr{Cvoid}}(C_NULL)
    ccall((:insight_jl_svd, LIB_INSIGHT), Cvoid,
          (Ptr{Cvoid}, Ptr{Ptr{Cvoid}}, Ptr{Ptr{Cvoid}}, Ptr{Ptr{Cvoid}}),
          x, u_ref, s_ref, vt_ref)
    u = InsightArray(u_ref[]); finalizer(_free, u)
    s = InsightArray(s_ref[]); finalizer(_free, s)
    vt = InsightArray(vt_ref[]); finalizer(_free, vt)
    return (u, s, vt)
end

function cholesky(x::InsightArray; lower::Bool=true)::InsightArray
    ptr = ccall((:insight_jl_cholesky, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32), x, lower ? Int32(1) : Int32(0))
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

# ============================================================================
# FFT
# ============================================================================

"""
    fft(x::InsightArray; n::Union{Int,Nothing}=nothing) -> InsightArray

1-D discrete Fourier transform.

# Arguments
- `x`: Input array (real or complex).
- `n`: Length of the transform (default: same as input length).

# Returns
- `InsightArray`: Complex-valued DFT result.
"""
function fft(x::InsightArray; n::Union{Int,Nothing}=nothing, axis::Union{Int,Nothing}=nothing)::InsightArray
    if axis !== nothing
        nv = n !== nothing ? Int64(n) : Int64(-1)
        ptr = ccall((:insight_jl_fft_axis, LIB_INSIGHT), Ptr{Cvoid},
                    (Ptr{Cvoid}, Int64, Int32), x, nv, Int32(_julia_axis(x, axis)))
    else
        has_n = n !== nothing ? Int32(1) : Int32(0)
        nv = n !== nothing ? Int64(n) : Int64(-1)
        ptr = ccall((:insight_jl_fft, LIB_INSIGHT), Ptr{Cvoid},
                    (Ptr{Cvoid}, Int32, Int64), x, has_n, nv)
    end
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

function ifft(x::InsightArray; n::Union{Int,Nothing}=nothing)::InsightArray
    has_n = n !== nothing ? Int32(1) : Int32(0)
    nv = n !== nothing ? Int64(n) : Int64(-1)
    ptr = ccall((:insight_jl_ifft, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32, Int64), x, has_n, nv)
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

"""
    rfft(x; n=nothing)

Real-input FFT (returns complex output of length n÷2+1).
"""
function rfft(x::InsightArray; n::Union{Int,Nothing}=nothing)::InsightArray
    has_n = n !== nothing ? Int32(1) : Int32(0)
    nv = n !== nothing ? Int64(n) : Int64(-1)
    ptr = ccall((:insight_jl_rfft, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32, Int64), x, has_n, nv)
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

"""
    irfft(x, n)

Inverse real FFT (complex input, real output of length n).
"""
function irfft(x::InsightArray, n::Int)::InsightArray
    ptr = ccall((:insight_jl_irfft, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int64), x, Int64(n))
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

# ============================================================================
# Random
# ============================================================================

function rand(dims::Vector{Int64}, dtype_val::Int32=float32,
              device::Int32=_current_device())::InsightArray
    ptr = ccall((:insight_jl_rand, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Int64}, Int32, Int32, Int32),
                dims, Int32(length(dims)), dtype_val, device)
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

function randn(dims::Vector{Int64}, dtype_val::Int32=float32,
               device::Int32=_current_device())::InsightArray

    ptr = ccall((:insight_jl_randn, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Int64}, Int32, Int32, Int32),
                dims, Int32(length(dims)), dtype_val, device)
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

# ============================================================================
# Cast
# ============================================================================

function cast(x::InsightArray, dtype_val::Int32)::InsightArray
    ptr = ccall((:insight_jl_cast, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32), x, dtype_val)
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

# ============================================================================
# Manipulation
# ============================================================================

function reshape(x::InsightArray, dims::Vector{Int64})::InsightArray
    ptr = ccall((:insight_jl_reshape, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Ptr{Int64}, Int32),
                x, dims, Int32(length(dims)))
    if ptr == C_NULL
        error("reshape failed: invalid dimensions")
    end
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

function transpose(x::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_transpose, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid},), x)
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

function Base.copy(x::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_copy, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid},), x)
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

"""
    squeeze(x::InsightArray) -> InsightArray

Remove dimensions of size 1 from the array shape.
"""
function squeeze(x::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_squeeze, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid},), x)
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

# ============================================================================
# Additional Manipulation (Phase D)
# ============================================================================

function permute(x::InsightArray, axes::Vector{Int32})::InsightArray
    insight_axes = _julia_axes(x, axes)
    ptr = ccall((:insight_jl_permute, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Ptr{Int32}, Int32),
                x, insight_axes, Int32(length(insight_axes)))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function swapaxes(x::InsightArray, axis1::Int, axis2::Int)::InsightArray
    ptr = ccall((:insight_jl_swapaxes, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32, Int32), x,
                Int32(_julia_axis(x, axis1)), Int32(_julia_axis(x, axis2)))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function moveaxis(x::InsightArray, source::Int, destination::Int)::InsightArray
    ptr = ccall((:insight_jl_moveaxis, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32, Int32), x,
                Int32(_julia_axis(x, source)),
                Int32(_julia_axis(x, destination)))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function fliplr(x::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_fliplr, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid},), x)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function flipud(x::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_flipud, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid},), x)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function rot90(x::InsightArray; k::Int=1,
               axes::Vector{Int32}=Int32[1, 2])::InsightArray
    insight_axes = _julia_axes(x, axes)
    ptr = ccall((:insight_jl_rot90, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32, Ptr{Int32}, Int32),
                x, Int32(k), insight_axes, Int32(length(insight_axes)))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function diag_fn(x::InsightArray; k::Int=0)::InsightArray
    ptr = ccall((:insight_jl_diag, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32), x, Int32(k))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function diagonal(x::InsightArray; offset::Int=0, axis1::Int=1,
                  axis2::Int=2)::InsightArray
    ptr = ccall((:insight_jl_diagonal, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32, Int32, Int32),
                x, Int32(offset), Int32(_julia_axis(x, axis1)),
                Int32(_julia_axis(x, axis2)))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function tril(x::InsightArray; k::Int=0)::InsightArray
    ptr = ccall((:insight_jl_tril, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32), x, Int32(k))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function triu(x::InsightArray; k::Int=0)::InsightArray
    ptr = ccall((:insight_jl_triu, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32), x, Int32(k))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function diff_fn(x::InsightArray; n::Int=1, axis::Int=-1)::InsightArray
    ptr = ccall((:insight_jl_diff, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32, Int32), x, Int32(n), Int32(_julia_axis(x, axis)))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

# ============================================================================
# Additional Indexing (Phase D)
# ============================================================================

function unique_ins(x::InsightArray; return_indices::Bool=false,
                    return_inverse::Bool=false,
                    return_counts::Bool=false)
    u_ref = Ref{Ptr{Cvoid}}(C_NULL)
    idx_ref = Ref{Ptr{Cvoid}}(C_NULL)
    inv_ref = Ref{Ptr{Cvoid}}(C_NULL)
    cnt_ref = Ref{Ptr{Cvoid}}(C_NULL)
    ccall((:insight_jl_unique, LIB_INSIGHT), Cvoid,
          (Ptr{Cvoid}, Int32, Int32, Int32,
           Ptr{Ptr{Cvoid}}, Ptr{Ptr{Cvoid}}, Ptr{Ptr{Cvoid}},
           Ptr{Ptr{Cvoid}}),
          x, return_indices ? Int32(1) : Int32(0),
          return_inverse ? Int32(1) : Int32(0),
          return_counts ? Int32(1) : Int32(0),
          u_ref, idx_ref, inv_ref, cnt_ref)
    u = InsightArray(u_ref[]); finalizer(_free, u)
    idx = InsightArray(idx_ref[]); finalizer(_free, idx)
    inv = InsightArray(inv_ref[]); finalizer(_free, inv)
    cnt = InsightArray(cnt_ref[]); finalizer(_free, cnt)
    return (unique=u, indices=idx, inverse=inv, counts=cnt)
end

function topk(x::InsightArray, k::Int; axis::Int=-1, largest::Bool=true,
              sorted::Bool=true)
    v_ref = Ref{Ptr{Cvoid}}(C_NULL)
    i_ref = Ref{Ptr{Cvoid}}(C_NULL)
    ccall((:insight_jl_topk, LIB_INSIGHT), Cvoid,
          (Ptr{Cvoid}, Int64, Int32, Int32, Int32,
           Ptr{Ptr{Cvoid}}, Ptr{Ptr{Cvoid}}),
          x, Int64(k), Int32(_julia_axis(x, axis)),
          largest ? Int32(1) : Int32(0),
          sorted ? Int32(1) : Int32(0), v_ref, i_ref)
    vals = InsightArray(v_ref[]); finalizer(_free, vals)
    idx = InsightArray(i_ref[]); finalizer(_free, idx)
    return (vals, idx)
end

function gather(x::InsightArray, dim::Int, index::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_gather, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32, Ptr{Cvoid}), x,
                Int32(_julia_axis(x, dim)), index)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function scatter(x::InsightArray, dim::Int, index::InsightArray,
                 src::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_scatter, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32, Ptr{Cvoid}, Ptr{Cvoid}),
                x, Int32(_julia_axis(x, dim)), index, src)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function scatter_add(x::InsightArray, dim::Int, index::InsightArray,
                     src::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_scatter_add, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32, Ptr{Cvoid}, Ptr{Cvoid}),
                x, Int32(_julia_axis(x, dim)), index, src)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function scatter_reduce(x::InsightArray, dim::Int, index::InsightArray,
                        src::InsightArray;
                        reduce::String="replace")::InsightArray
    ptr = ccall((:insight_jl_scatter_reduce, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32, Ptr{Cvoid}, Ptr{Cvoid}, Cstring),
                x, Int32(_julia_axis(x, dim)), index, src, reduce)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function interp(x::InsightArray, xp::InsightArray, fp::InsightArray;
                left::Union{Float64,Nothing}=nothing,
                right::Union{Float64,Nothing}=nothing)::InsightArray
    has_l = left !== nothing ? Int32(1) : Int32(0)
    lv = left !== nothing ? left : 0.0
    has_r = right !== nothing ? Int32(1) : Int32(0)
    rv = right !== nothing ? right : 0.0
    ptr = ccall((:insight_jl_interp, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid},
                 Int32, Float64, Int32, Float64),
                x, xp, fp, has_l, lv, has_r, rv)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function indices_fn(dims::Vector{Int64}; sparse::Bool=false)::InsightArray
    ptr = ccall((:insight_jl_indices, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Int64}, Int32, Int32),
                dims, Int32(length(dims)), sparse ? Int32(1) : Int32(0))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function ix_fn(arrays::Vector{InsightArray})
    n = length(arrays)
    ptrs = [a.ptr for a in arrays]
    out_ptrs = Vector{Ptr{Cvoid}}(undef, n)
    out_count = Ref{Int32}(0)
    ccall((:insight_jl_ix_, LIB_INSIGHT), Cvoid,
          (Ptr{Ptr{Cvoid}}, Int32, Ptr{Ptr{Cvoid}}, Ptr{Int32}),
          ptrs, Int32(n), out_ptrs, out_count)
    result = InsightArray[]
    for i in 1:out_count[]
        arr = InsightArray(out_ptrs[i])
        finalizer(_free, arr)
        push!(result, arr)
    end
    return result
end

# ============================================================================
# Additional Random (Phase D)
# ============================================================================

function seed(s::UInt64)
    ccall((:insight_jl_seed, LIB_INSIGHT), Cvoid, (UInt64,), s)
end
function seed(s::Integer)
    ccall((:insight_jl_seed, LIB_INSIGHT), Cvoid, (UInt64,), UInt64(s))
end

function get_seed()::UInt64
    ccall((:insight_jl_get_seed, LIB_INSIGHT), UInt64, ())
end

function rand_like(x::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_rand_like, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid},), x)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function randn_like(x::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_randn_like, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid},), x)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

"""
    item_flat(arr::InsightArray, idx::Integer) -> Float64

Get a single element by flat index (0-based). Transfers to CPU if needed.
"""
function item_flat(arr::InsightArray, idx::Integer)::Float64
    return ccall((:insight_jl_item_flat, LIB_INSIGHT), Float64,
                 (Ptr{Cvoid}, Int64), arr, Int64(idx))
end

"""
    nonzero(x::InsightArray) -> InsightArray

Return indices of non-zero elements. Returns [ndim, n_nonzero] array.
"""
function nonzero(x::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_nonzero, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid},), x)
    if ptr == C_NULL
        error("Insight: nonzero failed")
    end
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function exponential(scale::Float64, dims::Vector{Int64};
                     dtype_val::Int32=float32,
                     device::Int32=_current_device())::InsightArray
    ptr = ccall((:insight_jl_exponential, LIB_INSIGHT), Ptr{Cvoid},
                (Float64, Ptr{Int64}, Int32, Int32, Int32),
                scale, dims, Int32(length(dims)), dtype_val, device)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function gamma_dist(shape_param::Float64, rate::Float64,
                    dims::Vector{Int64}; dtype_val::Int32=float32,
                    device::Int32=_current_device())::InsightArray
    ptr = ccall((:insight_jl_gamma, LIB_INSIGHT), Ptr{Cvoid},
                (Float64, Float64, Ptr{Int64}, Int32, Int32, Int32),
                shape_param, rate, dims, Int32(length(dims)),
                dtype_val, device)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function beta_dist(a::Float64, b::Float64, dims::Vector{Int64};
                   dtype_val::Int32=float32,
                   device::Int32=_current_device())::InsightArray
    ptr = ccall((:insight_jl_beta_dist, LIB_INSIGHT), Ptr{Cvoid},
                (Float64, Float64, Ptr{Int64}, Int32, Int32, Int32),
                a, b, dims, Int32(length(dims)), dtype_val, device)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function binomial_dist(n::Int64, p::Float64, dims::Vector{Int64};
                       dtype_val::Int32=int64,
                       device::Int32=_current_device())::InsightArray
    ptr = ccall((:insight_jl_binomial, LIB_INSIGHT), Ptr{Cvoid},
                (Int64, Float64, Ptr{Int64}, Int32, Int32, Int32),
                n, p, dims, Int32(length(dims)), dtype_val, device)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function poisson_dist(lam::Float64, dims::Vector{Int64};
                      dtype_val::Int32=int64,
                      device::Int32=_current_device())::InsightArray
    ptr = ccall((:insight_jl_poisson, LIB_INSIGHT), Ptr{Cvoid},
                (Float64, Ptr{Int64}, Int32, Int32, Int32),
                lam, dims, Int32(length(dims)), dtype_val, device)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

# ============================================================================
# Additional FFT (Phase D)
# ============================================================================

function fftshift(x::InsightArray; axis::Int=-1)::InsightArray
    ptr = ccall((:insight_jl_fftshift, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32), x, Int32(_julia_axis(x, axis)))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function ifftshift(x::InsightArray; axis::Int=-1)::InsightArray
    ptr = ccall((:insight_jl_ifftshift, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32), x, Int32(_julia_axis(x, axis)))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

"""
    fftfreq(n::Int, d::Float64=1.0) -> InsightArray

Return the DFT sample frequencies for a signal of length `n` with sample spacing `d`.
"""
function fftfreq(n::Int, d::Float64=1.0)::InsightArray
    ptr = ccall((:insight_jl_fftfreq, LIB_INSIGHT), Ptr{Cvoid},
                (Int64, Float64), Int64(n), d)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function next_fast_len(target::Int)::Int
    Int(ccall((:insight_jl_next_fast_len, LIB_INSIGHT), Int32,
              (Int32,), Int32(target)))
end

function hfft(x::InsightArray; n::Union{Int,Nothing}=nothing)::InsightArray
    has_n = n !== nothing ? Int32(1) : Int32(0)
    nv = n !== nothing ? Int64(n) : Int64(-1)
    ptr = ccall((:insight_jl_hfft, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32, Int64), x, has_n, nv)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function ihfft(x::InsightArray; n::Union{Int,Nothing}=nothing)::InsightArray
    has_n = n !== nothing ? Int32(1) : Int32(0)
    nv = n !== nothing ? Int64(n) : Int64(-1)
    ptr = ccall((:insight_jl_ihfft, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32, Int64), x, has_n, nv)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function rfft2(x::InsightArray; s::Vector{Int64}=Int64[],
               axes::Vector{Int32}=Int32[-2, -1])::InsightArray
    insight_axes = _julia_axes(x, axes)
    ptr = ccall((:insight_jl_rfft2, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Ptr{Int64}, Int32, Ptr{Int32}, Int32),
                x, s, Int32(length(s)), insight_axes,
                Int32(length(insight_axes)))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function irfft2(x::InsightArray; s::Vector{Int64}=Int64[],
                axes::Vector{Int32}=Int32[-2, -1])::InsightArray
    insight_axes = _julia_axes(x, axes)
    ptr = ccall((:insight_jl_irfft2, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Ptr{Int64}, Int32, Ptr{Int32}, Int32),
                x, s, Int32(length(s)), insight_axes,
                Int32(length(insight_axes)))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function rfftn(x::InsightArray; s::Vector{Int64}=Int64[],
               axes::Vector{Int32}=Int32[])::InsightArray
    insight_axes = _julia_axes(x, axes)
    ptr = ccall((:insight_jl_rfftn, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Ptr{Int64}, Int32, Ptr{Int32}, Int32),
                x, s, Int32(length(s)), insight_axes,
                Int32(length(insight_axes)))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function irfftn(x::InsightArray; s::Vector{Int64}=Int64[],
                axes::Vector{Int32}=Int32[])::InsightArray
    insight_axes = _julia_axes(x, axes)
    ptr = ccall((:insight_jl_irfftn, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Ptr{Int64}, Int32, Ptr{Int32}, Int32),
                x, s, Int32(length(s)), insight_axes,
                Int32(length(insight_axes)))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

# ============================================================================
# Additional Linalg (Phase D)
# ============================================================================

function lstsq(a::InsightArray, b::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_lstsq, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Ptr{Cvoid}), a, b)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function cond_fn(x::InsightArray; p::Float64=2.0)::InsightArray
    ptr = ccall((:insight_jl_cond, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Float64), x, p)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function matrix_rank(x::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_matrix_rank, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid},), x)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function matrix_power(x::InsightArray, n::Int)::InsightArray
    ptr = ccall((:insight_jl_matrix_power, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32), x, Int32(n))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function slogdet(x::InsightArray)
    sign_ref = Ref{Ptr{Cvoid}}(C_NULL)
    logdet_ref = Ref{Ptr{Cvoid}}(C_NULL)
    ccall((:insight_jl_slogdet, LIB_INSIGHT), Cvoid,
          (Ptr{Cvoid}, Ptr{Ptr{Cvoid}}, Ptr{Ptr{Cvoid}}),
          x, sign_ref, logdet_ref)
    sign = InsightArray(sign_ref[]); finalizer(_free, sign)
    logdet = InsightArray(logdet_ref[]); finalizer(_free, logdet)
    return (sign, logdet)
end

function eigvalsh(x::InsightArray; uplo::String="L")::InsightArray
    ptr = ccall((:insight_jl_eigvalsh, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Cstring), x, uplo)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function pinv(x::InsightArray; rcond::Float64=-1.0)::InsightArray
    ptr = ccall((:insight_jl_pinv, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Float64), x, rcond)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

# ============================================================================
# Convenience functions (aliases and missing bindings)
# ============================================================================

"""
    item(arr::InsightArray, idx::Int) -> Number

Extract a scalar value from a 1-D array at 0-based index `idx`.
"""
function item(arr::InsightArray, idx::Int)
    data = to_array(arr)
    return data[idx + 1]
end

"""
    Base.getindex(arr::InsightArray, indices::Int...) -> InsightArray

Index an InsightArray with integer indices (1-based, Julia convention).
When fewer indices than dimensions are given, remaining dimensions are
kept as full slices (NumPy-style partial indexing).
"""
function Base.getindex(arr::InsightArray, indices::Int...)
    c_indices = [Int64(i - 1) for i in indices]
    n = Int32(length(c_indices))
    ptr = ccall((:insight_jl_at_index, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Ptr{Int64}, Int32), arr, c_indices, n)
    if ptr == C_NULL
        error("Insight: indexing failed (out of range or invalid indices)")
    end
    result = InsightArray(ptr)
    finalizer(_free, result)
    return result
end

"""
    to_data(arr::InsightArray) -> Array

Alias for `to_array`. Copies data from an InsightArray to a Julia Array.
"""
const to_data = to_array

"""
    Base.collect(arr::InsightArray) -> Vector

Extract all elements from an InsightArray as a flat Julia Vector.
Internally calls `to_array` and flattens with `vec()`.
"""
function Base.collect(arr::InsightArray)
    return vec(to_array(arr))
end

"""
    to(x::InsightArray, device_type::Int) -> InsightArray

Transfer array to a different device. device_type: 0=CPU, 1=GPU.
"""
function to(x::InsightArray, device_type::Int)::InsightArray
    ptr = ccall((:insight_jl_to_device, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32), x, Int32(device_type))
    if ptr == C_NULL
        error("Insight: device transfer failed (GPU backend not available?)")
    end
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

"""
    negative(x::InsightArray) -> InsightArray

Element-wise negation.
"""
function negative(x::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_negative, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid},), x)
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

"""
    abs(x::InsightArray) -> InsightArray

Element-wise absolute value. Alias for `abs_fn`.
"""
abs(x::InsightArray) = abs_fn(x)

"""
    sqrt(x::InsightArray) -> InsightArray

Element-wise square root. Alias for `sqrt_fn`.
"""
sqrt(x::InsightArray) = sqrt_fn(x)

"""
    exp(x::InsightArray) -> InsightArray

Element-wise exponential. Alias for `exp_fn`.
"""
exp(x::InsightArray) = exp_fn(x)

"""
    log(x::InsightArray) -> InsightArray

Element-wise natural logarithm. Alias for `log_fn`.
"""
log(x::InsightArray) = log_fn(x)

"""
    sin(x::InsightArray) -> InsightArray

Element-wise sine. Alias for `sin_fn`.
"""
sin(x::InsightArray) = sin_fn(x)

"""
    cos(x::InsightArray) -> InsightArray

Element-wise cosine. Alias for `cos_fn`.
"""
cos(x::InsightArray) = cos_fn(x)

"""
    tan(x::InsightArray) -> InsightArray

Element-wise tangent. Alias for `tan_fn`.
"""
tan(x::InsightArray) = tan_fn(x)

"""
    floor(x::InsightArray) -> InsightArray

Element-wise floor. Alias for `floor_fn`.
"""
floor(x::InsightArray) = floor_fn(x)

"""
    ceil(x::InsightArray) -> InsightArray

Element-wise ceil. Alias for `ceil_fn`.
"""
ceil(x::InsightArray) = ceil_fn(x)

"""
    round(x::InsightArray) -> InsightArray

Element-wise round. Alias for `round_fn`.
"""
round(x::InsightArray) = round_fn(x)

# --- Comparison operators ---
@jl_binary equal  :insight_jl_equal
@jl_binary greater :insight_jl_greater
@jl_binary less   :insight_jl_less

# Logical binary ops
@jl_binary logical_and :insight_jl_logical_and
@jl_binary logical_or  :insight_jl_logical_or
@jl_binary logical_xor :insight_jl_logical_xor

# Logical unary ops
@jl_unary logical_not :insight_jl_logical_not

# --- Additional reductions ---
function max(x::InsightArray; axis::Union{Int,Nothing}=nothing,
             keepdims::Bool=false)::InsightArray
    has_axis = axis !== nothing ? Int32(1) : Int32(0)
    ax = axis !== nothing ? Int32(_julia_axis(x, axis)) : Int32(0)
    kd = keepdims ? Int32(1) : Int32(0)
    ptr = ccall((:insight_jl_max, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32, Int32, Int32), x, has_axis, ax, kd)
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

function min(x::InsightArray; axis::Union{Int,Nothing}=nothing,
             keepdims::Bool=false)::InsightArray
    has_axis = axis !== nothing ? Int32(1) : Int32(0)
    ax = axis !== nothing ? Int32(_julia_axis(x, axis)) : Int32(0)
    kd = keepdims ? Int32(1) : Int32(0)
    ptr = ccall((:insight_jl_min, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32, Int32, Int32), x, has_axis, ax, kd)
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

function argmax(x::InsightArray; axis::Union{Int,Nothing}=nothing,
                keepdims::Bool=false)::InsightArray
    has_axis = axis !== nothing ? Int32(1) : Int32(0)
    ax = axis !== nothing ? Int32(_julia_axis(x, axis)) : Int32(0)
    kd = keepdims ? Int32(1) : Int32(0)
    ptr = ccall((:insight_jl_argmax, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32, Int32, Int32), x, has_axis, ax, kd)
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

function argmin(x::InsightArray; axis::Union{Int,Nothing}=nothing,
                keepdims::Bool=false)::InsightArray
    has_axis = axis !== nothing ? Int32(1) : Int32(0)
    ax = axis !== nothing ? Int32(_julia_axis(x, axis)) : Int32(0)
    kd = keepdims ? Int32(1) : Int32(0)
    ptr = ccall((:insight_jl_argmin, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32, Int32, Int32), x, has_axis, ax, kd)
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

function prod(x::InsightArray; axis::Union{Int,Nothing}=nothing,
              keepdims::Bool=false)::InsightArray
    has_axis = axis !== nothing ? Int32(1) : Int32(0)
    ax = axis !== nothing ? Int32(_julia_axis(x, axis)) : Int32(0)
    kd = keepdims ? Int32(1) : Int32(0)
    ptr = ccall((:insight_jl_prod, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32, Int32, Int32), x, has_axis, ax, kd)
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

# --- Additional linalg ---
"""
    dot(a::InsightArray, b::InsightArray) -> InsightArray

Dot product of two 1-D arrays.
"""
function dot(a::InsightArray, b::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_dot, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Ptr{Cvoid}), a, b)
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

"""
    outer(a::InsightArray, b::InsightArray) -> InsightArray

Outer product of two 1-D arrays.
"""
function outer(a::InsightArray, b::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_outer, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Ptr{Cvoid}), a, b)
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

"""
    norm(x::InsightArray; ord::Real=2.0) -> InsightArray

Matrix or vector norm.
"""
function norm(x::InsightArray; ord::Real=2.0)::InsightArray
    ptr = ccall((:insight_jl_norm, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Float64), x, Float64(ord))
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

"""
    trace(x::InsightArray) -> InsightArray

Sum of diagonal elements.
"""
function trace(x::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_trace, LIB_INSIGHT), Ptr{Cvoid}, (Ptr{Cvoid},), x)
    arr = InsightArray(ptr)
    finalizer(_free, arr)
    return arr
end

# ============================================================================
# Display
# ============================================================================

function Base.show(io::IO, a::InsightArray)
    if a.ptr == C_NULL
        print(io, "Array(<freed>)")
        return
    end
    cstr = ccall((:insight_jl_array_tostring, LIB_INSIGHT), Ptr{UInt8},
                 (Ptr{Cvoid},), a)
    if cstr != C_NULL
        s = unsafe_string(cstr)
        # NOTE: cstr points to a static buffer inside the DLL — do NOT free it
        print(io, s)
    else
        # Fallback to metadata-only
        s = shape(a)
        d = dtype(a)
        dt_name = d == float64 ? "float64" :
                  d == float32 ? "float32" :
                  d == int32   ? "int32" :
                  d == int64   ? "int64" :
                  d == int8    ? "int8" :
                  d == uint8   ? "uint8" :
                  d == bool    ? "bool" : "dtype($d)"
        p = device_type(a)
        place_name = p == GPU ? "gpu" : "cpu"
        print(io, "Array(shape=$s, dtype=$dt_name, place=$place_name)")
    end
end

function Base.show(io::IO, ::MIME"text/plain", a::InsightArray)
    show(io, a)
end

# ============================================================================
# Plot Module (conditionally available — requires INSIGHT_USE_MATPLOT)
# ============================================================================

module plot
    using ..Insight

    # Check at load time whether plot symbols are available
    const _has_plot = try
        ccall((:insight_jl_figure, Insight.LIB_INSIGHT), Cvoid,
              (Int32,), Int32(-1))
        true
    catch
        false
    end

    function _check()
        if !_has_plot
            error("Plot functions not available. Rebuild with INSIGHT_USE_MATPLOT=ON")
        end
    end

    function plotfn(y::Insight.InsightArray; format::String="")
        _check()
        ccall((:insight_jl_plot, Insight.LIB_INSIGHT), Cvoid,
              (Ptr{Cvoid}, Cstring), y, format)
    end

    function plotxy(x::Insight.InsightArray, y::Insight.InsightArray;
                  format::String="")
        _check()
        ccall((:insight_jl_plot_xy, Insight.LIB_INSIGHT), Cvoid,
              (Ptr{Cvoid}, Ptr{Cvoid}, Cstring), x, y, format)
    end

    function scatter(x::Insight.InsightArray, y::Insight.InsightArray;
                     size::Float64=20.0)
        _check()
        ccall((:insight_jl_plot_scatter, Insight.LIB_INSIGHT), Cvoid,
              (Ptr{Cvoid}, Ptr{Cvoid}, Float64), x, y, size)
    end

    function bar(y::Insight.InsightArray; width::Float64=0.8)
        _check()
        ccall((:insight_jl_bar, Insight.LIB_INSIGHT), Cvoid,
              (Ptr{Cvoid}, Float64), y, width)
    end

    function bar(x::Insight.InsightArray, y::Insight.InsightArray;
                 width::Float64=0.8)
        _check()
        ccall((:insight_jl_bar_xy, Insight.LIB_INSIGHT), Cvoid,
              (Ptr{Cvoid}, Ptr{Cvoid}, Float64), x, y, width)
    end

    function hist(data::Insight.InsightArray; bins::Int=10)
        _check()
        ccall((:insight_jl_hist, Insight.LIB_INSIGHT), Cvoid,
              (Ptr{Cvoid}, Int32), data, Int32(bins))
    end

    function hist(data::Insight.InsightArray, bins::Int)
        _check()
        ccall((:insight_jl_hist, Insight.LIB_INSIGHT), Cvoid,
              (Ptr{Cvoid}, Int32), data, Int32(bins))
    end

    function imshow(data::Insight.InsightArray)
        _check()
        ccall((:insight_jl_imshow, Insight.LIB_INSIGHT), Cvoid,
              (Ptr{Cvoid},), data)
    end

    function contour(X::Insight.InsightArray, Y::Insight.InsightArray,
                     Z::Insight.InsightArray; levels::Int=10)
        _check()
        ccall((:insight_jl_contour, Insight.LIB_INSIGHT), Cvoid,
              (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Int32),
              X, Y, Z, Int32(levels))
    end

    function subplot(rows::Int, cols::Int, index::Int)
        _check()
        ccall((:insight_jl_subplot, Insight.LIB_INSIGHT), Cvoid,
              (Int32, Int32, Int32), Int32(rows), Int32(cols), Int32(index))
    end

    function title(text::String)
        _check()
        ccall((:insight_jl_title, Insight.LIB_INSIGHT), Cvoid,
              (Cstring,), text)
    end

    function xlabel(text::String)
        _check()
        ccall((:insight_jl_xlabel, Insight.LIB_INSIGHT), Cvoid,
              (Cstring,), text)
    end

    function ylabel(text::String)
        _check()
        ccall((:insight_jl_ylabel, Insight.LIB_INSIGHT), Cvoid,
              (Cstring,), text)
    end

    function legend(labels::Vector{String})
        _check()
        ccall((:insight_jl_legend, Insight.LIB_INSIGHT), Cvoid,
              (Ptr{Cstring}, Int32), labels, Int32(length(labels)))
    end

    function savefig(filename::String)
        _check()
        ccall((:insight_jl_savefig, Insight.LIB_INSIGHT), Cvoid,
              (Cstring,), filename)
    end

    function figure(number::Int=-1)
        _check()
        ccall((:insight_jl_figure, Insight.LIB_INSIGHT), Cvoid,
              (Int32,), Int32(number))
    end

    function clf()
        _check()
        ccall((:insight_jl_clf, Insight.LIB_INSIGHT), Cvoid, ())
    end

    function grid(on::Bool=true)
        _check()
        ccall((:insight_jl_grid, Insight.LIB_INSIGHT), Cvoid,
              (Int32,), on ? Int32(1) : Int32(0))
    end

    function close()
        _check()
        ccall((:insight_jl_close, Insight.LIB_INSIGHT), Cvoid, ())
    end
end

# ============================================================================
# Signal Processing
# ============================================================================

function convolve(a::InsightArray, v::InsightArray;
                  mode::String="full")::InsightArray
    ptr = ccall((:insight_jl_convolve, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Ptr{Cvoid}, Cstring), a, v, mode)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function unwrap(p::InsightArray; axis::Int=-1, discont::Real=Float64(π),
                period::Real=Float64(2π))::InsightArray
    ptr = ccall((:insight_jl_unwrap, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Int32, Float64, Float64), p, Int32(_julia_axis(p, axis)),
                Float64(discont), Float64(period))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

function sinc(x::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_sinc, LIB_INSIGHT), Ptr{Cvoid}, (Ptr{Cvoid},), x)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

# ============================================================================
# Module documentation (stubs with docstrings for non-signal functions)
# ============================================================================
const _modules_dir = joinpath(@__DIR__, "modules")
include(joinpath(_modules_dir, "types.jl"))
include(joinpath(_modules_dir, "cast.jl"))
include(joinpath(_modules_dir, "elementwise.jl"))
include(joinpath(_modules_dir, "unary.jl"))
include(joinpath(_modules_dir, "complex.jl"))
include(joinpath(_modules_dir, "reduction.jl"))
include(joinpath(_modules_dir, "manipulation.jl"))
include(joinpath(_modules_dir, "linalg.jl"))
include(joinpath(_modules_dir, "fft.jl"))
include(joinpath(_modules_dir, "random.jl"))
include(joinpath(_modules_dir, "indexing.jl"))

# ============================================================================
# Signal processing (split into sub-module files aligned with C++ structure)
# ============================================================================

# Top-level signal functions (convolve, unwrap, sinc)
# convolve is defined above as ins::convolve wrapper
# unwrap is defined above
# sinc is defined above

# Include signal sub-module files (each contains ccall implementations + docstrings)
const _signal_dir = joinpath(@__DIR__, "modules", "signal")
include(joinpath(_signal_dir, "windows.jl"))
include(joinpath(_signal_dir, "waveforms.jl"))
include(joinpath(_signal_dir, "bsplines.jl"))
include(joinpath(_signal_dir, "filter_design.jl"))
include(joinpath(_signal_dir, "convolution.jl"))
include(joinpath(_signal_dir, "filtering.jl"))
include(joinpath(_signal_dir, "spectral.jl"))
include(joinpath(_signal_dir, "wavelets.jl"))
include(joinpath(_signal_dir, "acoustics.jl"))
include(joinpath(_signal_dir, "demod.jl"))
include(joinpath(_signal_dir, "peak_finding.jl"))
include(joinpath(_signal_dir, "radar.jl"))
include(joinpath(_signal_dir, "estimation.jl"))
include(joinpath(_signal_dir, "io.jl"))

# --- Signal submodule (convenience namespace) ---
module signal
    using ..Insight
    # Re-export all signal functions
    const hann = Insight.hann
    const hamming = Insight.hamming
    const blackman = Insight.blackman
    const kaiser = Insight.kaiser
    const gaussian = Insight.gaussian
    const boxcar = Insight.boxcar
    const triang = Insight.triang
    const bartlett = Insight.bartlett
    const flattop = Insight.flattop
    const nuttall = Insight.nuttall
    const blackmanharris = Insight.blackmanharris
    const tukey = Insight.tukey
    const chebwin = Insight.chebwin
    const taylor = Insight.taylor
    const get_window = Insight.get_window
    const sawtooth = Insight.sawtooth
    const square = Insight.square_wf
    const chirp = Insight.chirp
    const unit_impulse = Insight.unit_impulse
    const gauss_spline = Insight.gauss_spline
    const cubic = Insight.cubic
    const quadratic = Insight.quadratic
    const kaiser_beta = Insight.kaiser_beta
    const kaiser_atten = Insight.kaiser_atten
    const firwin = Insight.firwin
    const cmplx_sort = Insight.cmplx_sort
    const fftconvolve = Insight.fftconvolve
    const correlate = Insight.correlate
    const correlation_lags = Insight.correlation_lags
    const hilbert = Insight.hilbert
    const detrend = Insight.detrend
    const lfilter = Insight.lfilter
    const filtfilt = Insight.filtfilt
    const decimate = Insight.decimate
    const resample = Insight.resample
    const freq_shift = Insight.freq_shift
    const welch = Insight.welch_jl
    const periodogram = Insight.periodogram_jl
    const morlet = Insight.morlet
    const ricker = Insight.ricker
    const mel2hz = Insight.mel2hz
    const hz2mel = Insight.hz2mel
    const mel_frequencies = Insight.mel_frequencies
    const hz2bark = Insight.hz2bark
    const bark2hz = Insight.bark2hz
    const fm_demod = Insight.fm_demod
    const argrelmax = Insight.argrelmax
    const argrelmin = Insight.argrelmin
    const cfar_alpha = Insight.cfar_alpha
    const ca_cfar = Insight.ca_cfar
    const read_bin = Insight.read_bin
    const write_bin = Insight.write_bin
    const pack_bin = Insight.pack_bin
    const unpack_bin = Insight.unpack_bin
    const convolve = Insight.convolve
    const unwrap = Insight.unwrap
    const sinc_fn = Insight.sinc
    const csd = Insight.csd
    const coherence = Insight.coherence
    const spectrogram = Insight.spectrogram
    const stft = Insight.stft
    const vectorstrength = Insight.vectorstrength
    const lombscargle = Insight.lombscargle
    const choose_conv_method = Insight.choose_conv_method
    const firfilter_zi_state = Insight.firfilter_zi_state
    const pulse_compression = Insight.pulse_compression
    const pulse_doppler = Insight.pulse_doppler
    const mvdr = Insight.mvdr
    const convolve2d = Insight.convolve2d
    const correlate2d = Insight.correlate2d
    const hilbert2 = Insight.hilbert2
    const wiener = Insight.wiener
    const firfilter = Insight.firfilter
    const lfilter_zi = Insight.lfilter_zi
    const resample_poly = Insight.resample_poly
    const morlet2 = Insight.morlet2
    const firwin2 = Insight.firwin2
    const read_sigmf = Insight.read_sigmf
    const write_sigmf = Insight.write_sigmf
    const cosine_win = Insight.cosine_win
    const general_hamming = Insight.general_hamming
    const parzen_win = Insight.parzen_win
    const bohman_win = Insight.bohman_win
    const barthann_win = Insight.barthann_win
    const exponential_win = Insight.exponential_win
    const general_gaussian_win = Insight.general_gaussian_win
    const KalmanFilter = Insight.KalmanFilter
    const kf_predict = Insight.predict
    const kf_update = Insight.update
end

"""
    Timer

High-resolution timer for measuring execution time on CPU or GPU.

# Fields
- `handle::Ptr{Cvoid}` - Opaque pointer to the native timer

# Usage
```julia
t = Timer(0, 0)  # CPU
timer_start(t)
# ... work ...
timer_stop(t)
ms = timer_elapsed_ms(t)
timer_destroy(t)
```
"""
mutable struct Timer
    handle::Ptr{Cvoid}
end

"""
    Timer(device_type::Integer, device_id::Integer) -> Timer

Create a timer for the specified device.
- `device_type`: 0 for CPU, 1 for GPU
- `device_id`: Device ID (0 for first device)
"""
function Timer(device_type::Integer, device_id::Integer)
    handle = ccall((:insight_jl_timer_create, LIB_INSIGHT), Ptr{Cvoid},
                   (Cint, Cint), Cint(device_type), Cint(device_id))
    if handle == C_NULL
        error("Timer: failed to create timer (device not available)")
    end
    return Timer(handle)
end

"""
    timer_destroy(t::Timer)

Destroy a timer and release associated resources.
"""
function timer_destroy(t::Timer)
    if t.handle != C_NULL
        ccall((:insight_jl_timer_destroy, LIB_INSIGHT), Cvoid, (Ptr{Cvoid},), t.handle)
        t.handle = C_NULL
    end
end

"""
    timer_start(t::Timer)

Record a start event on the device.
"""
function timer_start(t::Timer)
    ccall((:insight_jl_timer_start, LIB_INSIGHT), Cvoid, (Ptr{Cvoid},), t.handle)
end

"""
    timer_stop(t::Timer)

Record a stop event and synchronize.
"""
function timer_stop(t::Timer)
    ccall((:insight_jl_timer_stop, LIB_INSIGHT), Cvoid, (Ptr{Cvoid},), t.handle)
end

"""
    timer_elapsed_ms(t::Timer) -> Float64

Get the elapsed time in milliseconds between start and stop.
"""
function timer_elapsed_ms(t::Timer)
    return ccall((:insight_jl_timer_elapsed_ms, LIB_INSIGHT), Float64,
                 (Ptr{Cvoid},), t.handle)
end

# ============================================================================
# Profiler
# ============================================================================

"""
    Profiler

Multi-event aggregated profiler for recording timing statistics.
Uses the device's native profiler mechanism to collect and aggregate
timing data for multiple named events.

# Fields
- `handle::Ptr{Cvoid}` - Opaque pointer to the native profiler

# Usage
```julia
prof = Profiler(0, 0)  # CPU
profiler_start(prof)
profiler_begin_event(prof, "fft")
# ... work ...
profiler_end_event(prof)
profiler_stop(prof)
profiler_report(prof)
profiler_destroy(prof)
```
"""
mutable struct Profiler
    handle::Ptr{Cvoid}
end

"""
    Profiler(device_type::Integer, device_id::Integer) -> Profiler

Create a profiler for the specified device.
- `device_type`: 0 for CPU, 1 for GPU
- `device_id`: Device ID (0 for first device)
"""
function Profiler(device_type::Integer, device_id::Integer)
    handle = ccall((:insight_jl_profiler_create, LIB_INSIGHT), Ptr{Cvoid},
                   (Cint, Cint, Cstring), Cint(device_type), Cint(device_id), C_NULL)
    if handle == C_NULL
        error("Profiler: failed to create profiler (device not available)")
    end
    return Profiler(handle)
end

"""
    profiler_destroy(prof::Profiler)

Destroy a profiler and release associated resources.
"""
function profiler_destroy(prof::Profiler)
    if prof.handle != C_NULL
        ccall((:insight_jl_profiler_destroy, LIB_INSIGHT), Cvoid, (Ptr{Cvoid},), prof.handle)
        prof.handle = C_NULL
    end
end

"""
    profiler_start(prof::Profiler)

Start recording profiler events.
"""
function profiler_start(prof::Profiler)
    ccall((:insight_jl_profiler_start, LIB_INSIGHT), Cvoid, (Ptr{Cvoid},), prof.handle)
end

"""
    profiler_stop(prof::Profiler)

Stop recording profiler events.
"""
function profiler_stop(prof::Profiler)
    ccall((:insight_jl_profiler_stop, LIB_INSIGHT), Cvoid, (Ptr{Cvoid},), prof.handle)
end

"""
    profiler_reset(prof::Profiler)

Clear all recorded data.
"""
function profiler_reset(prof::Profiler)
    ccall((:insight_jl_profiler_reset, LIB_INSIGHT), Cvoid, (Ptr{Cvoid},), prof.handle)
end

"""
    profiler_begin_event(prof::Profiler, name::String)

Begin a named event.
"""
function profiler_begin_event(prof::Profiler, name::String)
    ccall((:insight_jl_profiler_begin_event, LIB_INSIGHT), Cvoid,
          (Ptr{Cvoid}, Cstring), prof.handle, name)
end

"""
    profiler_end_event(prof::Profiler)

End the current event.
"""
function profiler_end_event(prof::Profiler)
    ccall((:insight_jl_profiler_end_event, LIB_INSIGHT), Cvoid, (Ptr{Cvoid},), prof.handle)
end

"""
    profiler_get_events(prof::Profiler) -> Vector{Dict}

Get aggregated event statistics as a vector of dicts.
Each dict has keys: name, calls, total_ms, min_ms, max_ms.
"""
function profiler_get_events(prof::Profiler)::Vector{Dict}
    json_str = ccall((:insight_jl_profiler_get_events, LIB_INSIGHT), Cstring,
                     (Ptr{Cvoid},), prof.handle)
    result = Vector{Dict}()
    if json_str != C_NULL
        raw = unsafe_string(json_str)
        ccall((:insight_jl_profiler_free_json, LIB_INSIGHT), Cvoid, (Cstring,), json_str)
        if length(raw) > 2
            # Simple JSON array parser for profiler events
            # Format: [{"name":"...","calls":N,"total_ms":F,"min_ms":F,"max_ms":F},...]
            s = strip(raw)
            if startswith(s, '[') && endswith(s, ']')
                s = s[2:end-1]  # strip [ ]
                while !isempty(s)
                    # Find one object { ... }
                    obj_start = findfirst('{', s)
                    obj_end = findfirst('}', s)
                    if obj_start === nothing || obj_end === nothing
                        break
                    end
                    obj_str = s[obj_start:obj_end]
                    s = s[obj_end+1:end]
                    # Parse key-value pairs
                    d = Dict{String,Any}()
                    # Extract name
                    m = match(r"\"name\":\"([^\"]*)\"", obj_str)
                    if m !== nothing; d["name"] = m.captures[1]; end
                    m = match(r"\"calls\":(\d+)", obj_str)
                    if m !== nothing; d["calls"] = parse(Int, m.captures[1]); end
                    m = match(r"\"total_ms\":([0-9.eE+-]+)", obj_str)
                    if m !== nothing; d["total_ms"] = parse(Float64, m.captures[1]); end
                    m = match(r"\"min_ms\":([0-9.eE+-]+)", obj_str)
                    if m !== nothing; d["min_ms"] = parse(Float64, m.captures[1]); end
                    m = match(r"\"max_ms\":([0-9.eE+-]+)", obj_str)
                    if m !== nothing; d["max_ms"] = parse(Float64, m.captures[1]); end
                    push!(result, d)
                end
            end
        end
    end
    return result
end

"""
    profiler_report(prof::Profiler)

Print a formatted timing report to stdout.
"""
function profiler_report(prof::Profiler)
    events = profiler_get_events(prof)
    if isempty(events)
        println("  [Profiler] no events recorded")
        return
    end
    println()
    println("  Event                Calls   Total(ms)     Avg(ms)     Max(ms)")
    println("  ", "\u2500"^59)
    for ev in events
        avg = ev["total_ms"] / Float64(ev["calls"])
        println(lpad(ev["name"], 20), lpad(string(ev["calls"]), 8),
                lpad(Base.round(ev["total_ms"], digits=3), 13),
                lpad(Base.round(avg, digits=4), 11),
                lpad(Base.round(ev["max_ms"], digits=4), 11))
    end
    println()
end

end # module Insight
