"""
Insight7 — Lightweight scientific computing framework for Python.

A NumPy-compatible array library with GPU acceleration support,
inspired by PaddlePaddle, Torch7, and NumPy/CuPy.

Quick Start::

    import insight as ins

    # Create arrays (Paddle-style API)
    a = ins.zeros([2, 3], ins.float32)
    b = ins.ones([2, 3], ins.float32)
    c = a + b

    # Reduction
    s = ins.sum(c, axis=0)

    # Linear algebra
    m = ins.matmul(a, b.T)

    # Signal processing
    w = ins.signal.hann(256)
    f, Pxx = ins.signal.welch(x, fs=1000)

    # NumPy interop
    import numpy as np
    arr = a.numpy()              # Insight -> NumPy
    d = ins.from_numpy(arr)      # NumPy -> Insight

DType Shortcuts (Paddle-style)::

    ins.float32, ins.float64, ins.int32, ins.int64, ins.bool, ...

Place Constructors::

    ins.CPUPlace()       # CPU device
    ins.GPUPlace(0)      # GPU device 0
"""

try:
    # Pre-load ALL backend .so files (CPU + GPU) from this package directory
    # so that dlopen inside the native module finds them.
    import ctypes as _ct
    import os as _os
    import glob as _gl

    _pkg_dir = _os.path.dirname(_os.path.abspath(__file__))

    # Windows: add DLL search directories (Python 3.8+)
    if _os.name == "nt" and hasattr(_os, "add_dll_directory"):
        _os.add_dll_directory(_pkg_dir)
        # Discover build*/backends/* directories for DLL resolution.
        _project_root = _os.path.normpath(_os.path.join(_pkg_dir, "..", "..", ".."))
        for _d in _gl.glob(_os.path.join(_project_root, "build*", "backends", "*")):
            if _os.path.isdir(_d):
                _os.add_dll_directory(_d)

    # Linux/macOS: ensure package directory is in LD_LIBRARY_PATH
    if _os.name != "nt":
        _ld = _os.environ.get("LD_LIBRARY_PATH", "")
        if _pkg_dir not in _ld.split(":"):
            _os.environ["LD_LIBRARY_PATH"] = _pkg_dir + ":" + _ld if _ld else _pkg_dir

    # Also chdir to package dir so discover_backends() (which scans ".")
    # finds the backends — needed for editable installs (pip install -e .)
    _saved_cwd = _os.getcwd()
    _os.chdir(_pkg_dir)

    _backend_patterns = [
        "libinsight_*_backend.so",  # Linux
        "libinsight_*_backend.dylib",  # macOS
        "insight_*_backend.dll",  # Windows
    ]
    for _pat in _backend_patterns:
        for _so in _gl.glob(_os.path.join(_pkg_dir, _pat)):
            try:
                _ct.CDLL(_so, mode=_ct.RTLD_GLOBAL)
            except OSError:
                pass
    from ._insight import *  # noqa: F401,F403

    _os.chdir(_saved_cwd)

    # Register package directory as backend search path, then init
    from ._insight import (
        init as _native_init,
        is_initialized as _is_init,
        add_backend_search_path as _add_search_path,
    )

    _add_search_path(_pkg_dir)

    # Also search directory of _insight.so (may differ in editable installs)
    _native_dir = None
    try:
        import importlib.util as _ilu

        _spec = _ilu.find_spec("insight._insight")
        if _spec and _spec.origin:
            _native_dir = _os.path.dirname(_os.path.abspath(_spec.origin))
            if _native_dir != _pkg_dir:
                _add_search_path(_native_dir)
    except Exception:
        pass

    # On Windows, register build*/backends/* as search paths so init() can
    # discover and load GPU backends (cuda, npu, etc.) by absolute path.
    if _os.name == "nt":
        _project_root = _os.path.normpath(_os.path.join(_pkg_dir, "..", "..", ".."))
        for _d in _gl.glob(_os.path.join(_project_root, "build*", "backends", "*")):
            if _os.path.isdir(_d):
                _add_search_path(_d)

    if not _is_init():
        _native_init()

    # Core types and infrastructure (native)
    from ._insight import (
        init,
        is_initialized,
        load_backend,
        # Signal top-level aliases (native)
        convolve,
        unwrap,
        sinc,
    )

    # Core types (wrapped with docstrings)
    from .types import DType, Place, Shape, Slice, CPUPlace, GPUPlace

    # Timer / Profiler
    from .timer import Timer, Profiler, ProfileBlock
    from .array import Array

    # Wrapper modules — each provides docstrings and argument validation
    from .creation import *  # noqa: F401,F403
    from .elementwise import *  # noqa: F401,F403
    from .unary import *  # noqa: F401,F403
    from .reduction import *  # noqa: F401,F403
    from .manipulation import *  # noqa: F401,F403
    from .linalg import *  # noqa: F401,F403
    from .fft import *  # noqa: F401,F403
    from .complex import *  # noqa: F401,F403
    from .random import *  # noqa: F401,F403
    from .cast import *  # noqa: F401,F403
    from .indexing import *  # noqa: F401,F403

    # Signal submodule (ins.signal.*) — wrapper layer with docstrings
    from . import signal as _signal_module

    signal = _signal_module

    # Import result types from native binding (not wrapped in submodule)
    from ._insight.signal import (
        SpectralResult,
        SpectrogramResult,
        GaussPulseResult,
        ChirpMethod,
    )

    # Re-export signal functions at top level for convenience
    from .signal import (  # noqa: F401
        # Windows
        general_cosine,
        get_window,
        boxcar,
        triang,
        parzen,
        bohman,
        bartlett,
        cosine,
        blackman,
        nuttall,
        blackmanharris,
        flattop,
        hann,
        general_hamming,
        hamming,
        tukey,
        barthann,
        kaiser,
        gaussian,
        general_gaussian,
        chebwin,
        taylor,
        qmf,
        # Waveforms
        sawtooth,
        square_wf,
        gausspulse,
        gausspulse_full,
        chirp,
        unit_impulse,
        # B-Splines
        gauss_spline,
        cubic,
        quadratic,
        # Filter Design
        kaiser_beta,
        kaiser_atten,
        firwin,
        firwin2,
        cmplx_sort,
        # Convolution
        fftconvolve,
        correlate,
        convolve2d,
        correlate2d,
        choose_conv_method,
        correlation_lags,
        # Filtering
        hilbert,
        hilbert2,
        detrend,
        wiener,
        firfilter,
        lfilter,
        lfilter_zi,
        filtfilt,
        decimate,
        resample,
        resample_poly,
        freq_shift,
        sosfilt,
        upfirdn,
        channelize_poly,
        # Spectral Analysis
        csd,
        welch,
        periodogram,
        coherence,
        spectrogram,
        stft,
        istft,
        vectorstrength,
        lombscargle,
        # Wavelets
        morlet,
        ricker,
        morlet2,
        cwt,
        # Acoustics
        mel2hz,
        hz2mel,
        mel_frequencies,
        hz2bark,
        bark2hz,
        # Demod
        fm_demod,
        # Peak Finding
        argrelextrema,
        argrelmax,
        argrelmin,
        # Radar
        pulse_compression,
        pulse_doppler,
        cfar_alpha,
        ca_cfar,
        mvdr,
        ambgfun,
        # Estimation
        KalmanFilter,
        # Signal I/O
        read_bin,
        write_bin,
        unpack_bin,
        pack_bin,
        read_sigmf,
        write_sigmf,
    )

    # Plot submodule (ins.plot.*) — may not exist if INSIGHT_USE_MATPLOT=OFF
    try:
        from ._insight import plot as _plot_module

        plot = _plot_module
    except ImportError:
        pass

except ImportError as e:
    raise ImportError(
        f"Failed to load Insight native module: {e}\n"
        "Make sure you have built the Python binding:\n"
        "  cmake --build build --target insight_python"
    ) from e

__version__ = "1.0.0"
__all__ = [
    # Types
    "DType",
    "Place",
    "Shape",
    "Slice",
    "Array",
    # Init
    "init",
    "is_initialized",
    "load_backend",
    # Place
    "CPUPlace",
    "GPUPlace",
    # Timer
    "Timer",
    # Profiler
    "Profiler",
    "ProfileBlock",
    # Dtypes
    "float32",
    "float64",
    "float16",
    "bfloat16",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "bool",
    "complex64",
    "complex128",
    # Creation
    "zeros",
    "ones",
    "full",
    "eye",
    "arange",
    "linspace",
    "logspace",
    "from_numpy",
    "from_array",
    "zeros_like",
    "ones_like",
    # Elementwise
    "add",
    "sub",
    "mul",
    "div",
    "pow",
    "mod",
    "maximum",
    "minimum",
    "equal",
    "not_equal",
    "greater",
    "less",
    "greater_equal",
    "less_equal",
    "logical_and",
    "logical_or",
    "logical_xor",
    "logical_not",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    # Unary
    "abs",
    "negative",
    "square",
    "sqrt",
    "exp",
    "log",
    "log2",
    "log10",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "sinh",
    "cosh",
    "tanh",
    "floor",
    "ceil",
    "round",
    "sign",
    "isnan",
    "isinf",
    "isfinite",
    "exp2",
    "expm1",
    "log1p",
    "cbrt",
    "reciprocal",
    "asinh",
    "acosh",
    "atanh",
    "trunc",
    "deg2rad",
    "rad2deg",
    "conj",
    "angle",
    "where",
    # Reduction
    "sum",
    "mean",
    "max",
    "min",
    "prod",
    "argmax",
    "argmin",
    "any",
    "all",
    "var",
    "std",
    "cumsum",
    "cumprod",
    "cummax",
    "cummin",
    "sem",
    "count_nonzero",
    "median",
    "quantile",
    "percentile",
    "nansum",
    "nanmean",
    "nanmax",
    "nanmin",
    "nanstd",
    "nanvar",
    # Manipulation
    "concat",
    "stack",
    "vstack",
    "hstack",
    "split",
    "tile",
    "repeat",
    "flip",
    "pad",
    "reshape",
    "flatten",
    "ravel",
    "squeeze",
    "unsqueeze",
    "roll",
    "permute",
    "transpose",
    "swapaxes",
    "moveaxis",
    "fliplr",
    "flipud",
    "rot90",
    "diag",
    "diagonal",
    "tril",
    "triu",
    "diff",
    # Linalg
    "matmul",
    "dot",
    "inner",
    "outer",
    "det",
    "slogdet",
    "inv",
    "pinv",
    "solve",
    "lstsq",
    "svd",
    "svdvals",
    "eigh",
    "eigvalsh",
    "eig",
    "eigvals",
    "cholesky",
    "qr",
    "lq",
    "lu",
    "norm",
    "cond",
    "matrix_rank",
    "matrix_power",
    "trace",
    "cov",
    # FFT
    "fft",
    "ifft",
    "fft2",
    "ifft2",
    "fftn",
    "ifftn",
    "rfft",
    "irfft",
    "fftfreq",
    "rfftfreq",
    # Complex
    "is_complex",
    "has_complex_shape",
    "to_complex",
    "as_complex",
    "as_real",
    "real",
    "imag",
    # Signal (top-level)
    "convolve",
    "unwrap",
    "sinc",
    # Signal submodule
    "signal",
    # Signal functions (top-level re-exports)
    "general_cosine",
    "get_window",
    "boxcar",
    "triang",
    "parzen",
    "bohman",
    "bartlett",
    "cosine",
    "blackman",
    "nuttall",
    "blackmanharris",
    "flattop",
    "hann",
    "general_hamming",
    "hamming",
    "tukey",
    "barthann",
    "kaiser",
    "gaussian",
    "general_gaussian",
    "chebwin",
    "taylor",
    "qmf",
    "sawtooth",
    "square_wf",
    "gausspulse",
    "gausspulse_full",
    "chirp",
    "unit_impulse",
    "ChirpMethod",
    "gauss_spline",
    "cubic",
    "quadratic",
    "kaiser_beta",
    "kaiser_atten",
    "firwin",
    "firwin2",
    "cmplx_sort",
    "fftconvolve",
    "correlate",
    "convolve2d",
    "correlate2d",
    "choose_conv_method",
    "correlation_lags",
    "hilbert",
    "hilbert2",
    "detrend",
    "wiener",
    "firfilter",
    "lfilter",
    "lfilter_zi",
    "filtfilt",
    "decimate",
    "resample",
    "resample_poly",
    "freq_shift",
    "sosfilt",
    "upfirdn",
    "channelize_poly",
    "SpectralResult",
    "SpectrogramResult",
    "GaussPulseResult",
    "csd",
    "welch",
    "periodogram",
    "coherence",
    "spectrogram",
    "stft",
    "istft",
    "vectorstrength",
    "lombscargle",
    "morlet",
    "ricker",
    "morlet2",
    "cwt",
    "mel2hz",
    "hz2mel",
    "mel_frequencies",
    "hz2bark",
    "bark2hz",
    # Random
    "rand",
    "randn",
    "randint",
    "normal",
    "uniform",
    "randperm",
    "seed",
    "get_seed",
    "rand_like",
    "randn_like",
    "exponential",
    "gamma",
    "beta",
    "binomial",
    "poisson",
    # Cast & Indexing
    "cast",
    "take",
    "nonzero",
    "flatnonzero",
    "argsort",
    "sort",
    "masked_select",
    "searchsorted",
    "unique",
    "topk",
    "gather",
    "scatter",
    "scatter_add",
    "scatter_reduce",
    "interp",
    "indices",
    "ix_",
    # Plot submodule
    "plot",
]
