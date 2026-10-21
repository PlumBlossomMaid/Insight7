"""Array creation functions.

Functions:
    zeros(shape, dtype, place) — Array filled with zeros
    ones(shape, dtype, place) — Array filled with ones
    full(shape, fill_value, dtype, place) — Array filled with scalar
    eye(n, m, k, dtype, place) — 2-D identity matrix
    arange(stop, dtype, place) — Evenly spaced values
    linspace(start, stop, num, dtype, place) — Evenly spaced over interval
    from_numpy(array) — Create from NumPy array
    from_array(data) — Create from Python list or NumPy array
    zeros_like(arr) — Zeros with same shape/dtype
    ones_like(arr) — Ones with same shape/dtype
"""

from insight._insight import zeros as _native_zeros
from insight._insight import ones as _native_ones
from insight._insight import full as _native_full
from insight._insight import eye as _native_eye
from insight._insight import arange as _native_arange
from insight._insight import linspace as _native_linspace
from insight._insight import logspace as _native_logspace
from insight._insight import Shape, DType, CPUPlace, get_device

# Import remaining functions directly (no Shape conversion needed)
from insight._insight import (
    from_numpy,
    from_array,
    zeros_like,
    ones_like,
)


def _to_shape(s):
    if isinstance(s, (list, tuple)):
        return Shape(list(s))
    return s


def zeros(shape, dtype=DType.float32, place=None):
    """Create an array filled with zeros.

    Args:
        shape: Shape as list of ints or Shape object.
        dtype: Data type (default float32).
        place: Device placement (default CPU).

    Returns:
        Array of zeros.
    """
    s = _to_shape(shape)
    if place is None:
        return _native_zeros(s, dtype, get_device())
    return _native_zeros(s, dtype, place)


def ones(shape, dtype=DType.float32, place=None):
    """Create an array filled with ones.

    Args:
        shape: Shape as list of ints or Shape object.
        dtype: Data type (default float32).
        place: Device placement (default CPU).

    Returns:
        Array of ones.
    """
    s = _to_shape(shape)
    if place is None:
        return _native_ones(s, dtype, get_device())
    return _native_ones(s, dtype, place)


def full(shape, fill_value, dtype=DType.float32, place=None):
    """Create an array filled with a scalar value.

    Args:
        shape: Shape as list of ints or Shape object.
        fill_value: Scalar fill value.
        dtype: Data type (default float32).
        place: Device placement (default CPU).

    Returns:
        Array filled with fill_value.
    """
    s = _to_shape(shape)
    if place is None:
        return _native_full(s, fill_value, dtype, get_device())
    return _native_full(s, fill_value, dtype, place)


def eye(n, m=-1, k=0, dtype=DType.float32, place=None):
    """Create a 2-D array with ones on a diagonal and zeros elsewhere.

    Args:
        n: Number of rows.
        m: Number of columns (default -1, meaning same as n).
        k: Diagonal offset (default 0).
        dtype: Data type (default float32).
        place: Device placement (default: current device).

    Returns:
        2-D identity-like array.
    """
    if place is None:
        place = get_device()
    return _native_eye(n, m, k, dtype, place)


def arange(*args, dtype=DType.int64, place=None):
    """Create an array of evenly spaced values.

    Called as arange(stop) or arange(start, stop, step).

    Args:
        dtype: Data type (default int64). May be passed positionally as the
            last argument for convenience (arange(20, float32)).
        place: Device placement (default: current device).

    Returns:
        1-D array of evenly spaced values.
    """
    if place is None:
        place = get_device()
    # Paddle-style convenience: allow dtype as a trailing positional argument.
    if args and isinstance(args[-1], DType):
        dtype = args[-1]
        args = args[:-1]
    if len(args) == 1:
        return _native_arange(float(args[0]), dtype=dtype, place=place)
    if len(args) == 2:
        return _native_arange(float(args[0]), float(args[1]), dtype=dtype, place=place)
    if len(args) == 3:
        return _native_arange(
            float(args[0]), float(args[1]), float(args[2]), dtype=dtype, place=place
        )
    raise TypeError(
        "arange expected (stop), (start, stop) or (start, stop, step), got "
        f"{len(args)} positional arguments"
    )


def linspace(start, stop, num, dtype=DType.float32, place=None):
    """Create an array of evenly spaced values over a closed interval.

    Args:
        start: First value.
        stop: Last value (inclusive).
        num: Number of samples.
        dtype: Data type (default float32).
        place: Device placement (default: current device).

    Returns:
        1-D array of num evenly spaced values.
    """
    if place is None:
        place = get_device()
    return _native_linspace(start, stop, num, dtype, place)


def logspace(start, stop, num, base=10.0, dtype=DType.float32, place=None):
    """Create an array of values evenly spaced on a log scale.

    Args:
        start: base ** start is the first value.
        stop: base ** stop is the last value.
        num: Number of samples.
        base: Log base (default 10.0).
        dtype: Data type (default float32).
        place: Device placement (default: current device).

    Returns:
        1-D array of num log-spaced values.
    """
    if place is None:
        place = get_device()
    return _native_logspace(start, stop, num, base, dtype, place)


__all__ = [
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
]
