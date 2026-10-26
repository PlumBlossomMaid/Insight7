"""Random number generation.

Functions:
    rand, randn, randint, normal, uniform, randperm
    seed, get_seed, rand_like, randn_like
    exponential, gamma, beta, binomial, poisson
"""

from __future__ import annotations

from typing import Optional, Union

from insight._insight import Array, DType, Place, Shape  # noqa: F401
from insight._insight import (
    rand as _native_rand,
    randn as _native_randn,
    randint as _native_randint,
    normal as _native_normal,
    uniform as _native_uniform,
    randperm as _native_randperm,
    seed as _native_seed,
    get_seed as _native_get_seed,
    rand_like as _native_rand_like,
    randn_like as _native_randn_like,
    randint_like as _native_randint_like,
    normal_like as _native_normal_like,
    uniform_like as _native_uniform_like,
    exponential as _native_exponential,
    gamma as _native_gamma,
    beta as _native_beta,
    binomial as _native_binomial,
    poisson as _native_poisson,
    chisquare as _native_chisquare,
    get_device as _get_device,
)


def _to_shape(s: Union[list, tuple, Shape]) -> Shape:
    """Convert a list/tuple to Shape if needed."""
    if isinstance(s, Shape):
        return s
    return Shape(s)


def seed(base_seed: int) -> None:
    """Set the random seed for all available devices.

    Each device receives ``base_seed + device_id`` as its seed.

    Args:
        base_seed: Base seed value.
    """
    _native_seed(base_seed)


def get_seed() -> int:
    """Get the current base random seed.

    Returns:
        Current base seed value.
    """
    return _native_get_seed()


def rand(
    shape: "Shape",
    dtype: "DType" = None,
    place: "Place" = None,
) -> "Array":
    """Create an array with uniformly distributed random values in [0, 1).

    Args:
        shape: Desired output shape.
        dtype: Data type (default: float32).
        place: Device placement (default: current device).

    Returns:
        Array of the given shape with random values in [0, 1).
    """
    from insight._insight import float32 as _f32, CPUPlace as _cpu

    if dtype is None:
        dtype = _f32
    if place is None:
        place = _get_device()
    return _native_rand(_to_shape(shape), dtype, place)


def randn(
    shape: "Shape",
    dtype: "DType" = None,
    place: "Place" = None,
) -> "Array":
    """Create an array with values from the standard normal distribution.

    Samples from a normal distribution with mean 0 and standard
    deviation 1.

    Args:
        shape: Desired output shape.
        dtype: Data type (default: float32).
        place: Device placement (default: current device).

    Returns:
        Array of the given shape with normally distributed values.
    """
    from insight._insight import float32 as _f32, CPUPlace as _cpu

    if dtype is None:
        dtype = _f32
    if place is None:
        place = _get_device()
    return _native_randn(_to_shape(shape), dtype, place)


def randint(
    low: int,
    high: int,
    shape: "Shape",
    dtype: "DType" = None,
    place: "Place" = None,
) -> "Array":
    """Create an array with random integers in [low, high).

    Args:
        low: Lowest integer to be drawn (inclusive).
        high: One above the highest integer to be drawn (exclusive).
        shape: Desired output shape.
        dtype: Data type (default: int64).
        place: Device placement (default: current device).

    Returns:
        Array of the given shape with random integer values.
    """
    from insight._insight import int64 as _i64, CPUPlace as _cpu

    if dtype is None:
        dtype = _i64
    if place is None:
        place = _get_device()
    return _native_randint(low, high, _to_shape(shape), dtype, place)


def normal(
    mean: float = 0.0,
    std: float = 1.0,
    shape: "Shape" = None,
    dtype: "DType" = None,
    place: "Place" = None,
) -> "Array":
    """Create an array with values from a normal distribution.

    Samples from ``N(mean, std^2)``.

    Args:
        mean: Mean of the distribution (default: 0.0).
        std: Standard deviation of the distribution (default: 1.0).
        shape: Desired output shape.
        dtype: Data type (default: float32).
        place: Device placement (default: current device).

    Returns:
        Array of the given shape with normally distributed values.
    """
    from insight._insight import float32 as _f32, CPUPlace as _cpu

    if dtype is None:
        dtype = _f32
    if place is None:
        place = _get_device()
    return _native_normal(mean, std, _to_shape(shape), dtype, place)


def uniform(
    low: float = 0.0,
    high: float = 1.0,
    shape: "Shape" = None,
    dtype: "DType" = None,
    place: "Place" = None,
) -> "Array":
    """Create an array with values from a uniform distribution.

    Samples from ``U(low, high)``.

    Args:
        low: Lower bound of the distribution (default: 0.0).
        high: Upper bound of the distribution (default: 1.0).
        shape: Desired output shape.
        dtype: Data type (default: float32).
        place: Device placement (default: current device).

    Returns:
        Array of the given shape with uniformly distributed values.
    """
    from insight._insight import float32 as _f32, CPUPlace as _cpu

    if dtype is None:
        dtype = _f32
    if place is None:
        place = _get_device()
    return _native_uniform(low, high, _to_shape(shape), dtype, place)


def randperm(
    n: int,
    dtype: "DType" = None,
    place: "Place" = None,
) -> "Array":
    """Create a random permutation of integers [0, n).

    Args:
        n: Upper bound (exclusive) of the permutation range.
        dtype: Data type (default: int64).
        place: Device placement (default: current device).

    Returns:
        1-D array of length ``n`` containing a random permutation of
        ``[0, n)``.
    """
    from insight._insight import int64 as _i64, CPUPlace as _cpu

    if dtype is None:
        dtype = _i64
    if place is None:
        place = _get_device()
    return _native_randperm(n, dtype, place)


def rand_like(x: "Array") -> "Array":
    """Create an array of uniform random values with the same shape.

    Returns an array with the same shape, dtype, and device as ``x``,
    filled with uniformly distributed random values in [0, 1).

    Args:
        x: Input array whose shape/dtype/device are matched.

    Returns:
        Array of uniform random values with the same properties as
        ``x``.
    """
    return _native_rand_like(x)


def randn_like(x: "Array") -> "Array":
    """Create an array of normal random values with the same shape.

    Returns an array with the same shape, dtype, and device as ``x``,
    filled with values from the standard normal distribution.

    Args:
        x: Input array whose shape/dtype/device are matched.

    Returns:
        Array of normally distributed random values with the same
        properties as ``x``.
    """
    return _native_randn_like(x)


def randint_like(x: "Array", low: int, high: int, dtype: "DType" = None) -> "Array":
    return (
        _native_randint_like(x, low, high, dtype)
        if dtype is not None
        else _native_randint_like(x, low, high)
    )


def normal_like(x: "Array", mean: float = 0.0, std: float = 1.0, dtype: "DType" = None) -> "Array":
    return (
        _native_normal_like(x, mean, std, dtype)
        if dtype is not None
        else _native_normal_like(x, mean, std)
    )


def uniform_like(x: "Array", low: float = 0.0, high: float = 1.0, dtype: "DType" = None) -> "Array":
    return (
        _native_uniform_like(x, low, high, dtype)
        if dtype is not None
        else _native_uniform_like(x, low, high)
    )


def exponential(
    scale: float,
    shape: "Shape",
    dtype: "DType" = None,
    place: "Place" = None,
) -> "Array":
    """Create an array from an exponential distribution.

    Samples from ``Exp(1/scale)``, where ``scale`` is the mean of the
    distribution.

    Args:
        scale: Scale parameter (mean of the distribution).
        shape: Desired output shape.
        dtype: Data type (default: float32).
        place: Device placement (default: current device).

    Returns:
        Array of the given shape with exponentially distributed values.
    """
    from insight._insight import float32 as _f32, CPUPlace as _cpu

    if dtype is None:
        dtype = _f32
    if place is None:
        place = _get_device()
    return _native_exponential(scale, _to_shape(shape), dtype, place)


def gamma(
    shape: float,
    rate: float,
    out_shape: "Shape",
    dtype: "DType" = None,
    place: "Place" = None,
) -> "Array":
    """Create an array from a gamma distribution.

    Args:
        shape: Shape parameter (alpha) of the gamma distribution.
        rate: Rate parameter (beta) of the gamma distribution.
        out_shape: Desired output shape.
        dtype: Data type (default: float32).
        place: Device placement (default: current device).

    Returns:
        Array of the given shape with gamma-distributed values.
    """
    from insight._insight import float32 as _f32, CPUPlace as _cpu

    if dtype is None:
        dtype = _f32
    if place is None:
        place = _get_device()
    return _native_gamma(shape, rate, _to_shape(out_shape), dtype, place)


def beta(
    a: float,
    b: float,
    shape: "Shape",
    dtype: "DType" = None,
    place: "Place" = None,
) -> "Array":
    """Create an array from a beta distribution.

    Args:
        a: Alpha parameter.
        b: Beta parameter.
        shape: Desired output shape.
        dtype: Data type (default: float32).
        place: Device placement (default: current device).

    Returns:
        Array of the given shape with beta-distributed values.
    """
    from insight._insight import float32 as _f32, CPUPlace as _cpu

    if dtype is None:
        dtype = _f32
    if place is None:
        place = _get_device()
    return _native_beta(a, b, _to_shape(shape), dtype, place)


def binomial(
    n: int,
    p: float,
    shape: "Shape",
    dtype: "DType" = None,
    place: "Place" = None,
) -> "Array":
    """Create an array from a binomial distribution.

    Args:
        n: Number of trials.
        p: Probability of success for each trial.
        shape: Desired output shape.
        dtype: Data type (default: int64).
        place: Device placement (default: current device).

    Returns:
        Array of the given shape with binomially distributed values.
    """
    from insight._insight import int64 as _i64, CPUPlace as _cpu

    if dtype is None:
        dtype = _i64
    if place is None:
        place = _get_device()
    return _native_binomial(n, p, _to_shape(shape), dtype, place)


def poisson(
    lam: float,
    shape: "Shape",
    dtype: "DType" = None,
    place: "Place" = None,
) -> "Array":
    """Create an array from a Poisson distribution.

    Args:
        lam: Lambda parameter (expected number of events).
        shape: Desired output shape.
        dtype: Data type (default: int64).
        place: Device placement (default: current device).

    Returns:
        Array of the given shape with Poisson-distributed values.
    """
    from insight._insight import int64 as _i64, CPUPlace as _cpu

    if dtype is None:
        dtype = _i64
    if place is None:
        place = _get_device()
    return _native_poisson(lam, _to_shape(shape), dtype, place)


def chisquare(
    df: float,
    shape: "Shape",
    dtype: "DType" = None,
    place: "Place" = None,
) -> "Array":
    """Create an array from a chi-square distribution.

    Args:
        df: Degrees of freedom.
        shape: Desired output shape.
        dtype: Data type (default: float32).
        place: Device placement (default: current device).

    Returns:
        Array of the given shape with chi-square-distributed values.
    """
    from insight._insight import float32 as _f32, CPUPlace as _cpu

    if dtype is None:
        dtype = _f32
    if place is None:
        place = _get_device()
    return _native_chisquare(df, _to_shape(shape), dtype, place)


__all__ = [
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
    "randint_like",
    "normal_like",
    "uniform_like",
    "exponential",
    "gamma",
    "beta",
    "binomial",
    "poisson",
    "chisquare",
]
