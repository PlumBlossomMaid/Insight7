"""Reduction operations.

Functions:
    sum, mean, max, min, prod, argmax, argmin
    any, all, var, std, cumsum, cumprod
    cummax, cummin, sem, count_nonzero, median
    quantile, percentile
    nansum, nanmean, nanmax, nanmin, nanstd, nanvar
"""

from __future__ import annotations

from typing import Optional

from insight._insight import Array  # noqa: F401
from insight._insight import (
    sum as _native_sum,
    mean as _native_mean,
    max as _native_max,
    min as _native_min,
    prod as _native_prod,
    argmax as _native_argmax,
    argmin as _native_argmin,
    any as _native_any,
    all as _native_all,
    var as _native_var,
    std as _native_std,
    cumsum as _native_cumsum,
    cumprod as _native_cumprod,
    cummax as _native_cummax,
    cummin as _native_cummin,
    sem as _native_sem,
    count_nonzero as _native_count_nonzero,
    median as _native_median,
    quantile as _native_quantile,
    percentile as _native_percentile,
    nansum as _native_nansum,
    nanmean as _native_nanmean,
    nanmax as _native_nanmax,
    nanmin as _native_nanmin,
    nanstd as _native_nanstd,
    nanvar as _native_nanvar,
    nanmedian as _native_nanmedian,
    nanquantile as _native_nanquantile,
    bincount as _native_bincount,
)


def sum(
    x: "Array",
    axis: Optional[int] = None,
    keepdims: bool = False,
) -> "Array":
    """Sum of array elements along a given axis.

    Args:
        x: Input array.
        axis: Axis along which to compute the sum. If None, the array is
            flattened before computing the sum.
        keepdims: If True, the reduced axes are retained as dimensions
            with size 1.

    Returns:
        Array containing the sum values. If ``axis`` is None, a scalar
        array is returned.
    """
    return _native_sum(x, axis, keepdims)


def mean(
    x: "Array",
    axis: Optional[int] = None,
    keepdims: bool = False,
) -> "Array":
    """Mean of array elements along a given axis.

    Args:
        x: Input array.
        axis: Axis along which to compute the mean. If None, the array is
            flattened before computing the mean.
        keepdims: If True, the reduced axes are retained as dimensions
            with size 1.

    Returns:
        Array containing the mean values.
    """
    return _native_mean(x, axis, keepdims)


def max(
    x: "Array",
    axis: Optional[int] = None,
    keepdims: bool = False,
) -> "Array":
    """Maximum of array elements along a given axis.

    Args:
        x: Input array.
        axis: Axis along which to compute the maximum. If None, the array
            is flattened before computing the maximum.
        keepdims: If True, the reduced axes are retained as dimensions
            with size 1.

    Returns:
        Array containing the maximum values.
    """
    return _native_max(x, axis, keepdims)


def min(
    x: "Array",
    axis: Optional[int] = None,
    keepdims: bool = False,
) -> "Array":
    """Minimum of array elements along a given axis.

    Args:
        x: Input array.
        axis: Axis along which to compute the minimum. If None, the array
            is flattened before computing the minimum.
        keepdims: If True, the reduced axes are retained as dimensions
            with size 1.

    Returns:
        Array containing the minimum values.
    """
    return _native_min(x, axis, keepdims)


def prod(
    x: "Array",
    axis: Optional[int] = None,
    keepdims: bool = False,
) -> "Array":
    """Product of array elements along a given axis.

    Args:
        x: Input array.
        axis: Axis along which to compute the product. If None, the array
            is flattened before computing the product.
        keepdims: If True, the reduced axes are retained as dimensions
            with size 1.

    Returns:
        Array containing the product values.
    """
    return _native_prod(x, axis, keepdims)


def argmax(
    x: "Array",
    axis: Optional[int] = None,
    keepdims: bool = False,
) -> "Array":
    """Indices of maximum values along a given axis.

    Returns the index of the maximum value along the specified axis.

    Args:
        x: Input array.
        axis: Axis along which to find the argmax. If None, the array is
            flattened before computing.
        keepdims: If True, the reduced axes are retained as dimensions
            with size 1.

    Returns:
        Array of indices (dtype: int64) of the maximum values.
    """
    return _native_argmax(x, axis, keepdims)


def argmin(
    x: "Array",
    axis: Optional[int] = None,
    keepdims: bool = False,
) -> "Array":
    """Indices of minimum values along a given axis.

    Returns the index of the minimum value along the specified axis.

    Args:
        x: Input array.
        axis: Axis along which to find the argmin. If None, the array is
            flattened before computing.
        keepdims: If True, the reduced axes are retained as dimensions
            with size 1.

    Returns:
        Array of indices (dtype: int64) of the minimum values.
    """
    return _native_argmin(x, axis, keepdims)


def any(
    x: "Array",
    axis: Optional[int] = None,
    keepdims: bool = False,
) -> "Array":
    """Test whether any element along an axis is non-zero.

    Returns True if any element in the input array (along the given
    axis) is non-zero.

    Args:
        x: Input array.
        axis: Axis along which to test. If None, the array is flattened
            before testing.
        keepdims: If True, the reduced axes are retained as dimensions
            with size 1.

    Returns:
        Boolean array indicating whether any element is non-zero.
    """
    return _native_any(x, axis, keepdims)


def all(
    x: "Array",
    axis: Optional[int] = None,
    keepdims: bool = False,
) -> "Array":
    """Test whether all elements along an axis are non-zero.

    Returns True only if every element in the input array (along the
    given axis) is non-zero.

    Args:
        x: Input array.
        axis: Axis along which to test. If None, the array is flattened
            before testing.
        keepdims: If True, the reduced axes are retained as dimensions
            with size 1.

    Returns:
        Boolean array indicating whether all elements are non-zero.
    """
    return _native_all(x, axis, keepdims)


def var(
    x: "Array",
    axis: Optional[int] = None,
    keepdims: bool = False,
    ddof: int = 0,
) -> "Array":
    """Variance of array elements along a given axis.

    Args:
        x: Input array.
        axis: Axis along which to compute the variance. If None, the
            array is flattened before computing.
        keepdims: If True, the reduced axes are retained as dimensions
            with size 1.
        ddof: Delta degrees of freedom. The divisor is ``N - ddof``,
            where ``N`` is the number of elements. Use 0 for population
            variance and 1 for sample variance.

    Returns:
        Array containing the variance values.
    """
    return _native_var(x, axis, keepdims, ddof)


def std(
    x: "Array",
    axis: Optional[int] = None,
    keepdims: bool = False,
    ddof: int = 0,
) -> "Array":
    """Standard deviation of array elements along a given axis.

    Args:
        x: Input array.
        axis: Axis along which to compute the standard deviation. If
            None, the array is flattened before computing.
        keepdims: If True, the reduced axes are retained as dimensions
            with size 1.
        ddof: Delta degrees of freedom. The divisor is ``N - ddof``,
            where ``N`` is the number of elements. Use 0 for population
            standard deviation and 1 for sample standard deviation.

    Returns:
        Array containing the standard deviation values.
    """
    return _native_std(x, axis, keepdims, ddof)


def cumsum(x: "Array", axis: int) -> "Array":
    """Cumulative sum along a given axis.

    Computes the cumulative sum of elements along the specified axis.

    Args:
        x: Input array.
        axis: Axis along which to compute the cumulative sum.

    Returns:
        Array with the same shape and dtype as the input, containing
        the cumulative sum values.
    """
    return _native_cumsum(x, axis)


def cumprod(x: "Array", axis: int) -> "Array":
    """Cumulative product along a given axis.

    Computes the cumulative product of elements along the specified axis.

    Args:
        x: Input array.
        axis: Axis along which to compute the cumulative product.

    Returns:
        Array with the same shape and dtype as the input, containing
        the cumulative product values.
    """
    return _native_cumprod(x, axis)


def cummax(x: "Array", axis: int) -> "Array":
    """Cumulative maximum along a given axis.

    Computes the running maximum of elements along the specified axis.

    Args:
        x: Input array.
        axis: Axis along which to compute the cumulative maximum.

    Returns:
        Array with the same shape as the input, containing the
        cumulative maximum values.
    """
    return _native_cummax(x, axis)


def cummin(x: "Array", axis: int) -> "Array":
    """Cumulative minimum along a given axis.

    Computes the running minimum of elements along the specified axis.

    Args:
        x: Input array.
        axis: Axis along which to compute the cumulative minimum.

    Returns:
        Array with the same shape as the input, containing the
        cumulative minimum values.
    """
    return _native_cummin(x, axis)


def sem(
    x: "Array",
    axis: Optional[int] = None,
    keepdims: bool = False,
    ddof: int = 0,
) -> "Array":
    """Standard error of the mean along a given axis.

    Computes ``std(x, ddof=ddof) / sqrt(N)``, where N is the number of
    elements along the axis.

    Args:
        x: Input array.
        axis: Axis along which to compute. If None, the array is
            flattened before computing.
        keepdims: If True, the reduced axes are retained as dimensions
            with size 1.
        ddof: Delta degrees of freedom. The divisor for the standard
            deviation is ``N - ddof``.

    Returns:
        Array containing the standard error of the mean values.
    """
    return _native_sem(x, axis, keepdims, ddof)


def count_nonzero(
    x: "Array",
    axis: Optional[int] = None,
    keepdims: bool = False,
) -> "Array":
    """Count the number of non-zero elements along a given axis.

    Args:
        x: Input array.
        axis: Axis along which to count. If None, the array is
            flattened before counting.
        keepdims: If True, the reduced axes are retained as dimensions
            with size 1.

    Returns:
        Array of counts (dtype: int64).
    """
    return _native_count_nonzero(x, axis, keepdims)


def median(
    x: "Array",
    axis: Optional[int] = None,
    keepdims: bool = False,
) -> "Array":
    """Median of array elements along a given axis.

    Args:
        x: Input array.
        axis: Axis along which to compute the median. If None, the array
            is flattened before computing.
        keepdims: If True, the reduced axes are retained as dimensions
            with size 1.

    Returns:
        Array containing the median values.
    """
    return _native_median(x, axis, keepdims)


def quantile(
    x: "Array",
    q: "Array",
    axis: Optional[int] = None,
    keepdims: bool = False,
) -> "Array":
    """Quantile values along a given axis.

    Computes the q-th quantile of the data along the specified axis,
    where ``q`` is in the range [0, 1].

    Args:
        x: Input array.
        q: Quantile(s) to compute. A scalar float in [0, 1], or an
            array of quantile values.
        axis: Axis along which to compute the quantile. If None, the
            array is flattened before computing.
        keepdims: If True, the reduced axes are retained as dimensions
            with size 1.

    Returns:
        Array containing the quantile values.
    """
    return _native_quantile(x, q, axis, keepdims)


def percentile(
    x: "Array",
    q: float,
    axis: Optional[int] = None,
    keepdims: bool = False,
) -> "Array":
    """Percentile values along a given axis.

    Computes the q-th percentile of the data along the specified axis,
    where ``q`` is in the range [0, 100].

    Args:
        x: Input array.
        q: Percentile to compute, in the range [0, 100].
        axis: Axis along which to compute the percentile. If None, the
            array is flattened before computing.
        keepdims: If True, the reduced axes are retained as dimensions
            with size 1.

    Returns:
        Array containing the percentile values.
    """
    return _native_percentile(x, q, axis, keepdims)


def nansum(
    x: "Array",
    axis: Optional[int] = None,
    keepdims: bool = False,
) -> "Array":
    """Sum of array elements, ignoring NaN values.

    Args:
        x: Input array.
        axis: Axis along which to compute the sum. If None, the array is
            flattened before computing.
        keepdims: If True, the reduced axes are retained as dimensions
            with size 1.

    Returns:
        Array containing the sum values. Returns NaN if all elements
        along the axis are NaN.
    """
    return _native_nansum(x, axis, keepdims)


def nanmean(
    x: "Array",
    axis: Optional[int] = None,
    keepdims: bool = False,
) -> "Array":
    """Mean of array elements, ignoring NaN values.

    Args:
        x: Input array.
        axis: Axis along which to compute the mean. If None, the array is
            flattened before computing.
        keepdims: If True, the reduced axes are retained as dimensions
            with size 1.

    Returns:
        Array containing the mean values. Returns NaN if all elements
        along the axis are NaN.
    """
    return _native_nanmean(x, axis, keepdims)


def nanmax(
    x: "Array",
    axis: Optional[int] = None,
    keepdims: bool = False,
) -> "Array":
    """Maximum of array elements, ignoring NaN values.

    Args:
        x: Input array.
        axis: Axis along which to compute the maximum. If None, the
            array is flattened before computing.
        keepdims: If True, the reduced axes are retained as dimensions
            with size 1.

    Returns:
        Array containing the maximum values. Returns NaN if all elements
        along the axis are NaN.
    """
    return _native_nanmax(x, axis, keepdims)


def nanmin(
    x: "Array",
    axis: Optional[int] = None,
    keepdims: bool = False,
) -> "Array":
    """Minimum of array elements, ignoring NaN values.

    Args:
        x: Input array.
        axis: Axis along which to compute the minimum. If None, the
            array is flattened before computing.
        keepdims: If True, the reduced axes are retained as dimensions
            with size 1.

    Returns:
        Array containing the minimum values. Returns NaN if all elements
        along the axis are NaN.
    """
    return _native_nanmin(x, axis, keepdims)


def nanstd(
    x: "Array",
    axis: Optional[int] = None,
    keepdims: bool = False,
    ddof: int = 0,
) -> "Array":
    """Standard deviation of array elements, ignoring NaN values.

    Args:
        x: Input array.
        axis: Axis along which to compute the standard deviation. If
            None, the array is flattened before computing.
        keepdims: If True, the reduced axes are retained as dimensions
            with size 1.
        ddof: Delta degrees of freedom. The divisor is ``N - ddof``,
            where ``N`` is the number of non-NaN elements.

    Returns:
        Array containing the standard deviation values. Returns NaN if
        all elements along the axis are NaN.
    """
    return _native_nanstd(x, axis, keepdims, ddof)


def nanvar(
    x: "Array",
    axis: Optional[int] = None,
    keepdims: bool = False,
    ddof: int = 0,
) -> "Array":
    """Variance of array elements, ignoring NaN values.

    Args:
        x: Input array.
        axis: Axis along which to compute the variance. If None, the
            array is flattened before computing.
        keepdims: If True, the reduced axes are retained as dimensions
            with size 1.
        ddof: Delta degrees of freedom. The divisor is ``N - ddof``,
            where ``N`` is the number of non-NaN elements.

    Returns:
        Array containing the variance values. Returns NaN if all
        elements along the axis are NaN.
    """
    return _native_nanvar(x, axis, keepdims, ddof)


def nanmedian(x: "Array", axis: Optional[int] = None, keepdims: bool = False) -> "Array":
    return _native_nanmedian(x, axis, keepdims)


def nanquantile(x: "Array", q, axis: Optional[int] = None, keepdims: bool = False) -> "Array":
    return _native_nanquantile(x, q, axis, keepdims)


def bincount(x: "Array", weights: Optional["Array"] = None, minlength: int = 0) -> "Array":
    return _native_bincount(x, weights, minlength)


__all__ = [
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
    "nanmedian",
    "nanquantile",
    "bincount",
]
