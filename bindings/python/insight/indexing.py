"""Indexing and sorting operations.

Functions:
    take, nonzero, flatnonzero, argsort, sort
    masked_select, searchsorted, unique, topk
    gather, scatter, scatter_add, scatter_reduce
    interp, indices, ix_

Types:
    UniqueResult — Result of unique() with values, indices, inverse, counts
"""

from __future__ import annotations

from typing import List, Optional, Union

from insight._insight import Array, Shape  # noqa: F401
from insight._insight import (
    take as _native_take,
    take_along_axis as _native_take_along_axis,
    put as _native_put,
    put_along_axis as _native_put_along_axis,
    compress as _native_compress,
    partition as _native_partition,
    argpartition as _native_argpartition,
    lexsort as _native_lexsort,
    nonzero as _native_nonzero,
    flatnonzero as _native_flatnonzero,
    argsort as _native_argsort,
    sort as _native_sort,
    masked_select as _native_masked_select,
    searchsorted as _native_searchsorted,
    unique as _native_unique,
    UniqueResult,
    topk as _native_topk,
    gather as _native_gather,
    scatter as _native_scatter,
    scatter_add as _native_scatter_add,
    scatter_reduce as _native_scatter_reduce,
    interp as _native_interp,
    indices as _native_indices,
    ix_ as _native_ix_,
)


def _to_shape(s: Union[list, tuple, Shape]) -> Shape:
    """Convert a list/tuple to Shape if needed."""
    if isinstance(s, Shape):
        return s
    return Shape(s)


def take(
    x: "Array",
    indices: "Array",
    axis: Optional[int] = None,
) -> "Array":
    """Take elements from an array at the given indices.

    Args:
        x: Input array.
        indices: Array of indices to take.
        axis: Axis along which to take elements. If None, the input
            array is flattened first.

    Returns:
        Array containing the taken elements.
    """
    return _native_take(x, indices, axis)


def take_along_axis(x: "Array", indices: "Array", axis: int) -> "Array":
    return _native_take_along_axis(x, indices, axis)


def put(x: "Array", indices: "Array", values: "Array", axis=None) -> "Array":
    return _native_put(x, indices, values, axis)


def put_along_axis(x: "Array", indices: "Array", values: "Array", axis: int) -> "Array":
    return _native_put_along_axis(x, indices, values, axis)


def compress(x: "Array", condition: "Array", axis=None) -> "Array":
    return _native_compress(x, condition, axis)


def partition(x: "Array", kth: int, axis: int = -1) -> "Array":
    return _native_partition(x, kth, axis)


def argpartition(x: "Array", kth: int, axis: int = -1) -> "Array":
    return _native_argpartition(x, kth, axis)


def lexsort(keys: list, axis: int = -1) -> "Array":
    return _native_lexsort(keys, axis)


def nonzero(x: "Array") -> "Array":
    """Return the indices of non-zero elements.

    Args:
        x: Input array.

    Returns:
        Array of shape ``(ndim, n)`` where ``n`` is the number of
        non-zero elements. Each row contains the indices along the
        corresponding axis.
    """
    return _native_nonzero(x)


def flatnonzero(x: "Array") -> "Array":
    """Return the flattened indices of non-zero elements.

    Args:
        x: Input array.

    Returns:
        1-D array of indices (into the flattened array) where the
        elements are non-zero.
    """
    return _native_flatnonzero(x)


def argsort(
    x: "Array",
    axis: int = -1,
    descending: bool = False,
) -> "Array":
    """Return the indices that would sort an array.

    Args:
        x: Input array.
        axis: Axis along which to sort (default: -1, the last axis).
        descending: If True, sort in descending order.

    Returns:
        Array of indices (dtype: int64) that would sort the input.
    """
    return _native_argsort(x, axis, descending)


def sort(
    x: "Array",
    axis: int = -1,
    descending: bool = False,
) -> "Array":
    """Sort an array along the given axis.

    Args:
        x: Input array.
        axis: Axis along which to sort (default: -1, the last axis).
        descending: If True, sort in descending order.

    Returns:
        Sorted array.
    """
    return _native_sort(x, axis, descending)


def masked_select(x: "Array", mask: "Array") -> "Array":
    """Select elements where the mask is True.

    Args:
        x: Input array.
        mask: Boolean mask array (broadcastable to the shape of ``x``).

    Returns:
        1-D array of selected elements.
    """
    return _native_masked_select(x, mask)


def searchsorted(
    x: "Array",
    v: "Array",
    side: str = "left",
) -> "Array":
    """Find indices where elements should be inserted to maintain order.

    Args:
        x: Sorted 1-D input array.
        v: Values to search for.
        side: If ``'left'`` (default), returns the first suitable
            insertion index. If ``'right'``, returns the last.

    Returns:
        Array of insertion indices (same shape as ``v``).
    """
    return _native_searchsorted(x, v, side)


def unique(
    x: "Array",
    return_indices: bool = False,
    return_inverse: bool = False,
    return_counts: bool = False,
) -> "UniqueResult":
    """Return the unique elements of an array.

    Args:
        x: Input array.
        return_indices: If True, also return the indices of the first
            occurrence of each unique value.
        return_inverse: If True, also return the indices to reconstruct
            the original array from the unique values.
        return_counts: If True, also return the count of each unique
            value.

    Returns:
        A ``UniqueResult`` object with attributes:
            - ``unique``: Array of unique elements.
            - ``indices``: (optional) Indices of first occurrences.
            - ``inverse``: (optional) Indices to reconstruct input.
            - ``counts``: (optional) Count of each unique element.
    """
    return _native_unique(x, return_indices, return_inverse, return_counts)


def topk(
    x: "Array",
    k: int,
    axis: int = -1,
    largest: bool = True,
    sorted: bool = True,
) -> tuple:
    """Return the k largest (or smallest) elements and their indices.

    Args:
        x: Input array.
        k: Number of top elements to return.
        axis: Axis along which to find the top elements (default: -1).
        largest: If True (default), return the largest elements. If
            False, return the smallest.
        sorted: If True (default), return the results in sorted order.

    Returns:
        Tuple of ``(values, indices)``.
    """
    return _native_topk(x, k, axis, largest, sorted)


def gather(
    x: "Array",
    index: "Array",
    dim: int,
) -> "Array":
    """Gather values along an axis using indices.

    Collects values from ``x`` at the positions specified by ``index``
    along the given dimension.

    Args:
        x: Input array.
        index: Array of indices (must have the same number of
            dimensions as ``x``).
        dim: Axis along which to gather.

    Returns:
        Array with the same shape as ``index``, containing the
        gathered values.
    """
    return _native_gather(x, dim, index)


def scatter(
    x: "Array",
    index: "Array",
    src: "Array",
    dim: int,
) -> "Array":
    """Scatter values into an array at the given indices.

    Writes values from ``src`` into the output at positions specified
    by ``index`` along dimension ``dim``.

    Args:
        x: Input array (defines the output shape).
        index: Array of indices.
        src: Source values to scatter.
        dim: Axis along which to scatter.

    Returns:
        Array with the same shape as ``x``, with values from ``src``
        placed at the specified indices.
    """
    return _native_scatter(x, dim, index, src)


def scatter_add(
    x: "Array",
    index: "Array",
    src: "Array",
    dim: int,
) -> "Array":
    """Scatter-add values into an array at the given indices.

    Accumulates (adds) values from ``src`` into the output at positions
    specified by ``index`` along dimension ``dim``.

    Args:
        x: Input array (defines the output shape).
        index: Array of indices.
        src: Source values to add.
        dim: Axis along which to scatter.

    Returns:
        Array with the same shape as ``x``, with values from ``src``
        accumulated at the specified indices.
    """
    return _native_scatter_add(x, dim, index, src)


def scatter_reduce(
    x: "Array",
    index: "Array",
    src: "Array",
    reduce: str = "replace",
    dim: int = 0,
) -> "Array":
    """Scatter values into an array with a reduction operation.

    Args:
        x: Input array (defines the output shape).
        index: Array of indices.
        src: Source values.
        reduce: Reduction mode. One of ``'replace'``, ``'add'``,
            ``'mul'``, ``'max'``, ``'min'``.
        dim: Axis along which to scatter.

    Returns:
        Array with the same shape as ``x``, with values from ``src``
        reduced at the specified indices.
    """
    return _native_scatter_reduce(x, dim, index, src, reduce)


def interp(
    x: "Array",
    xp: "Array",
    fp: "Array",
    left: Optional[float] = None,
    right: Optional[float] = None,
) -> "Array":
    """One-dimensional linear interpolation.

    Returns the interpolated values at ``x`` given data points
    ``(xp, fp)``.

    Args:
        x: The x-coordinates at which to evaluate the interpolated
            values.
        xp: The x-coordinates of the data points (must be increasing).
        fp: The y-coordinates of the data points (same length as
            ``xp``).
        left: Value to return for ``x < xp[0]``. If None, defaults to
            ``fp[0]``.
        right: Value to return for ``x > xp[-1]``. If None, defaults
            to ``fp[-1]``.

    Returns:
        Interpolated values, same shape as ``x``.
    """
    return _native_interp(x, xp, fp, left, right)


def indices(shape: "Shape", sparse: bool = False) -> "Array":
    """Return an array representing a grid of indices.

    Args:
        shape: Shape of the grid.
        sparse: If True, return a list of 1-D arrays (one per
            dimension) instead of a single stacked array.

    Returns:
        Array of shape ``(ndim, *shape)`` where the first dimension
        indexes the coordinate axis.
    """
    return _native_indices(_to_shape(shape), sparse)


def ix_(arrays: List["Array"]) -> List["Array"]:
    """Construct an open mesh from multiple sequences.

    Given 1-D arrays, returns broadcastable arrays that can be used
    for advanced indexing.

    Args:
        arrays: List of 1-D arrays.

    Returns:
        List of arrays, each with ``ndim == len(arrays)``, suitable for
        use in advanced indexing expressions.
    """
    return _native_ix_(arrays)


__all__ = [
    "take",
    "take_along_axis",
    "put",
    "put_along_axis",
    "compress",
    "partition",
    "argpartition",
    "lexsort",
    "flatnonzero",
    "argsort",
    "sort",
    "masked_select",
    "searchsorted",
    "unique",
    "UniqueResult",
    "topk",
    "gather",
    "scatter",
    "scatter_add",
    "scatter_reduce",
    "interp",
    "indices",
    "ix_",
]
