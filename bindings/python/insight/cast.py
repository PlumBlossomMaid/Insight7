"""Type casting.

Functions:
    cast(x, dtype) — Cast array to a different dtype
"""

from __future__ import annotations

from insight._insight import Array, DType  # noqa: F401
from insight._insight import cast as _native_cast
from insight._insight import cast_like as _native_cast_like


def cast(x: "Array", dtype: "DType", copy: bool = True) -> "Array":
    """Cast an array to a different data type.

    Converts each element of the input array to the target data type.
    The shape and device of the output are the same as the input.

    Args:
        x: Input array.
        dtype: Target data type (e.g., ``ins.float32``, ``ins.int64``).
        copy: If True (default), always returns a new array. If False
            and the dtype already matches, returns the input array
            itself.

    Returns:
        Array with the target dtype, same shape and device as the input.
    """
    return _native_cast(x, dtype, copy)


def cast_like(input: "Array", other: "Array", copy: bool = True) -> "Array":
    """Cast ``input`` to the dtype of ``other``."""
    return _native_cast_like(input, other, copy)


__all__ = ["cast", "cast_like"]
