"""Broadcasting helpers."""

from __future__ import annotations

from insight._insight import Array, Shape  # noqa: F401
from insight._insight import (
    broadcast_shape as _native_broadcast_shape,
    broadcast_to as _native_broadcast_to,
    broadcast_arrays as _native_broadcast_arrays,
)


def broadcast_shape(a: "Shape", b: "Shape") -> "Shape":
    return _native_broadcast_shape(a, b)


def broadcast_to(x: "Array", shape) -> "Array":
    if isinstance(shape, (list, tuple)):
        shape = Shape(shape)
    return _native_broadcast_to(x, shape)


def broadcast_arrays(arrays: list) -> list:
    return _native_broadcast_arrays(arrays)


__all__ = ["broadcast_shape", "broadcast_to", "broadcast_arrays"]
