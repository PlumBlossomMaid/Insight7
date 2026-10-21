"""Elementwise binary operations.

Functions:
    add(a, b), sub(a, b), mul(a, b), div(a, b), pow(a, b), mod(a, b)
    maximum(a, b), minimum(a, b)
    equal(a, b), not_equal(a, b), greater(a, b), less(a, b)
    greater_equal(a, b), less_equal(a, b)
    logical_and(a, b), logical_or(a, b), logical_xor(a, b)
    logical_not(x)
"""

from __future__ import annotations

from insight._insight import Array  # noqa: F401
from insight._insight import (
    add as _native_add,
    sub as _native_sub,
    mul as _native_mul,
    div as _native_div,
    pow as _native_pow,
    mod as _native_mod,
    maximum as _native_maximum,
    minimum as _native_minimum,
    equal as _native_equal,
    not_equal as _native_not_equal,
    greater as _native_greater,
    less as _native_less,
    greater_equal as _native_greater_equal,
    less_equal as _native_less_equal,
    logical_and as _native_logical_and,
    logical_or as _native_logical_or,
    logical_xor as _native_logical_xor,
    logical_not as _native_logical_not,
    bitwise_and as _native_bitwise_and,
    bitwise_or as _native_bitwise_or,
    bitwise_xor as _native_bitwise_xor,
    greater_than as _native_greater_than,
    less_than as _native_less_than,
    bitwise_left_shift as _native_bitwise_left_shift,
    bitwise_right_shift as _native_bitwise_right_shift,
    bitwise_not as _native_bitwise_not,
)


def add(a: "Array", b: "Array") -> "Array":
    """Element-wise addition of two arrays.

    Computes ``a + b`` with broadcasting support.

    Args:
        a: First input array.
        b: Second input array.

    Returns:
        Array containing the element-wise sum.
    """
    return _native_add(a, b)


def sub(a: "Array", b: "Array") -> "Array":
    """Element-wise subtraction of two arrays.

    Computes ``a - b`` with broadcasting support.

    Args:
        a: First input array.
        b: Second input array.

    Returns:
        Array containing the element-wise difference.
    """
    return _native_sub(a, b)


def mul(a: "Array", b: "Array") -> "Array":
    """Element-wise multiplication of two arrays.

    Computes ``a * b`` with broadcasting support.

    Args:
        a: First input array.
        b: Second input array.

    Returns:
        Array containing the element-wise product.
    """
    return _native_mul(a, b)


def div(a: "Array", b: "Array") -> "Array":
    """Element-wise division of two arrays.

    Computes ``a / b`` with broadcasting support.

    Args:
        a: First input array (numerator).
        b: Second input array (denominator).

    Returns:
        Array containing the element-wise quotient.
    """
    return _native_div(a, b)


def pow(a: "Array", b: "Array") -> "Array":
    """Element-wise exponentiation of two arrays.

    Computes ``a ** b`` with broadcasting support.

    Args:
        a: First input array (base).
        b: Second input array (exponent).

    Returns:
        Array containing ``a`` raised to the power of ``b`` element-wise.
    """
    return _native_pow(a, b)


def mod(a: "Array", b: "Array") -> "Array":
    """Element-wise modulo of two arrays.

    Computes ``a % b`` with broadcasting support.

    Args:
        a: First input array (dividend).
        b: Second input array (divisor).

    Returns:
        Array containing the element-wise remainder.
    """
    return _native_mod(a, b)


def maximum(a: "Array", b: "Array") -> "Array":
    """Element-wise maximum of two arrays.

    Returns the larger of each pair of corresponding elements from ``a``
    and ``b``, with broadcasting support.

    Args:
        a: First input array.
        b: Second input array.

    Returns:
        Array containing the element-wise maximum values.
    """
    return _native_maximum(a, b)


def minimum(a: "Array", b: "Array") -> "Array":
    """Element-wise minimum of two arrays.

    Returns the smaller of each pair of corresponding elements from ``a``
    and ``b``, with broadcasting support.

    Args:
        a: First input array.
        b: Second input array.

    Returns:
        Array containing the element-wise minimum values.
    """
    return _native_minimum(a, b)


def equal(a: "Array", b: "Array") -> "Array":
    """Element-wise equality comparison.

    Computes ``a == b`` with broadcasting support.

    Args:
        a: First input array.
        b: Second input array.

    Returns:
        Boolean array where each element is True if the corresponding
        elements are equal.
    """
    return _native_equal(a, b)


def not_equal(a: "Array", b: "Array") -> "Array":
    """Element-wise inequality comparison.

    Computes ``a != b`` with broadcasting support.

    Args:
        a: First input array.
        b: Second input array.

    Returns:
        Boolean array where each element is True if the corresponding
        elements are not equal.
    """
    return _native_not_equal(a, b)


def greater(a: "Array", b: "Array") -> "Array":
    """Element-wise greater-than comparison.

    Computes ``a > b`` with broadcasting support.

    Args:
        a: First input array.
        b: Second input array.

    Returns:
        Boolean array where each element is True if the corresponding
        element of ``a`` is greater than that of ``b``.
    """
    return _native_greater(a, b)


def less(a: "Array", b: "Array") -> "Array":
    """Element-wise less-than comparison.

    Computes ``a < b`` with broadcasting support.

    Args:
        a: First input array.
        b: Second input array.

    Returns:
        Boolean array where each element is True if the corresponding
        element of ``a`` is less than that of ``b``.
    """
    return _native_less(a, b)


def greater_equal(a: "Array", b: "Array") -> "Array":
    """Element-wise greater-or-equal comparison.

    Computes ``a >= b`` with broadcasting support.

    Args:
        a: First input array.
        b: Second input array.

    Returns:
        Boolean array where each element is True if the corresponding
        element of ``a`` is greater than or equal to that of ``b``.
    """
    return _native_greater_equal(a, b)


def less_equal(a: "Array", b: "Array") -> "Array":
    """Element-wise less-or-equal comparison.

    Computes ``a <= b`` with broadcasting support.

    Args:
        a: First input array.
        b: Second input array.

    Returns:
        Boolean array where each element is True if the corresponding
        element of ``a`` is less than or equal to that of ``b``.
    """
    return _native_less_equal(a, b)


def logical_and(a: "Array", b: "Array") -> "Array":
    """Element-wise logical AND.

    Computes the logical AND of two boolean arrays with broadcasting
    support. Returns True where both ``a`` and ``b`` are non-zero.

    Args:
        a: First input array.
        b: Second input array.

    Returns:
        Boolean array containing the element-wise logical AND result.
    """
    return _native_logical_and(a, b)


def logical_or(a: "Array", b: "Array") -> "Array":
    """Element-wise logical OR.

    Computes the logical OR of two boolean arrays with broadcasting
    support. Returns True where either ``a`` or ``b`` is non-zero.

    Args:
        a: First input array.
        b: Second input array.

    Returns:
        Boolean array containing the element-wise logical OR result.
    """
    return _native_logical_or(a, b)


def logical_xor(a: "Array", b: "Array") -> "Array":
    """Element-wise logical exclusive OR.

    Computes the logical XOR of two boolean arrays with broadcasting
    support. Returns True where exactly one of ``a`` or ``b`` is non-zero.

    Args:
        a: First input array.
        b: Second input array.

    Returns:
        Boolean array containing the element-wise logical XOR result.
    """
    return _native_logical_xor(a, b)


def logical_not(x: "Array") -> "Array":
    """Element-wise logical NOT.

    Computes the logical negation of the input array. Returns True where
    the element is zero and False where it is non-zero.

    Args:
        x: Input array.

    Returns:
        Boolean array containing the element-wise logical NOT result.
    """
    return _native_logical_not(x)


def bitwise_and(a: "Array", b: "Array") -> "Array":
    """Element-wise bitwise AND.

    Computes ``a & b`` with broadcasting support. Both inputs must be
    integer or boolean arrays.

    Args:
        a: First input array.
        b: Second input array.

    Returns:
        Array containing the element-wise bitwise AND result.
    """
    return _native_bitwise_and(a, b)


def bitwise_or(a: "Array", b: "Array") -> "Array":
    """Element-wise bitwise OR.

    Computes ``a | b`` with broadcasting support. Both inputs must be
    integer or boolean arrays.

    Args:
        a: First input array.
        b: Second input array.

    Returns:
        Array containing the element-wise bitwise OR result.
    """
    return _native_bitwise_or(a, b)


def bitwise_xor(a: "Array", b: "Array") -> "Array":
    """Element-wise bitwise XOR.

    Computes ``a ^ b`` with broadcasting support. Both inputs must be
    integer or boolean arrays.

    Args:
        a: First input array.
        b: Second input array.

    Returns:
        Array containing the element-wise bitwise XOR result.
    """
    return _native_bitwise_xor(a, b)


def greater_than(a: "Array", b: "Array") -> "Array":
    return _native_greater_than(a, b)


def less_than(a: "Array", b: "Array") -> "Array":
    return _native_less_than(a, b)


def bitwise_left_shift(a: "Array", b: "Array") -> "Array":
    return _native_bitwise_left_shift(a, b)


def bitwise_right_shift(a: "Array", b: "Array") -> "Array":
    return _native_bitwise_right_shift(a, b)


def bitwise_not(x: "Array") -> "Array":
    return _native_bitwise_not(x)


__all__ = [
    "add",
    "sub",
    "div",
    "pow",
    "mod",
    "maximum",
    "minimum",
    "equal",
    "not_equal",
    "greater",
    "greater_than",
    "less",
    "less_than",
    "greater_equal",
    "less_equal",
    "logical_and",
    "logical_or",
    "logical_xor",
    "logical_not",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "bitwise_left_shift",
    "bitwise_right_shift",
    "bitwise_not",
]
