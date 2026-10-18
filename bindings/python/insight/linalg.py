"""Linear algebra operations.

Functions:
    matmul, dot, inner, outer, det, slogdet, inv, pinv
    solve, lstsq, svd, svdvals, eigh, eigvalsh, eig, eigvals
    cholesky, qr, lq, lu, norm, cond, matrix_rank
    matrix_power, trace, cov
"""

from __future__ import annotations

from typing import Optional

from insight._insight import Array  # noqa: F401
from insight._insight import (
    matmul as _native_matmul,
    dot as _native_dot,
    inner as _native_inner,
    outer as _native_outer,
    det as _native_det,
    slogdet as _native_slogdet,
    inv as _native_inv,
    pinv as _native_pinv,
    solve as _native_solve,
    cholesky_solve as _native_cholesky_solve,
    solve_triangular as _native_solve_triangular,
    lstsq as _native_lstsq,
    svd as _native_svd,
    svdvals as _native_svdvals,
    eigh as _native_eigh,
    eigvalsh as _native_eigvalsh,
    eig as _native_eig,
    eigvals as _native_eigvals,
    cholesky as _native_cholesky,
    qr as _native_qr,
    lq as _native_lq,
    lu as _native_lu,
    norm as _native_norm,
    cond as _native_cond,
    matrix_rank as _native_matrix_rank,
    matrix_power as _native_matrix_power,
    trace as _native_trace,
    cov as _native_cov,
)


def matmul(a: "Array", b: "Array") -> "Array":
    """Matrix multiplication.

    Computes the matrix product of ``a`` and ``b``. Both inputs must be
    2-D arrays (or batched matrices with compatible leading dimensions).

    Args:
        a: First input matrix (2-D).
        b: Second input matrix (2-D). The number of rows must equal the
            number of columns in ``a``.

    Returns:
        Array containing the matrix product.
    """
    return _native_matmul(a, b)


def dot(a: "Array", b: "Array") -> "Array":
    """Dot product of two 1-D arrays.

    Computes the scalar dot product of two vectors.

    Args:
        a: First input vector (1-D).
        b: Second input vector (1-D). Must have the same length as
            ``a``.

    Returns:
        Scalar array containing the dot product.
    """
    return _native_dot(a, b)


def inner(a: "Array", b: "Array") -> "Array":
    """Inner product of two 1-D arrays.

    Equivalent to ``dot(a, b)`` for 1-D inputs.

    Args:
        a: First input vector (1-D).
        b: Second input vector (1-D).

    Returns:
        Scalar array containing the inner product.
    """
    return _native_inner(a, b)


def outer(a: "Array", b: "Array") -> "Array":
    """Outer product of two 1-D arrays.

    Computes the outer product ``a[i] * b[j]`` for all pairs.

    Args:
        a: First input vector (1-D).
        b: Second input vector (1-D).

    Returns:
        2-D array of shape ``(len(a), len(b))`` containing the outer
        product.
    """
    return _native_outer(a, b)


def det(x: "Array") -> "Array":
    """Determinant of a square matrix.

    Args:
        x: Input square matrix (2-D).

    Returns:
        Scalar array containing the determinant value.
    """
    return _native_det(x)


def slogdet(x: "Array") -> tuple:
    """Sign and natural log of the absolute value of the determinant.

    This is useful when the determinant is too large or too small to
    represent directly. The determinant can be recovered as
    ``sign * exp(logabsdet)``.

    Args:
        x: Input square matrix (2-D).

    Returns:
        Tuple of ``(sign, logabsdet)`` where ``sign`` is -1, 0, or 1
        and ``logabsdet`` is the natural log of the absolute value of
        the determinant.
    """
    return _native_slogdet(x)


def inv(x: "Array") -> "Array":
    """Inverse of a square matrix.

    Args:
        x: Input non-singular square matrix (2-D).

    Returns:
        2-D array containing the inverse matrix.
    """
    return _native_inv(x)


def pinv(x: "Array", rcond: float = -1.0) -> "Array":
    """Pseudo-inverse (Moore-Penrose inverse) of a matrix.

    Computed via singular value decomposition.

    Args:
        x: Input matrix (2-D).
        rcond: Cutoff for small singular values. Singular values smaller
            than ``rcond * max(singular_values)`` are set to zero. If
            negative, defaults to ``max(shape) * eps``.

    Returns:
        2-D array containing the pseudo-inverse.
    """
    return _native_pinv(x, rcond)


def solve(a: "Array", b: "Array") -> "Array":
    """Solve a linear system ``AX = B``.

    Finds the exact solution of the system of linear equations
    ``a @ x = b``.

    Args:
        a: Coefficient matrix (2-D, square).
        b: Right-hand side (1-D or 2-D).

    Returns:
        Array containing the solution ``X``.
    """
    return _native_solve(a, b)


def cholesky_solve(A: "Array", B: "Array", lower: bool = True) -> "Array":
    return _native_cholesky_solve(A, B, lower)


def solve_triangular(A: "Array", B: "Array", lower: bool = True, unit_diag: bool = False) -> "Array":
    return _native_solve_triangular(A, B, lower, unit_diag)


def lstsq(a: "Array", b: "Array", rcond: float = -1.0) -> "Array":
    """Least-squares solution to ``AX = B``.

    Computes the vector ``x`` that approximately solves ``a @ x = b``
    by minimizing the Euclidean norm ``||a @ x - b||``.

    Args:
        a: Coefficient matrix (2-D, m x n).
        b: Right-hand side (1-D or 2-D).
        rcond: Cutoff for small singular values. If negative, defaults
            to ``max(shape) * eps``.

    Returns:
        Array containing the least-squares solution.
    """
    return _native_lstsq(a, b, rcond)


def svd(x: "Array", full_matrices: bool = True) -> tuple:
    """Singular value decomposition.

    Factors the matrix as ``U @ diag(S) @ V^T``.

    Args:
        x: Input matrix (2-D).
        full_matrices: If True, ``U`` and ``V^T`` have the shapes
            ``(m, m)`` and ``(n, n)``. If False, the shapes are
            ``(m, k)`` and ``(k, n)`` where ``k = min(m, n)``.

    Returns:
        Tuple of ``(U, S, V_T)`` where ``S`` contains the singular
        values in descending order.
    """
    return _native_svd(x, full_matrices)


def svdvals(x: "Array") -> "Array":
    """Singular values of a matrix.

    Args:
        x: Input matrix (2-D).

    Returns:
        1-D array of singular values in descending order.
    """
    return _native_svdvals(x)


def eigh(x: "Array", uplo: str = "L") -> tuple:
    """Eigenvalues and eigenvectors of a symmetric (Hermitian) matrix.

    Args:
        x: Input symmetric or Hermitian matrix (2-D).
        uplo: Which triangle to use. ``'L'`` (default) uses the lower
            triangle; ``'U'`` uses the upper triangle.

    Returns:
        Tuple of ``(eigenvalues, eigenvectors)`` where eigenvalues are
        in ascending order and eigenvectors are columns of the returned
        matrix.
    """
    return _native_eigh(x, uplo)


def eigvalsh(x: "Array", uplo: str = "L") -> "Array":
    """Eigenvalues of a symmetric (Hermitian) matrix.

    Args:
        x: Input symmetric or Hermitian matrix (2-D).
        uplo: Which triangle to use. ``'L'`` (default) uses the lower
            triangle; ``'U'`` uses the upper triangle.

    Returns:
        1-D array of eigenvalues in ascending order.
    """
    return _native_eigvalsh(x, uplo)


def eig(x: "Array") -> tuple:
    """Eigenvalues and eigenvectors of a general matrix.

    Args:
        x: Input square matrix (2-D).

    Returns:
        Tuple of ``(eigenvalues, eigenvectors)``.
    """
    return _native_eig(x)


def eigvals(x: "Array") -> "Array":
    """Eigenvalues of a general matrix.

    Args:
        x: Input square matrix (2-D).

    Returns:
        1-D array of eigenvalues.
    """
    return _native_eigvals(x)


def cholesky(x: "Array", lower: bool = True) -> "Array":
    """Cholesky decomposition of a symmetric positive-definite matrix.

    Factors the matrix as ``L @ L^T`` (lower) or ``U^T @ U`` (upper).

    Args:
        x: Input symmetric positive-definite matrix (2-D).
        lower: If True (default), returns the lower-triangular
            Cholesky factor ``L``. If False, returns the upper-
            triangular factor ``U``.

    Returns:
        2-D array containing the Cholesky factor.
    """
    return _native_cholesky(x, lower)


def qr(x: "Array", mode: str = "reduced") -> tuple:
    """QR decomposition.

    Factors the matrix as ``Q @ R`` where ``Q`` is orthogonal and ``R``
    is upper-triangular.

    Args:
        x: Input matrix (2-D).
        mode: ``'reduced'`` (default) returns ``Q`` of shape
            ``(m, k)`` and ``R`` of shape ``(k, n)`` where
            ``k = min(m, n)``. ``'complete'`` returns ``Q`` of shape
            ``(m, m)`` and ``R`` of shape ``(m, n)``.

    Returns:
        Tuple of ``(Q, R)``.
    """
    return _native_qr(x, mode)


def lq(x: "Array", mode: str = "reduced") -> tuple:
    """LQ decomposition.

    Factors the matrix as ``L @ Q`` where ``L`` is lower-triangular
    and ``Q`` is orthogonal.

    Args:
        x: Input matrix (2-D).
        mode: ``'reduced'`` (default) returns ``L`` of shape
            ``(m, k)`` and ``Q`` of shape ``(k, n)`` where
            ``k = min(m, n)``. ``'complete'`` returns full factors.

    Returns:
        Tuple of ``(L, Q)``.
    """
    return _native_lq(x, mode)


def lu(x: "Array", pivot: bool = True) -> tuple:
    """LU decomposition.

    Factors the matrix as ``P @ L @ U`` (with pivoting) or ``L @ U``
    (without pivoting).

    Args:
        x: Input matrix (2-D).
        pivot: If True (default), perform partial pivoting and return
            pivot indices alongside the LU factors.

    Returns:
        Tuple of ``(LU, pivots)``. ``LU`` contains both ``L`` and ``U``
        packed into a single matrix.
    """
    return _native_lu(x, pivot)


def norm(x: "Array", ord: float = 2.0) -> "Array":
    """Matrix or vector norm.

    Args:
        x: Input array (1-D or 2-D).
        ord: Order of the norm. For vectors: 1, 2 (default), inf, etc.
            For matrices: 1, 2 (default), inf, ``'fro'``, etc.

    Returns:
        Scalar array containing the norm value.
    """
    return _native_norm(x, ord)


def cond(x: "Array", p: float = 2.0) -> "Array":
    """Condition number of a matrix.

    Computes the condition number with respect to the given norm.

    Args:
        x: Input matrix (2-D).
        p: Norm type: ``1``, ``2`` (default), ``inf``, or ``'fro'``.

    Returns:
        Scalar array containing the condition number.
    """
    return _native_cond(x, p)


def matrix_rank(x: "Array", tol: float = -1.0) -> "Array":
    """Matrix rank computed via SVD.

    Args:
        x: Input matrix (2-D).
        tol: Tolerance below which singular values are considered zero.
            If negative, defaults to ``max(shape) * eps * max(s)``.

    Returns:
        Scalar array (int64) containing the rank.
    """
    return _native_matrix_rank(x, tol)


def matrix_power(x: "Array", n: int) -> "Array":
    """Raise a square matrix to an integer power.

    Computes ``x^n`` by repeated matrix multiplication.

    Args:
        x: Input square matrix (2-D).
        n: Non-negative integer exponent.

    Returns:
        2-D array containing ``x`` raised to the ``n``-th power.
    """
    return _native_matrix_power(x, n)


def trace(x: "Array") -> "Array":
    """Trace of a matrix (sum of diagonal elements).

    Args:
        x: Input matrix (2-D).

    Returns:
        Scalar array containing the sum of the diagonal elements.
    """
    return _native_trace(x)


def cov(
    x: "Array",
    rowvar: bool = True,
    ddof: int = 1,
) -> "Array":
    """Covariance matrix.

    Args:
        x: Input array (2-D). If ``rowvar`` is True, each row
            represents a variable and each column an observation.
        rowvar: If True (default), each row is a variable. If False,
            each column is a variable (the input is transposed
            internally).
        ddof: Delta degrees of freedom. The divisor is ``N - ddof``
            where ``N`` is the number of observations. Default is 1
            (sample covariance).

    Returns:
        2-D square covariance matrix.
    """
    return _native_cov(x, rowvar, ddof)


__all__ = [
    "matmul",
    "dot",
    "inner",
    "outer",
    "det",
    "slogdet",
    "inv",
    "pinv",
    "solve",
    "cholesky_solve",
    "solve_triangular",
    "lstsq",
    "svd",
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
]
