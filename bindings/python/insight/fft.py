"""FFT operations (ins::fft namespace).

Functions:
    fft, ifft, fft2, ifft2, fftn, ifftn
    rfft, irfft, rfft2, irfft2, rfftn, irfftn
    hfft, ihfft
    fftfreq, rfftfreq, fftshift, ifftshift, next_fast_len
"""

from __future__ import annotations

from typing import List, Optional

from insight._insight import Array  # noqa: F401
from insight._insight import (
    fft as _native_fft,
    ifft as _native_ifft,
    fft2 as _native_fft2,
    ifft2 as _native_ifft2,
    fftn as _native_fftn,
    ifftn as _native_ifftn,
    rfft as _native_rfft,
    irfft as _native_irfft,
    fftfreq as _native_fftfreq,
    rfftfreq as _native_rfftfreq,
    fftshift as _native_fftshift,
    ifftshift as _native_ifftshift,
    next_fast_len as _native_next_fast_len,
    rfft2 as _native_rfft2,
    irfft2 as _native_irfft2,
    rfftn as _native_rfftn,
    irfftn as _native_irfftn,
    hfft as _native_hfft,
    ihfft as _native_ihfft,
)


def fft(
    x: "Array",
    n: int = -1,
    axis: int = -1,
    norm: str = "backward",
) -> "Array":
    """1-D discrete Fourier transform.

    Computes the discrete Fourier transform along the specified axis.

    Args:
        x: Input array (real or complex).
        n: FFT length. If ``n`` is larger than the input length, the
            input is zero-padded. If smaller, the input is truncated.
            If -1 (default), uses the input length.
        axis: Axis along which to compute the FFT (default: -1).
        norm: Normalization mode. ``'backward'`` (default) applies no
            normalization on the forward transform and ``1/n`` on the
            inverse. ``'forward'`` applies ``1/n`` on the forward.
            ``'ortho'`` applies ``1/sqrt(n)`` in both directions.

    Returns:
        Complex array of the same shape (or adjusted for ``n``) with
        the frequency-domain representation.
    """
    return _native_fft(x, n, axis, norm)


def ifft(
    x: "Array",
    n: int = -1,
    axis: int = -1,
    norm: str = "backward",
) -> "Array":
    """1-D inverse discrete Fourier transform.

    Args:
        x: Input complex array.
        n: IFFT length. If -1 (default), uses the input length.
        axis: Axis along which to compute the IFFT (default: -1).
        norm: Normalization mode (same convention as ``fft``).

    Returns:
        Complex array with the time-domain representation.
    """
    return _native_ifft(x, n, axis, norm)


def rfft(
    x: "Array",
    n: int = -1,
    axis: int = -1,
    norm: str = "backward",
) -> "Array":
    """1-D real-input discrete Fourier transform.

    Computes the FFT of a real-valued input. The output contains only
    the non-negative frequency terms (the first half of the full FFT
    output), since the spectrum of real input is conjugate-symmetric.

    Args:
        x: Real-valued input array.
        n: FFT length. If -1 (default), uses the input length.
        axis: Axis along which to compute the FFT (default: -1).
        norm: Normalization mode (default: ``'backward'``).

    Returns:
        Complex array whose transformed axis has length ``n//2+1``.
    """
    return _native_rfft(x, n, axis, norm)


def irfft(
    x: "Array",
    n: int = -1,
    axis: int = -1,
    norm: str = "backward",
) -> "Array":
    """1-D inverse real-input discrete Fourier transform.

    Computes the inverse of ``rfft``. The input must be the
    half-spectrum produced by ``rfft``.

    Args:
        x: Complex input array (half-spectrum).
        n: Output length. If -1, uses ``2*(m-1)`` where ``m`` is the
            length of the transformed axis.
        axis: Axis along which to compute the IFFT (default: -1).
        norm: Normalization mode (default: ``'backward'``).

    Returns:
        Real-valued array.
    """
    return _native_irfft(x, n, axis, norm)


def fft2(
    x: "Array",
    s: Optional[List[int]] = None,
    axes: Optional[List[int]] = None,
    norm: str = "backward",
) -> "Array":
    """2-D discrete Fourier transform.

    Computes the 2-D FFT over the last two axes by default.

    Args:
        x: Input array (real or complex).
        s: Shape of the FFT (optional). Pads or truncates the
            specified axes to this size.
        axes: Axes over which to compute the 2-D FFT. Default is
            ``[-2, -1]``.
        norm: Normalization mode (default: ``'backward'``).

    Returns:
        Complex array with the 2-D frequency-domain representation.
    """
    if s is None:
        s = []
    if axes is None:
        axes = [-2, -1]
    return _native_fft2(x, s, axes, norm)


def ifft2(
    x: "Array",
    s: Optional[List[int]] = None,
    axes: Optional[List[int]] = None,
    norm: str = "backward",
) -> "Array":
    """2-D inverse discrete Fourier transform.

    Args:
        x: Input complex array.
        s: Shape of the IFFT (optional).
        axes: Axes over which to compute the 2-D IFFT. Default is
            ``[-2, -1]``.
        norm: Normalization mode (default: ``'backward'``).

    Returns:
        Complex array with the 2-D time-domain representation.
    """
    if s is None:
        s = []
    if axes is None:
        axes = [-2, -1]
    return _native_ifft2(x, s, axes, norm)


def rfft2(
    x: "Array",
    s: Optional[List[int]] = None,
    axes: Optional[List[int]] = None,
    norm: str = "backward",
) -> "Array":
    """2-D real-input discrete Fourier transform.

    Args:
        x: Real-valued input array.
        s: Shape of the FFT (optional).
        axes: Axes over which to compute the 2-D FFT. Default is
            ``[-2, -1]``.
        norm: Normalization mode (default: ``'backward'``).

    Returns:
        Complex array with the 2-D frequency-domain representation
        (non-negative frequencies only).
    """
    if s is None:
        s = []
    if axes is None:
        axes = [-2, -1]
    return _native_rfft2(x, s, axes, norm)


def irfft2(
    x: "Array",
    s: Optional[List[int]] = None,
    axes: Optional[List[int]] = None,
    norm: str = "backward",
) -> "Array":
    """2-D inverse real-input discrete Fourier transform.

    Args:
        x: Complex input array (half-spectrum).
        s: Output shape (optional).
        axes: Axes over which to compute the 2-D IFFT. Default is
            ``[-2, -1]``.
        norm: Normalization mode (default: ``'backward'``).

    Returns:
        Real-valued array with the 2-D time-domain representation.
    """
    if s is None:
        s = []
    if axes is None:
        axes = [-2, -1]
    return _native_irfft2(x, s, axes, norm)


def fftn(
    x: "Array",
    s: Optional[List[int]] = None,
    axes: Optional[List[int]] = None,
    norm: str = "backward",
) -> "Array":
    """N-dimensional discrete Fourier transform.

    Args:
        x: Input array (real or complex).
        s: Shape of the FFT along each axis (optional). Pads or
            truncates the specified axes to these sizes.
        axes: Axes over which to compute the FFT. Default is all axes.
        norm: Normalization mode (default: ``'backward'``).

    Returns:
        Complex array with the N-D frequency-domain representation.
    """
    if s is None:
        s = []
    if axes is None:
        axes = []
    return _native_fftn(x, s, axes, norm)


def ifftn(
    x: "Array",
    s: Optional[List[int]] = None,
    axes: Optional[List[int]] = None,
    norm: str = "backward",
) -> "Array":
    """N-dimensional inverse discrete Fourier transform.

    Args:
        x: Input complex array.
        s: Shape of the IFFT along each axis (optional).
        axes: Axes over which to compute the IFFT. Default is all axes.
        norm: Normalization mode (default: ``'backward'``).

    Returns:
        Complex array with the N-D time-domain representation.
    """
    if s is None:
        s = []
    if axes is None:
        axes = []
    return _native_ifftn(x, s, axes, norm)


def rfftn(
    x: "Array",
    s: Optional[List[int]] = None,
    axes: Optional[List[int]] = None,
    norm: str = "backward",
) -> "Array":
    """N-dimensional real-input discrete Fourier transform.

    Args:
        x: Real-valued input array.
        s: Shape of the FFT along each axis (optional).
        axes: Axes over which to compute the FFT. Default is all axes.
        norm: Normalization mode (default: ``'backward'``).

    Returns:
        Complex array with the N-D frequency-domain representation
        (non-negative frequencies only).
    """
    if s is None:
        s = []
    if axes is None:
        axes = []
    return _native_rfftn(x, s, axes, norm)


def irfftn(
    x: "Array",
    s: Optional[List[int]] = None,
    axes: Optional[List[int]] = None,
    norm: str = "backward",
) -> "Array":
    """N-dimensional inverse real-input discrete Fourier transform.

    Args:
        x: Complex input array (half-spectrum).
        s: Output shape along each axis (optional).
        axes: Axes over which to compute the IFFT. Default is all axes.
        norm: Normalization mode (default: ``'backward'``).

    Returns:
        Real-valued array with the N-D time-domain representation.
    """
    if s is None:
        s = []
    if axes is None:
        axes = []
    return _native_irfftn(x, s, axes, norm)


def hfft(
    x: "Array",
    n: int = -1,
    axis: int = -1,
    norm: str = "backward",
) -> "Array":
    """1-D FFT of a Hermitian-symmetric complex input."""
    return _native_hfft(x, n, axis, norm)


def ihfft(
    x: "Array",
    n: int = -1,
    axis: int = -1,
    norm: str = "backward",
) -> "Array":
    """1-D inverse Hermitian FFT from real input."""
    return _native_ihfft(x, n, axis, norm)


def fftfreq(n: int, d: float = 1.0) -> "Array":
    """Compute the discrete Fourier transform sample frequencies.

    Returns the frequency bins corresponding to the output of ``fft``.

    Args:
        n: Window length (number of samples).
        d: Sample spacing (inverse of the sampling rate). Default is 1.0.

    Returns:
        1-D array of length ``n`` containing the frequency bins.
    """
    return _native_fftfreq(n, d)


def rfftfreq(n: int, d: float = 1.0) -> "Array":
    """Compute the sample frequencies for ``rfft``.

    Returns the frequency bins corresponding to the output of ``rfft``
    (non-negative frequencies only).

    Args:
        n: Window length (number of samples).
        d: Sample spacing (inverse of the sampling rate). Default is 1.0.

    Returns:
        1-D array of length ``n//2+1`` containing the non-negative
        frequency bins.
    """
    return _native_rfftfreq(n, d)


def fftshift(x: "Array", axis: int = -1) -> "Array":
    """Shift the zero-frequency component to the center of the spectrum.

    Swaps the two halves of the FFT output so that the zero-frequency
    component is in the middle.

    Args:
        x: Input array (typically the output of ``fft``).
        axis: Axis along which to shift (default: -1).

    Returns:
        Shifted array (view).
    """
    return _native_fftshift(x, axis)


def ifftshift(x: "Array", axis: int = -1) -> "Array":
    """Inverse of ``fftshift``.

    Shifts the zero-frequency component back to the beginning of the
    array.

    Args:
        x: Input array (typically the output of ``fftshift``).
        axis: Axis along which to shift (default: -1).

    Returns:
        Shifted array (view).
    """
    return _native_ifftshift(x, axis)


def next_fast_len(target: int) -> int:
    """Find the smallest fast FFT length greater than or equal to target.

    Returns a number that is highly composite (e.g., products of small
    primes), which allows FFT algorithms to run most efficiently.

    Args:
        target: Desired minimum FFT length.

    Returns:
        Integer >= ``target`` that is efficient for FFT computation.
    """
    return _native_next_fast_len(target)


__all__ = [
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
    "fftshift",
    "ifftshift",
    "next_fast_len",
    "rfft2",
    "irfft2",
    "rfftn",
    "irfftn",
    "hfft",
    "ihfft",
]
