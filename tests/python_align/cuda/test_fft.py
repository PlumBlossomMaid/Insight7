"""FFT alignment tests — Insight CUDA vs NumPy.

Run with:
    PYTHONPATH=bindings/python:build/bindings/python \\
    LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda \\
    python -m pytest tests/python_align/cuda/test_fft.py -v
"""

import sys
import os
import pytest

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
sys.path.insert(0, os.path.join(_root, "bindings", "python"))
sys.path.insert(0, os.path.join(_root, "build", "bindings", "python"))

try:
    import insight as ins
    import numpy as np
    from numpy.testing import assert_allclose
except ImportError:
    pytest.skip("Insight or NumPy not available", allow_module_level=True)

# Skip if CUDA not available
try:
    ins.init()
except Exception:
    pytest.skip("GPU backend not available", allow_module_level=True)

GPU = ins.GPUPlace(0)
CPU = ins.CPUPlace()


def to_gpu(np_arr):
    return ins.from_numpy(np_arr).to(GPU)


def to_numpy(gpu_arr):
    return gpu_arr.to(CPU).numpy()


class TestFFTAlignmentGPU:
    """Insight FFT on GPU vs NumPy FFT."""

    def test_fft(self):
        x_np = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float64)
        x = to_gpu(x_np)
        result = ins.fft(x)
        assert_allclose(to_numpy(result), np.fft.fft(x_np), atol=1e-6)

    def test_ifft(self):
        x_np = np.array([1, 2, 3, 4], dtype=np.complex128)
        x = to_gpu(x_np)
        result = ins.ifft(x)
        assert_allclose(to_numpy(result), np.fft.ifft(x_np), atol=1e-6)

    def test_fft_ifft_roundtrip(self):
        x_np = np.random.randn(16).astype(np.float64)
        x = to_gpu(x_np)
        result = ins.ifft(ins.fft(x))
        assert_allclose(to_numpy(result).real, x_np, atol=1e-6)

    def test_rfft(self):
        x_np = np.array([1, 2, 3, 4], dtype=np.float64)
        x = to_gpu(x_np)
        result = ins.rfft(x)
        assert_allclose(to_numpy(result), np.fft.rfft(x_np), atol=1e-6)

    def test_fftfreq(self):
        result = ins.fftfreq(8, 0.1)
        assert_allclose(result.numpy(), np.fft.fftfreq(8, 0.1))

    def test_rfftfreq(self):
        result = ins.rfftfreq(8, 0.1)
        assert_allclose(result.numpy(), np.fft.rfftfreq(8, 0.1))


class TestFFTExtendedGPU:
    """Extended FFT ops on GPU vs NumPy FFT."""

    def test_fft2(self):
        x_np = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
        x = to_gpu(x_np)
        result = ins.fft2(x)
        assert_allclose(to_numpy(result), np.fft.fft2(x_np), atol=1e-6)

    def test_ifft2(self):
        x_np = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.complex128)
        x = to_gpu(x_np)
        result = ins.ifft2(x)
        assert_allclose(to_numpy(result), np.fft.ifft2(x_np), atol=1e-6)

    def test_fftn(self):
        x_np = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        x = to_gpu(x_np)
        result = ins.fftn(x)
        assert_allclose(to_numpy(result), np.fft.fftn(x_np), atol=1e-6)

    def test_ifftn(self):
        x_np = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        x = to_gpu(x_np)
        result = ins.ifftn(x)
        assert_allclose(to_numpy(result), np.fft.ifftn(x_np), atol=1e-6)

    def test_rfft_irfft_roundtrip(self):
        x_np = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float64)
        x = to_gpu(x_np)
        freq = ins.rfft(x)
        result = ins.irfft(freq, 8)
        assert_allclose(to_numpy(result), x_np, atol=1e-6)


class TestFFTShiftGPU:
    """fftshift / ifftshift on GPU vs NumPy."""

    def test_fftshift_1d(self):
        x_np = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float64)
        x = to_gpu(x_np)
        result = ins.fftshift(x)
        assert_allclose(to_numpy(result), np.fft.fftshift(x_np), atol=1e-10)

    def test_fftshift_fftfreq(self):
        x = ins.fftfreq(32, 100e-6).to(GPU)
        result = ins.fftshift(x)
        assert_allclose(to_numpy(result), np.fft.fftshift(np.fft.fftfreq(32, 100e-6)), atol=1e-10)

    def test_fftshift_2d(self):
        x_np = np.arange(24, dtype=np.float64).reshape(4, 6)
        x = to_gpu(x_np)
        result = ins.fftshift(x)
        assert_allclose(to_numpy(result), np.fft.fftshift(x_np), atol=1e-10)

    def test_fftshift_axis(self):
        x_np = np.arange(24, dtype=np.float64).reshape(4, 6)
        x = to_gpu(x_np)
        result = ins.fftshift(x, axis=1)
        assert_allclose(to_numpy(result), np.fft.fftshift(x_np, axes=1), atol=1e-10)

    def test_ifftshift_1d(self):
        x_np = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float64)
        x = to_gpu(x_np)
        result = ins.ifftshift(x)
        assert_allclose(to_numpy(result), np.fft.ifftshift(x_np), atol=1e-10)

    def test_ifftshift_roundtrip(self):
        x_np = np.random.randn(16).astype(np.float64)
        x = to_gpu(x_np)
        result = ins.fftshift(ins.ifftshift(x))
        assert_allclose(to_numpy(result), x_np, atol=1e-10)
