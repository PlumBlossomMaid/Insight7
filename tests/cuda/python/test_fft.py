"""FFT CUDA binding tests."""

import sys
import os
import pytest

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
sys.path.insert(0, os.path.join(_root, "bindings", "python"))
sys.path.insert(0, os.path.join(_root, "build", "bindings", "python"))

try:
    import insight as ins
    import numpy as np
except ImportError:
    pytest.skip("Insight or NumPy not available", allow_module_level=True)

try:
    ins.init()
    if ins.device_count() == 0:
        pytest.skip("No GPU available", allow_module_level=True)
except Exception:
    pytest.skip("GPU backend not available", allow_module_level=True)

GPU = ins.GPUPlace(0)
CPU = ins.CPUPlace()


def to_gpu(np_arr, dtype=None):
    if dtype is not None:
        np_arr = np_arr.astype(dtype)
    return ins.from_numpy(np_arr).to(GPU)


def to_numpy(gpu_arr):
    return gpu_arr.to(CPU).numpy()


class TestFFTCUDA:
    def test_fft(self):
        x_np = np.array([1, 2, 3, 4], dtype=np.float64)
        result = ins.fft(to_gpu(x_np))
        expected = np.fft.fft(x_np)
        np.testing.assert_allclose(to_numpy(result), expected, atol=1e-8)

    def test_ifft(self):
        x_np = np.array([1, 2, 3, 4], dtype=np.float64)
        fft_np = np.fft.fft(x_np)
        result = ins.ifft(to_gpu(fft_np.astype(np.complex128)))
        expected = np.fft.ifft(fft_np)
        np.testing.assert_allclose(to_numpy(result), expected, atol=1e-8)

    def test_fft_ifft_roundtrip(self):
        x_np = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float64)
        X = ins.fft(to_gpu(x_np))
        x_back = ins.ifft(X)
        np.testing.assert_allclose(to_numpy(x_back).real, x_np, atol=1e-8)

    def test_rfft(self):
        x_np = np.array([1, 2, 3, 4], dtype=np.float64)
        result = ins.rfft(to_gpu(x_np))
        expected = np.fft.rfft(x_np)
        assert to_numpy(result).shape[0] == expected.shape[0]

    def test_irfft(self):
        x_np = np.array([1, 2, 3, 4], dtype=np.float64)
        rfft_np = np.fft.rfft(x_np)
        result = ins.irfft(to_gpu(rfft_np.astype(np.complex128)))
        expected = np.fft.irfft(rfft_np)
        assert to_numpy(result).shape[0] == expected.shape[0]

    def test_fftfreq(self):
        result = ins.fftfreq(8, 1.0)
        expected = np.fft.fftfreq(8, 1.0)
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-8)

    def test_rfftfreq(self):
        result = ins.rfftfreq(8, 1.0)
        expected = np.fft.rfftfreq(8, 1.0)
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-8)

    def test_fft2(self):
        x_np = np.array([[1, 2], [3, 4]], dtype=np.float64)
        x_complex = ins.from_numpy(x_np.astype(np.complex128)).to(GPU)
        result = ins.fft2(x_complex)
        expected = np.fft.fft2(x_np)
        np.testing.assert_allclose(to_numpy(result), expected, atol=1e-8)

    def test_ifft2(self):
        x_np = np.array([[1, 2], [3, 4]], dtype=np.float64)
        fft_np = np.fft.fft2(x_np)
        result = ins.ifft2(to_gpu(fft_np.astype(np.complex128)))
        expected = np.fft.ifft2(fft_np)
        np.testing.assert_allclose(to_numpy(result), expected, atol=1e-8)

    def test_fft_longer_signal(self):
        np.random.seed(42)
        x_np = np.random.randn(256)
        result = ins.fft(to_gpu(x_np))
        expected = np.fft.fft(x_np)
        np.testing.assert_allclose(to_numpy(result), expected, atol=1e-6)

    def test_fft_complex_input(self):
        x_np = np.array([1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j], dtype=np.complex128)
        result = ins.fft(to_gpu(x_np))
        expected = np.fft.fft(x_np)
        np.testing.assert_allclose(to_numpy(result), expected, atol=1e-8)

    def test_fft_ones(self):
        x_np = np.ones(8, dtype=np.float64)
        result = ins.fft(to_gpu(x_np))
        expected = np.fft.fft(x_np)
        np.testing.assert_allclose(to_numpy(result), expected, atol=1e-8)

    def test_fft_power_of_two(self):
        x_np = np.random.randn(16)
        result = ins.fft(to_gpu(x_np))
        expected = np.fft.fft(x_np)
        np.testing.assert_allclose(to_numpy(result), expected, atol=1e-8)

    def test_fft_non_power_of_two(self):
        x_np = np.random.randn(10)
        result = ins.fft(to_gpu(x_np))
        expected = np.fft.fft(x_np)
        np.testing.assert_allclose(to_numpy(result), expected, atol=1e-8)

    def test_fftfreq_custom_spacing(self):
        result = ins.fftfreq(16, 0.5)
        expected = np.fft.fftfreq(16, 0.5)
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
