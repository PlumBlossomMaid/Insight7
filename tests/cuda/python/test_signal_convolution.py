"""Signal Convolution CUDA binding tests.

Run with:
    PYTHONPATH=bindings/python:build/bindings/python \
    LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda \
    python -m pytest tests/cuda/python/test_signal_convolution.py -v
"""

import sys
import os
import pytest
import math

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


class TestFftconvolve:
    """FFT convolution — tests 1-7."""

    def test_fftconvolve_full(self):
        a = ins.from_numpy(np.array([1, 2, 3], dtype=np.float64))
        b = ins.from_numpy(np.array([1, 1], dtype=np.float64))
        c = ins.signal.fftconvolve(a, b, "full")
        np.testing.assert_allclose(c.numpy(), [1, 3, 5, 3], atol=1e-10)

    def test_fftconvolve_same(self):
        a = ins.from_numpy(np.array([1, 2, 3, 4, 5], dtype=np.float64))
        b = ins.from_numpy(np.array([1, 1, 1], dtype=np.float64))
        c = ins.signal.fftconvolve(a, b, "same")
        np.testing.assert_allclose(c.numpy(), [3, 6, 9, 12, 9], atol=1e-10)

    def test_fftconvolve_valid(self):
        a = ins.from_numpy(np.array([1, 2, 3, 4, 5], dtype=np.float64))
        b = ins.from_numpy(np.array([1, 1, 1], dtype=np.float64))
        c = ins.signal.fftconvolve(a, b, "valid")
        np.testing.assert_allclose(c.numpy(), [6, 9, 12], atol=1e-10)

    def test_fftconvolve_commutative(self):
        a = ins.from_numpy(np.array([1, 2, 3], dtype=np.float64))
        b = ins.from_numpy(np.array([4, 5], dtype=np.float64))
        c1 = ins.signal.fftconvolve(a, b, "full")
        c2 = ins.signal.fftconvolve(b, a, "full")
        np.testing.assert_allclose(c1.numpy(), c2.numpy(), atol=1e-10)

    def test_fftconvolve_impulse(self):
        a = ins.from_numpy(np.array([1, 2, 3, 4, 5], dtype=np.float64))
        b = ins.from_numpy(np.array([1.0], dtype=np.float64))
        c = ins.signal.fftconvolve(a, b, "full")
        np.testing.assert_allclose(c.numpy(), [1, 2, 3, 4, 5], atol=1e-10)

    def test_fftconvolve_2d(self):
        a_np = np.arange(1, 10, dtype=np.float64).reshape(3, 3)
        b_np = np.array([[1, 0], [0, 1]], dtype=np.float64)
        a = ins.from_numpy(a_np)
        b = ins.from_numpy(b_np)
        c = ins.signal.fftconvolve(a, b, "full")
        assert c.numpy().shape == (4, 4)

    def test_fftconvolve_2d_same(self):
        a_np = np.arange(1, 10, dtype=np.float64).reshape(3, 3)
        b_np = np.array([[1, 0], [0, 1]], dtype=np.float64)
        a = ins.from_numpy(a_np)
        b = ins.from_numpy(b_np)
        c = ins.signal.fftconvolve(a, b, "same")
        assert c.numpy().shape == (3, 3)


class TestCorrelate:
    """Cross-correlation — tests 8-10."""

    def test_correlate_full(self):
        a = ins.from_numpy(np.array([1, 2, 3], dtype=np.float64))
        b = ins.from_numpy(np.array([1, 1], dtype=np.float64))
        c = ins.signal.correlate(a, b, "full")
        np.testing.assert_allclose(c.numpy(), [1, 3, 5, 3], atol=1e-10)

    def test_correlate_same(self):
        a = ins.from_numpy(np.array([1, 2, 3, 4, 5], dtype=np.float64))
        b = ins.from_numpy(np.array([1, 1, 1], dtype=np.float64))
        c = ins.signal.correlate(a, b, "same")
        np.testing.assert_allclose(c.numpy(), [3, 6, 9, 12, 9], atol=1e-10)

    def test_correlate_autocorrelation(self):
        a = ins.from_numpy(np.array([1, 0, -1], dtype=np.float64))
        c = ins.signal.correlate(a, a, "full")
        d = c.numpy()
        assert len(d) == 5
        assert abs(d[0] - (-1.0)) < 1e-10
        assert abs(d[2] - 2.0) < 1e-10
        assert abs(d[4] - (-1.0)) < 1e-10


class TestConvolve2d:
    """2D convolution — test 11."""

    def test_convolve2d_basic(self):
        a_np = np.arange(1, 10, dtype=np.float64).reshape(3, 3)
        b_np = np.array([[1, 0], [0, 1]], dtype=np.float64)
        a = ins.from_numpy(a_np)
        b = ins.from_numpy(b_np)
        c = ins.signal.convolve2d(a, b, "full")
        assert c.numpy().shape == (4, 4)


class TestCorrelate2d:
    """2D cross-correlation — test 12."""

    def test_correlate2d_valid(self):
        a_np = np.arange(1, 10, dtype=np.float64).reshape(3, 3)
        b_np = np.array([[1, 1], [1, 1]], dtype=np.float64)
        a = ins.from_numpy(a_np)
        b = ins.from_numpy(b_np)
        c = ins.signal.correlate2d(a, b, "valid")
        assert c.numpy().shape == (2, 2)


class TestChooseConvMethod:
    """choose_conv_method — test 13."""

    def test_choose_conv_method_small(self):
        a = ins.from_numpy(np.array([1, 2, 3], dtype=np.float64))
        b = ins.from_numpy(np.array([1, 1], dtype=np.float64))
        method = ins.signal.choose_conv_method(a, b, "full")
        assert method == "direct"


class TestCorrelationLags:
    """correlation_lags — tests 14-15."""

    def test_correlation_lags_full(self):
        lags = ins.signal.correlation_lags(5, 3, "full")
        lags_np = lags.numpy()
        assert len(lags_np) == 7
        assert lags_np[0] == -2
        assert lags_np[6] == 4

    def test_correlation_lags_same(self):
        lags = ins.signal.correlation_lags(5, 3, "same")
        lags_np = lags.numpy()
        assert len(lags_np) == 5
        assert lags_np[0] == -1
        assert lags_np[4] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
