"""Signal Filtering CUDA binding tests.

Run with:
    PYTHONPATH=bindings/python:build/bindings/python \
    LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda \
    python -m pytest tests/cuda/python/test_signal_filtering.py -v
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


class TestHilbert:
    """Hilbert transform — tests 1-3."""

    def test_hilbert_basic(self):
        N = 64
        x_np = np.cos(2 * np.pi * np.arange(N) / N)
        x = ins.from_numpy(x_np.astype(np.float64))
        y = ins.signal.hilbert(x, N)
        assert y.numel == N
        # Analytic signal should have non-zero magnitude
        y_np = y.numpy()
        for i in range(N):
            assert abs(y_np[i]) > 0.01

    def test_hilbert_dc(self):
        x_np = np.ones(8, dtype=np.float64)
        x = ins.from_numpy(x_np)
        y = ins.signal.hilbert(x, 8)
        y_np = y.numpy()
        for i in range(8):
            assert abs(y_np[i].real - 1.0) < 1e-10
            assert abs(y_np[i].imag) < 1e-10

    def test_hilbert_default_n(self):
        x = ins.from_numpy(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64))
        y = ins.signal.hilbert(x)
        assert y.numel == 4


class TestDetrend:
    """Detrend — tests 4-6."""

    def test_detrend_constant(self):
        x = ins.from_numpy(np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64))
        y = ins.signal.detrend(x, type="constant")
        np.testing.assert_allclose(y.numpy(), [-2, -1, 0, 1, 2], atol=1e-10)

    def test_detrend_linear(self):
        x = ins.from_numpy(np.array([0.0, 3.0, 6.0, 9.0, 12.0], dtype=np.float64))
        y = ins.signal.detrend(x, type="linear")
        np.testing.assert_allclose(y.numpy(), 0.0, atol=1e-10)

    def test_detrend_linear_noisy(self):
        x = ins.from_numpy(np.array([0.1, 3.0, 5.9, 9.1, 12.0], dtype=np.float64))
        y = ins.signal.detrend(x, type="linear")
        np.testing.assert_allclose(y.numpy(), 0.0, atol=0.2)


class TestLfilter:
    """IIR/FIR filter — tests 7-9."""

    def test_lfilter_fir_basic(self):
        b = ins.from_numpy(np.array([1.0, 1.0], dtype=np.float64))
        a = ins.from_numpy(np.array([1.0], dtype=np.float64))
        x = ins.from_numpy(np.array([1, 0, 0, 0, 0], dtype=np.float64))
        y = ins.signal.lfilter(b, a, x)
        assert y.numel == 6
        d = y.numpy()
        assert abs(d[0] - 1.0) < 1e-10
        assert abs(d[1] - 1.0) < 1e-10
        for i in range(2, 6):
            assert abs(d[i]) < 1e-10

    def test_lfilter_fir_moving_avg(self):
        b = ins.from_numpy(np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float64))
        a = ins.from_numpy(np.array([1.0], dtype=np.float64))
        x = ins.from_numpy(np.array([1, 2, 3, 4, 5], dtype=np.float64))
        y = ins.signal.lfilter(b, a, x)
        d = y.numpy()
        assert abs(d[1] - 1.0) < 1e-10
        assert abs(d[2] - 2.0) < 1e-10
        assert abs(d[3] - 3.0) < 1e-10
        assert abs(d[4] - 4.0) < 1e-10

    def test_lfilter_iir_impulse(self):
        # y[n] = x[n] + 0.5*y[n-1], impulse response
        b = ins.from_numpy(np.array([1.0], dtype=np.float64))
        a = ins.from_numpy(np.array([1.0, -0.5], dtype=np.float64))
        x = ins.from_numpy(np.array([1, 0, 0, 0, 0], dtype=np.float64))
        y = ins.signal.lfilter(b, a, x)
        d = y.numpy()
        assert abs(d[0] - 1.0) < 1e-10
        assert abs(d[1] - 0.5) < 1e-10
        assert abs(d[2] - 0.25) < 1e-10
        assert abs(d[3] - 0.125) < 1e-10
        assert abs(d[4] - 0.0625) < 1e-10


class TestFiltfilt:
    """Zero-phase filtering — test 10."""

    def test_filtfilt_fir(self):
        b = ins.from_numpy(np.array([1.0, 1.0], dtype=np.float64))
        a = ins.from_numpy(np.array([1.0], dtype=np.float64))
        x = ins.from_numpy(np.array([1, 2, 3, 4, 5], dtype=np.float64))
        y = ins.signal.filtfilt(b, a, x)
        assert y.numel == x.numel


class TestFreqShift:
    """Frequency shift — tests 11-12."""

    def test_freq_shift_zero(self):
        N = 64
        x = ins.from_numpy(np.ones(N, dtype=np.float64))
        y = ins.signal.freq_shift(x, 0.0, 1.0)
        y_np = y.numpy()
        for i in range(N):
            assert abs(abs(y_np[i]) - 1.0) < 1e-10

    def test_freq_shift_positive(self):
        N = 16
        x = ins.from_numpy(np.ones(N, dtype=np.float64))
        fs = 1.0
        y = ins.signal.freq_shift(x, fs / 4.0, fs)
        y_np = y.numpy()
        for i in range(N):
            expected_real = math.cos(2 * math.pi * 0.25 * i)
            expected_imag = math.sin(2 * math.pi * 0.25 * i)
            assert abs(y_np[i].real - expected_real) < 1e-10
            assert abs(y_np[i].imag - expected_imag) < 1e-10


class TestDecimate:
    """Decimation — tests 13-14."""

    def test_decimate_basic(self):
        N = 100
        x_np = np.sin(2 * np.pi * np.arange(N) / N)
        x = ins.from_numpy(x_np.astype(np.float64))
        y = ins.signal.decimate(x, 2, zero_phase=False)
        assert y.numel == 50

    def test_decimate_zero_phase(self):
        N = 100
        x_np = np.sin(2 * np.pi * np.arange(N) / N)
        x = ins.from_numpy(x_np.astype(np.float64))
        y = ins.signal.decimate(x, 4, zero_phase=True)
        assert y.numel == 25


class TestResample:
    """Resampling — tests 15-16."""

    def test_resample_identity(self):
        x = ins.from_numpy(np.array([1, 2, 3, 4, 5], dtype=np.float64))
        y = ins.signal.resample(x, 5)
        assert y.numel == 5
        np.testing.assert_allclose(y.numpy(), [1, 2, 3, 4, 5], atol=1e-6)

    def test_resample_upsample(self):
        x = ins.from_numpy(np.array([1, 2, 3, 4], dtype=np.float64))
        y = ins.signal.resample(x, 8)
        assert y.numel == 8


class TestResamplePoly:
    """Polyphase resampling — tests 17-18."""

    def test_resample_poly_identity(self):
        x = ins.from_numpy(np.array([1, 2, 3, 4, 5], dtype=np.float64))
        y = ins.signal.resample_poly(x, 1, 1)
        assert y.numel == 5
        np.testing.assert_allclose(y.numpy(), [1, 2, 3, 4, 5], atol=1e-10)

    def test_resample_poly_upsample(self):
        x = ins.from_numpy(np.array([1, 0, 1, 0], dtype=np.float64))
        y = ins.signal.resample_poly(x, 2, 1)
        assert y.numel >= 7
        assert y.numel <= 9


class TestWiener:
    """Wiener filter — test 19."""

    def test_wiener_basic(self):
        # wiener requires 1D input (convolve only supports 1D tensors)
        x = ins.from_numpy(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float64))
        y = ins.signal.wiener(x)
        assert y.numel == 9


class TestFirfilter:
    """FIR filter — test 20."""

    def test_firfilter_basic(self):
        b = ins.from_numpy(np.array([1, 2, 1], dtype=np.float64))
        x = ins.from_numpy(np.array([1, 0, 0, 0, 0], dtype=np.float64))
        y = ins.signal.firfilter(b, x)
        assert y.numel == 7
        d = y.numpy()
        assert abs(d[0] - 1.0) < 1e-10
        assert abs(d[1] - 2.0) < 1e-10
        assert abs(d[2] - 1.0) < 1e-10
        for i in range(3, 7):
            assert abs(d[i]) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
