"""Signal Filter Design CUDA binding tests.

Run with:
    PYTHONPATH=bindings/python:build/bindings/python \
    LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda \
    python -m pytest tests/cuda/python/test_signal_filter_design.py -v
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


class TestKaiserBeta:
    """Kaiser beta parameter — tests 1-4."""

    def test_kaiser_beta_low(self):
        # a <= 21 -> beta = 0
        assert ins.signal.kaiser_beta(0.0) == 0.0
        assert ins.signal.kaiser_beta(21.0) == 0.0

    def test_kaiser_beta_mid(self):
        a = 40.0
        expected = 0.5842 * (a - 21.0) ** 0.4 + 0.07886 * (a - 21.0)
        assert abs(ins.signal.kaiser_beta(a) - expected) < 1e-10

    def test_kaiser_beta_high(self):
        a = 60.0
        expected = 0.1102 * (a - 8.7)
        assert abs(ins.signal.kaiser_beta(a) - expected) < 1e-10

    def test_kaiser_beta_boundary(self):
        beta = ins.signal.kaiser_beta(50.0)
        assert beta > 4.0
        assert beta < 6.0


class TestKaiserAtten:
    """Kaiser attenuation — test 5."""

    def test_kaiser_atten_basic(self):
        atten = ins.signal.kaiser_atten(101, 0.1)
        expected = 2.285 * 100.0 * math.pi * 0.1 + 7.95
        assert abs(atten - expected) < 1e-10


class TestFirwin:
    """FIR filter design — tests 6-12."""

    def test_firwin_lowpass_basic(self):
        h = ins.signal.firwin(11, [0.3], window="hamming", pass_zero="lowpass")
        assert h.numel == 11
        d = h.numpy()
        # Symmetric (Type I: odd numtaps)
        for i in range(5):
            assert abs(d[i] - d[10 - i]) < 1e-10

    def test_firwin_lowpass_dc_gain(self):
        h = ins.signal.firwin(21, [0.25], window="hamming", pass_zero="lowpass")
        dc = float(np.sum(h.numpy()))
        assert abs(dc - 1.0) < 1e-6

    def test_firwin_highpass_basic(self):
        h = ins.signal.firwin(11, [0.3], window="hamming", pass_zero="highpass")
        assert h.numel == 11
        d = h.numpy()
        for i in range(5):
            assert abs(d[i] - d[10 - i]) < 1e-10

    def test_firwin_highpass_nyquist_gain(self):
        h = ins.signal.firwin(21, [0.25], window="hamming", pass_zero="highpass")
        d = h.numpy()
        nyquist_gain = sum(d[i] * (1.0 if i % 2 == 0 else -1.0) for i in range(21))
        assert abs(nyquist_gain - 1.0) < 1e-6

    def test_firwin_bandpass(self):
        h = ins.signal.firwin(21, [0.2, 0.5], window="hamming", pass_zero="bandpass")
        assert h.numel == 21
        dc = float(np.sum(h.numpy()))
        assert abs(dc) < 0.01  # DC gain ~0 for bandpass

    def test_firwin_different_windows(self):
        h1 = ins.signal.firwin(11, [0.3], window="hann", pass_zero="lowpass")
        h2 = ins.signal.firwin(11, [0.3], window="blackman", pass_zero="lowpass")
        assert h1.numel == 11
        assert h2.numel == 11
        assert h1.numpy()[0] != h2.numpy()[0]

    def test_firwin_unscaled(self):
        h = ins.signal.firwin(21, [0.25], window="hamming", pass_zero="lowpass", scale=False)
        dc = float(np.sum(h.numpy()))
        assert dc != 1.0


class TestFirwin2:
    """FIR filter design (arbitrary response) — tests 13-14."""

    def test_firwin2_lowpass(self):
        freq = [0.0, 0.5, 0.5, 1.0]
        gain = [1.0, 1.0, 0.0, 0.0]
        h = ins.signal.firwin2(15, freq, gain)
        assert h.numel == 15
        dc = float(np.sum(h.numpy()))
        assert abs(dc - 1.0) < 0.1

    def test_firwin2_different_lengths(self):
        freq = [0.0, 0.5, 1.0]
        gain = [1.0, 1.0, 0.0]
        h_short = ins.signal.firwin2(7, freq, gain)
        h_long = ins.signal.firwin2(31, freq, gain)
        assert h_short.numel == 7
        assert h_long.numel == 31


class TestCmplxSort:
    """Complex sort — tests 15-16."""

    def test_cmplx_sort_real(self):
        a = ins.from_numpy(np.array([3.0, -1.0, 2.0, -4.0], dtype=np.float64))
        result = ins.signal.cmplx_sort(a)
        # cmplx_sort returns (sorted, indices)
        if isinstance(result, tuple):
            sorted_arr = result[0]
        else:
            sorted_arr = result
        d = sorted_arr.numpy()
        assert abs(d[0] - (-1.0)) < 1e-10
        assert abs(d[1] - 2.0) < 1e-10
        assert abs(d[2] - 3.0) < 1e-10
        assert abs(d[3] - (-4.0)) < 1e-10

    def test_cmplx_sort_single(self):
        a = ins.from_numpy(np.array([42.0], dtype=np.float64))
        result = ins.signal.cmplx_sort(a)
        if isinstance(result, tuple):
            sorted_arr = result[0]
        else:
            sorted_arr = result
        assert abs(sorted_arr.numpy()[0] - 42.0) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
