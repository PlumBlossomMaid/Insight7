"""Signal Waveforms CUDA binding tests.

Run with:
    PYTHONPATH=bindings/python:build/bindings/python \
    LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda \
    python -m pytest tests/cuda/python/test_signal_waveforms.py -v
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


class TestSawtooth:
    """Sawtooth waveform — tests 1-3."""

    def test_sawtooth_basic(self):
        t_np = np.array([0.0, math.pi, 2 * math.pi - 1e-10], dtype=np.float64)
        t = ins.from_numpy(t_np)
        y = ins.signal.sawtooth(t, width=1.0)
        d = y.numpy()
        assert abs(d[0] - (-1.0)) < 1e-6
        assert abs(d[1] - 0.0) < 1e-6
        assert abs(d[2] - 1.0) < 1e-6

    def test_sawtooth_triangle(self):
        t_np = np.array([0.0, math.pi / 2, math.pi], dtype=np.float64)
        t = ins.from_numpy(t_np)
        y = ins.signal.sawtooth(t, width=0.5)
        d = y.numpy()
        assert abs(d[0] - (-1.0)) < 1e-6
        assert abs(d[1] - 0.0) < 1e-6
        assert abs(d[2] - 1.0) < 1e-6

    def test_sawtooth_periodicity(self):
        t_np = np.array([0.5, 0.5 + 2 * math.pi, 0.5 + 4 * math.pi], dtype=np.float64)
        t = ins.from_numpy(t_np)
        y = ins.signal.sawtooth(t, width=1.0)
        d = y.numpy()
        assert abs(d[0] - d[1]) < 1e-6
        assert abs(d[0] - d[2]) < 1e-6


class TestSquare:
    """Square waveform — tests 4-5."""

    def test_square_basic(self):
        t_np = np.array([0.0, math.pi, 2 * math.pi - 1e-10], dtype=np.float64)
        t = ins.from_numpy(t_np)
        y = ins.signal.square_wf(t, duty=0.5)
        d = y.numpy()
        assert abs(d[0] - 1.0) < 1e-6
        assert abs(d[1] - (-1.0)) < 1e-6
        assert abs(d[2] - (-1.0)) < 1e-6

    def test_square_periodicity(self):
        t_np = np.array([1.0, 1.0 + 2 * math.pi, 1.0 + 4 * math.pi], dtype=np.float64)
        t = ins.from_numpy(t_np)
        y = ins.signal.square_wf(t, duty=0.5)
        d = y.numpy()
        assert abs(d[0] - d[1]) < 1e-6
        assert abs(d[0] - d[2]) < 1e-6


class TestGausspulse:
    """Gaussian pulse — tests 6-7."""

    def test_gausspulse_peak(self):
        t = ins.from_numpy(np.array([0.0], dtype=np.float64))
        y = ins.signal.gausspulse(t, fc=1000.0)
        assert abs(y.numpy()[0] - 1.0) < 1e-6

    def test_gausspulse_decay(self):
        t = ins.from_numpy(np.array([0.0, 0.001, 0.01], dtype=np.float64))
        y = ins.signal.gausspulse(t, fc=1000.0, bw=0.5)
        d = y.numpy()
        assert abs(d[0] - 1.0) < 1e-6  # peak at t=0
        assert abs(d[0]) > abs(d[2])  # decay


class TestChirp:
    """Chirp signal — tests 8-9."""

    def test_chirp_linear(self):
        t = ins.from_numpy(np.array([0.0], dtype=np.float64))
        y = ins.signal.chirp(t, f0=6.0, t1=1.0, f1=10.0, method="linear")
        assert abs(y.numpy()[0] - 1.0) < 1e-6

    def test_chirp_with_phase(self):
        # cos(0 + pi/2) = 0
        t = ins.from_numpy(np.array([0.0], dtype=np.float64))
        y = ins.signal.chirp(t, f0=6.0, t1=1.0, f1=10.0, method="linear", phi=math.pi / 2)
        assert abs(y.numpy()[0]) < 1e-6


class TestUnitImpulse:
    """Unit impulse — test 10."""

    def test_unit_impulse_center(self):
        y = ins.signal.unit_impulse(ins.Shape([11]))
        assert y.numel == 11
        d = y.numpy()
        for i in range(11):
            if i == 5:
                assert abs(d[i] - 1.0) < 1e-10
            else:
                assert abs(d[i]) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
