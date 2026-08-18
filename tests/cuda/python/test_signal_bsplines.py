"""Signal Bsplines CUDA binding tests.

Run with:
    PYTHONPATH=bindings/python:build/bindings/python \
    LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda \
    python -m pytest tests/cuda/python/test_signal_bsplines.py -v
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


class TestGaussSpline:
    """Gaussian B-spline — tests 1-3."""

    def test_gauss_spline_basic(self):
        # At x=0: gauss_spline(0, n) = 1/sqrt(2*pi*sigma^2), sigma^2=(n+1)/12
        x = ins.from_numpy(np.array([0.0], dtype=np.float64))
        y = ins.signal.gauss_spline(x, 3)
        sigma_sq = 4.0 / 12.0
        expected = 1.0 / math.sqrt(2 * math.pi * sigma_sq)
        assert abs(y.numpy()[0] - expected) < 1e-10

    def test_gauss_spline_symmetry(self):
        x = ins.from_numpy(np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64))
        y = ins.signal.gauss_spline(x, 2)
        d = y.numpy()
        assert abs(d[0] - d[4]) < 1e-10
        assert abs(d[1] - d[3]) < 1e-10
        assert d[2] > d[0]
        assert d[2] > d[1]

    def test_gauss_spline_decay(self):
        x = ins.from_numpy(np.array([0.0, 1.0, 2.0, 3.0, 5.0], dtype=np.float64))
        y = ins.signal.gauss_spline(x, 3)
        d = y.numpy()
        for i in range(1, 5):
            assert d[i] < d[i - 1]


class TestCubic:
    """Cubic B-spline — tests 4-8."""

    def test_cubic_basic(self):
        # cubic(0) = 2/3
        x = ins.from_numpy(np.array([0.0], dtype=np.float64))
        y = ins.signal.cubic(x)
        assert abs(y.numpy()[0] - 2.0 / 3.0) < 1e-10

    def test_cubic_symmetry(self):
        x = ins.from_numpy(np.array([-1.5, -0.5, 0.0, 0.5, 1.5], dtype=np.float64))
        y = ins.signal.cubic(x)
        d = y.numpy()
        assert abs(d[0] - d[4]) < 1e-10
        assert abs(d[1] - d[3]) < 1e-10

    def test_cubic_zero_outside(self):
        x = ins.from_numpy(np.array([-2.5, -2.0, 2.0, 2.5], dtype=np.float64))
        y = ins.signal.cubic(x)
        np.testing.assert_allclose(y.numpy(), 0.0, atol=1e-10)

    def test_cubic_region1(self):
        # |x| < 1: 2/3 - 0.5*|x|^2*(2-|x|)
        x = ins.from_numpy(np.array([0.5], dtype=np.float64))
        y = ins.signal.cubic(x)
        expected = 2.0 / 3.0 - 0.5 * 0.25 * 1.5
        assert abs(y.numpy()[0] - expected) < 1e-10

    def test_cubic_region2(self):
        # 1 <= |x| < 2: (1/6)*(2-|x|)^3
        x = ins.from_numpy(np.array([1.5], dtype=np.float64))
        y = ins.signal.cubic(x)
        expected = (1.0 / 6.0) * 0.125
        assert abs(y.numpy()[0] - expected) < 1e-10


class TestQuadratic:
    """Quadratic B-spline — tests 9-10."""

    def test_quadratic_basic(self):
        # quadratic(0) = 0.75
        x = ins.from_numpy(np.array([0.0], dtype=np.float64))
        y = ins.signal.quadratic(x)
        assert abs(y.numpy()[0] - 0.75) < 1e-10

    def test_quadratic_symmetry(self):
        x = ins.from_numpy(np.array([-1.0, -0.25, 0.0, 0.25, 1.0], dtype=np.float64))
        y = ins.signal.quadratic(x)
        d = y.numpy()
        assert abs(d[0] - d[4]) < 1e-10
        assert abs(d[1] - d[3]) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
