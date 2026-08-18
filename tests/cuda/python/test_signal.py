"""Signal CUDA binding tests (base signal functions).

Run with:
    PYTHONPATH=bindings/python:build/bindings/python \\
    LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda \\
    python -m pytest tests/cuda/python/test_signal.py -v
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


class TestUnwrap:
    """Unwrap phase signal — tests 1-5."""

    def test_unwrap_basic(self):
        data = [0.0, math.pi / 2, math.pi, 3 * math.pi / 2, 2 * math.pi, 3 * math.pi]
        a = ins.from_numpy(np.array(data, dtype=np.float64))
        u = ins.unwrap(a)
        np.testing.assert_allclose(u.numpy(), data, atol=1e-6)

    def test_unwrap_with_jumps(self):
        data = [0.0, 0.1, 3.2, 3.3, 6.4, 6.5]
        a = ins.from_numpy(np.array(data, dtype=np.float64))
        u = ins.unwrap(a)
        np.testing.assert_allclose(u.numpy(), data, atol=1e-6)

    def test_unwrap_scalar(self):
        a = ins.from_numpy(np.array([math.pi], dtype=np.float64))
        u = ins.unwrap(a)
        assert abs(u.numpy()[0] - math.pi) < 1e-6

    def test_unwrap_2d_axis1(self):
        data = [0.0, math.pi / 2, math.pi, 3 * math.pi / 2, 0.1, math.pi / 2, 3.2, 3.3]
        a = ins.from_numpy(np.array(data, dtype=np.float64).reshape(2, 4))
        u = ins.unwrap(a, axis=1)
        assert u.numel == 8
        u_np = u.numpy()
        np.testing.assert_allclose(u_np[0], [0.0, math.pi / 2, math.pi, 3 * math.pi / 2], atol=1e-6)

    def test_unwrap_continuity(self):
        n = 20
        original = [i * 0.6 for i in range(n)]
        wrapped = [x % (2 * math.pi) for x in original]
        a = ins.from_numpy(np.array(wrapped, dtype=np.float64))
        recovered = ins.unwrap(a)
        np.testing.assert_allclose(recovered.numpy(), original, atol=1e-6)


class TestSinc:
    """Sinc function — tests 6-7."""

    def test_sinc_values(self):
        data = [0.0, 1.0, 2.0, -1.0, -2.0, 0.5]
        a = ins.from_numpy(np.array(data, dtype=np.float32))
        y = ins.sinc(a)
        y_np = y.numpy()
        assert abs(y_np[0] - 1.0) < 1e-5
        assert abs(y_np[1]) < 1e-5
        assert abs(y_np[2]) < 1e-5
        assert abs(y_np[3]) < 1e-5
        assert abs(y_np[5] - 0.6366198) < 1e-5

    def test_sinc_numpy_alignment(self):
        data = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float64)
        a = ins.from_numpy(data)
        y = ins.sinc(a)
        expected = np.sinc(data)
        np.testing.assert_allclose(y.numpy(), expected, atol=1e-10)


class TestConvolve:
    """Convolve — tests 8-9."""

    def test_convolve_full(self):
        a_np = np.array([1, 2, 3], dtype=np.float64)
        v_np = np.array([1, 1], dtype=np.float64)
        a = ins.from_numpy(a_np)
        v = ins.from_numpy(v_np)
        try:
            full = ins.convolve(a, v, "full")
            expected = np.convolve(a_np, v_np, "full")
            np.testing.assert_allclose(full.numpy(), expected, atol=1e-5)
        except RuntimeError:
            pytest.skip("convolve requires FFTW3")

    def test_convolve_same_valid(self):
        a_np = np.array([1, 2, 3], dtype=np.float64)
        v_np = np.array([1, 1], dtype=np.float64)
        a = ins.from_numpy(a_np)
        v = ins.from_numpy(v_np)
        try:
            same = ins.convolve(a, v, "same")
            expected = np.convolve(a_np, v_np, "same")
            np.testing.assert_allclose(same.numpy(), expected, atol=1e-5)
        except RuntimeError:
            pytest.skip("convolve requires FFTW3")


class TestCorrelate:
    """Cross-correlation — test 10."""

    def test_correlate_full(self):
        a_np = np.array([1, 2, 3], dtype=np.float64)
        b_np = np.array([1, 1], dtype=np.float64)
        a = ins.from_numpy(a_np)
        b = ins.from_numpy(b_np)
        try:
            c = ins.signal.correlate(a, b, "full")
            assert c.numel == 4
        except RuntimeError:
            pytest.skip("correlate requires FFTW3")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
