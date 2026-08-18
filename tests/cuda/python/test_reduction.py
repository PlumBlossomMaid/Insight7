"""Reduction CUDA binding tests — aligned with C++ test_reduction.cpp.

Run with:
    PYTHONPATH=bindings/python:build/bindings/python
    LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda
    python -m pytest tests/cuda/python/test_reduction.py -v
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


class TestReductionCUDA:
    """Reduction operations — CUDA backend."""

    def test_sum(self):
        a = to_gpu(np.array([1, 2, 3, 4], dtype=np.float64))
        result = ins.sum(a)
        assert result.numpy().item() == pytest.approx(10.0)

    def test_sum_axis(self):
        a_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        result = ins.sum(to_gpu(a_np), axis=0)
        np.testing.assert_allclose(to_numpy(result), np.sum(a_np, axis=0))

    def test_mean(self):
        a = to_gpu(np.array([1, 2, 3, 4], dtype=np.float64))
        result = ins.mean(a)
        assert result.numpy().item() == pytest.approx(2.5)

    def test_mean_axis(self):
        a_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        result = ins.mean(to_gpu(a_np), axis=1)
        np.testing.assert_allclose(to_numpy(result), np.mean(a_np, axis=1))

    def test_max(self):
        a = to_gpu(np.array([3, 1, 4, 1, 5], dtype=np.float64))
        result = ins.max(a)
        assert result.numpy().item() == pytest.approx(5.0)

    def test_min(self):
        a = to_gpu(np.array([3, 1, 4, 1, 5], dtype=np.float64))
        result = ins.min(a)
        assert result.numpy().item() == pytest.approx(1.0)

    def test_prod(self):
        a = to_gpu(np.array([1, 2, 3, 4], dtype=np.float64))
        result = ins.prod(a)
        assert result.numpy().item() == pytest.approx(24.0)

    def test_argmax(self):
        a = to_gpu(np.array([1, 5, 3, 2], dtype=np.float64))
        result = ins.argmax(a)
        assert result.numpy().item() == 1

    def test_argmin(self):
        a = to_gpu(np.array([5, 1, 3, 2], dtype=np.float64))
        result = ins.argmin(a)
        assert result.numpy().item() == 1

    def test_cumsum(self):
        a_np = np.array([1, 2, 3, 4], dtype=np.float64)
        result = ins.cumsum(to_gpu(a_np), axis=0)
        np.testing.assert_allclose(to_numpy(result), np.cumsum(a_np))

    def test_cumprod(self):
        a_np = np.array([1, 2, 3, 4], dtype=np.float64)
        result = ins.cumprod(to_gpu(a_np), axis=0)
        np.testing.assert_allclose(to_numpy(result), np.cumprod(a_np))

    def test_cummax(self):
        a_np = np.array([3, 1, 4, 1, 5], dtype=np.float64)
        result = ins.cummax(to_gpu(a_np), axis=0)
        np.testing.assert_allclose(to_numpy(result), np.maximum.accumulate(a_np))

    def test_cummin(self):
        a_np = np.array([3, 1, 4, 1, 5], dtype=np.float64)
        result = ins.cummin(to_gpu(a_np), axis=0)
        np.testing.assert_allclose(to_numpy(result), np.minimum.accumulate(a_np))

    def test_any(self):
        a = to_gpu(np.array([True, False, True], dtype=np.bool_))
        result = ins.any(a)
        assert result.numpy().item()

    def test_all(self):
        a = to_gpu(np.array([True, False, True], dtype=np.bool_))
        result = ins.all(a)
        assert not result.numpy().item()

    def test_var(self):
        a_np = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        result = ins.var(to_gpu(a_np))
        assert result.numpy().item() == pytest.approx(np.var(a_np), rel=1e-5)

    def test_std(self):
        a_np = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        result = ins.std(to_gpu(a_np))
        assert result.numpy().item() == pytest.approx(np.std(a_np), rel=1e-5)

    def test_count_nonzero(self):
        a = to_gpu(np.array([0, 1, 0, 3, 0], dtype=np.float64))
        result = ins.count_nonzero(a)
        assert result.numpy().item() == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
