"""Creation CUDA binding tests — aligned with C++ test_creation.cpp.

Run with:
    PYTHONPATH=bindings/python:build/bindings/python
    LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda
    python -m pytest tests/cuda/python/test_creation.py -v
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


class TestCreationCUDA:
    """Array creation — CUDA backend."""

    def test_zeros_1d(self):
        a = ins.zeros([5], ins.float32).to(GPU)
        assert a.numel == 5
        np.testing.assert_allclose(to_numpy(a), np.zeros(5, dtype=np.float32))

    def test_zeros_2d(self):
        a = ins.zeros([3, 4], ins.float64).to(GPU)
        assert a.numel == 12
        assert a.ndim == 2
        np.testing.assert_allclose(to_numpy(a), np.zeros([3, 4], dtype=np.float64))

    def test_ones_1d(self):
        a = ins.ones([4], ins.float32).to(GPU)
        assert a.numel == 4
        np.testing.assert_allclose(to_numpy(a), np.ones(4, dtype=np.float32))

    def test_ones_2d(self):
        a = ins.ones([2, 5], ins.float64).to(GPU)
        assert a.numel == 10
        np.testing.assert_allclose(to_numpy(a), np.ones([2, 5], dtype=np.float64))

    def test_full(self):
        a = ins.full([3, 3], 7.0, ins.float32).to(GPU)
        assert a.numel == 9
        np.testing.assert_allclose(to_numpy(a), np.full([3, 3], 7.0, dtype=np.float32))

    def test_full_negative(self):
        a = ins.full([2, 2], -3.5, ins.float64).to(GPU)
        np.testing.assert_allclose(to_numpy(a), np.full([2, 2], -3.5, dtype=np.float64))

    def test_eye(self):
        a = ins.eye(4).to(GPU)
        assert a.numel == 16
        np.testing.assert_allclose(to_numpy(a), np.eye(4, dtype=np.float32))

    def test_arange(self):
        a = ins.arange(10, ins.float32).to(GPU)
        assert a.numel == 10
        np.testing.assert_allclose(to_numpy(a), np.arange(10, dtype=np.float32))

    def test_linspace(self):
        a = ins.linspace(0.0, 1.0, 11, ins.float64).to(GPU)
        assert a.numel == 11
        np.testing.assert_allclose(to_numpy(a), np.linspace(0, 1, 11, dtype=np.float64), rtol=1e-6)

    def test_from_numpy(self):
        data = np.array([1.5, 2.5, 3.5], dtype=np.float64)
        a = ins.from_numpy(data).to(GPU)
        assert a.numel == 3
        np.testing.assert_allclose(to_numpy(a), data)

    def test_from_numpy_2d(self):
        data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        a = ins.from_numpy(data).to(GPU)
        assert a.numel == 6
        assert a.ndim == 2
        np.testing.assert_allclose(to_numpy(a), data)

    def test_zeros_like(self):
        a = ins.ones([3, 3], ins.float32).to(GPU)
        b = ins.zeros_like(a)
        assert b.numel == 9
        np.testing.assert_allclose(to_numpy(b), np.zeros([3, 3]))

    def test_ones_like(self):
        a = ins.zeros([2, 4], ins.float64).to(GPU)
        b = ins.ones_like(a)
        assert b.numel == 8
        np.testing.assert_allclose(to_numpy(b), np.ones([2, 4]))

    def test_dtype_float64(self):
        a = ins.zeros([3], ins.float64).to(GPU)
        assert a.dtype == ins.float64

    def test_dtype_int32(self):
        a = ins.ones([3], ins.int32).to(GPU)
        assert a.dtype == ins.int32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
