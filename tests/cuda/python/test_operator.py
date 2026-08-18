"""Operator CUDA binding tests — aligned with C++ test_operator.cpp.

Run with:
    PYTHONPATH=bindings/python:build/bindings/python
    LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda
    python -m pytest tests/cuda/python/test_operator.py -v
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


class TestOperatorCUDA:
    """Arithmetic, comparison, and indexing operators — CUDA backend."""

    def test_add_operator(self):
        a_np = np.array([1, 2, 3], dtype=np.float64)
        b_np = np.array([4, 5, 6], dtype=np.float64)
        np.testing.assert_allclose(to_numpy(to_gpu(a_np) + to_gpu(b_np)), a_np + b_np, rtol=1e-10)

    def test_sub_operator(self):
        a_np = np.array([10, 20, 30], dtype=np.float64)
        b_np = np.array([1, 2, 3], dtype=np.float64)
        np.testing.assert_allclose(to_numpy(to_gpu(a_np) - to_gpu(b_np)), a_np - b_np, rtol=1e-10)

    def test_mul_operator(self):
        a_np = np.array([2, 3, 4], dtype=np.float64)
        b_np = np.array([5, 6, 7], dtype=np.float64)
        np.testing.assert_allclose(to_numpy(to_gpu(a_np) * to_gpu(b_np)), a_np * b_np, rtol=1e-10)

    def test_div_operator(self):
        a_np = np.array([10, 20, 30], dtype=np.float64)
        b_np = np.array([2, 4, 5], dtype=np.float64)
        np.testing.assert_allclose(to_numpy(to_gpu(a_np) / to_gpu(b_np)), a_np / b_np, rtol=1e-10)

    def test_negation(self):
        a_np = np.array([1, -2, 3], dtype=np.float64)
        np.testing.assert_allclose(to_numpy(-to_gpu(a_np)), -a_np, rtol=1e-10)

    def test_scalar_add(self):
        a_np = np.array([1, 2, 3], dtype=np.float64)
        np.testing.assert_allclose(to_numpy(to_gpu(a_np) + 10.0), a_np + 10.0, rtol=1e-10)

    def test_scalar_mul(self):
        a_np = np.array([1, 2, 3], dtype=np.float64)
        np.testing.assert_allclose(to_numpy(to_gpu(a_np) * 2.0), a_np * 2.0, rtol=1e-10)

    def test_eq_operator(self):
        a = to_gpu(np.array([1, 2, 3], dtype=np.float64))
        b = to_gpu(np.array([1, 0, 3], dtype=np.float64))
        result = a == b
        np.testing.assert_array_equal(to_numpy(result), [True, False, True])

    def test_lt_operator(self):
        a = to_gpu(np.array([1, 5, 3], dtype=np.float64))
        b = to_gpu(np.array([2, 4, 5], dtype=np.float64))
        result = a < b
        np.testing.assert_array_equal(to_numpy(result), [True, False, True])

    def test_le_operator(self):
        a = to_gpu(np.array([1, 5, 3], dtype=np.float64))
        b = to_gpu(np.array([2, 4, 5], dtype=np.float64))
        result = a <= b
        np.testing.assert_array_equal(to_numpy(result), [True, False, True])

    def test_gt_operator(self):
        a = to_gpu(np.array([3, 1, 5], dtype=np.float64))
        b = to_gpu(np.array([2, 4, 5], dtype=np.float64))
        result = a > b
        np.testing.assert_array_equal(to_numpy(result), [True, False, False])

    def test_indexing(self):
        a = to_gpu(np.array([10, 20, 30, 40], dtype=np.float64))
        b = a["0"]
        assert b.numel == 1

    def test_string_slice(self):
        a = ins.ones([4, 4], ins.float32).to(GPU)
        b = a[":,1:-1"]
        assert b.defined
        assert b.numel == 8

    def test_inplace_add(self):
        a_np = np.array([1, 2, 3], dtype=np.float64)
        a = to_gpu(a_np)
        a += to_gpu(np.array([10, 20, 30], dtype=np.float64))
        np.testing.assert_allclose(to_numpy(a), [11, 22, 33], rtol=1e-10)

    def test_inplace_mul(self):
        a_np = np.array([2, 3, 4], dtype=np.float64)
        a = to_gpu(a_np)
        a *= to_gpu(np.array([5, 6, 7], dtype=np.float64))
        np.testing.assert_allclose(to_numpy(a), [10, 18, 28], rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
