"""DType CUDA binding tests — aligned with C++ test_dtype.cpp.

Run with:
    PYTHONPATH=bindings/python:build/bindings/python
    LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda
    python -m pytest tests/cuda/python/test_dtype.py -v
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


class TestDTypeCUDA:
    """DType and cast operations — CUDA backend."""

    def test_dtype_shortcuts(self):
        assert ins.float32 is not None
        assert ins.float64 is not None
        assert ins.int32 is not None
        assert ins.int64 is not None
        assert ins.bool is not None

    def test_dtype_enum(self):
        assert ins.DType.float32 == ins.float32
        assert ins.DType.int64 == ins.int64

    def test_cast_f32_to_f64(self):
        a = ins.ones([3], ins.float32).to(GPU)
        b = ins.cast(a, ins.float64)
        assert b.dtype == ins.float64
        np.testing.assert_allclose(to_numpy(b), np.ones(3, dtype=np.float64))

    def test_cast_f64_to_i32(self):
        a_np = np.array([1.9, 2.5, 3.1], dtype=np.float64)
        a = to_gpu(a_np)
        b = ins.cast(a, ins.int32)
        assert b.dtype == ins.int32

    def test_cast_i32_to_f32(self):
        a = to_gpu(np.array([1, 2, 3], dtype=np.int32))
        b = ins.cast(a, ins.float32)
        assert b.dtype == ins.float32
        np.testing.assert_allclose(to_numpy(b), [1.0, 2.0, 3.0])

    def test_cast_to_bool(self):
        a = to_gpu(np.array([0, 1, 2], dtype=np.float64))
        b = ins.cast(a, ins.bool)
        assert b.dtype == ins.bool

    def test_cast_i64_to_f64(self):
        a = to_gpu(np.array([100, 200, 300], dtype=np.int64))
        b = ins.cast(a, ins.float64)
        assert b.dtype == ins.float64
        np.testing.assert_allclose(to_numpy(b), [100.0, 200.0, 300.0])

    def test_cast_preserves_values(self):
        a_np = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        a = to_gpu(a_np)
        b = ins.cast(a, ins.float32)
        c = ins.cast(b, ins.float64)
        np.testing.assert_allclose(to_numpy(c), a_np, rtol=1e-5)

    def test_array_dtype_property(self):
        a = ins.zeros([3], ins.float32).to(GPU)
        assert a.dtype == ins.float32
        b = ins.zeros([3], ins.int64).to(GPU)
        assert b.dtype == ins.int64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
