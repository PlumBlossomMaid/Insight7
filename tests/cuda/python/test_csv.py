"""CSV CUDA binding tests — aligned with C++ test_csv.cpp.

Note: CSV read/write is not exposed in the Python bindings. This test verifies
GPU data roundtrip via from_numpy/to_numpy which is the binding-level equivalent.

Run with:
    PYTHONPATH=bindings/python:build/bindings/python
    LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda
    python -m pytest tests/cuda/python/test_csv.py -v
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


class TestCSVCUDA:
    """CSV / I/O roundtrip — CUDA backend.

    CSV write/read is not exposed in bindings; these tests verify GPU data
    roundtrip via from_numpy/to_numpy.
    """

    def test_roundtrip_1d(self):
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        a = to_gpu(data)
        result = to_numpy(a)
        np.testing.assert_array_equal(result, data)

    def test_roundtrip_2d(self):
        data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        a = to_gpu(data)
        result = to_numpy(a)
        np.testing.assert_array_equal(result, data)

    def test_roundtrip_int(self):
        data = np.array([10, 20, 30], dtype=np.int32)
        a = to_gpu(data)
        result = to_numpy(a)
        np.testing.assert_array_equal(result, data)

    def test_roundtrip_float64(self):
        data = np.array([1.1, 2.2, 3.3], dtype=np.float64)
        a = to_gpu(data)
        result = to_numpy(a)
        np.testing.assert_allclose(result, data, rtol=1e-10)

    def test_roundtrip_after_ops(self):
        data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        a = to_gpu(data)
        b = ins.add(a, a)
        result = to_numpy(b)
        np.testing.assert_allclose(result, data * 2, rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
