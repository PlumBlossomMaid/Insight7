"""Cast alignment tests — Insight CUDA vs NumPy.

Run with:
    PYTHONPATH=bindings/python:build/bindings/python \\
    LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda \\
    python -m pytest tests/python_align/cuda/test_cast.py -v
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
    from numpy.testing import assert_allclose
except ImportError:
    pytest.skip("Insight or NumPy not available", allow_module_level=True)

# Skip if CUDA not available
try:
    ins.init()
except Exception:
    pytest.skip("GPU backend not available", allow_module_level=True)

GPU = ins.GPUPlace(0)
CPU = ins.CPUPlace()


def to_gpu(np_arr):
    return ins.from_numpy(np_arr).to(GPU)


def to_numpy(gpu_arr):
    return gpu_arr.to(CPU).numpy()


class TestCastAlignmentGPU:
    """Insight cast on GPU vs NumPy dtype conversion."""

    def test_float32_to_float64(self):
        x_np = np.array([1.5, 2.5, 3.5], dtype=np.float32)
        x = to_gpu(x_np)
        result = ins.cast(x, ins.float64)
        assert_allclose(to_numpy(result), x_np.astype(np.float64))

    def test_float64_to_float32(self):
        x_np = np.array([1.5, 2.5, 3.5], dtype=np.float64)
        x = to_gpu(x_np)
        result = ins.cast(x, ins.float32)
        assert_allclose(to_numpy(result), x_np.astype(np.float32))

    def test_float64_to_int32(self):
        x_np = np.array([1.9, 2.1, -3.7, 0.0], dtype=np.float64)
        x = to_gpu(x_np)
        result = ins.cast(x, ins.int32)
        expected = x_np.astype(np.int32)
        assert_allclose(to_numpy(result), expected)

    def test_float64_to_int64(self):
        x_np = np.array([1.9, 2.1, -3.7, 0.0], dtype=np.float64)
        x = to_gpu(x_np)
        result = ins.cast(x, ins.int64)
        expected = x_np.astype(np.int64)
        assert_allclose(to_numpy(result), expected)

    def test_float64_to_bool(self):
        x_np = np.array([0.0, 1.0, -1.0, 0.0, 3.14], dtype=np.float64)
        x = to_gpu(x_np)
        result = ins.cast(x, ins.bool)
        expected = x_np.astype(bool)
        assert_allclose(to_numpy(result), expected)

    def test_int32_to_float64(self):
        x_np = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        x = to_gpu(x_np)
        result = ins.cast(x, ins.float64)
        assert_allclose(to_numpy(result), x_np.astype(np.float64))

    def test_bool_to_float64(self):
        x_np = np.array([True, False, True, False], dtype=bool)
        x = to_gpu(x_np)
        result = ins.cast(x, ins.float64)
        assert_allclose(to_numpy(result), x_np.astype(np.float64))
