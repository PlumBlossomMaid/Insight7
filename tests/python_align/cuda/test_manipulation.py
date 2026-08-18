"""Manipulation alignment tests — Insight CUDA vs NumPy.

Run with:
    PYTHONPATH=bindings/python:build/bindings/python \\
    LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda \\
    python -m pytest tests/python_align/cuda/test_manipulation.py -v
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


class TestManipulationAlignmentGPU:
    """Insight manipulation ops vs NumPy."""

    def test_reshape(self):
        a_np = np.arange(12, dtype=np.float64)
        a = ins.from_numpy(a_np)
        result = ins.reshape(a, [3, 4])
        assert_allclose(result.numpy(), a_np.reshape(3, 4))

    def test_flatten(self):
        a_np = np.array([[1, 2], [3, 4]], dtype=np.float64)
        a = ins.from_numpy(a_np)
        result = ins.flatten(a)
        assert_allclose(result.numpy(), a_np.flatten())

    def test_concat(self):
        a_np = np.array([1, 2, 3], dtype=np.float64)
        b_np = np.array([4, 5, 6], dtype=np.float64)
        a = ins.from_numpy(a_np)
        b = ins.from_numpy(b_np)
        result = ins.concat([a, b])
        assert_allclose(result.numpy(), np.concatenate([a_np, b_np]))

    def test_flip(self):
        a_np = np.array([1, 2, 3, 4], dtype=np.float64)
        a = ins.from_numpy(a_np)
        result = ins.flip(a)
        assert_allclose(result.numpy(), np.flip(a_np))

    def test_squeeze(self):
        a_np = np.array([[[1, 2, 3]]], dtype=np.float64)  # shape (1,1,3)
        a = ins.from_numpy(a_np)
        result = ins.squeeze(a)
        assert_allclose(result.numpy(), np.squeeze(a_np))

    def test_transpose(self):
        a_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        a = ins.from_numpy(a_np)
        result = ins.transpose(a)
        assert_allclose(result.numpy(), np.transpose(a_np))
