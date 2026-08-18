"""Creation alignment tests — Insight CUDA vs NumPy.

Run with:
    PYTHONPATH=bindings/python:build/bindings/python \\
    LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda \\
    python -m pytest tests/python_align/cuda/test_creation.py -v
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


class TestCreationAlignmentGPU:
    """Insight creation on GPU vs NumPy."""

    def test_zeros(self):
        a = ins.zeros([3, 4], ins.float32).to(GPU)
        assert_allclose(to_numpy(a), np.zeros([3, 4], dtype=np.float32))

    def test_ones(self):
        a = ins.ones([3, 4], ins.float32).to(GPU)
        assert_allclose(to_numpy(a), np.ones([3, 4], dtype=np.float32))

    def test_full(self):
        a = ins.full([2, 3], 3.14, ins.float64).to(GPU)
        assert_allclose(to_numpy(a), np.full([2, 3], 3.14))

    def test_eye(self):
        a = ins.eye(4).to(GPU)
        assert_allclose(to_numpy(a), np.eye(4, dtype=np.float32))

    def test_arange(self):
        a = ins.arange(20, ins.float64).to(GPU)
        assert_allclose(to_numpy(a), np.arange(20, dtype=np.float64))

    def test_linspace(self):
        a = ins.linspace(0.0, 1.0, 50, ins.float64).to(GPU)
        assert_allclose(to_numpy(a), np.linspace(0, 1, 50), atol=1e-10)
