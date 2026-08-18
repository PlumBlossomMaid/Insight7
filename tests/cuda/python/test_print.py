"""Print CUDA binding tests — aligned with C++ test_print.cpp.

Run with:
    PYTHONPATH=bindings/python:build/bindings/python
    LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda
    python -m pytest tests/cuda/python/test_print.py -v
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


class TestPrintCUDA:
    """Print / repr / str operations — CUDA backend."""

    def test_repr_1d(self):
        a = ins.zeros([5], ins.float32).to(GPU)
        r = repr(a)
        assert isinstance(r, str) and len(r) > 0

    def test_repr_2d(self):
        a = ins.ones([3, 4], ins.float64).to(GPU)
        r = repr(a)
        assert isinstance(r, str) and len(r) > 0

    def test_repr_int(self):
        a = ins.zeros([3], ins.int32).to(GPU)
        r = repr(a)
        assert isinstance(r, str) and len(r) > 0

    def test_str_1d(self):
        a = ins.from_numpy(np.array([1, 2, 3], dtype=np.float32)).to(GPU)
        s = str(a)
        assert isinstance(s, str) and len(s) > 0

    def test_str_2d(self):
        a = ins.from_numpy(np.array([[1, 2], [3, 4]], dtype=np.float64)).to(GPU)
        s = str(a)
        assert isinstance(s, str) and len(s) > 0

    def test_print_does_not_crash_large(self):
        a = ins.ones([100, 100], ins.float32).to(GPU)
        _ = repr(a)

    def test_print_empty(self):
        a = ins.zeros([0], ins.float32).to(GPU)
        _ = repr(a)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
