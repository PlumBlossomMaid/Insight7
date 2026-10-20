"""Random alignment tests — Insight vs NumPy.

Run with:
    PYTHONPATH=bindings/python:build/bindings/python python -m pytest tests/python_align/cpu/test_random.py -v
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
    pytest.skip("Insight, NumPy, or SciPy not available", allow_module_level=True)


class TestRandomBatch5:
    """Batch 5 random ops vs NumPy (multinomial/choice/shuffle)."""

    def test_multinomial_shape(self):
        p_np = np.array([0.1, 0.2, 0.7], dtype=np.float64)
        p = ins.from_numpy(p_np)
        result = ins.multinomial(p, 10, True)
        assert list(result.shape) == [10]
        assert result.dtype == ins.DType.int64

    def test_multinomial_in_range(self):
        p_np = np.array([0.2, 0.3, 0.5], dtype=np.float64)
        p = ins.from_numpy(p_np)
        result = ins.multinomial(p, 5000, True)
        vals = result.numpy()
        assert vals.min() >= 0
        assert vals.max() <= 2

    def test_multinomial_without_replacement_unique(self):
        p_np = np.ones(5, dtype=np.float64)
        p = ins.from_numpy(p_np)
        result = ins.multinomial(p, 5, False)
        vals = result.numpy()
        assert len(np.unique(vals)) == 5

    def test_multinomial_weighted_dominant(self):
        p_np = np.array([0.05, 0.05, 0.9], dtype=np.float64)
        p = ins.from_numpy(p_np)
        result = ins.multinomial(p, 20000, True)
        vals = result.numpy()
        assert (vals == 2).sum() > 15000

    def test_choice_shape_and_dtype(self):
        x_np = np.arange(10, dtype=np.float32)
        x = ins.from_numpy(x_np)
        result = ins.choice(x, 5)
        assert list(result.shape) == [5]
        assert result.dtype == ins.DType.float32

    def test_choice_values_in_range(self):
        x_np = np.arange(10, dtype=np.int64)
        x = ins.from_numpy(x_np)
        result = ins.choice(x, 5000, True)
        vals = result.numpy()
        assert vals.min() >= 0
        assert vals.max() <= 9

    def test_choice_with_p(self):
        x_np = np.arange(4, dtype=np.int64)
        p_np = np.array([0.05, 0.05, 0.05, 0.85], dtype=np.float64)
        x = ins.from_numpy(x_np)
        p = ins.from_numpy(p_np)
        result = ins.choice(x, 20000, True, p)
        vals = result.numpy()
        assert (vals == 3).sum() > 15000

    def test_choice_without_replacement_unique(self):
        x_np = np.arange(8, dtype=np.int64)
        x = ins.from_numpy(x_np)
        result = ins.choice(x, 8, False)
        vals = result.numpy()
        assert len(np.unique(vals)) == 8

    def test_shuffle_preserves_set(self):
        x_np = np.arange(100, dtype=np.int64)
        x = ins.from_numpy(x_np)
        result = ins.shuffle(x)
        assert sorted(result.numpy().tolist()) == sorted(x_np.tolist())

    def test_shuffle_shape(self):
        x_np = np.arange(50, dtype=np.float32)
        x = ins.from_numpy(x_np)
        result = ins.shuffle(x)
        assert list(result.shape) == [50]
        assert result.dtype == ins.DType.float32
