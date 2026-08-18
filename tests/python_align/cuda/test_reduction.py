"""Reduction alignment tests — Insight CUDA vs NumPy.

Run with:
    PYTHONPATH=bindings/python:build/bindings/python \\
    LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda \\
    python -m pytest tests/python_align/cuda/test_reduction.py -v
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


class TestReductionAlignmentGPU:
    """Insight reduction ops vs NumPy."""

    @pytest.fixture
    def data_1d(self):
        return np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64)

    @pytest.fixture
    def data_2d(self):
        return np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)

    def test_sum(self, data_1d):
        x = ins.from_numpy(data_1d)
        assert_allclose(ins.sum(x).numpy(), np.sum(data_1d))

    def test_mean(self, data_1d):
        x = ins.from_numpy(data_1d)
        assert_allclose(ins.mean(x).numpy(), np.mean(data_1d))

    def test_max(self, data_1d):
        x = ins.from_numpy(data_1d)
        assert_allclose(ins.max(x).numpy(), np.max(data_1d))

    def test_min(self, data_1d):
        x = ins.from_numpy(data_1d)
        assert_allclose(ins.min(x).numpy(), np.min(data_1d))

    def test_sum_axis(self, data_2d):
        x = ins.from_numpy(data_2d)
        assert_allclose(ins.sum(x, axis=0).numpy(), np.sum(data_2d, axis=0))
        assert_allclose(ins.sum(x, axis=1).numpy(), np.sum(data_2d, axis=1))

    def test_mean_axis(self, data_2d):
        x = ins.from_numpy(data_2d)
        result = ins.mean(x, axis=0)
        expected = np.mean(data_2d, axis=0)
        # Insight may keep reduced dimension; squeeze to match NumPy
        assert_allclose(result.numpy().squeeze(), expected)

    def test_argmax(self, data_1d):
        x = ins.from_numpy(data_1d)
        assert_allclose(ins.argmax(x).numpy(), np.argmax(data_1d))

    def test_argmin(self, data_1d):
        x = ins.from_numpy(data_1d)
        assert_allclose(ins.argmin(x).numpy(), np.argmin(data_1d))

    def test_prod(self, data_1d):
        x = ins.from_numpy(data_1d)
        assert_allclose(ins.prod(x).numpy(), np.prod(data_1d), rtol=1e-5)

    def test_var(self, data_1d):
        x = ins.from_numpy(data_1d)
        assert_allclose(ins.var(x).numpy(), np.var(data_1d), rtol=1e-5)

    def test_std(self, data_1d):
        x = ins.from_numpy(data_1d)
        assert_allclose(ins.std(x).numpy(), np.std(data_1d), rtol=1e-5)

    def test_cumsum(self, data_1d):
        x = ins.from_numpy(data_1d)
        assert_allclose(ins.cumsum(x, 0).numpy(), np.cumsum(data_1d))


class TestReductionExtendedGPU:
    """Extended reduction ops vs NumPy."""

    @pytest.fixture
    def data_1d(self):
        return np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64)

    @pytest.fixture
    def data_2d(self):
        return np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)

    def test_cumprod(self, data_1d):
        x = ins.from_numpy(data_1d)
        assert_allclose(ins.cumprod(x, 0).numpy(), np.cumprod(data_1d), rtol=1e-5)

    def test_cummax(self, data_1d):
        x = ins.from_numpy(data_1d)
        result = ins.cummax(x, 0)
        expected = np.maximum.accumulate(data_1d)
        assert_allclose(result.numpy(), expected)

    def test_cummin(self, data_1d):
        x = ins.from_numpy(data_1d)
        result = ins.cummin(x, 0)
        expected = np.minimum.accumulate(data_1d)
        assert_allclose(result.numpy(), expected)

    def test_median_odd(self):
        x_np = np.array([3, 1, 4, 1, 5], dtype=np.float64)
        x = ins.from_numpy(x_np)
        assert_allclose(ins.median(x).numpy(), np.median(x_np))

    def test_median_even(self):
        x_np = np.array([3, 1, 4, 1, 5, 9], dtype=np.float64)
        x = ins.from_numpy(x_np)
        assert_allclose(ins.median(x).numpy(), np.median(x_np))

    def test_quantile(self, data_1d):
        x = ins.from_numpy(data_1d)
        result = ins.quantile(x, 0.5)
        assert_allclose(result.numpy(), np.quantile(data_1d, 0.5), rtol=1e-5)

    def test_quantile_75(self, data_1d):
        x = ins.from_numpy(data_1d)
        result = ins.quantile(x, 0.75)
        assert_allclose(result.numpy(), np.quantile(data_1d, 0.75), rtol=1e-5)

    def test_count_nonzero(self):
        x_np = np.array([0, 1, 0, 3, 0, 5], dtype=np.float64)
        x = ins.from_numpy(x_np)
        result = ins.count_nonzero(x)
        # count_nonzero returns scalar
        assert_allclose(result.numpy(), np.count_nonzero(x_np))

    def test_count_nonzero_2d(self):
        x_np = np.array([[0, 1], [2, 0], [0, 3]], dtype=np.float64)
        x = ins.from_numpy(x_np)
        result = ins.count_nonzero(x)
        assert_allclose(result.numpy(), np.count_nonzero(x_np))

    def test_nansum(self):
        x_np = np.array([1.0, np.nan, 3.0, np.nan, 5.0], dtype=np.float64)
        x = ins.from_numpy(x_np)
        assert_allclose(ins.nansum(x).numpy(), np.nansum(x_np), rtol=1e-8)

    def test_nanmean(self):
        x_np = np.array([1.0, np.nan, 3.0, np.nan, 5.0], dtype=np.float64)
        x = ins.from_numpy(x_np)
        assert_allclose(ins.nanmean(x).numpy(), np.nanmean(x_np), rtol=1e-8)

    def test_nanstd(self):
        x_np = np.array([1.0, np.nan, 3.0, np.nan, 5.0], dtype=np.float64)
        x = ins.from_numpy(x_np)
        assert_allclose(ins.nanstd(x).numpy(), np.nanstd(x_np), rtol=1e-5)

    def test_nanvar(self):
        x_np = np.array([1.0, np.nan, 3.0, np.nan, 5.0], dtype=np.float64)
        x = ins.from_numpy(x_np)
        assert_allclose(ins.nanvar(x).numpy(), np.nanvar(x_np), rtol=1e-5)

    def test_any(self):
        x_np = np.array([False, False, True, False])
        x = ins.from_numpy(x_np)
        assert_allclose(ins.any(x).numpy(), np.any(x_np))

    def test_all(self):
        x_np = np.array([True, True, True, False])
        x = ins.from_numpy(x_np)
        assert_allclose(ins.all(x).numpy(), np.all(x_np))

    def test_sem(self, data_1d):
        x = ins.from_numpy(data_1d)
        # standard error of the mean = std / sqrt(n)
        expected = np.std(data_1d) / np.sqrt(len(data_1d))
        assert_allclose(ins.sem(x).numpy(), expected, rtol=1e-5)
