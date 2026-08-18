"""Unary alignment tests — Insight CUDA vs NumPy.

Run with:
    PYTHONPATH=bindings/python:build/bindings/python \\
    LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda \\
    python -m pytest tests/python_align/cuda/test_unary.py -v
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


class TestUnaryAlignmentGPU:
    """Insight unary ops vs NumPy."""

    @pytest.fixture
    def data(self):
        return np.linspace(0.1, 5.0, 20).astype(np.float64)

    def test_abs(self):
        x_np = np.array([-3, -1, 0, 2, 4], dtype=np.float64)
        x = ins.from_numpy(x_np)
        assert_allclose(ins.abs(x).numpy(), np.abs(x_np))

    def test_sqrt(self, data):
        x = ins.from_numpy(data)
        assert_allclose(ins.sqrt(x).numpy(), np.sqrt(data), rtol=1e-8)

    def test_exp(self, data):
        x = ins.from_numpy(data[:5])  # small values to avoid overflow
        assert_allclose(ins.exp(x).numpy(), np.exp(data[:5]), rtol=1e-6)

    def test_log(self, data):
        x = ins.from_numpy(data)
        assert_allclose(ins.log(x).numpy(), np.log(data), rtol=1e-8)

    def test_sin(self, data):
        x = ins.from_numpy(data)
        assert_allclose(ins.sin(x).numpy(), np.sin(data), rtol=1e-8)

    def test_cos(self, data):
        x = ins.from_numpy(data)
        assert_allclose(ins.cos(x).numpy(), np.cos(data), rtol=1e-8)

    def test_tan(self):
        x_np = np.array([0.1, 0.5, 1.0], dtype=np.float64)
        x = ins.from_numpy(x_np)
        assert_allclose(ins.tan(x).numpy(), np.tan(x_np), rtol=1e-6)

    def test_floor(self):
        x_np = np.array([1.1, 2.7, -0.3, 3.0], dtype=np.float64)
        x = ins.from_numpy(x_np)
        assert_allclose(ins.floor(x).numpy(), np.floor(x_np))

    def test_ceil(self):
        x_np = np.array([1.1, 2.7, -0.3, 3.0], dtype=np.float64)
        x = ins.from_numpy(x_np)
        assert_allclose(ins.ceil(x).numpy(), np.ceil(x_np))

    def test_round(self):
        x_np = np.array([1.1, 2.5, 3.7, -1.3], dtype=np.float64)
        x = ins.from_numpy(x_np)
        assert_allclose(ins.round(x).numpy(), np.round(x_np))

    def test_sign(self):
        x_np = np.array([-3, -1, 0, 1, 3], dtype=np.float64)
        x = ins.from_numpy(x_np)
        assert_allclose(ins.sign(x).numpy(), np.sign(x_np))

    def test_exp2(self):
        x_np = np.array([0, 1, 2, 3], dtype=np.float64)
        x = ins.from_numpy(x_np)
        assert_allclose(ins.exp2(x).numpy(), np.exp2(x_np), rtol=1e-8)

    def test_expm1(self):
        x_np = np.array([0, 1e-10, 0.5], dtype=np.float64)
        x = ins.from_numpy(x_np)
        assert_allclose(ins.expm1(x).numpy(), np.expm1(x_np), rtol=1e-8)

    def test_log1p(self):
        x_np = np.array([0, 1e-10, 0.5], dtype=np.float64)
        x = ins.from_numpy(x_np)
        assert_allclose(ins.log1p(x).numpy(), np.log1p(x_np), rtol=1e-8)

    def test_cbrt(self):
        x_np = np.array([1, 8, 27, 64], dtype=np.float64)
        x = ins.from_numpy(x_np)
        assert_allclose(ins.cbrt(x).numpy(), np.cbrt(x_np), rtol=1e-8)

    def test_deg2rad(self):
        x_np = np.array([0, 90, 180, 360], dtype=np.float64)
        x = ins.from_numpy(x_np)
        assert_allclose(ins.deg2rad(x).numpy(), np.deg2rad(x_np), rtol=1e-10)

    def test_rad2deg(self):
        x_np = np.array([0, np.pi / 4, np.pi / 2, np.pi], dtype=np.float64)
        x = ins.from_numpy(x_np)
        assert_allclose(ins.rad2deg(x).numpy(), np.rad2deg(x_np), rtol=1e-10)

    def test_where(self):
        cond_np = np.array([True, False, True, False])
        x_np = np.array([1, 2, 3, 4], dtype=np.float64)
        y_np = np.array([5, 6, 7, 8], dtype=np.float64)
        cond = ins.from_numpy(cond_np)
        x = ins.from_numpy(x_np)
        y = ins.from_numpy(y_np)
        assert_allclose(ins.where(cond, x, y).numpy(), np.where(cond_np, x_np, y_np))


class TestUnaryExtendedGPU:
    """Extended unary ops vs NumPy."""

    def test_negative(self):
        x_np = np.array([1.5, -2.3, 0.0, 4.7], dtype=np.float64)
        x = ins.from_numpy(x_np)
        assert_allclose(ins.negative(x).numpy(), np.negative(x_np))

    def test_reciprocal(self):
        x_np = np.array([1.0, 2.0, 4.0, 0.5], dtype=np.float64)
        x = ins.from_numpy(x_np)
        assert_allclose(ins.reciprocal(x).numpy(), np.reciprocal(x_np), rtol=1e-8)

    def test_asin(self):
        x_np = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float64)
        x = ins.from_numpy(x_np)
        assert_allclose(ins.asin(x).numpy(), np.arcsin(x_np), rtol=1e-8)

    def test_acos(self):
        x_np = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float64)
        x = ins.from_numpy(x_np)
        assert_allclose(ins.acos(x).numpy(), np.arccos(x_np), rtol=1e-8)

    def test_atan(self):
        x_np = np.array([-10.0, -1.0, 0.0, 1.0, 10.0], dtype=np.float64)
        x = ins.from_numpy(x_np)
        assert_allclose(ins.atan(x).numpy(), np.arctan(x_np), rtol=1e-8)

    def test_sinh(self):
        x_np = np.array([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=np.float64)
        x = ins.from_numpy(x_np)
        assert_allclose(ins.sinh(x).numpy(), np.sinh(x_np), rtol=1e-8)

    def test_cosh(self):
        x_np = np.array([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=np.float64)
        x = ins.from_numpy(x_np)
        assert_allclose(ins.cosh(x).numpy(), np.cosh(x_np), rtol=1e-8)

    def test_tanh(self):
        x_np = np.array([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=np.float64)
        x = ins.from_numpy(x_np)
        assert_allclose(ins.tanh(x).numpy(), np.tanh(x_np), rtol=1e-8)

    def test_asinh(self):
        x_np = np.array([-5.0, -1.0, 0.0, 1.0, 5.0], dtype=np.float64)
        x = ins.from_numpy(x_np)
        assert_allclose(ins.asinh(x).numpy(), np.arcsinh(x_np), rtol=1e-8)

    def test_acosh(self):
        # acosh requires x >= 1
        x_np = np.array([1.0, 1.5, 2.0, 5.0, 10.0], dtype=np.float64)
        x = ins.from_numpy(x_np)
        assert_allclose(ins.acosh(x).numpy(), np.arccosh(x_np), rtol=1e-8)

    def test_atanh(self):
        # atanh requires |x| < 1
        x_np = np.array([-0.9, -0.5, 0.0, 0.5, 0.9], dtype=np.float64)
        x = ins.from_numpy(x_np)
        assert_allclose(ins.atanh(x).numpy(), np.arctanh(x_np), rtol=1e-8)

    def test_trunc(self):
        x_np = np.array([1.7, -2.3, 0.5, 3.0, -0.1], dtype=np.float64)
        x = ins.from_numpy(x_np)
        assert_allclose(ins.trunc(x).numpy(), np.trunc(x_np))

    def test_isnan(self):
        x_np = np.array([1.0, np.nan, 3.0, np.nan], dtype=np.float64)
        x = ins.from_numpy(x_np)
        assert_allclose(ins.isnan(x).numpy(), np.isnan(x_np))

    def test_isinf(self):
        x_np = np.array([1.0, np.inf, 3.0, -np.inf], dtype=np.float64)
        x = ins.from_numpy(x_np)
        assert_allclose(ins.isinf(x).numpy(), np.isinf(x_np))

    def test_isfinite(self):
        x_np = np.array([1.0, np.nan, np.inf, -np.inf, 3.0], dtype=np.float64)
        x = ins.from_numpy(x_np)
        assert_allclose(ins.isfinite(x).numpy(), np.isfinite(x_np))

    def test_logical_not(self):
        a_np = np.array([True, False, True, False])
        a = ins.from_numpy(a_np)
        assert_allclose(ins.logical_not(a).numpy(), np.logical_not(a_np))

    def test_square(self):
        x_np = np.array([-3.0, -1.0, 0.0, 2.0, 4.0], dtype=np.float64)
        x = ins.from_numpy(x_np)
        assert_allclose(ins.square(x).numpy(), np.square(x_np))

    def test_log2(self):
        x_np = np.array([1.0, 2.0, 4.0, 8.0, 16.0], dtype=np.float64)
        x = ins.from_numpy(x_np)
        assert_allclose(ins.log2(x).numpy(), np.log2(x_np), rtol=1e-8)

    def test_log10(self):
        x_np = np.array([1.0, 10.0, 100.0, 1000.0], dtype=np.float64)
        x = ins.from_numpy(x_np)
        assert_allclose(ins.log10(x).numpy(), np.log10(x_np), rtol=1e-8)
