"""Unary CUDA binding tests — aligned with C++ test_unary.cpp.

Run with:
    PYTHONPATH=bindings/python:build/bindings/python
    LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda
    python -m pytest tests/cuda/python/test_unary.py -v
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


class TestUnaryCUDA:
    """Unary math operations — CUDA backend."""

    def test_abs(self):
        a_np = np.array([-3, -1, 0, 2, 4], dtype=np.float64)
        result = ins.abs(to_gpu(a_np))
        np.testing.assert_allclose(to_numpy(result), np.abs(a_np), rtol=1e-10)

    def test_sqrt(self):
        a_np = np.array([1, 4, 9, 16], dtype=np.float64)
        result = ins.sqrt(to_gpu(a_np))
        np.testing.assert_allclose(to_numpy(result), np.sqrt(a_np), rtol=1e-10)

    def test_exp(self):
        a_np = np.array([0, 1, 2], dtype=np.float64)
        result = ins.exp(to_gpu(a_np))
        np.testing.assert_allclose(to_numpy(result), np.exp(a_np), rtol=1e-10)

    def test_log(self):
        a_np = np.array([1, 2, 3], dtype=np.float64)
        result = ins.log(to_gpu(a_np))
        np.testing.assert_allclose(to_numpy(result), np.log(a_np), rtol=1e-10)

    def test_sin_cos_tan(self):
        a_np = np.array([0, 0.5, 1.0], dtype=np.float64)
        np.testing.assert_allclose(to_numpy(ins.sin(to_gpu(a_np))), np.sin(a_np), rtol=1e-10)
        np.testing.assert_allclose(to_numpy(ins.cos(to_gpu(a_np))), np.cos(a_np), rtol=1e-10)
        np.testing.assert_allclose(to_numpy(ins.tan(to_gpu(a_np))), np.tan(a_np), rtol=1e-10)

    def test_floor_ceil_round(self):
        a_np = np.array([1.2, 2.5, 3.7, -1.3], dtype=np.float64)
        np.testing.assert_allclose(to_numpy(ins.floor(to_gpu(a_np))), np.floor(a_np), rtol=1e-10)
        np.testing.assert_allclose(to_numpy(ins.ceil(to_gpu(a_np))), np.ceil(a_np), rtol=1e-10)
        np.testing.assert_allclose(to_numpy(ins.round(to_gpu(a_np))), np.round(a_np), rtol=1e-10)

    def test_sign(self):
        a_np = np.array([-3, -1, 0, 2, 4], dtype=np.float64)
        result = ins.sign(to_gpu(a_np))
        np.testing.assert_allclose(to_numpy(result), np.sign(a_np), rtol=1e-10)

    def test_negative(self):
        a_np = np.array([1, -2, 3], dtype=np.float64)
        result = ins.negative(to_gpu(a_np))
        np.testing.assert_allclose(to_numpy(result), -a_np, rtol=1e-10)

    def test_square(self):
        a_np = np.array([1, 2, 3, 4], dtype=np.float64)
        result = ins.square(to_gpu(a_np))
        np.testing.assert_allclose(to_numpy(result), a_np**2, rtol=1e-10)

    def test_reciprocal(self):
        a_np = np.array([1, 2, 4, 5], dtype=np.float64)
        result = ins.reciprocal(to_gpu(a_np))
        np.testing.assert_allclose(to_numpy(result), 1.0 / a_np, rtol=1e-10)

    def test_log2_log10(self):
        a_np = np.array([1, 2, 8, 100], dtype=np.float64)
        np.testing.assert_allclose(to_numpy(ins.log2(to_gpu(a_np))), np.log2(a_np), rtol=1e-10)
        np.testing.assert_allclose(to_numpy(ins.log10(to_gpu(a_np))), np.log10(a_np), rtol=1e-10)

    def test_asin_acos_atan(self):
        a_np = np.array([-0.5, 0, 0.5], dtype=np.float64)
        np.testing.assert_allclose(to_numpy(ins.asin(to_gpu(a_np))), np.arcsin(a_np), rtol=1e-10)
        np.testing.assert_allclose(to_numpy(ins.acos(to_gpu(a_np))), np.arccos(a_np), rtol=1e-10)
        np.testing.assert_allclose(to_numpy(ins.atan(to_gpu(a_np))), np.arctan(a_np), rtol=1e-10)

    def test_sinh_cosh_tanh(self):
        a_np = np.array([-1, 0, 1], dtype=np.float64)
        np.testing.assert_allclose(to_numpy(ins.sinh(to_gpu(a_np))), np.sinh(a_np), rtol=1e-10)
        np.testing.assert_allclose(to_numpy(ins.cosh(to_gpu(a_np))), np.cosh(a_np), rtol=1e-10)
        np.testing.assert_allclose(to_numpy(ins.tanh(to_gpu(a_np))), np.tanh(a_np), rtol=1e-10)

    def test_deg2rad_rad2deg(self):
        a_np = np.array([0, 90, 180, 360], dtype=np.float64)
        np.testing.assert_allclose(
            to_numpy(ins.deg2rad(to_gpu(a_np))), np.deg2rad(a_np), rtol=1e-10
        )
        b_np = np.array([0, np.pi / 2, np.pi, 2 * np.pi], dtype=np.float64)
        np.testing.assert_allclose(
            to_numpy(ins.rad2deg(to_gpu(b_np))), np.rad2deg(b_np), rtol=1e-10
        )

    def test_exp2_expm1_log1p(self):
        a_np = np.array([0, 1, 2, 3], dtype=np.float64)
        np.testing.assert_allclose(to_numpy(ins.exp2(to_gpu(a_np))), np.exp2(a_np), rtol=1e-10)
        np.testing.assert_allclose(to_numpy(ins.expm1(to_gpu(a_np))), np.expm1(a_np), rtol=1e-10)
        np.testing.assert_allclose(to_numpy(ins.log1p(to_gpu(a_np))), np.log1p(a_np), rtol=1e-10)

    def test_cbrt(self):
        a_np = np.array([1, 8, 27], dtype=np.float64)
        result = ins.cbrt(to_gpu(a_np))
        np.testing.assert_allclose(to_numpy(result), np.cbrt(a_np), rtol=1e-10)

    def test_trunc(self):
        a_np = np.array([1.2, -2.7, 3.5], dtype=np.float64)
        result = ins.trunc(to_gpu(a_np))
        np.testing.assert_allclose(to_numpy(result), np.trunc(a_np), rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
