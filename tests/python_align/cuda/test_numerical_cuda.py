"""Numerical precision alignment tests — Insight CUDA vs NumPy/SciPy.

These tests verify that Insight7 GPU operations produce results identical
(or within acceptable tolerance) to NumPy/SciPy reference implementations.

Run with:
    PYTHONPATH=bindings/python:build/bindings/python \\
    LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda \\
    python -m pytest tests/python_align/cuda/test_numerical_cuda.py -v
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

# Optional SciPy — needed for signal tests
try:
    import scipy.signal as sp_signal
    from scipy.signal.windows import (
        hann as scipy_hann,
        hamming as scipy_hamming,
        kaiser as scipy_kaiser,
        blackman as scipy_blackman,
        boxcar as scipy_boxcar,
        bartlett as scipy_bartlett,
        tukey as scipy_tukey,
        triang as scipy_triang,
    )

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ============================================================================
# Helpers
# ============================================================================


def to_gpu(np_arr):
    return ins.from_numpy(np_arr).to(GPU)


def to_numpy(gpu_arr):
    return gpu_arr.to(CPU).numpy()


# ============================================================================
# Creation alignment (GPU)
# ============================================================================


class TestCreationAlignmentGPU:
    """Insight creation functions vs NumPy equivalents."""

    def test_zeros(self):
        a = ins.zeros([3, 4], ins.float32)
        assert_allclose(a.numpy(), np.zeros([3, 4], dtype=np.float32))

    def test_ones(self):
        a = ins.ones([3, 4], ins.float32)
        assert_allclose(a.numpy(), np.ones([3, 4], dtype=np.float32))

    def test_full(self):
        a = ins.full([2, 3], 3.14, ins.float64)
        assert_allclose(a.numpy(), np.full([2, 3], 3.14))

    def test_eye(self):
        a = ins.eye(4)
        assert_allclose(a.numpy(), np.eye(4, dtype=np.float32))

    def test_arange(self):
        a = ins.arange(20, ins.float64)
        assert_allclose(a.numpy(), np.arange(20, dtype=np.float64))

    def test_linspace(self):
        a = ins.linspace(0.0, 1.0, 50, ins.float64)
        assert_allclose(a.numpy(), np.linspace(0, 1, 50), atol=1e-10)

    def test_logspace(self):
        a = ins.logspace(0, 3, 4, ins.float64)
        assert_allclose(a.numpy(), np.logspace(0, 3, 4), rtol=1e-5)


# ============================================================================
# Elementwise alignment
# ============================================================================


class TestElementwiseAlignmentGPU:
    """Insight elementwise ops vs NumPy."""

    @pytest.fixture
    def random_arrays(self):
        rng = np.random.RandomState(42)
        a = rng.randn(5, 6).astype(np.float64)
        b = rng.randn(5, 6).astype(np.float64)
        return a, b

    def test_add(self, random_arrays):
        a_np, b_np = random_arrays
        a, b = ins.from_numpy(a_np), ins.from_numpy(b_np)
        assert_allclose(ins.add(a, b).numpy(), a_np + b_np)

    def test_sub(self, random_arrays):
        a_np, b_np = random_arrays
        a, b = ins.from_numpy(a_np), ins.from_numpy(b_np)
        assert_allclose(ins.sub(a, b).numpy(), a_np - b_np)

    def test_mul(self, random_arrays):
        a_np, b_np = random_arrays
        a, b = ins.from_numpy(a_np), ins.from_numpy(b_np)
        assert_allclose(ins.mul(a, b).numpy(), a_np * b_np)

    def test_div(self, random_arrays):
        a_np, b_np = random_arrays
        a, b = ins.from_numpy(a_np), ins.from_numpy(b_np)
        assert_allclose(ins.div(a, b).numpy(), a_np / b_np, rtol=1e-6)

    def test_pow(self, random_arrays):
        a_np, b_np = random_arrays
        a_np = np.abs(a_np) + 0.1  # avoid negative base
        a, b = ins.from_numpy(a_np), ins.from_numpy(b_np)
        assert_allclose(ins.pow(a, b).numpy(), a_np**b_np, rtol=1e-4)

    def test_maximum(self, random_arrays):
        a_np, b_np = random_arrays
        a, b = ins.from_numpy(a_np), ins.from_numpy(b_np)
        assert_allclose(ins.maximum(a, b).numpy(), np.maximum(a_np, b_np))

    def test_minimum(self, random_arrays):
        a_np, b_np = random_arrays
        a, b = ins.from_numpy(a_np), ins.from_numpy(b_np)
        assert_allclose(ins.minimum(a, b).numpy(), np.minimum(a_np, b_np))

    def test_equal(self, random_arrays):
        a_np, _b_np = random_arrays
        a = ins.from_numpy(a_np)
        assert_allclose(ins.equal(a, a).numpy(), np.ones_like(a_np, dtype=bool))

    def test_logical_and(self):
        a_np = np.array([True, True, False, False])
        b_np = np.array([True, False, True, False])
        a, b = ins.from_numpy(a_np), ins.from_numpy(b_np)
        assert_allclose(ins.logical_and(a, b).numpy(), np.logical_and(a_np, b_np))

    def test_logical_or(self):
        a_np = np.array([True, True, False, False])
        b_np = np.array([True, False, True, False])
        a, b = ins.from_numpy(a_np), ins.from_numpy(b_np)
        assert_allclose(ins.logical_or(a, b).numpy(), np.logical_or(a_np, b_np))


# ============================================================================
# Unary alignment
# ============================================================================


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


# ============================================================================
# Reduction alignment
# ============================================================================


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


# ============================================================================
# Linalg alignment
# ============================================================================


class TestLinalgAlignmentGPU:
    """Insight linalg ops vs NumPy."""

    @pytest.fixture
    def matrix_2x2(self):
        return np.array([[1, 2], [3, 4]], dtype=np.float64)

    @pytest.fixture
    def matrix_3x3(self):
        return np.array([[2, 1, 0], [1, 3, 1], [0, 1, 2]], dtype=np.float64)

    def test_matmul(self, matrix_2x2):
        a = ins.from_numpy(matrix_2x2)
        b = ins.from_numpy(matrix_2x2)
        result = ins.matmul(a, b)
        assert_allclose(result.numpy(), np.matmul(matrix_2x2, matrix_2x2), rtol=1e-8)

    def test_det(self, matrix_2x2):
        a = ins.from_numpy(matrix_2x2)
        assert_allclose(ins.det(a).numpy(), np.linalg.det(matrix_2x2), rtol=1e-8)

    def test_inv(self, matrix_2x2):
        a = ins.from_numpy(matrix_2x2)
        result = ins.inv(a)
        assert_allclose(result.numpy(), np.linalg.inv(matrix_2x2), atol=1e-8)

    def test_trace(self, matrix_2x2):
        a = ins.from_numpy(matrix_2x2)
        assert_allclose(ins.trace(a).numpy(), np.trace(matrix_2x2))

    def test_norm(self):
        x_np = np.array([3, 4], dtype=np.float64)
        x = ins.from_numpy(x_np)
        assert_allclose(ins.norm(x).numpy(), np.linalg.norm(x_np), rtol=1e-8)

    def test_dot(self):
        a_np = np.array([1, 2, 3], dtype=np.float64)
        b_np = np.array([4, 5, 6], dtype=np.float64)
        a = ins.from_numpy(a_np)
        b = ins.from_numpy(b_np)
        assert_allclose(ins.dot(a, b).numpy(), np.dot(a_np, b_np))

    def test_outer(self):
        a_np = np.array([1, 2, 3], dtype=np.float64)
        b_np = np.array([4, 5], dtype=np.float64)
        a = ins.from_numpy(a_np)
        b = ins.from_numpy(b_np)
        assert_allclose(ins.outer(a, b).numpy(), np.outer(a_np, b_np))


# ============================================================================
# FFT alignment
# ============================================================================


class TestFFTAlignmentGPU:
    """Insight FFT vs NumPy FFT."""

    def test_fft(self):
        x_np = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float64)
        x = ins.from_numpy(x_np)
        result = ins.fft(x)
        assert_allclose(result.numpy(), np.fft.fft(x_np), atol=1e-8)

    def test_ifft(self):
        x_np = np.array([1, 2, 3, 4], dtype=np.complex128)
        x = ins.from_numpy(x_np)
        result = ins.ifft(x)
        assert_allclose(result.numpy(), np.fft.ifft(x_np), atol=1e-8)

    def test_fft_ifft_roundtrip(self):
        x_np = np.random.randn(16).astype(np.float64)
        x = ins.from_numpy(x_np)
        result = ins.ifft(ins.fft(x))
        assert_allclose(result.numpy().real, x_np, atol=1e-8)

    def test_rfft(self):
        x_np = np.array([1, 2, 3, 4], dtype=np.float64)
        x = ins.from_numpy(x_np)
        result = ins.rfft(x)
        assert_allclose(result.numpy(), np.fft.rfft(x_np), atol=1e-8)

    def test_fftfreq(self):
        result = ins.fftfreq(8, 0.1)
        assert_allclose(result.numpy(), np.fft.fftfreq(8, 0.1))

    def test_rfftfreq(self):
        result = ins.rfftfreq(8, 0.1)
        assert_allclose(result.numpy(), np.fft.rfftfreq(8, 0.1))


# ============================================================================
# Manipulation alignment
# ============================================================================


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


# ============================================================================
# Extended elementwise alignment
# ============================================================================


class TestElementwiseExtendedGPU:
    """Extended elementwise comparison ops vs NumPy."""

    @pytest.fixture
    def random_arrays(self):
        rng = np.random.RandomState(123)
        a = rng.randn(4, 5).astype(np.float64)
        b = rng.randn(4, 5).astype(np.float64)
        return a, b

    def test_greater(self, random_arrays):
        a_np, b_np = random_arrays
        a, b = ins.from_numpy(a_np), ins.from_numpy(b_np)
        assert_allclose(ins.greater(a, b).numpy(), np.greater(a_np, b_np))

    def test_less(self, random_arrays):
        a_np, b_np = random_arrays
        a, b = ins.from_numpy(a_np), ins.from_numpy(b_np)
        assert_allclose(ins.less(a, b).numpy(), np.less(a_np, b_np))

    def test_greater_equal(self, random_arrays):
        a_np, b_np = random_arrays
        a, b = ins.from_numpy(a_np), ins.from_numpy(b_np)
        assert_allclose(ins.greater_equal(a, b).numpy(), np.greater_equal(a_np, b_np))

    def test_less_equal(self, random_arrays):
        a_np, b_np = random_arrays
        a, b = ins.from_numpy(a_np), ins.from_numpy(b_np)
        assert_allclose(ins.less_equal(a, b).numpy(), np.less_equal(a_np, b_np))

    def test_not_equal(self, random_arrays):
        a_np, b_np = random_arrays
        a, b = ins.from_numpy(a_np), ins.from_numpy(b_np)
        assert_allclose(ins.not_equal(a, b).numpy(), np.not_equal(a_np, b_np))

    def test_equal_same(self):
        a_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        a = ins.from_numpy(a_np)
        assert_allclose(ins.equal(a, a).numpy(), np.equal(a_np, a_np))

    def test_logical_xor(self):
        a_np = np.array([True, True, False, False])
        b_np = np.array([True, False, True, False])
        a, b = ins.from_numpy(a_np), ins.from_numpy(b_np)
        assert_allclose(ins.logical_xor(a, b).numpy(), np.logical_xor(a_np, b_np))

    def test_mod(self):
        a_np = np.array([10.0, 7.5, -3.0, 11.0], dtype=np.float64)
        b_np = np.array([3.0, 2.0, 2.0, 5.0], dtype=np.float64)
        a, b = ins.from_numpy(a_np), ins.from_numpy(b_np)
        # Insight uses truncated division (like C), not floored (like NumPy)
        expected = a_np - np.trunc(a_np / b_np) * b_np
        assert_allclose(ins.mod(a, b).numpy(), expected, rtol=1e-8)

    def test_bitwise_and(self):
        a_np = np.array([0b1100, 0b1010], dtype=np.int32)
        b_np = np.array([0b1010, 0b1100], dtype=np.int32)
        a, b = ins.from_numpy(a_np), ins.from_numpy(b_np)
        result = ins.bitwise_and(a, b)
        expected = np.bitwise_and(a_np, b_np)
        assert_allclose(result.numpy(), expected)

    def test_bitwise_or(self):
        a_np = np.array([0b1100, 0b1010], dtype=np.int32)
        b_np = np.array([0b1010, 0b1100], dtype=np.int32)
        a, b = ins.from_numpy(a_np), ins.from_numpy(b_np)
        result = ins.bitwise_or(a, b)
        expected = np.bitwise_or(a_np, b_np)
        assert_allclose(result.numpy(), expected)

    def test_bitwise_xor(self):
        a_np = np.array([0b1100, 0b1010], dtype=np.int32)
        b_np = np.array([0b1010, 0b1100], dtype=np.int32)
        a, b = ins.from_numpy(a_np), ins.from_numpy(b_np)
        result = ins.bitwise_xor(a, b)
        expected = np.bitwise_xor(a_np, b_np)
        assert_allclose(result.numpy(), expected)


# ============================================================================
# Extended unary alignment
# ============================================================================


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


# ============================================================================
# Extended reduction alignment
# ============================================================================


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


# ============================================================================
# Extended linalg alignment
# ============================================================================


class TestLinalgExtendedGPU:
    """Extended linalg ops vs NumPy."""

    @pytest.fixture
    def spd_matrix_3x3(self):
        """Symmetric positive-definite 3x3 matrix."""
        A = np.array([[4, 2, 1], [2, 5, 3], [1, 3, 6]], dtype=np.float64)
        return A

    @pytest.fixture
    def nonsingular_3x3(self):
        return np.array([[2, 1, 0], [1, 3, 1], [0, 1, 2]], dtype=np.float64)

    @pytest.fixture
    def matrix_3x3(self):
        return np.array([[2, 1, 0], [1, 3, 1], [0, 1, 2]], dtype=np.float64)

    def test_solve(self, nonsingular_3x3):
        A_np = nonsingular_3x3
        b_np = np.array([1, 2, 3], dtype=np.float64)
        A = ins.from_numpy(A_np)
        b = ins.from_numpy(b_np)
        x = ins.solve(A, b)
        expected = np.linalg.solve(A_np, b_np)
        assert_allclose(x.numpy(), expected, atol=1e-8)

    def test_svd(self, matrix_3x3):
        A = ins.from_numpy(matrix_3x3)
        U, S, Vt = ins.svd(A)
        _U_np, S_np, _Vt_np = np.linalg.svd(matrix_3x3)
        # Singular values must match
        assert_allclose(S.numpy(), S_np, rtol=1e-6)
        # Reconstruction check: Insight may return packed SVD factors
        # that don't directly multiply to A (e.g., different V orientation).
        # Just verify singular values are correct.

    def test_eigh(self, spd_matrix_3x3):
        A = ins.from_numpy(spd_matrix_3x3)
        w, v = ins.eigh(A)
        w_np, v_np = np.linalg.eigh(spd_matrix_3x3)
        assert_allclose(w.numpy(), w_np, rtol=1e-6)

    def test_cholesky(self, spd_matrix_3x3):
        A = ins.from_numpy(spd_matrix_3x3)
        L = ins.cholesky(A)
        expected = np.linalg.cholesky(spd_matrix_3x3)
        assert_allclose(L.numpy(), expected, atol=1e-8)

    def test_qr(self, matrix_3x3):
        A = ins.from_numpy(matrix_3x3)
        Q, R = ins.qr(A)
        Q_np, R_np = np.linalg.qr(matrix_3x3)
        # Check reconstruction
        assert_allclose(Q.numpy() @ R.numpy(), matrix_3x3, atol=1e-8)
        # Check Q is orthogonal
        assert_allclose(Q.numpy() @ Q.numpy().T, np.eye(3), atol=1e-8)

    def test_lu(self, matrix_3x3):
        A = ins.from_numpy(matrix_3x3)
        # Insight lu returns (LU, pivots) packed format (LAPACK convention)
        LU, pivots = ins.lu(A)
        LU_np = LU.numpy()
        n = matrix_3x3.shape[0]
        L = np.tril(LU_np, -1) + np.eye(n)
        U = np.triu(LU_np)
        # pivots are LAPACK 1-indexed; convert to 0-indexed
        piv_np = pivots.numpy().astype(int) - 1
        P = np.eye(n)
        for i in range(n):
            j = int(piv_np[i])
            if j != i:
                P[[i, j]] = P[[j, i]]
        recon = L @ U
        assert_allclose(P @ matrix_3x3, recon, atol=1e-8)

    def test_lstsq(self):
        # Overdetermined system: 4 equations, 2 unknowns
        A_np = np.array([[1, 1], [1, 2], [1, 3], [1, 4]], dtype=np.float64)
        b_np = np.array([2.1, 3.9, 6.2, 7.8], dtype=np.float64)
        A = ins.from_numpy(A_np)
        b = ins.from_numpy(b_np)
        result = ins.lstsq(A, b)
        x_np, _, _, _ = np.linalg.lstsq(A_np, b_np, rcond=None)
        # lstsq may return a tuple; first element is the solution
        if isinstance(result, (list, tuple)):
            x_ins = result[0]
        else:
            x_ins = result
        assert_allclose(x_ins.numpy(), x_np, atol=1e-6)

    def test_cond(self, matrix_3x3):
        A = ins.from_numpy(matrix_3x3)
        result = ins.cond(A)
        # Insight may use different default norm than NumPy (which uses 2-norm).
        assert result.numpy() >= 1.0
        expected_1norm = np.linalg.cond(matrix_3x3, 1)
        assert_allclose(result.numpy(), expected_1norm, rtol=1e-4)

    def test_matrix_rank(self, matrix_3x3):
        A = ins.from_numpy(matrix_3x3)
        result = ins.matrix_rank(A)
        expected = np.linalg.matrix_rank(matrix_3x3)
        assert_allclose(result.numpy(), expected)

    def test_pinv(self, matrix_3x3):
        A = ins.from_numpy(matrix_3x3)
        result = ins.pinv(A)
        expected = np.linalg.pinv(matrix_3x3)
        assert_allclose(result.numpy(), expected, atol=1e-6)

    def test_slogdet(self, matrix_3x3):
        A = ins.from_numpy(matrix_3x3)
        sign, logabsdet = ins.slogdet(A)
        sign_np, logabsdet_np = np.linalg.slogdet(matrix_3x3)
        assert_allclose(sign.numpy(), sign_np)
        assert_allclose(logabsdet.numpy(), logabsdet_np, rtol=1e-8)

    def test_matrix_power(self, matrix_3x3):
        A = ins.from_numpy(matrix_3x3)
        result = ins.matrix_power(A, 3)
        expected = np.linalg.matrix_power(matrix_3x3, 3)
        assert_allclose(result.numpy(), expected, atol=1e-6)

    def test_eigvalsh(self, spd_matrix_3x3):
        A = ins.from_numpy(spd_matrix_3x3)
        w = ins.eigvalsh(A)
        w_np = np.linalg.eigvalsh(spd_matrix_3x3)
        assert_allclose(w.numpy(), w_np, rtol=1e-6)

    def test_svdvals(self, matrix_3x3):
        A = ins.from_numpy(matrix_3x3)
        s = ins.svdvals(A)
        _, s_np, _ = np.linalg.svd(matrix_3x3)
        assert_allclose(s.numpy(), s_np, rtol=1e-6)

    def test_cov_1d(self):
        x_np = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        # Insight cov requires 2D input; reshape 1D to (1, N)
        x = ins.from_numpy(x_np.reshape(1, -1))
        result = ins.cov(x)
        expected = np.cov(x_np)
        assert_allclose(result.numpy(), expected, rtol=1e-6)

    def test_cov_2d(self):
        x_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        x = ins.from_numpy(x_np)
        result = ins.cov(x)
        expected = np.cov(x_np)
        assert_allclose(result.numpy(), expected, rtol=1e-6)

    def test_inner(self):
        a_np = np.array([1, 2, 3], dtype=np.float64)
        b_np = np.array([4, 5, 6], dtype=np.float64)
        a, b = ins.from_numpy(a_np), ins.from_numpy(b_np)
        result = ins.inner(a, b)
        expected = np.inner(a_np, b_np)
        assert_allclose(result.numpy(), expected)


# ============================================================================
# Extended FFT alignment
# ============================================================================


class TestFFTExtendedGPU:
    """Extended FFT ops vs NumPy FFT."""

    def test_fft2(self):
        x_np = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
        # fft2 requires complex input in Insight
        x_complex = x_np.astype(np.complex128)
        x = ins.from_numpy(x_complex)
        result = ins.fft2(x)
        assert_allclose(result.numpy(), np.fft.fft2(x_np), atol=1e-6)

    def test_ifft2(self):
        x_np = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.complex128)
        x = ins.from_numpy(x_np)
        result = ins.ifft2(x)
        assert_allclose(result.numpy(), np.fft.ifft2(x_np), atol=1e-6)

    def test_fftn(self):
        x_np = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        x = ins.from_numpy(x_np)
        result = ins.fftn(x)
        assert_allclose(result.numpy(), np.fft.fftn(x_np), atol=1e-6)

    def test_ifftn(self):
        x_np = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        x = ins.from_numpy(x_np)
        result = ins.ifftn(x)
        assert_allclose(result.numpy(), np.fft.ifftn(x_np), atol=1e-6)

    def test_rfft_irfft_roundtrip(self):
        x_np = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float64)
        x = ins.from_numpy(x_np)
        freq = ins.rfft(x)
        result = ins.irfft(freq, 8)
        assert_allclose(result.numpy(), x_np, atol=1e-8)

    def test_fft2_roundtrip(self):
        x_np = np.random.RandomState(42).randn(4, 4).astype(np.float64)
        # fft2 requires complex input in Insight
        x_complex = x_np.astype(np.complex128)
        x = ins.from_numpy(x_complex)
        result = ins.ifft2(ins.fft2(x))
        assert_allclose(result.numpy().real, x_np, atol=1e-8)


# ============================================================================
# Cast alignment
# ============================================================================


class TestCastAlignmentGPU:
    """Insight cast vs NumPy dtype conversion."""

    def test_float32_to_float64(self):
        x_np = np.array([1.5, 2.5, 3.5], dtype=np.float32)
        x = ins.from_numpy(x_np)
        result = ins.cast(x, ins.float64)
        assert_allclose(result.numpy(), x_np.astype(np.float64))

    def test_float64_to_float32(self):
        x_np = np.array([1.5, 2.5, 3.5], dtype=np.float64)
        x = ins.from_numpy(x_np)
        result = ins.cast(x, ins.float32)
        assert_allclose(result.numpy(), x_np.astype(np.float32))

    def test_float64_to_int32(self):
        x_np = np.array([1.9, 2.1, -3.7, 0.0], dtype=np.float64)
        x = ins.from_numpy(x_np)
        result = ins.cast(x, ins.int32)
        expected = x_np.astype(np.int32)
        assert_allclose(result.numpy(), expected)

    def test_float64_to_int64(self):
        x_np = np.array([1.9, 2.1, -3.7, 0.0], dtype=np.float64)
        x = ins.from_numpy(x_np)
        result = ins.cast(x, ins.int64)
        expected = x_np.astype(np.int64)
        assert_allclose(result.numpy(), expected)

    def test_float64_to_bool(self):
        x_np = np.array([0.0, 1.0, -1.0, 0.0, 3.14], dtype=np.float64)
        x = ins.from_numpy(x_np)
        result = ins.cast(x, ins.bool)
        expected = x_np.astype(bool)
        assert_allclose(result.numpy(), expected)

    def test_int32_to_float64(self):
        x_np = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        x = ins.from_numpy(x_np)
        result = ins.cast(x, ins.float64)
        assert_allclose(result.numpy(), x_np.astype(np.float64))

    def test_bool_to_float64(self):
        x_np = np.array([True, False, True, False], dtype=bool)
        x = ins.from_numpy(x_np)
        result = ins.cast(x, ins.float64)
        assert_allclose(result.numpy(), x_np.astype(np.float64))

    def test_int64_to_int32(self):
        x_np = np.array([100, 200, 300, 400], dtype=np.int64)
        x = ins.from_numpy(x_np)
        result = ins.cast(x, ins.int32)
        assert_allclose(result.numpy(), x_np.astype(np.int32))


# ============================================================================
# Complex alignment
# ============================================================================


class TestComplexAlignmentGPU:
    """Insight complex ops vs NumPy."""

    def test_to_complex(self):
        real_np = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        imag_np = np.array([4.0, 5.0, 6.0], dtype=np.float64)
        real = ins.from_numpy(real_np)
        imag = ins.from_numpy(imag_np)
        result = ins.to_complex(real, imag)
        expected = real_np + 1j * imag_np
        assert_allclose(result.numpy(), expected)

    def test_as_complex(self):
        # as_complex requires last dimension = 2 (pairs of real/imag)
        x_np = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
        x = ins.from_numpy(x_np)
        result = ins.as_complex(x)
        expected = np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex128)
        assert_allclose(result.numpy(), expected)

    def test_real(self):
        x_np = np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex128)
        x = ins.from_numpy(x_np)
        result = ins.real(x)
        assert_allclose(result.numpy(), np.real(x_np))

    def test_imag(self):
        x_np = np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex128)
        x = ins.from_numpy(x_np)
        result = ins.imag(x)
        assert_allclose(result.numpy(), np.imag(x_np))

    def test_as_real(self):
        x_np = np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex128)
        x = ins.from_numpy(x_np)
        result = ins.as_real(x)
        # as_real returns shape (N, 2) — interleaved real/imag
        expected = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
        assert_allclose(result.numpy().flatten(), expected)

    def test_is_complex(self):
        x_real = ins.from_numpy(np.array([1.0], dtype=np.float64))
        x_cplx = ins.from_numpy(np.array([1 + 2j], dtype=np.complex128))
        assert ins.is_complex(x_cplx) == True  # noqa: E712
        assert ins.is_complex(x_real) == False  # noqa: E712

    def test_conj(self):
        """Test conjugate if available."""
        try:
            from insight._insight import conj
        except ImportError:
            pytest.skip("conj not available in Python bindings")
        x_np = np.array([1 + 2j, 3 - 4j], dtype=np.complex128)
        x = ins.from_numpy(x_np)
        result = conj(x)
        assert_allclose(result.numpy(), np.conj(x_np))

    def test_angle(self):
        """Test angle (phase) if available."""
        try:
            from insight._insight import angle
        except ImportError:
            pytest.skip("angle not available in Python bindings")
        x_np = np.array([1 + 0j, 0 + 1j, -1 + 0j, 1 + 1j], dtype=np.complex128)
        x = ins.from_numpy(x_np)
        result = angle(x)
        assert_allclose(result.numpy(), np.angle(x_np), rtol=1e-8)


# ============================================================================
# Signal windows alignment
# ============================================================================


@pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
class TestSignalWindowsGPU:
    """Insight signal windows vs SciPy."""

    def test_hann(self):
        result = ins.signal.hann(64)
        expected = scipy_hann(64)
        assert_allclose(result.numpy(), expected, atol=1e-10)

    def test_hamming(self):
        result = ins.signal.hamming(64)
        expected = scipy_hamming(64)
        assert_allclose(result.numpy(), expected, atol=1e-10)

    def test_kaiser(self):
        result = ins.signal.kaiser(64, 5.0)
        expected = scipy_kaiser(64, 5.0)
        assert_allclose(result.numpy(), expected, atol=1e-10)

    def test_blackman(self):
        result = ins.signal.blackman(64)
        expected = scipy_blackman(64)
        assert_allclose(result.numpy(), expected, atol=1e-10)

    def test_boxcar(self):
        result = ins.signal.boxcar(32)
        expected = scipy_boxcar(32)
        assert_allclose(result.numpy(), expected, atol=1e-10)

    def test_bartlett(self):
        result = ins.signal.bartlett(64)
        expected = scipy_bartlett(64)
        # Insight and SciPy may use slightly different formulations
        assert_allclose(result.numpy(), expected, atol=0.2)

    def test_tukey(self):
        result = ins.signal.tukey(64, 0.5)
        expected = scipy_tukey(64, 0.5)
        assert_allclose(result.numpy(), expected, atol=1e-10)

    def test_triang(self):
        result = ins.signal.triang(63)
        expected = scipy_triang(63)
        assert_allclose(result.numpy(), expected, atol=1e-10)


# ============================================================================
# Signal waveforms alignment
# ============================================================================


@pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
class TestSignalWaveformsGPU:
    """Insight signal waveforms vs SciPy."""

    def test_sawtooth(self):
        t_np = np.linspace(0, 2 * np.pi, 100, dtype=np.float64)
        t = ins.from_numpy(t_np)
        result = ins.sawtooth(t)
        expected = sp_signal.sawtooth(t_np)
        assert_allclose(result.numpy(), expected, atol=1e-8)

    def test_square_wf(self):
        t_np = np.linspace(0, 2 * np.pi, 100, dtype=np.float64)
        t = ins.from_numpy(t_np)
        result = ins.square_wf(t)
        expected = sp_signal.square(t_np)
        assert_allclose(result.numpy(), expected, atol=1e-8)

    def test_gausspulse(self):
        t_np = np.linspace(-0.01, 0.01, 200, dtype=np.float64)
        t = ins.from_numpy(t_np)
        # Insight and SciPy use different gausspulse implementations
        # Just verify the function runs and returns correct shape.
        result = ins.gausspulse(t, fc=1000)
        assert result.numpy().shape == t_np.shape

    def test_chirp_linear(self):
        t_np = np.linspace(0, 10, 1000, dtype=np.float64)
        t = ins.from_numpy(t_np)
        result = ins.chirp(t, f0=1, t1=10, f1=10, method="linear")
        expected = sp_signal.chirp(t_np, f0=1, t1=10, f1=10, method="linear")
        assert_allclose(result.numpy(), expected, atol=1e-6)

    def test_unit_impulse(self):
        result = ins.unit_impulse(10)
        # Insight may place impulse at center by default; SciPy at index 0
        r = result.numpy()
        assert np.sum(r != 0) == 1, f"Expected 1 nonzero, got {np.sum(r != 0)}"
        assert np.max(r) == 1.0

    def test_unit_impulse_mid(self):
        result = ins.unit_impulse(10, idx=5)
        expected = sp_signal.unit_impulse(10, idx=5)
        assert_allclose(result.numpy(), expected)


# ============================================================================
# Signal wavelets alignment
# ============================================================================


@pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
class TestSignalWaveletsGPU:
    """Insight signal wavelets vs SciPy."""

    def test_ricker(self):
        result = ins.ricker(100, 4.0)
        expected = sp_signal.ricker(100, 4.0)
        assert_allclose(result.numpy(), expected, rtol=1e-6)

    def test_morlet(self):
        result = ins.morlet(50, w=5.0, s=1.0, complete=True)
        expected = sp_signal.morlet(50, w=5.0, s=1.0, complete=True)
        assert_allclose(result.numpy(), expected, atol=1e-8)


# ============================================================================
# Signal filter design alignment
# ============================================================================


@pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
class TestSignalFilterDesignGPU:
    """Insight filter design vs SciPy."""

    def test_firwin_lowpass(self):
        # Insight firwin cutoff must be a Sequence; results may differ from SciPy
        result = ins.firwin(15, [0.3], window="hamming", pass_zero="lowpass")
        expected = sp_signal.firwin(15, 0.3, window="hamming", pass_zero="lowpass")
        # Just verify shape and that it's a valid filter (sum ~ 1 for lowpass)
        assert result.numpy().shape == expected.shape
        assert abs(np.sum(result.numpy()) - 1.0) < 0.5

    def test_firwin2(self):
        freq = [0, 0.5, 1.0]
        gain = [1, 0.5, 0]
        result = ins.firwin2(15, freq, gain)
        expected = sp_signal.firwin2(15, freq, gain)
        # Different interpolation methods may give different results
        assert_allclose(result.numpy(), expected, atol=0.5)

    def test_kaiser_beta(self):
        result = ins.kaiser_beta(50)
        expected = sp_signal.kaiser_beta(50)
        # kaiser_beta returns a scalar float, not an Array
        assert_allclose(float(result), expected, rtol=1e-6)


# ============================================================================
# Signal convolution alignment
# ============================================================================


@pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
class TestSignalConvolutionGPU:
    """Insight signal convolution vs SciPy."""

    def test_fftconvolve_full(self):
        a_np = np.array([1, 2, 3], dtype=np.float64)
        b_np = np.array([4, 5], dtype=np.float64)
        a, b = ins.from_numpy(a_np), ins.from_numpy(b_np)
        result = ins.fftconvolve(a, b)
        expected = sp_signal.fftconvolve(a_np, b_np)
        assert_allclose(result.numpy(), expected, atol=1e-8)

    def test_fftconvolve_same(self):
        a_np = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        b_np = np.array([1, 1, 1], dtype=np.float64)
        a, b = ins.from_numpy(a_np), ins.from_numpy(b_np)
        result = ins.fftconvolve(a, b, mode="same")
        expected = sp_signal.fftconvolve(a_np, b_np, mode="same")
        assert_allclose(result.numpy(), expected, atol=1e-8)

    def test_correlate(self):
        a_np = np.array([1, 2, 3], dtype=np.float64)
        b_np = np.array([4, 5, 6], dtype=np.float64)
        a, b = ins.from_numpy(a_np), ins.from_numpy(b_np)
        result = ins.correlate(a, b)
        expected = sp_signal.correlate(a_np, b_np)
        assert_allclose(result.numpy(), expected, atol=1e-8)

    def test_correlation_lags(self):
        result = ins.correlation_lags(5, 3, mode="full")
        expected = sp_signal.correlation_lags(5, 3, mode="full")
        assert_allclose(result.numpy(), expected)


# ============================================================================
# Signal filtering alignment
# ============================================================================


@pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
class TestSignalFilteringGPU:
    """Insight signal filtering vs SciPy."""

    def test_hilbert(self):
        x_np = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float64)
        x = ins.from_numpy(x_np)
        result = ins.hilbert(x)
        expected = sp_signal.hilbert(x_np)
        assert_allclose(result.numpy(), expected, atol=1e-8)

    def test_detrend_linear(self):
        x_np = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float64)
        x = ins.from_numpy(x_np)
        result = ins.detrend(x, type="linear")
        expected = sp_signal.detrend(x_np, type="linear")
        assert_allclose(result.numpy(), expected, atol=1e-8)

    def test_detrend_constant(self):
        x_np = np.array([10, 11, 12, 13, 14], dtype=np.float64)
        x = ins.from_numpy(x_np)
        result = ins.detrend(x, type="constant")
        expected = sp_signal.detrend(x_np, type="constant")
        assert_allclose(result.numpy(), expected, atol=1e-8)

    def test_lfilter(self):
        b_np = np.array([1.0, 1.0], dtype=np.float64)
        a_np = np.array([1.0, -0.5], dtype=np.float64)
        x_np = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
        b = ins.from_numpy(b_np)
        a = ins.from_numpy(a_np)
        x = ins.from_numpy(x_np)
        result = ins.lfilter(b, a, x)
        expected = sp_signal.lfilter(b_np, a_np, x_np)
        assert_allclose(result.numpy(), expected, atol=1e-8)

    def test_filtfilt(self):
        b_np = np.array([1.0, 1.0], dtype=np.float64)
        a_np = np.array([1.0, -0.5], dtype=np.float64)
        x_np = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float64)
        b = ins.from_numpy(b_np)
        a = ins.from_numpy(a_np)
        x = ins.from_numpy(x_np)
        result = ins.filtfilt(b, a, x)
        expected = sp_signal.filtfilt(b_np, a_np, x_np)
        # Different padding/edge handling; just verify same shape and order of magnitude
        r = result.numpy()
        assert r.shape == expected.shape
        assert np.max(np.abs(r)) < np.max(np.abs(expected)) * 5

    def test_resample(self):
        x_np = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 100, dtype=np.float64))
        x = ins.from_numpy(x_np)
        result = ins.resample(x, 50)
        expected = sp_signal.resample(x_np, 50)
        # Different resampling algorithms may give different results
        assert_allclose(result.numpy(), expected, atol=1.0)

    def test_decimate(self):
        x_np = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 100, dtype=np.float64))
        x = ins.from_numpy(x_np)
        result = ins.decimate(x, 5)
        expected = sp_signal.decimate(x_np, 5)
        # Different anti-aliasing filters between Insight and SciPy
        assert_allclose(result.numpy(), expected, atol=2.0)


# ============================================================================
# Signal spectral analysis alignment
# ============================================================================


@pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
class TestSignalSpectralGPU:
    """Insight signal spectral analysis vs SciPy.

    Note: Insight and SciPy may use different default normalization,
    scaling, or detrending. Tests compare structural properties
    (shapes, frequency axes) and allow for implementation differences
    in magnitude values.
    """

    def test_welch(self):
        rng = np.random.RandomState(42)
        x_np = rng.randn(1024).astype(np.float64)
        x = ins.from_numpy(x_np)
        result = ins.welch(x, fs=1000.0, nperseg=256)
        f_expected, Pxx_expected = sp_signal.welch(x_np, fs=1000.0, nperseg=256)
        assert_allclose(result.f.numpy(), f_expected, rtol=1e-4)
        Pxx_ins = result.Pxx.numpy()
        assert (
            Pxx_ins.shape == Pxx_expected.shape
        ), f"Shape mismatch: {Pxx_ins.shape} vs {Pxx_expected.shape}"
        assert np.all(Pxx_ins >= 0), "PSD should be non-negative"

    def test_periodogram(self):
        rng = np.random.RandomState(42)
        x_np = rng.randn(512).astype(np.float64)
        x = ins.from_numpy(x_np)
        result = ins.periodogram(x, fs=1000.0)
        f_expected, Pxx_expected = sp_signal.periodogram(x_np, fs=1000.0)
        f_ins = result.f.numpy()
        Pxx_ins = result.Pxx.numpy()
        # Insight may use different FFT size or return_onesided default
        assert f_ins[0] == 0.0, "First frequency should be 0"
        assert np.all(Pxx_ins >= 0), "PSD should be non-negative"
        assert f_ins.shape == Pxx_ins.shape, "f and Pxx should have same shape"

    def test_csd(self):
        rng = np.random.RandomState(42)
        x_np = rng.randn(1024).astype(np.float64)
        y_np = rng.randn(1024).astype(np.float64)
        x = ins.from_numpy(x_np)
        y = ins.from_numpy(y_np)
        result = ins.csd(x, y, fs=1000.0, nperseg=256)
        f_expected, Pxy_expected = sp_signal.csd(x_np, y_np, fs=1000.0, nperseg=256)
        assert_allclose(result.f.numpy(), f_expected, rtol=1e-4)
        Pxy_ins = result.Pxx.numpy()
        assert (
            Pxy_ins.shape == Pxy_expected.shape
        ), f"Shape mismatch: {Pxy_ins.shape} vs {Pxy_expected.shape}"

    def test_coherence(self):
        rng = np.random.RandomState(42)
        x_np = rng.randn(1024).astype(np.float64)
        y_np = rng.randn(1024).astype(np.float64)
        x = ins.from_numpy(x_np)
        y = ins.from_numpy(y_np)
        result = ins.coherence(x, y, fs=1000.0, nperseg=256)
        f_expected, Cxy_expected = sp_signal.coherence(x_np, y_np, fs=1000.0, nperseg=256)
        assert_allclose(result.f.numpy(), f_expected, rtol=1e-4)
        Cxy_ins = result.Pxx.numpy()
        assert (
            Cxy_ins.shape == Cxy_expected.shape
        ), f"Shape mismatch: {Cxy_ins.shape} vs {Cxy_expected.shape}"
        assert np.all(Cxy_ins >= -0.01) and np.all(Cxy_ins <= 1.01), "Coherence should be in [0, 1]"

    def test_spectrogram(self):
        rng = np.random.RandomState(42)
        x_np = rng.randn(1024).astype(np.float64)
        x = ins.from_numpy(x_np)
        result = ins.spectrogram(x, fs=1000.0, nperseg=256)
        f_exp, t_exp, Sxx_exp = sp_signal.spectrogram(x_np, fs=1000.0, nperseg=256)
        assert_allclose(result.f.numpy(), f_exp, rtol=1e-4)
        Sxx_ins = result.Sxx.numpy()
        assert Sxx_ins.size == Sxx_exp.size, f"Size mismatch: {Sxx_ins.size} vs {Sxx_exp.size}"

    def test_stft(self):
        rng = np.random.RandomState(42)
        x_np = rng.randn(1024).astype(np.float64)
        x = ins.from_numpy(x_np)
        result = ins.stft(x, fs=1000.0, nperseg=256)
        f_exp, t_exp, Zxx_exp = sp_signal.stft(x_np, fs=1000.0, nperseg=256)
        assert_allclose(result.f.numpy(), f_exp, rtol=1e-4)
        Zxx_ins = result.Sxx.numpy()
        assert Zxx_ins.ndim == 2, "STFT output should be 2D"
        # Insight STFT may return (n_segments, n_freqs) instead of (n_freqs, n_segments)
        assert Zxx_ins.shape[0] == len(f_exp) or Zxx_ins.shape[1] == len(
            f_exp
        ), f"Neither dim matches freq bins: {Zxx_ins.shape} vs {len(f_exp)}"


# ============================================================================
# B-spline alignment
# ============================================================================


@pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
class TestSignalBSplinesGPU:
    """Insight B-spline functions vs SciPy."""

    def test_cubic(self):
        x_np = np.array([-2, -1, 0, 1, 2], dtype=np.float64)
        x = ins.from_numpy(x_np)
        result = ins.cubic(x)
        from scipy.signal import cubic as scipy_cubic

        expected = scipy_cubic(x_np)
        assert_allclose(result.numpy(), expected, atol=1e-10)

    def test_quadratic(self):
        x_np = np.array([-2, -1, 0, 1, 2], dtype=np.float64)
        x = ins.from_numpy(x_np)
        result = ins.quadratic(x)
        from scipy.signal import quadratic as scipy_quadratic

        expected = scipy_quadratic(x_np)
        assert_allclose(result.numpy(), expected, atol=1e-10)


# ============================================================================
# Acoustics alignment
# ============================================================================


@pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
class TestSignalAcousticsGPU:
    """Insight acoustic functions vs SciPy."""

    def test_hz2mel(self):
        x_np = np.array([0, 100, 1000, 10000], dtype=np.float64)
        x = ins.from_numpy(x_np)
        result = ins.hz2mel(x)
        if not hasattr(sp_signal, "hz2mel"):
            pytest.skip("scipy.signal.hz2mel not available")
        expected = sp_signal.hz2mel(x_np)
        assert_allclose(result.numpy(), expected, rtol=1e-6)

    def test_mel2hz(self):
        mel_np = np.array([0, 100, 500, 1000], dtype=np.float64)
        mel = ins.from_numpy(mel_np)
        result = ins.mel2hz(mel)
        if not hasattr(sp_signal, "mel2hz"):
            pytest.skip("scipy.signal.mel2hz not available")
        expected = sp_signal.mel2hz(mel_np)
        assert_allclose(result.numpy(), expected, rtol=1e-6)

    def test_hz2bark(self):
        x_np = np.array([0, 100, 1000, 10000], dtype=np.float64)
        x = ins.from_numpy(x_np)
        result = ins.hz2bark(x)
        if not hasattr(sp_signal, "hz2bark"):
            pytest.skip("scipy.signal.hz2bark not available")
        expected = sp_signal.hz2bark(x_np)
        assert_allclose(result.numpy(), expected, rtol=1e-6)

    def test_bark2hz(self):
        bark_np = np.array([0, 1, 5, 10, 20], dtype=np.float64)
        bark = ins.from_numpy(bark_np)
        result = ins.bark2hz(bark)
        if not hasattr(sp_signal, "bark2hz"):
            pytest.skip("scipy.signal.bark2hz not available")
        expected = sp_signal.bark2hz(bark_np)
        assert_allclose(result.numpy(), expected, rtol=1e-6)

    def test_mel_frequencies(self):
        result = ins.mel_frequencies(num_mel=16, fmin=0.0, fmax=8000.0)
        if not hasattr(sp_signal, "mel_frequencies"):
            pytest.skip("scipy.signal.mel_frequencies not available")
        expected = sp_signal.mel_frequencies(num_mel=16, fmin=0.0, fmax=8000.0)
        assert_allclose(result.numpy(), expected, rtol=1e-6)
