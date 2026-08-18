"""CUDA binding tests — 35 tests aligned with CPU Python/Lua/Julia.

Run with:
    PYTHONPATH=bindings/python:build/bindings/python \\
    LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda \\
    python -m pytest tests/cuda/python/test_cuda_basic.py -v
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
    return gpu_arr.contiguous().to(CPU).numpy()


# ============================================================================
# Creation (7 tests)
# ============================================================================


class TestCreation:
    """Array creation on GPU — tests 1-7."""

    def test_zeros(self):
        a = ins.zeros([2, 3], ins.float32).to(GPU)
        assert a.numel == 6
        assert np.allclose(to_numpy(a), 0)

    def test_ones(self):
        a = ins.ones([2, 3], ins.float32).to(GPU)
        assert a.numel == 6
        np.testing.assert_allclose(to_numpy(a), np.ones([2, 3]))

    def test_full(self):
        a = ins.full([2, 3], 7.0, ins.float32).to(GPU)
        assert a.numel == 6
        np.testing.assert_allclose(to_numpy(a), np.full([2, 3], 7.0))

    def test_eye(self):
        a = ins.eye(3).to(GPU)
        assert a.numel == 9
        np.testing.assert_allclose(to_numpy(a), np.eye(3))

    def test_arange(self):
        a = ins.arange(10, ins.float32).to(GPU)
        assert a.numel == 10
        np.testing.assert_allclose(to_numpy(a), np.arange(10, dtype=np.float32))

    def test_linspace(self):
        a = ins.linspace(0.0, 1.0, 5, ins.float64).to(GPU)
        assert a.numel == 5
        np.testing.assert_allclose(to_numpy(a), np.linspace(0, 1, 5), rtol=1e-6)

    def test_from_numpy(self):
        data = np.array([1.5, 2.5, 3.5], dtype=np.float64)
        a = ins.from_numpy(data).to(GPU)
        assert a.numel == 3
        np.testing.assert_allclose(to_numpy(a), data)


# ============================================================================
# Arithmetic (5 tests)
# ============================================================================


class TestArithmetic:
    """Elementwise arithmetic on GPU — tests 8-12."""

    def test_add(self):
        a_np = np.array([1, 2, 3], dtype=np.float64)
        b_np = np.array([4, 5, 6], dtype=np.float64)
        result = ins.add(to_gpu(a_np), to_gpu(b_np))
        np.testing.assert_allclose(to_numpy(result), a_np + b_np, rtol=1e-10)

    def test_sub(self):
        a_np = np.array([10, 20, 30], dtype=np.float64)
        b_np = np.array([1, 2, 3], dtype=np.float64)
        result = ins.sub(to_gpu(a_np), to_gpu(b_np))
        np.testing.assert_allclose(to_numpy(result), a_np - b_np, rtol=1e-10)

    def test_mul(self):
        a_np = np.array([2, 3, 4], dtype=np.float64)
        b_np = np.array([5, 6, 7], dtype=np.float64)
        result = ins.mul(to_gpu(a_np), to_gpu(b_np))
        np.testing.assert_allclose(to_numpy(result), a_np * b_np, rtol=1e-10)

    def test_div(self):
        a_np = np.array([10, 20, 30], dtype=np.float64)
        b_np = np.array([2, 4, 5], dtype=np.float64)
        result = ins.div(to_gpu(a_np), to_gpu(b_np))
        np.testing.assert_allclose(to_numpy(result), a_np / b_np, rtol=1e-10)

    def test_neg(self):
        a_np = np.array([1, -2, 3], dtype=np.float64)
        result = ins.negative(to_gpu(a_np))
        np.testing.assert_allclose(to_numpy(result), -a_np, rtol=1e-10)


# ============================================================================
# Unary (5 tests)
# ============================================================================


class TestUnary:
    """Unary math on GPU — tests 13-17."""

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

    def test_sin(self):
        a_np = np.array([0, 0.5, 1.0], dtype=np.float64)
        result = ins.sin(to_gpu(a_np))
        np.testing.assert_allclose(to_numpy(result), np.sin(a_np), rtol=1e-10)


# ============================================================================
# Reduction (5 tests)
# ============================================================================


class TestReduction:
    """Reduction ops on GPU — tests 18-22."""

    def test_sum(self):
        a = to_gpu(np.array([1, 2, 3, 4], dtype=np.float64))
        result = ins.sum(a)
        assert result.numpy().item() == pytest.approx(10.0)

    def test_mean(self):
        a = to_gpu(np.array([1, 2, 3, 4], dtype=np.float64))
        result = ins.mean(a)
        assert result.numpy().item() == pytest.approx(2.5)

    def test_max(self):
        a = to_gpu(np.array([3, 1, 4, 1, 5], dtype=np.float64))
        result = ins.max(a)
        assert result.numpy().item() == pytest.approx(5.0)

    def test_min(self):
        a = to_gpu(np.array([3, 1, 4, 1, 5], dtype=np.float64))
        result = ins.min(a)
        assert result.numpy().item() == pytest.approx(1.0)

    def test_argmax(self):
        a = to_gpu(np.array([1, 5, 3, 2], dtype=np.float64))
        result = ins.argmax(a)
        assert result.numpy().item() == 1


# ============================================================================
# Comparison (4 tests)
# ============================================================================


class TestComparison:
    """Comparison ops on GPU — tests 23-26."""

    def test_equal(self):
        a = to_gpu(np.array([1, 2, 3], dtype=np.float64))
        b = to_gpu(np.array([1, 0, 3], dtype=np.float64))
        result = ins.equal(a, b)
        np.testing.assert_array_equal(to_numpy(result), np.array([True, False, True]))

    def test_greater(self):
        a = to_gpu(np.array([3, 1, 5], dtype=np.float64))
        b = to_gpu(np.array([2, 4, 5], dtype=np.float64))
        result = ins.greater(a, b)
        np.testing.assert_array_equal(to_numpy(result), np.array([True, False, False]))

    def test_less(self):
        a = to_gpu(np.array([1, 5, 3], dtype=np.float64))
        b = to_gpu(np.array([2, 4, 5], dtype=np.float64))
        result = ins.less(a, b)
        np.testing.assert_array_equal(to_numpy(result), np.array([True, False, True]))

    def test_logical_and(self):
        a = to_gpu(np.array([True, True, False], dtype=np.bool_))
        b = to_gpu(np.array([True, False, False], dtype=np.bool_))
        result = ins.logical_and(a, b)
        np.testing.assert_array_equal(to_numpy(result), np.array([True, False, False]))


# ============================================================================
# Manipulation (3 tests)
# ============================================================================


class TestManipulation:
    """Manipulation ops on GPU — tests 27-29."""

    def test_reshape(self):
        a = to_gpu(np.arange(6, dtype=np.float64))
        b = ins.reshape(a, [2, 3])
        assert b.numel == 6

    def test_transpose(self):
        a = to_gpu(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64))
        b = ins.permute(a, [1, 0])
        np.testing.assert_allclose(
            to_numpy(b), np.array([[1, 4], [2, 5], [3, 6]], dtype=np.float64)
        )

    def test_squeeze(self):
        a = to_gpu(np.zeros([1, 3, 1], dtype=np.float64))
        b = ins.squeeze(a)
        assert b.numel == 3


# ============================================================================
# Linalg (3 tests)
# ============================================================================


class TestLinalg:
    """Linear algebra on GPU — tests 30-32."""

    def test_matmul(self):
        a_np = np.array([[1, 2], [3, 4]], dtype=np.float64)
        b_np = np.array([[5, 6], [7, 8]], dtype=np.float64)
        result = ins.matmul(to_gpu(a_np), to_gpu(b_np))
        np.testing.assert_allclose(to_numpy(result), a_np @ b_np, rtol=1e-10)

    def test_det(self):
        a_np = np.array([[1, 2], [3, 4]], dtype=np.float32)
        try:
            result = ins.det(to_gpu(a_np))
            assert result.numpy().item() == pytest.approx(-2.0, abs=1e-3)
        except RuntimeError:
            pytest.skip("det requires cuBLAS")

    def test_inv(self):
        a_np = np.array([[4, 7], [2, 6]], dtype=np.float32)
        try:
            result = ins.inv(to_gpu(a_np))
            np.testing.assert_allclose(to_numpy(result), np.linalg.inv(a_np), atol=1e-3)
        except RuntimeError:
            pytest.skip("inv requires cuBLAS")


# ============================================================================
# FFT (2 tests)
# ============================================================================


class TestFFT:
    """FFT on GPU — tests 33-34."""

    def test_fft(self):
        x_np = np.array([1, 2, 3, 4], dtype=np.float64)
        result = ins.fft(to_gpu(x_np))
        expected = np.fft.fft(x_np)
        np.testing.assert_allclose(to_numpy(result), expected, atol=1e-8)

    def test_ifft(self):
        x_np = np.array([1, 2, 3, 4], dtype=np.float64)
        fft_np = np.fft.fft(x_np)
        result = ins.ifft(to_gpu(fft_np.astype(np.complex128)))
        expected = np.fft.ifft(fft_np)
        np.testing.assert_allclose(to_numpy(result), expected, atol=1e-8)


# ============================================================================
# Signal (1 test)
# ============================================================================


class TestSignal:
    """Signal — test 35."""

    def test_hann(self):
        w = ins.signal.hann(16)
        assert w.numel == 16
        assert w.numpy()[0] == pytest.approx(0.0, abs=1e-5)
        assert w.numpy()[8] == pytest.approx(1.0, abs=1e-1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
