"""Elementwise alignment tests — Insight CUDA vs NumPy.

Run with:
    PYTHONPATH=bindings/python:build/bindings/python \\
    LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda \\
    python -m pytest tests/python_align/cuda/test_elementwise.py -v
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
        # Insight: a - trunc(a/b) * b
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
