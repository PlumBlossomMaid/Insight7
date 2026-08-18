"""Complex number CUDA binding tests."""

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


class TestComplexCUDA:
    def test_is_complex_real(self):
        a = to_gpu(np.array([1, 2, 3], dtype=np.float64))
        assert not ins.is_complex(a)

    def test_is_complex_complex(self):
        a = to_gpu(np.array([1 + 1j, 2 + 2j], dtype=np.complex128))
        assert ins.is_complex(a)

    def test_to_complex(self):
        real = to_gpu(np.array([1, 2, 3], dtype=np.float64))
        result = ins.to_complex(real)
        assert ins.is_complex(result)
        assert result.numel == 3

    def test_to_complex_with_imag(self):
        real = to_gpu(np.array([1, 2, 3], dtype=np.float64))
        imag = to_gpu(np.array([4, 5, 6], dtype=np.float64))
        result = ins.to_complex(real, imag)
        assert ins.is_complex(result)
        assert result.numel == 3

    def test_real(self):
        a = to_gpu(np.array([1 + 2j, 3 + 4j], dtype=np.complex128))
        r = ins.real(a)
        np.testing.assert_allclose(to_numpy(r), np.array([1.0, 3.0]), rtol=1e-10)

    def test_imag(self):
        a = to_gpu(np.array([1 + 2j, 3 + 4j], dtype=np.complex128))
        i = ins.imag(a)
        np.testing.assert_allclose(to_numpy(i), np.array([2.0, 4.0]), rtol=1e-10)

    def test_as_complex(self):
        a = to_gpu(np.array([1, 2, 3, 4], dtype=np.float64).reshape(2, 2))
        result = ins.as_complex(a)
        assert ins.is_complex(result)

    def test_as_real(self):
        a = to_gpu(np.array([1 + 2j, 3 + 4j], dtype=np.complex128))
        result = ins.as_real(a)
        assert not ins.is_complex(result)
        assert result.numel == 4

    def test_complex_abs(self):
        a_np = np.array([3 + 4j, 1 + 0j], dtype=np.complex128)
        a = to_gpu(a_np)
        result = ins.abs(a)
        expected = np.abs(a_np)
        np.testing.assert_allclose(to_numpy(result), expected, rtol=1e-5)

    def test_complex_conj(self):
        a_np = np.array([1 + 2j, 3 + 4j], dtype=np.complex128)
        a = to_gpu(a_np)
        result = ins.conj(a)
        expected = np.conj(a_np)
        np.testing.assert_allclose(to_numpy(result), expected, atol=1e-10)

    def test_complex_angle(self):
        a_np = np.array([1 + 0j, 0 + 1j, -1 + 0j], dtype=np.complex128)
        a = to_gpu(a_np)
        result = ins.angle(a)
        expected = np.angle(a_np)
        np.testing.assert_allclose(to_numpy(result), expected, atol=1e-10)

    def test_complex_add(self):
        a = to_gpu(np.array([1 + 2j, 3 + 4j], dtype=np.complex128))
        b = to_gpu(np.array([5 + 6j, 7 + 8j], dtype=np.complex128))
        result = ins.add(a, b)
        expected = np.array([6 + 8j, 10 + 12j])
        np.testing.assert_allclose(to_numpy(result), expected, atol=1e-10)

    def test_complex_mul(self):
        a = to_gpu(np.array([1 + 2j, 3 + 4j], dtype=np.complex128))
        b = to_gpu(np.array([5 + 6j, 7 + 8j], dtype=np.complex128))
        result = ins.mul(a, b)
        expected = np.array([1 + 2j, 3 + 4j]) * np.array([5 + 6j, 7 + 8j])
        np.testing.assert_allclose(to_numpy(result), expected, atol=1e-10)

    def test_complex_exp(self):
        # exp kernel does not support complex dtype; test with real input
        a_np = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        a = to_gpu(a_np)
        result = ins.exp(a)
        expected = np.exp(a_np)
        np.testing.assert_allclose(to_numpy(result), expected, atol=1e-8)

    def test_as_complex_as_real_roundtrip(self):
        a = to_gpu(np.array([1, 2, 3, 4], dtype=np.float64).reshape(2, 2))
        c = ins.as_complex(a)
        assert ins.is_complex(c)
        r = ins.as_real(c)
        assert not ins.is_complex(r)
        assert r.numel == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
