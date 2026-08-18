"""Manipulation CUDA binding tests — aligned with C++ test_manipulation.cpp.

Run with:
    PYTHONPATH=bindings/python:build/bindings/python
    LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda
    python -m pytest tests/cuda/python/test_manipulation.py -v
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


class TestManipulationCUDA:
    """Array manipulation operations — CUDA backend."""

    def test_reshape(self):
        a = to_gpu(np.arange(6, dtype=np.float64))
        b = ins.reshape(a, [2, 3])
        assert b.numel == 6
        assert b.ndim == 2

    def test_transpose(self):
        a_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        b = ins.permute(to_gpu(a_np), [1, 0])
        result = to_numpy(b)
        np.testing.assert_allclose(result, a_np.T)

    def test_squeeze(self):
        a = to_gpu(np.zeros([1, 3, 1], dtype=np.float64))
        b = ins.squeeze(a)
        assert b.numel == 3

    def test_unsqueeze(self):
        a = to_gpu(np.zeros([3], dtype=np.float64))
        b = ins.unsqueeze(a, 0)
        assert b.ndim == 2

    def test_concat(self):
        a = to_gpu(np.array([1, 2], dtype=np.float64))
        b = to_gpu(np.array([3, 4], dtype=np.float64))
        c = ins.concat([a, b])
        assert c.numel == 4
        np.testing.assert_allclose(to_numpy(c), [1, 2, 3, 4])

    def test_concat_axis(self):
        a_np = np.array([[1, 2], [3, 4]], dtype=np.float64)
        b_np = np.array([[5, 6]], dtype=np.float64)
        c = ins.concat([to_gpu(a_np), to_gpu(b_np)], axis=0)
        np.testing.assert_allclose(to_numpy(c), np.concatenate([a_np, b_np], axis=0))

    def test_split(self):
        a = to_gpu(np.arange(6, dtype=np.float64))
        parts = ins.split(a, 3, axis=0)
        assert len(parts) == 3
        assert parts[0].numel == 2

    def test_tile(self):
        a = to_gpu(np.array([1, 2, 3], dtype=np.float64))
        b = ins.tile(a, [2])
        assert b.numel == 6

    def test_repeat(self):
        a = to_gpu(np.array([1, 2, 3], dtype=np.float64))
        b = ins.repeat(a, 2)
        assert b.numel == 6

    def test_flip(self):
        a_np = np.array([1, 2, 3, 4], dtype=np.float64)
        result = ins.flip(to_gpu(a_np))
        np.testing.assert_allclose(to_numpy(result), np.flip(a_np))

    def test_roll(self):
        a_np = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        result = ins.roll(to_gpu(a_np), 2)
        np.testing.assert_allclose(to_numpy(result), np.roll(a_np, 2))

    def test_pad(self):
        a = to_gpu(np.array([1, 2, 3], dtype=np.float64))
        b = ins.pad(a, [2, 3])
        assert b.numel == 8

    def test_diag(self):
        a = to_gpu(np.array([1, 2, 3], dtype=np.float64))
        b = ins.diag(a)
        assert b.numel == 9
        assert b.ndim == 2

    def test_tril(self):
        a = to_gpu(np.ones([3, 3], dtype=np.float64))
        b = ins.tril(a)
        np.testing.assert_allclose(to_numpy(b), np.tril(np.ones([3, 3])))

    def test_triu(self):
        a = to_gpu(np.ones([3, 3], dtype=np.float64))
        b = ins.triu(a)
        np.testing.assert_allclose(to_numpy(b), np.triu(np.ones([3, 3])))

    def test_diff(self):
        a_np = np.array([1, 3, 6, 10], dtype=np.float64)
        result = ins.diff(to_gpu(a_np))
        np.testing.assert_allclose(to_numpy(result), np.diff(a_np))

    def test_swapaxes(self):
        a = to_gpu(np.ones([2, 3, 4], dtype=np.float64))
        b = ins.swapaxes(a, 0, 2)
        assert b.numel == 24

    def test_moveaxis(self):
        a = to_gpu(np.ones([2, 3, 4], dtype=np.float64))
        b = ins.moveaxis(a, 0, 2)
        assert b.numel == 24

    def test_fliplr(self):
        a_np = np.array([[1, 2], [3, 4]], dtype=np.float64)
        result = ins.fliplr(to_gpu(a_np))
        np.testing.assert_allclose(to_numpy(result), np.fliplr(a_np))

    def test_flipud(self):
        a_np = np.array([[1, 2], [3, 4]], dtype=np.float64)
        result = ins.flipud(to_gpu(a_np))
        np.testing.assert_allclose(to_numpy(result), np.flipud(a_np))

    def test_rot90(self):
        a_np = np.array([[1, 2], [3, 4]], dtype=np.float64)
        result = ins.rot90(to_gpu(a_np))
        np.testing.assert_allclose(to_numpy(result), np.rot90(a_np))

    def test_flatten(self):
        a = to_gpu(np.ones([2, 3], dtype=np.float64))
        b = ins.flatten(a)
        assert b.numel == 6
        assert b.ndim == 1

    def test_ravel(self):
        a = to_gpu(np.ones([2, 3], dtype=np.float64))
        b = ins.ravel(a)
        assert b.numel == 6
        assert b.ndim == 1

    # --- In-place assignment (__setitem__) ---

    def test_setitem_scalar_index(self):
        """a[1] = 5.0 on a 1-D GPU array."""
        a = to_gpu(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64))
        a[1] = 5.0
        np.testing.assert_allclose(to_numpy(a), [1.0, 5.0, 3.0, 4.0])

    def test_setitem_slice_scalar(self):
        """a[0:3] = 9.0 — fill a GPU slice with a scalar."""
        a = to_gpu(np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64))
        a[0:3] = 9.0
        np.testing.assert_allclose(to_numpy(a), [9.0, 9.0, 9.0, 4.0, 5.0])

    def test_setitem_slice_array(self):
        """a[1:4] = arr — copy an array into a GPU slice."""
        a = to_gpu(np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64))
        src = to_gpu(np.array([7.0, 8.0, 9.0], dtype=np.float64))
        a[1:4] = src
        np.testing.assert_allclose(to_numpy(a), [1.0, 7.0, 8.0, 9.0, 5.0])

    def test_setitem_2d_scalar(self):
        """a[0:2, 1:3] = 5.0 — fill a 2-D GPU sub-region with a scalar."""
        a = to_gpu(np.zeros([3, 4], dtype=np.float64))
        a[0:2, 1:3] = 5.0
        expected = np.zeros([3, 4], dtype=np.float64)
        expected[0:2, 1:3] = 5.0
        np.testing.assert_allclose(to_numpy(a), expected)

    def test_setitem_2d_array(self):
        """a[0:2, 1:3] = src — copy an array into a 2-D GPU sub-region."""
        a = to_gpu(np.zeros([3, 4], dtype=np.float64))
        src = to_gpu(np.array([[7.0, 8.0], [9.0, 10.0]], dtype=np.float64))
        a[0:2, 1:3] = src
        expected = np.zeros([3, 4], dtype=np.float64)
        expected[0:2, 1:3] = [[7.0, 8.0], [9.0, 10.0]]
        np.testing.assert_allclose(to_numpy(a), expected)

    def test_setitem_string_spec(self):
        """a["1:3"] = 7.0 — string-spec in-place assignment on GPU."""
        a = to_gpu(np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64))
        a["1:3"] = 7.0
        np.testing.assert_allclose(to_numpy(a), [1.0, 7.0, 7.0, 4.0, 5.0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
