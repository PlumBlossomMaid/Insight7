"""Manipulation alignment tests — Insight vs NumPy.

Run with:
    PYTHONPATH=bindings/python:build/bindings/python python -m pytest tests/python_align/cpu/test_manipulation.py -v
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


class TestManipulationAlignment:
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


class TestManipulationBatch3:
    """Batch 3 manipulation ops vs NumPy (ediff1d/meshgrid/trapz/gradient)."""

    def test_ediff1d(self):
        x_np = np.array([1.0, 4.0, 6.0, 9.0], dtype=np.float64)
        x = ins.from_numpy(x_np)
        assert_allclose(ins.ediff1d(x).numpy(), np.ediff1d(x_np))

    def test_ediff1d_to_end_to_begin(self):
        x_np = np.array([1.0, 4.0, 6.0], dtype=np.float64)
        x = ins.from_numpy(x_np)
        result = ins.ediff1d(x, to_end=ins.from_numpy(np.array([100.0])),
                             to_begin=ins.from_numpy(np.array([-1.0])))
        expected = np.ediff1d(x_np, to_end=100.0, to_begin=-1.0)
        assert_allclose(result.numpy(), expected)

    def test_meshgrid_xy(self):
        x_np = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        y_np = np.array([4.0, 5.0], dtype=np.float64)
        x = ins.from_numpy(x_np)
        y = ins.from_numpy(y_np)
        X, Y = ins.meshgrid([x, y], "xy")
        Xn, Yn = np.meshgrid(x_np, y_np, indexing="xy")
        assert_allclose(X.numpy(), Xn)
        assert_allclose(Y.numpy(), Yn)

    def test_trapz(self):
        y_np = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        y = ins.from_numpy(y_np)
        # np.trapz was renamed trapezoid in NumPy 2.0.
        assert_allclose(ins.trapz(y).numpy(), np.trapezoid(y_np), rtol=1e-10)

    def test_trapz_with_x(self):
        y_np = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        x_np = np.array([0.0, 1.0, 3.0], dtype=np.float64)
        y, x = ins.from_numpy(y_np), ins.from_numpy(x_np)
        assert_allclose(ins.trapz(y, x=x).numpy(), np.trapezoid(y_np, x_np),
                        rtol=1e-10)

    def test_gradient_1d(self):
        f_np = np.array([1.0, 2.0, 4.0, 7.0, 11.0], dtype=np.float64)
        f = ins.from_numpy(f_np)
        result = ins.gradient(f)
        # gradient with default dx=1 row-major: 1D returns single array.
        if isinstance(result, (list, tuple)):
            result = result[0]
        assert_allclose(result.numpy(), np.gradient(f_np), rtol=1e-10)

    def test_gradient_2d_all_axes(self):
        f_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
        f = ins.from_numpy(f_np)
        result = ins.gradient(f)
        assert isinstance(result, (list, tuple))
        assert len(result) == 2
        g0, g1 = np.gradient(f_np)
        assert_allclose(result[0].numpy(), g0, rtol=1e-10)
        assert_allclose(result[1].numpy(), g1, rtol=1e-10)


class TestManipulationBatch4:
    """Batch 4 ops vs NumPy (repeat_interleave/one_hot/histogram)."""

    def test_repeat_interleave(self):
        x_np = np.array([1, 2, 3], dtype=np.int64)
        x = ins.from_numpy(x_np)
        result = ins.repeat_interleave(x, 3)
        assert_allclose(result.numpy(), np.repeat(x_np, 3))

    def test_repeat_interleave_axis(self):
        x_np = np.array([[1, 2], [3, 4]], dtype=np.int64)
        x = ins.from_numpy(x_np)
        result = ins.repeat_interleave(x, 2, 0)
        assert_allclose(result.numpy(), np.repeat(x_np, 2, axis=0))

    def test_one_hot(self):
        idx_np = np.array([0, 2, 1], dtype=np.int64)
        idx = ins.from_numpy(idx_np)
        result = ins.one_hot(idx, 3)
        expected = np.eye(3, dtype=np.float32)[idx_np]
        assert_allclose(result.numpy(), expected, atol=1e-6)

    def test_one_hot_default_classes(self):
        idx_np = np.array([0, 1, 4], dtype=np.int64)
        idx = ins.from_numpy(idx_np)
        result = ins.one_hot(idx, 5)
        expected = np.eye(5, dtype=np.float32)[idx_np]
        assert_allclose(result.numpy(), expected, atol=1e-6)

    def test_histogram(self):
        x_np = np.array([0.2, 0.4, 0.6, 0.8, 1.2, 1.5, 2.3], dtype=np.float64)
        x = ins.from_numpy(x_np)
        hist, edges = ins.histogram(x, bins=4)
        h_np, e_np = np.histogram(x_np, bins=4)
        assert_allclose(hist.numpy(), h_np)
        assert_allclose(edges.numpy(), e_np, rtol=1e-10)
