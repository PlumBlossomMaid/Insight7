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


class TestRandomBitExact:
    """Fixed-seed bit-exact alignment with NumPy legacy RandomState.

    Insight's random ops are built on a verbatim port of NumPy 2.4.6's
    MT19937 stack (randomkit.c / mt19937.c / distributions.c /
    legacy-distributions.c), consumed as a single per-device stream like
    np.random's global RandomState. With the same seed, every op must
    reproduce NumPy's output exactly, bit for bit.

    Insight seed semantics: seed(base) gives each device base + device_id;
    on CPU (device 0) the device seed equals base, so ins.seed(42) is the
    same stream as np.random.seed(42).
    """

    CPU = ins.CPUPlace()

    @staticmethod
    def _seed_and_sync(seed=42):
        np.random.seed(seed)
        ins.seed(seed)

    # ---- core draws ----------------------------------------------------

    def test_rand_f64(self):
        self._seed_and_sync()
        np_a = np.random.rand(50)
        a = ins.rand(ins.Shape([50]), ins.DType.float64, self.CPU)
        np.testing.assert_array_equal(a.numpy(), np_a)

    def test_randn_f64(self):
        self._seed_and_sync()
        np_a = np.random.randn(50)
        a = ins.randn(ins.Shape([50]), ins.DType.float64, self.CPU)
        np.testing.assert_array_equal(a.numpy(), np_a)

    def test_randint_i64(self):
        self._seed_and_sync()
        np_a = np.random.randint(0, 100, size=50)
        a = ins.randint(0, 100, ins.Shape([50]), ins.DType.int64, self.CPU)
        np.testing.assert_array_equal(a.numpy(), np_a)

    def test_randint_negative_low(self):
        self._seed_and_sync(7)
        np_a = np.random.randint(-5, 5, size=40)
        a = ins.randint(-5, 5, ins.Shape([40]), ins.DType.int64, self.CPU)
        np.testing.assert_array_equal(a.numpy(), np_a)

    def test_randperm(self):
        self._seed_and_sync()
        np_a = np.random.permutation(30)
        a = ins.randperm(30, ins.DType.int64, self.CPU)
        np.testing.assert_array_equal(a.numpy(), np_a)

    def test_uniform_f64(self):
        self._seed_and_sync()
        np_a = np.random.uniform(-3.0, 7.0, size=50)
        a = ins.uniform(-3.0, 7.0, ins.Shape([50]), ins.DType.float64, self.CPU)
        np.testing.assert_array_equal(a.numpy(), np_a)

    def test_normal_f64(self):
        self._seed_and_sync()
        np_a = np.random.normal(2.0, 0.5, size=50)
        a = ins.normal(2.0, 0.5, ins.Shape([50]), ins.DType.float64, self.CPU)
        np.testing.assert_array_equal(a.numpy(), np_a)

    # ---- choice / shuffle / multinomial -------------------------------

    def test_choice_replace_true_no_p(self):
        self._seed_and_sync()
        np_a = np.random.choice(10, size=20, replace=True)
        a = ins.choice(ins.arange(10, dtype=ins.DType.int64, place=self.CPU), 20, True)
        np.testing.assert_array_equal(a.numpy(), np_a)

    def test_choice_replace_true_with_p(self):
        self._seed_and_sync()
        probs = np.array([0.05, 0.05, 0.3, 0.6], dtype=np.float64)
        np_a = np.random.choice(4, size=30, replace=True, p=probs)
        a = ins.choice(
            ins.arange(4, dtype=ins.DType.int64, place=self.CPU), 30, True, ins.from_numpy(probs)
        )
        np.testing.assert_array_equal(a.numpy(), np_a)

    def test_choice_replace_false_no_p(self):
        self._seed_and_sync()
        np_a = np.random.choice(10, size=10, replace=False)
        a = ins.choice(ins.arange(10, dtype=ins.DType.int64, place=self.CPU), 10, False)
        np.testing.assert_array_equal(a.numpy(), np_a)

    def test_choice_replace_false_with_p(self):
        self._seed_and_sync()
        probs = np.array([0.1, 0.1, 0.2, 0.2, 0.4], dtype=np.float64)
        np_a = np.random.choice(5, size=5, replace=False, p=probs)
        a = ins.choice(
            ins.arange(5, dtype=ins.DType.int64, place=self.CPU), 5, False, ins.from_numpy(probs)
        )
        np.testing.assert_array_equal(a.numpy(), np_a)

    def test_shuffle_matches_numpy(self):
        self._seed_and_sync()
        arr = np.arange(40)
        np.random.shuffle(arr)
        a = ins.shuffle(ins.arange(40, dtype=ins.DType.int64, place=self.CPU))
        np.testing.assert_array_equal(a.numpy(), arr)

    def test_multinomial_replace_true(self):
        self._seed_and_sync()
        probs = np.array([0.1, 0.2, 0.7], dtype=np.float64)
        # Insight.multinomial samples *indices* (paddle semantics), which is
        # exactly np.random.choice(arange(d), num, replace=True, p=probs).
        np_a = np.random.choice(3, size=30, replace=True, p=probs)
        a = ins.multinomial(ins.from_numpy(probs), 30, True, self.CPU)
        np.testing.assert_array_equal(a.numpy(), np_a)

    def test_multinomial_replace_false(self):
        self._seed_and_sync()
        probs = np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float64)
        np_a = np.random.choice(5, size=5, replace=False, p=probs)
        a = ins.multinomial(ins.from_numpy(probs), 5, False, self.CPU)
        np.testing.assert_array_equal(a.numpy(), np_a)

    # ---- parameterized distributions -----------------------------------

    def test_exponential_f64(self):
        self._seed_and_sync()
        np_a = np.random.exponential(0.5, size=50)
        a = ins.exponential(0.5, ins.Shape([50]), ins.DType.float64, self.CPU)
        np.testing.assert_array_equal(a.numpy(), np_a)

    def test_gamma_f64(self):
        self._seed_and_sync()
        # Insight gamma(shape, rate); NumPy gamma(shape, scale=1/rate).
        np_a = np.random.gamma(2.0, 0.5, size=50)
        a = ins.gamma(2.0, 2.0, ins.Shape([50]), ins.DType.float64, self.CPU)
        np.testing.assert_array_equal(a.numpy(), np_a)

    def test_beta_f64(self):
        self._seed_and_sync()
        np_a = np.random.beta(2.0, 5.0, size=50)
        a = ins.beta(2.0, 5.0, ins.Shape([50]), ins.DType.float64, self.CPU)
        np.testing.assert_array_equal(a.numpy(), np_a)

    def test_binomial(self):
        self._seed_and_sync()
        np_a = np.random.binomial(10, 0.4, size=50)
        a = ins.binomial(10, 0.4, ins.Shape([50]), ins.DType.int64, self.CPU)
        np.testing.assert_array_equal(a.numpy(), np_a)

    def test_poisson(self):
        self._seed_and_sync()
        np_a = np.random.poisson(3.0, size=50)
        a = ins.poisson(3.0, ins.Shape([50]), ins.DType.int64, self.CPU)
        np.testing.assert_array_equal(a.numpy(), np_a)

    def test_chisquare_f64(self):
        self._seed_and_sync()
        np_a = np.random.chisquare(5.0, size=50)
        a = ins.chisquare(5.0, ins.Shape([50]), ins.DType.float64, self.CPU)
        np.testing.assert_array_equal(a.numpy(), np_a)

    # ---- stream continuation (single global state, like np.random) ----

    def test_stream_continues_across_calls(self):
        self._seed_and_sync(123)
        # np.random is a single persistent RandomState: two sequential calls
        # keep consuming the same stream. Insight must do the same.
        np_a1 = np.random.rand(5)
        np_a2 = np.random.rand(3)
        a1 = ins.rand(ins.Shape([5]), ins.DType.float64, self.CPU)
        a2 = ins.rand(ins.Shape([3]), ins.DType.float64, self.CPU)
        np.testing.assert_array_equal(a1.numpy(), np_a1)
        np.testing.assert_array_equal(a2.numpy(), np_a2)

    def test_reseed_resets_stream(self):
        np.random.seed(42)
        np_1 = np.random.rand(4)
        np.random.seed(42)
        np_2 = np.random.rand(4)
        ins.seed(42)
        i1 = ins.rand(ins.Shape([4]), ins.DType.float64, self.CPU)
        ins.seed(42)
        i2 = ins.rand(ins.Shape([4]), ins.DType.float64, self.CPU)
        np.testing.assert_array_equal(i1.numpy(), np_1)
        np.testing.assert_array_equal(i2.numpy(), np_2)
        np.testing.assert_array_equal(i1.numpy(), np_2)
