"""Random number generation CUDA binding tests."""

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


def to_numpy(gpu_arr):
    return gpu_arr.to(CPU).numpy()


class TestRandomCUDA:
    def test_rand_shape(self):
        a = ins.rand([3, 4], ins.float64, GPU)
        assert a.numel == 12

    def test_rand_range(self):
        a = ins.rand([1000], ins.float64, GPU)
        data = to_numpy(a)
        assert np.all(data >= 0) and np.all(data < 1)

    def test_randn_shape(self):
        a = ins.randn([3, 4], ins.float64, GPU)
        assert a.numel == 12

    def test_randn_statistics(self):
        a = ins.randn([10000], ins.float64, GPU)
        data = to_numpy(a)
        assert abs(np.mean(data)) < 0.1
        assert abs(np.std(data) - 1.0) < 0.1

    def test_seed_determinism(self):
        ins.seed(42)
        a = ins.rand([5], ins.float64, GPU)
        ins.seed(42)
        b = ins.rand([5], ins.float64, GPU)
        np.testing.assert_array_equal(to_numpy(a), to_numpy(b))

    def test_get_seed(self):
        ins.seed(12345)
        s = ins.get_seed()
        assert s == 12345

    def test_randint_shape(self):
        a = ins.randint(0, 10, [3, 3], ins.int64, GPU)
        assert a.numel == 9

    def test_randint_range(self):
        a = ins.randint(0, 10, [1000], ins.int64, GPU)
        data = to_numpy(a)
        assert np.all(data >= 0) and np.all(data < 10)

    def test_rand_like(self):
        template = ins.zeros([3, 4], ins.float64, GPU)
        a = ins.rand_like(template)
        assert a.numel == 12

    def test_randn_like(self):
        template = ins.zeros([3, 4], ins.float64, GPU)
        a = ins.randn_like(template)
        assert a.numel == 12

    def test_exponential_shape(self):
        a = ins.exponential(1.0, [100], ins.float64, GPU)
        assert a.numel == 100
        data = to_numpy(a)
        assert np.all(data >= 0)

    def test_gamma_shape(self):
        a = ins.gamma(2.0, 1.0, [100], ins.float64, GPU)
        assert a.numel == 100
        data = to_numpy(a)
        assert np.all(data >= 0)

    def test_beta_shape(self):
        a = ins.beta(2.0, 5.0, [100], ins.float64, GPU)
        assert a.numel == 100
        data = to_numpy(a)
        assert np.all(data >= 0) and np.all(data <= 1)

    def test_binomial_shape(self):
        a = ins.binomial(10, 0.5, [100], ins.int64, GPU)
        assert a.numel == 100

    def test_poisson_shape(self):
        a = ins.poisson(5.0, [100], ins.int64, GPU)
        assert a.numel == 100
        data = to_numpy(a)
        assert np.all(data >= 0)

    def test_rand_3d(self):
        a = ins.rand([2, 3, 4], ins.float64, GPU)
        assert a.numel == 24

    def test_randperm(self):
        a = ins.randperm(10, ins.int64, GPU)
        assert a.numel == 10
        data = sorted(to_numpy(a).tolist())
        assert data == list(range(10))

    def test_uniform(self):
        a = ins.uniform(0.0, 1.0, [100], ins.float64, GPU)
        assert a.numel == 100

    def test_normal(self):
        a = ins.normal(0.0, 1.0, [100], ins.float64, GPU)
        assert a.numel == 100

    def test_rand_float32(self):
        a = ins.rand([10], ins.float32, GPU)
        assert a.numel == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
