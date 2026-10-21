"""Signal Windows CUDA binding tests.

Run with:
    PYTHONPATH=bindings/python:build/bindings/python \
    LD_LIBRARY_PATH=build/backends/cpu:build/backends/cuda \
    python -m pytest tests/cuda/python/test_signal_windows.py -v
"""

import sys
import os
import pytest
import math

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


class TestBoxcar:
    """Boxcar window — tests 1-2."""

    def test_boxcar_basic(self):
        w = ins.signal.boxcar(5)
        assert w.numel == 5
        np.testing.assert_allclose(w.numpy(), [1, 1, 1, 1, 1], atol=1e-10)

    def test_boxcar_size1(self):
        w = ins.signal.boxcar(1)
        assert w.numel == 1
        assert abs(w.numpy()[0] - 1.0) < 1e-10


class TestTriang:
    """Triangular window — tests 3-4."""

    def test_triang_odd(self):
        # scipy.signal.windows.triang(5) = [1/3, 2/3, 1, 2/3, 1/3]
        w = ins.signal.triang(5)
        assert w.numel == 5
        np.testing.assert_allclose(w.numpy(), [1 / 3, 2 / 3, 1, 2 / 3, 1 / 3], atol=1e-10)

    def test_triang_even(self):
        # scipy.signal.windows.triang(4) = [0.25, 0.75, 0.75, 0.25]
        w = ins.signal.triang(4)
        assert w.numel == 4
        np.testing.assert_allclose(w.numpy(), [0.25, 0.75, 0.75, 0.25], atol=1e-10)


class TestParzen:
    """Parzen window — test 5."""

    def test_parzen_basic(self):
        w = ins.signal.parzen(8)
        assert w.numel == 8
        d = w.numpy()
        assert abs(d[0] - 0.00390625) < 1e-6
        assert abs(d[1] - 0.10546875) < 1e-6
        assert abs(d[2] - 0.47265625) < 1e-6
        assert abs(d[3] - 0.91796875) < 1e-6
        # Symmetric
        assert abs(d[4] - d[3]) < 1e-10
        assert abs(d[7] - d[0]) < 1e-10


class TestBohman:
    """Bohman window — test 6."""

    def test_bohman_endpoints(self):
        w = ins.signal.bohman(11)
        d = w.numpy()
        assert abs(d[0]) < 1e-10  # endpoints = 0
        assert abs(d[10]) < 1e-10
        assert abs(d[5] - 1.0) < 1e-6  # center = 1


class TestBartlett:
    """Bartlett window — test 7."""

    def test_bartlett_basic(self):
        # scipy.signal.windows.bartlett(5) = [0, 0.5, 1, 0.5, 0]
        w = ins.signal.bartlett(5)
        np.testing.assert_allclose(w.numpy(), [0, 0.5, 1, 0.5, 0], atol=1e-10)


class TestCosine:
    """Cosine window — test 8."""

    def test_cosine_basic(self):
        w = ins.signal.cosine(4)
        d = w.numpy()
        assert abs(d[0] - math.sin(math.pi / 8)) < 1e-10
        assert abs(d[1] - math.sin(3 * math.pi / 8)) < 1e-10
        assert abs(d[2] - math.sin(5 * math.pi / 8)) < 1e-10
        assert abs(d[3] - math.sin(7 * math.pi / 8)) < 1e-10


class TestExponential:
    """Exponential window — test 9."""

    def test_exponential_basic(self):
        w = ins.signal.exponential(5, tau=2.0)
        d = w.numpy()
        assert w.numel == 5
        assert all(v > 0 for v in d)  # all positive
        assert d[2] == max(d)  # center is maximum


class TestBlackman:
    """Blackman window — test 10."""

    def test_blackman_basic(self):
        w = ins.signal.blackman(5)
        d = w.numpy()
        assert abs(d[0]) < 1e-10  # endpoints ~0
        assert abs(d[4]) < 1e-10
        assert abs(d[2] - 1.0) < 1e-6  # center ~1
        assert abs(d[1] - d[3]) < 1e-10  # symmetric


class TestNuttall:
    """Nuttall window — test 11."""

    def test_nuttall_basic(self):
        w = ins.signal.nuttall(5)
        d = w.numpy()
        assert abs(d[0] - 0.0003628) < 1e-4
        assert abs(d[4] - 0.0003628) < 1e-4
        assert abs(d[2] - 1.0) < 1e-6


class TestBlackmanharris:
    """Blackman-Harris window — test 12."""

    def test_blackmanharris_basic(self):
        w = ins.signal.blackmanharris(5)
        d = w.numpy()
        assert abs(d[0] - 6.0e-5) < 1e-4
        assert abs(d[4] - 6.0e-5) < 1e-4
        assert abs(d[2] - 1.0) < 1e-6


class TestFlattop:
    """Flat-top window — test 13."""

    def test_flattop_basic(self):
        w = ins.signal.flattop(5)
        assert w.numel == 5
        d = w.numpy()
        assert abs(d[0] - d[4]) < 1e-10  # symmetric
        assert abs(d[1] - d[3]) < 1e-10


class TestHann:
    """Hann window — tests 14-15."""

    def test_hann_basic(self):
        w = ins.signal.hann(5)
        d = w.numpy()
        assert abs(d[0]) < 1e-10
        assert abs(d[4]) < 1e-10
        assert abs(d[2] - 1.0) < 1e-10
        assert abs(d[1] - 0.5) < 1e-10
        assert abs(d[3] - 0.5) < 1e-10

    def test_hann_size16(self):
        w = ins.signal.hann(16)
        assert w.numel == 16
        assert abs(w.numpy()[0]) < 1e-5


class TestHamming:
    """Hamming window — test 16."""

    def test_hamming_basic(self):
        w = ins.signal.hamming(5)
        d = w.numpy()
        assert abs(d[0] - 0.08) < 1e-10  # endpoints = 0.08
        assert abs(d[4] - 0.08) < 1e-10
        assert abs(d[2] - 1.0) < 1e-10  # center
        assert abs(d[1] - d[3]) < 1e-10


class TestTukey:
    """Tukey window — tests 17-18."""

    def test_tukey_alpha0(self):
        # alpha=0 -> rectangular
        w = ins.signal.tukey(10, alpha=0.0)
        np.testing.assert_allclose(w.numpy(), 1.0, atol=1e-10)

    def test_tukey_alpha1(self):
        # alpha=1 -> Hann
        w_tukey = ins.signal.tukey(10, alpha=1.0)
        w_hann = ins.signal.hann(10)
        np.testing.assert_allclose(w_tukey.numpy(), w_hann.numpy(), atol=1e-10)


class TestBarthann:
    """Bartlett-Hann window — test 19."""

    def test_barthann_basic(self):
        w = ins.signal.barthann(11)
        d = w.numpy()
        assert abs(d[5] - 1.0) < 0.02  # center ~1
        # Symmetric
        for i in range(5):
            assert abs(d[i] - d[10 - i]) < 1e-10


class TestKaiser:
    """Kaiser window — tests 20."""

    def test_kaiser_beta0(self):
        # beta=0 -> rectangular
        w = ins.signal.kaiser(5, 0.0)
        np.testing.assert_allclose(w.numpy(), 1.0, atol=1e-10)


class TestGaussian:
    """Gaussian window — test 21."""

    def test_gaussian_basic(self):
        w = ins.signal.gaussian(5, 1.0)
        d = w.numpy()
        assert abs(d[2] - 1.0) < 1e-10  # center = 1
        assert abs(d[0] - d[4]) < 1e-10  # symmetric
        assert abs(d[0] - math.exp(-2.0)) < 1e-10


class TestGeneralGaussian:
    """Generalized Gaussian window — test 22."""

    def test_general_gaussian_basic(self):
        # p=1, sig=1 -> same as gaussian with std=1
        w = ins.signal.general_gaussian(5, 1.0, 1.0)
        d = w.numpy()
        assert abs(d[2] - 1.0) < 1e-10
        assert abs(d[0] - math.exp(-2.0)) < 1e-10


class TestChebwin:
    """Dolph-Chebyshev window — test 23."""

    def test_chebwin_basic(self):
        w = ins.signal.chebwin(11, 50.0)
        d = w.numpy()
        assert w.numel == 11
        # Symmetric
        for i in range(5):
            assert abs(d[i] - d[10 - i]) < 1e-6
        # All values in [0, 1]
        for v in d:
            assert v >= 0.0
            assert v <= 1.0 + 1e-6
        # Max should be 1
        assert abs(max(d) - 1.0) < 1e-6


class TestTaylor:
    """Taylor window — test 24."""

    def test_taylor_basic(self):
        w = ins.signal.taylor(32, nbar=4, sll=-30.0, norm=True)
        d = w.numpy()
        assert w.numel == 32
        # Symmetric
        for i in range(16):
            assert abs(d[i] - d[31 - i]) < 1e-6
        # All values in [0, 1]
        for v in d:
            assert v >= 0.0
            assert v <= 1.0 + 1e-6
        # Max should be 1
        assert abs(max(d) - 1.0) < 1e-6


class TestGetWindow:
    """get_window — tests 25-26."""

    def test_get_window_boxcar(self):
        w = ins.signal.get_window("boxcar", 8)
        assert w.numel == 8
        np.testing.assert_allclose(w.numpy(), 1.0, atol=1e-10)

    def test_get_window_hann(self):
        # get_window returns a window of the requested size
        w = ins.signal.get_window("hann", 5)
        assert w.numel == 5
        d = w.numpy()
        assert d[0] < d[2]  # center is larger than edge


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
