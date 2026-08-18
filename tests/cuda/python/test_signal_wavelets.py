"""Signal wavelets CUDA binding tests."""

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


class TestSignalWaveletsCUDA:
    """Wavelet functions — signal_wavelets CUDA tests."""

    def test_morlet(self):
        w = ins.signal.morlet(32, w=5.0, s=1.0, complete=True)
        w_gpu = w.to(GPU)
        assert w_gpu is not None
        assert w_gpu.numel == 32

    def test_morlet_short(self):
        w = ins.signal.morlet(16, w=5.0, s=1.0, complete=True)
        w_gpu = w.to(GPU)
        assert w_gpu is not None
        assert w_gpu.numel == 16

    def test_morlet_incomplete(self):
        w = ins.signal.morlet(32, w=5.0, s=1.0, complete=False)
        w_gpu = w.to(GPU)
        assert w_gpu is not None
        assert w_gpu.numel == 32

    def test_ricker(self):
        w = ins.signal.ricker(100, 4.0)
        w_gpu = w.to(GPU)
        assert w_gpu is not None
        assert w_gpu.numel == 100

    def test_ricker_narrow(self):
        w = ins.signal.ricker(50, 2.0)
        w_gpu = w.to(GPU)
        assert w_gpu is not None
        assert w_gpu.numel == 50

    def test_morlet2(self):
        w = ins.signal.morlet2(32, s=1.0, w=5.0)
        w_gpu = w.to(GPU)
        assert w_gpu is not None
        assert w_gpu.numel == 32

    def test_morlet2_scaled(self):
        w = ins.signal.morlet2(64, s=2.0, w=5.0)
        w_gpu = w.to(GPU)
        assert w_gpu is not None
        assert w_gpu.numel == 64

    def test_cwt(self):
        x_np = np.sin(2 * np.pi * 5 * np.arange(256) / 256.0).astype(np.float64)
        x = ins.from_numpy(x_np).to(GPU)
        widths = [1, 3, 5, 10, 20]
        result = ins.signal.cwt(x, ins.signal.ricker, widths)
        assert result is not None
        assert result.numel > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
