"""Signal spectral CUDA binding tests."""

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


class TestSignalSpectralCUDA:
    """Spectral analysis — signal_spectral CUDA tests."""

    def _make_sine(self, freq=10.0, fs=256.0, n=1024):
        t = np.arange(n) / fs
        return np.sin(2 * np.pi * freq * t).astype(np.float64)

    def test_welch(self):
        x_np = self._make_sine()
        x = ins.from_numpy(x_np).to(GPU)
        result = ins.signal.welch(x, fs=256.0, nperseg=256)
        assert result is not None
        f = result.f
        pxx = result.Pxx
        assert f.numel > 0
        assert pxx.numel > 0
        assert f.numel == pxx.numel

    def test_periodogram(self):
        x_np = self._make_sine(n=512)
        x = ins.from_numpy(x_np).to(GPU)
        result = ins.signal.periodogram(x, fs=256.0)
        assert result is not None
        f = result.f
        pxx = result.Pxx
        assert f.numel > 0
        assert pxx.numel > 0

    def test_csd(self):
        x_np = self._make_sine(freq=10.0)
        y_np = self._make_sine(freq=10.0)
        x = ins.from_numpy(x_np).to(GPU)
        y = ins.from_numpy(y_np).to(GPU)
        result = ins.signal.csd(x, y, fs=256.0, nperseg=256)
        assert result is not None
        f = result.f
        pxx = result.Pxx
        assert f.numel > 0
        assert pxx.numel > 0

    def test_coherence(self):
        x_np = self._make_sine(freq=10.0)
        y_np = self._make_sine(freq=10.0) + 0.1 * np.random.randn(1024)
        x = ins.from_numpy(x_np).to(GPU)
        y = ins.from_numpy(y_np).to(GPU)
        result = ins.signal.coherence(x, y, fs=256.0, nperseg=256)
        assert result is not None
        f = result.f
        pxx = result.Pxx
        assert f.numel > 0
        assert pxx.numel > 0

    def test_spectrogram(self):
        x_np = self._make_sine()
        x = ins.from_numpy(x_np).to(GPU)
        result = ins.signal.spectrogram(x, fs=256.0, nperseg=256)
        assert result is not None
        assert result.f.numel > 0
        assert result.t.numel > 0
        assert result.Sxx.numel > 0

    def test_stft(self):
        x_np = self._make_sine()
        x = ins.from_numpy(x_np).to(GPU)
        result = ins.signal.stft(x, fs=256.0, nperseg=256)
        assert result is not None
        assert result.f.numel > 0
        assert result.t.numel > 0
        assert result.Sxx.numel > 0

    def test_vectorstrength(self):
        events = ins.from_numpy(np.array([0.0, 1.0, 2.0, 3.0])).to(GPU)
        result = ins.signal.vectorstrength(events, 1.0)
        assert result is not None

    def test_welch_custom_params(self):
        x_np = self._make_sine(n=2048)
        x = ins.from_numpy(x_np).to(GPU)
        result = ins.signal.welch(x, fs=256.0, nperseg=512, noverlap=256, scaling="spectrum")
        assert result is not None
        assert result.f.numel > 0
        assert result.Pxx.numel > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
