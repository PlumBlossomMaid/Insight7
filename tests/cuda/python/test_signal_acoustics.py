"""Signal acoustics CUDA binding tests."""

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


def to_gpu(np_arr):
    return ins.from_numpy(np_arr).to(GPU)


def to_numpy(gpu_arr):
    return gpu_arr.to(CPU).numpy()


class TestSignalAcousticsCUDA:
    """Acoustic scale conversions — signal_acoustics CUDA tests."""

    def test_mel2hz(self):
        mel = to_gpu(np.array([0.0, 1000.0, 2000.0]))
        hz = ins.signal.mel2hz(mel)
        assert hz is not None
        assert hz.numel == 3

    def test_hz2mel(self):
        hz = to_gpu(np.array([0.0, 1000.0, 4000.0]))
        mel = ins.signal.hz2mel(hz)
        assert mel is not None
        assert mel.numel == 3

    def test_mel_roundtrip(self):
        hz_orig = np.array([100.0, 500.0, 1000.0, 4000.0, 8000.0])
        mel = ins.signal.hz2mel(to_gpu(hz_orig))
        hz_back = ins.signal.mel2hz(mel)
        np.testing.assert_allclose(to_numpy(hz_back), hz_orig, rtol=1e-5)

    def test_mel_frequencies(self):
        freqs = ins.signal.mel_frequencies(128, fmin=0.0, fmax=11025.0)
        freqs_gpu = freqs.to(GPU)
        assert freqs_gpu is not None
        assert freqs_gpu.numel == 128

    def test_mel_frequencies_custom(self):
        freqs = ins.signal.mel_frequencies(64, fmin=20.0, fmax=8000.0)
        freqs_gpu = freqs.to(GPU)
        assert freqs_gpu is not None
        assert freqs_gpu.numel == 64

    def test_hz2bark(self):
        hz = to_gpu(np.array([100.0, 500.0, 2000.0]))
        bark = ins.signal.hz2bark(hz)
        assert bark is not None
        assert bark.numel == 3

    def test_bark2hz(self):
        bark = to_gpu(np.array([1.0, 5.0, 15.0]))
        hz = ins.signal.bark2hz(bark)
        assert hz is not None
        assert hz.numel == 3

    def test_bark_roundtrip(self):
        hz_orig = np.array([100.0, 500.0, 2000.0, 6000.0])
        bark = ins.signal.hz2bark(to_gpu(hz_orig))
        hz_back = ins.signal.bark2hz(bark)
        np.testing.assert_allclose(to_numpy(hz_back), hz_orig, rtol=1e-3)

    def test_mel_frequencies_sorted(self):
        freqs = ins.signal.mel_frequencies(128, fmin=0.0, fmax=11025.0)
        freqs_np = freqs.to(CPU).numpy()
        assert np.all(np.diff(freqs_np) >= 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
