"""Signal peak_finding CUDA binding tests."""

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


class TestSignalPeakFindingCUDA:
    """Peak finding — signal_peak_finding CUDA tests."""

    def test_argrelmax(self):
        data_np = np.array([0, 2, 1, 3, 0, 4, 1], dtype=np.float64)
        x = to_gpu(data_np)
        result = ins.signal.argrelmax(x)
        assert result is not None
        assert len(result) > 0

    def test_argrelmin(self):
        data_np = np.array([3, 1, 4, 0, 5, 2, 6], dtype=np.float64)
        x = to_gpu(data_np)
        result = ins.signal.argrelmin(x)
        assert result is not None
        assert len(result) > 0

    def test_argrelextrema_greater(self):
        data_np = np.array([0, 2, 1, 3, 0, 4, 1], dtype=np.float64)
        x = to_gpu(data_np)
        result = ins.signal.argrelextrema(x, comparator="greater")
        assert result is not None

    def test_argrelextrema_less(self):
        data_np = np.array([3, 1, 4, 0, 5, 2, 6], dtype=np.float64)
        x = to_gpu(data_np)
        result = ins.signal.argrelextrema(x, comparator="less")
        assert result is not None

    def test_argrelmax_order2(self):
        data_np = np.array([0, 1, 3, 2, 1, 5, 2, 1, 0], dtype=np.float64)
        x = to_gpu(data_np)
        result = ins.signal.argrelmax(x, order=2)
        assert result is not None

    def test_argrelmin_order2(self):
        data_np = np.array([5, 3, 1, 2, 4, 0, 3, 4, 5], dtype=np.float64)
        x = to_gpu(data_np)
        result = ins.signal.argrelmin(x, order=2)
        assert result is not None

    def test_argrelmax_no_peaks(self):
        data_np = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        x = to_gpu(data_np)
        result = ins.signal.argrelmax(x)
        assert result is not None

    def test_argrelmin_no_valleys(self):
        data_np = np.array([5, 4, 3, 2, 1], dtype=np.float64)
        x = to_gpu(data_np)
        result = ins.signal.argrelmin(x)
        assert result is not None

    def test_argrelmax_sine(self):
        data_np = np.sin(np.linspace(0, 4 * np.pi, 100)).astype(np.float64)
        x = to_gpu(data_np)
        result = ins.signal.argrelmax(x)
        assert result is not None

    def test_argrelmin_sine(self):
        data_np = np.sin(np.linspace(0, 4 * np.pi, 100)).astype(np.float64)
        x = to_gpu(data_np)
        result = ins.signal.argrelmin(x)
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
