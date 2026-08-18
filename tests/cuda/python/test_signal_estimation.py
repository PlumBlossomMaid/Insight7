"""Signal estimation CUDA binding tests."""

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


class TestSignalEstimationCUDA:
    """Kalman filter — signal_estimation CUDA tests."""

    def test_kalman_filter_constructor(self):
        kf = ins.signal.KalmanFilter(2, 1)
        assert kf is not None
        assert kf.dim_x == 2
        assert kf.dim_z == 1

    def test_kalman_filter_state_shape(self):
        kf = ins.signal.KalmanFilter(2, 1)
        x = kf.x
        assert x is not None
        assert x.numel == 2

    def test_kalman_filter_covariance_shape(self):
        kf = ins.signal.KalmanFilter(2, 1)
        P = kf.P
        assert P is not None
        assert P.numel == 4

    def test_kalman_filter_set_F(self):
        kf = ins.signal.KalmanFilter(2, 1)
        F = ins.from_numpy(np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float64))
        kf.F = F
        assert kf.F is not None
        assert kf.F.numel == 4

    def test_kalman_filter_set_H(self):
        kf = ins.signal.KalmanFilter(2, 1)
        H = ins.from_numpy(np.array([[1.0, 0.0]], dtype=np.float64))
        kf.H = H
        assert kf.H is not None
        assert kf.H.numel == 2

    def test_kalman_filter_predict(self):
        kf = ins.signal.KalmanFilter(2, 1)
        F = ins.from_numpy(np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float64))
        H = ins.from_numpy(np.array([[1.0, 0.0]], dtype=np.float64))
        kf.F = F
        kf.H = H
        kf.predict()
        x_after = kf.x
        assert x_after is not None

    def test_kalman_filter_update(self):
        kf = ins.signal.KalmanFilter(2, 1)
        F = ins.from_numpy(np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float64))
        H = ins.from_numpy(np.array([[1.0, 0.0]], dtype=np.float64))
        kf.F = F
        kf.H = H
        kf.predict()
        z = ins.from_numpy(np.array([1.0], dtype=np.float64))
        kf.update(z)
        x_after = kf.x
        assert x_after is not None

    def test_kalman_filter_predict_update_cycle(self):
        kf = ins.signal.KalmanFilter(2, 1)
        F = ins.from_numpy(np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float64))
        H = ins.from_numpy(np.array([[1.0, 0.0]], dtype=np.float64))
        kf.F = F
        kf.H = H
        for _ in range(5):
            kf.predict()
            z = ins.from_numpy(np.array([1.0], dtype=np.float64))
            kf.update(z)
        assert kf.x is not None
        assert kf.x.numel == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
