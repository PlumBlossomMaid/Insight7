"""Signal io CUDA binding tests."""

import sys
import os
import tempfile
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


class TestSignalIOCUDA:
    """Binary I/O — signal_io CUDA tests."""

    def test_write_read_bin_roundtrip(self):
        data_np = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        data = to_gpu(data_np)
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            path = f.name
        try:
            ins.signal.write_bin(path, data)
            result = ins.signal.read_bin(path, dtype="float64")
            assert result is not None
            assert result.numel == 5
            np.testing.assert_allclose(to_numpy(result), data_np, rtol=1e-10)
        finally:
            os.unlink(path)

    def test_write_read_bin_float32(self):
        data_np = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        data = to_gpu(data_np)
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            path = f.name
        try:
            ins.signal.write_bin(path, data)
            result = ins.signal.read_bin(path, dtype="float32")
            assert result is not None
            assert result.numel == 3
        finally:
            os.unlink(path)

    def test_write_read_bin_int16(self):
        data_np = np.array([100, 200, 300, 400], dtype=np.int16)
        data = to_gpu(data_np)
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            path = f.name
        try:
            ins.signal.write_bin(path, data)
            result = ins.signal.read_bin(path, dtype="int16")
            assert result is not None
            assert result.numel == 4
        finally:
            os.unlink(path)

    def test_write_read_bin_large(self):
        data_np = np.random.randn(1024).astype(np.float64)
        data = to_gpu(data_np)
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            path = f.name
        try:
            ins.signal.write_bin(path, data)
            result = ins.signal.read_bin(path, dtype="float64")
            assert result is not None
            assert result.numel == 1024
            np.testing.assert_allclose(to_numpy(result), data_np, rtol=1e-10)
        finally:
            os.unlink(path)

    def test_pack_bin(self):
        data_np = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        data = to_gpu(data_np)
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            path = f.name
        try:
            ins.signal.pack_bin(path, data)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_unpack_bin(self):
        data_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        data = to_gpu(data_np)
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            path = f.name
        try:
            ins.signal.write_bin(path, data)
            result = ins.signal.unpack_bin(path, count=32)
            assert result is not None
            assert result.numel > 0
        finally:
            os.unlink(path)

    def test_write_sigmf(self):
        data_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        data = to_gpu(data_np)
        with tempfile.NamedTemporaryFile(suffix=".sigmf-data", delete=False) as f:
            path = f.name
        try:
            ins.signal.write_sigmf(path, data)
            assert os.path.exists(path)
        finally:
            os.unlink(path)

    def test_write_sigmf_with_metadata(self):
        data_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        data = to_gpu(data_np)
        with tempfile.NamedTemporaryFile(suffix=".sigmf-data", delete=False) as f:
            path = f.name
        try:
            meta = {"core:datatype": "cf64_le", "core:sample_rate": 1e6}
            ins.signal.write_sigmf(path, data, metadata=meta)
            assert os.path.exists(path)
            meta_path = path.replace(".sigmf-data", ".sigmf-meta")
            assert os.path.exists(meta_path)
        finally:
            if os.path.exists(path):
                os.unlink(path)
            meta_path = path.replace(".sigmf-data", ".sigmf-meta")
            if os.path.exists(meta_path):
                os.unlink(meta_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
