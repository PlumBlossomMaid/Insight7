"""Python interoperability protocol tests."""

import os
import sys

import pytest

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
sys.path.insert(0, os.path.join(_root, "bindings", "python"))
sys.path.insert(0, os.path.join(_root, "build", "bindings", "python"))

try:
    import insight as ins
    import numpy as np
except ImportError:
    pytest.skip("Insight or NumPy not available", allow_module_level=True)


class TestPythonInteropCPU:
    def test_array_protocol(self):
        a = ins.from_numpy(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
        out = np.asarray(a)

        assert out.shape == (2, 3)
        assert out.dtype == np.float32
        np.testing.assert_allclose(out, [[1, 2, 3], [4, 5, 6]])

    def test_array_protocol_dtype(self):
        a = ins.from_numpy(np.array([1, 2, 3], dtype=np.int32))
        out = np.asarray(a, dtype=np.float64)

        assert out.dtype == np.float64
        np.testing.assert_allclose(out, [1.0, 2.0, 3.0])

    def test_array_interface_cpu_metadata(self):
        a = ins.from_numpy(np.array([[1, 2], [3, 4]], dtype=np.float64))
        iface = a.__array_interface__

        assert iface["shape"] == (2, 2)
        assert iface["typestr"] == np.dtype(np.float64).str
        assert iface["version"] == 3
        assert iface["data"][0] != 0
        assert iface["data"][1] is False

    def test_numpy_export_keeps_storage_alive(self):
        a = ins.from_numpy(np.arange(6, dtype=np.float64).reshape(2, 3))
        out = a.numpy()
        del a

        np.testing.assert_allclose(out, np.arange(6, dtype=np.float64).reshape(2, 3))

    def test_dlpack_device_cpu(self):
        a = ins.from_numpy(np.array([1, 2, 3], dtype=np.float32))

        assert a.__dlpack_device__() == (1, 0)

    @pytest.mark.skipif(not hasattr(np, "from_dlpack"), reason="NumPy lacks from_dlpack")
    def test_dlpack_roundtrip_cpu(self):
        a = ins.from_numpy(np.array([1, 2, 3], dtype=np.float32))
        out = np.from_dlpack(a)

        assert out.dtype == np.float32
        np.testing.assert_allclose(out, [1, 2, 3])
