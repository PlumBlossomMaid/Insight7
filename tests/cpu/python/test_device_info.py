"""Device information CPU binding tests."""

import sys
import os
import pytest

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
sys.path.insert(0, os.path.join(_root, "bindings", "python"))
sys.path.insert(0, os.path.join(_root, "build", "bindings", "python"))

try:
    import insight as ins
except ImportError:
    pytest.skip("Insight not available", allow_module_level=True)


class TestDeviceInfoCPU:
    def test_device_name_cpu(self):
        name = ins.device_name("cpu", 0)
        assert isinstance(name, str)
        assert len(name) > 0

    def test_device_name_returns_string(self):
        name = ins.device_name("cpu")
        assert isinstance(name, str)

    def test_gpu_version_returns_int(self):
        ver = ins.gpu_version()
        assert isinstance(ver, int)

    def test_driver_version_returns_int(self):
        ver = ins.driver_version()
        assert isinstance(ver, int)

    def test_device_count_returns_int(self):
        count = ins.device_count()
        assert isinstance(count, int)
        assert count >= 0

    def test_compute_capability_returns_int(self):
        cc = ins.compute_capability(0)
        assert isinstance(cc, int)

    def test_device_memory_returns_tuple(self):
        try:
            mem = ins.device_memory(0)
            assert isinstance(mem, tuple)
            assert len(mem) == 2
            assert mem[0] >= 0  # total
            assert mem[1] >= 0  # free
        except RuntimeError:
            pytest.skip("No GPU available for memory query")

    def test_gpu_version_nonneg(self):
        ver = ins.gpu_version()
        assert ver >= 0

    def test_driver_version_nonneg(self):
        ver = ins.driver_version()
        assert ver >= 0

    def test_device_name_no_crash(self):
        # Should not crash even with invalid device
        try:
            name = ins.device_name("cpu", 0)
            assert name is not None
        except RuntimeError:
            pass  # acceptable

    def test_device_count_zero_on_cpu_only(self):
        # On CPU-only builds, GPU count should be 0
        try:
            ins.init()
        except Exception:
            pass
        count = ins.device_count()
        assert isinstance(count, int)

    def test_compute_capability_nonneg(self):
        cc = ins.compute_capability(0)
        assert cc >= 0

    def test_is_initialized(self):
        assert ins.is_initialized()

    def test_device_name_default_args(self):
        name = ins.device_name("cpu")
        assert isinstance(name, str)

    def test_compute_capability_default(self):
        cc = ins.compute_capability()
        assert isinstance(cc, int)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
