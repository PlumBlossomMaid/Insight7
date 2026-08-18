// tests/cpu/test_device_info.cpp
#include "insight/core/place.h"
#include "insight/init.h"
#include <cstdlib>
#include <gtest/gtest.h>

using namespace ins;

class DeviceInfoTestCPU : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
#ifdef INSIGHT_WITH_SDAA
    setenv("INSIGHT_GPU_BACKEND", "sdaa", 1);
#endif
    ins::init();
  }
};

TEST_F(DeviceInfoTestCPU, DeviceCount) {
  size_t count = device_count(DeviceKind::CPU);
  EXPECT_GE(count, 0);
}

TEST_F(DeviceInfoTestCPU, DeviceNameCPU) {
  std::string name = device_name(DeviceKind::CPU);
  EXPECT_EQ(name, "CPU");
}

TEST_F(DeviceInfoTestCPU, GpuRuntimeVersionReturnsInt) {
  int ver = gpu_runtime_version();
  EXPECT_GE(ver, 0);
}

TEST_F(DeviceInfoTestCPU, DriverVersionNoGPU) {
  int ver = driver_version();
  EXPECT_GE(ver, 0);
}

TEST_F(DeviceInfoTestCPU, ComputeCapabilityNoGPU) {
  int cap = compute_capability(0);
  EXPECT_GE(cap, 0);
}

TEST_F(DeviceInfoTestCPU, DeviceMemoryNoGPU) {
  DeviceMemoryInfo info = device_memory(0);
  EXPECT_GE(info.total, static_cast<size_t>(0));
  EXPECT_GE(info.free, static_cast<size_t>(0));
}

TEST_F(DeviceInfoTestCPU, IsDeviceAvailableCPU) {
  EXPECT_TRUE(is_device_available(DeviceKind::CPU));
}

#ifdef INSIGHT_WITH_SDAA
TEST_F(DeviceInfoTestCPU, ActiveGpuBackendSdaa) {
  if (is_device_available(DeviceKind::GPU)) {
    EXPECT_EQ(active_gpu_backend_name(), "sdaa");
    EXPECT_EQ(active_gpu_backend_version(), "runtime");
  }
}

TEST_F(DeviceInfoTestCPU, InitOptionsCanRequestSdaaBackend) {
  InitOptions options;
  options.gpu_backend = "sdaa";
  ins::init(options);

  if (is_device_available(DeviceKind::GPU)) {
    EXPECT_EQ(active_gpu_backend_name(), "sdaa");
  }
}
#endif
