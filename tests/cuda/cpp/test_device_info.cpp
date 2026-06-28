// tests/cuda/test_device_info.cu
#include "insight/core/place.h"
#include "insight/init.h"
#include <gtest/gtest.h>
#include <string>

using namespace ins;

class DeviceInfoTestGPU : public ::testing::Test {
protected:
  static void SetUpTestSuite() { ins::init({"cpu", "iluvatar"}); }
};

TEST_F(DeviceInfoTestGPU, GPUDeviceCount) {
  size_t count = device_count(DeviceKind::GPU);
  EXPECT_GE(count, 1);
}

TEST_F(DeviceInfoTestGPU, GPUDeviceName) {
  std::string name = device_name(DeviceKind::GPU, 0);
  EXPECT_FALSE(name.empty());
  EXPECT_NE(name, "CPU");
}

TEST_F(DeviceInfoTestGPU, CudaVersion) {
  int ver = cuda_version();
  EXPECT_GT(ver, 0);
}

TEST_F(DeviceInfoTestGPU, DriverVersion) {
  int ver = driver_version();
  EXPECT_GT(ver, 0);
}

TEST_F(DeviceInfoTestGPU, ComputeCapability) {
  int cap = compute_capability(0);
  EXPECT_GT(cap, 0);
  EXPECT_GE(cap, 50);
}

TEST_F(DeviceInfoTestGPU, DeviceMemory) {
  DeviceMemoryInfo info = device_memory(0);
  EXPECT_GT(info.total, static_cast<size_t>(0));
  EXPECT_GT(info.free, static_cast<size_t>(0));
  EXPECT_GE(info.total, info.free);
}

TEST_F(DeviceInfoTestGPU, IsDeviceAvailableGPU) {
  EXPECT_TRUE(is_device_available(DeviceKind::GPU));
}

TEST_F(DeviceInfoTestGPU, GPUPlace) {
  Place p = GPUPlace(0);
  EXPECT_TRUE(p.is_gpu());
  EXPECT_EQ(p.device_id(), 0);
}
