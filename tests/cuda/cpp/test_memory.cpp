// tests/cuda/cpp/test_memory.cpp
#include "insight/c_api/memory.h"
#include "insight/core/place.h"
#include "insight/insight.h"
#include "gtest/gtest.h"

class MemoryTestGPU : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    ins::init();
    try {
      ins::set_device(ins::GPUPlace(0));
    } catch (...) {
      GTEST_SKIP() << "GPU not available";
    }
  }
};

// ========== C++ API ==========

TEST_F(MemoryTestGPU, DeviceMemoryInfoGPU) {
  auto info = ins::device_memory_info(ins::DeviceKind::GPU, 0);
  EXPECT_GT(info.total, 0);
  EXPECT_GT(info.free, 0);
  // GPU total should be reasonable (e.g., > 1 MB)
  EXPECT_GT(info.total, 1024 * 1024);
}

TEST_F(MemoryTestGPU, DeviceMemoryInfoGPUConvenience) {
  auto info = ins::device_memory(0);
  EXPECT_GT(info.total, 0);
  EXPECT_GT(info.free, 0);
}

// ========== C API ==========

TEST_F(MemoryTestGPU, CDeviceMemoryInfoGPU) {
  size_t total = 0, free_mem = 0;
  C_Status status = insight_device_memory_info(1, 0, &total, &free_mem);
  EXPECT_EQ(status, C_SUCCESS);
  EXPECT_GT(total, 0);
  EXPECT_GT(free_mem, 0);
}
