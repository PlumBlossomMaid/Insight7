#include "insight/core/place.h"
#include "insight/core/profiler.h"
#include "insight/init.h"
#include "gtest/gtest.h"
#include <cstdint>
#include <vector>

using namespace ins;

class SdaaRuntimeTestGPU : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    ins::init();
    if (!ins::is_device_available(ins::DeviceKind::GPU)) {
      GTEST_SKIP() << "GPU not available";
    }
    if (ins::active_gpu_backend_name() != "sdaa") {
      GTEST_SKIP() << "SDAA backend not active";
    }
  }
};

TEST_F(SdaaRuntimeTestGPU, DeviceInfo) {
  EXPECT_EQ(active_gpu_backend_name(), "sdaa");
  EXPECT_EQ(active_gpu_backend_version(), "runtime");
  EXPECT_GT(device_count(DeviceKind::GPU), 0);
  EXPECT_FALSE(device_name(DeviceKind::GPU, 0).empty());
  EXPECT_GT(gpu_runtime_version(), 0);
  EXPECT_GT(driver_version(), 0);
}

TEST_F(SdaaRuntimeTestGPU, MemoryStats) {
  auto info = device_memory_info(DeviceKind::GPU, 0);
  EXPECT_GT(info.total, 0);
  EXPECT_GT(info.free, 0);
  EXPECT_GE(info.total, info.free);
}

TEST_F(SdaaRuntimeTestGPU, HostDeviceCopyRoundTrip) {
  Place device_place = GPUPlace(0);
  std::vector<uint32_t> host{1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<uint32_t> out(host.size(), 0);

  void *device = device_place.allocate(host.size() * sizeof(uint32_t));
  device_place.copy_from_host(device, host.data(), host.size() * sizeof(uint32_t));
  device_place.copy_to_host(out.data(), device, out.size() * sizeof(uint32_t));
  device_place.deallocate(device, host.size() * sizeof(uint32_t));

  EXPECT_EQ(out, host);
}

TEST_F(SdaaRuntimeTestGPU, DeviceToDeviceCopyRoundTrip) {
  Place device_place = GPUPlace(0);
  std::vector<uint32_t> host{8, 7, 6, 5, 4, 3, 2, 1};
  std::vector<uint32_t> out(host.size(), 0);
  size_t bytes = host.size() * sizeof(uint32_t);

  void *src = device_place.allocate(bytes);
  void *dst = device_place.allocate(bytes);
  device_place.copy_from_host(src, host.data(), bytes);
  device_place.copy_on_device(dst, src, bytes);
  device_place.copy_to_host(out.data(), dst, bytes);
  device_place.deallocate(dst, bytes);
  device_place.deallocate(src, bytes);

  EXPECT_EQ(out, host);
}

TEST_F(SdaaRuntimeTestGPU, TimerEventSmoke) {
  Timer timer(GPUPlace(0));
  timer.start();
  GPUPlace(0).synchronize();
  timer.stop();
  EXPECT_GE(timer.elapsed_ms(), 0.0f);
}
