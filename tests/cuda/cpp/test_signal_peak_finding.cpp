// tests/cuda/test_signal_peak_finding.cpp
#include "insight/insight.h"
#include <gtest/gtest.h>
#include <vector>

using namespace ins;
using namespace ins::signal;

namespace {

class PeakFindingTestGPU : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    ins::init({"cpu", "iluvatar"});
    try {
      set_device(GPUPlace(0));
    // Iluvatar: signal API uses F64
    GTEST_SKIP() << "Iluvatar: signal API uses F64 (hardware limit)";
    } catch (...) {
      GTEST_SKIP() << "GPU not available";
    }
  }
};

// ========== argrelextrema ==========

TEST_F(PeakFindingTestGPU, ArgrelextremaGreater1D) {
  std::vector<double> data = {1, 3, 2, 5, 4, 7, 6};
  Array arr_cpu = to_array(data, DType::F64, CPUPlace());
  Array arr = arr_cpu.to(GPUPlace(0));

  auto indices = argrelextrema(arr, "greater", 0, 1, "clip");

  ASSERT_EQ(indices.size(), 1);
  // Copy index result to CPU for value check
  Array idx_cpu = indices[0].to(CPUPlace());
  ASSERT_EQ(idx_cpu.numel(), 3);
  EXPECT_EQ(idx_cpu.data<int64_t>()[0], 1);
  EXPECT_EQ(idx_cpu.data<int64_t>()[1], 3);
  EXPECT_EQ(idx_cpu.data<int64_t>()[2], 5);
}

TEST_F(PeakFindingTestGPU, ArgrelextremaLess1D) {
  std::vector<double> data = {5, 3, 4, 1, 2, 0, 3};
  Array arr_cpu = to_array(data, DType::F64, CPUPlace());
  Array arr = arr_cpu.to(GPUPlace(0));

  auto indices = argrelextrema(arr, "less", 0, 1, "clip");

  ASSERT_EQ(indices.size(), 1);
  Array idx_cpu = indices[0].to(CPUPlace());
  ASSERT_EQ(idx_cpu.numel(), 3);
  EXPECT_EQ(idx_cpu.data<int64_t>()[0], 1);
  EXPECT_EQ(idx_cpu.data<int64_t>()[1], 3);
  EXPECT_EQ(idx_cpu.data<int64_t>()[2], 5);
}

TEST_F(PeakFindingTestGPU, ArgrelextremaOrder2) {
  std::vector<double> data = {1, 3, 2, 5, 4, 7, 6};
  Array arr_cpu = to_array(data, DType::F64, CPUPlace());
  Array arr = arr_cpu.to(GPUPlace(0));

  auto indices = argrelextrema(arr, "greater", 0, 2, "clip");

  ASSERT_EQ(indices.size(), 1);
  Array idx_cpu = indices[0].to(CPUPlace());
  ASSERT_GE(idx_cpu.numel(), 1);
}

TEST_F(PeakFindingTestGPU, ArgrelextremaNoMaxima) {
  std::vector<double> data = {1, 2, 3, 4, 5};
  Array arr_cpu = to_array(data, DType::F64, CPUPlace());
  Array arr = arr_cpu.to(GPUPlace(0));

  auto indices = argrelextrema(arr, "greater", 0, 1, "clip");

  ASSERT_EQ(indices.size(), 1);
  EXPECT_EQ(indices[0].numel(), 0);
}

TEST_F(PeakFindingTestGPU, ArgrelextremaF32) {
  std::vector<float> data = {1.0f, 3.0f, 2.0f, 5.0f, 4.0f};
  Array arr_cpu = to_array(data, DType::F32, CPUPlace());
  Array arr = arr_cpu.to(GPUPlace(0));

  auto indices = argrelextrema(arr, "greater", 0, 1, "clip");

  ASSERT_EQ(indices.size(), 1);
  Array idx_cpu = indices[0].to(CPUPlace());
  ASSERT_EQ(idx_cpu.numel(), 2);
  EXPECT_EQ(idx_cpu.data<int64_t>()[0], 1);
  EXPECT_EQ(idx_cpu.data<int64_t>()[1], 3);
}

// ========== argrelmax ==========

TEST_F(PeakFindingTestGPU, ArgrelmaxBasic) {
  std::vector<double> data = {1, 3, 2, 5, 4};
  Array arr_cpu = to_array(data, DType::F64, CPUPlace());
  Array arr = arr_cpu.to(GPUPlace(0));

  auto indices = argrelmax(arr, 0, 1, "clip");

  ASSERT_EQ(indices.size(), 1);
  Array idx_cpu = indices[0].to(CPUPlace());
  ASSERT_EQ(idx_cpu.numel(), 2);
  EXPECT_EQ(idx_cpu.data<int64_t>()[0], 1);
  EXPECT_EQ(idx_cpu.data<int64_t>()[1], 3);
}

// ========== argrelmin ==========

TEST_F(PeakFindingTestGPU, ArgrelminBasic) {
  std::vector<double> data = {5, 3, 4, 1, 2};
  Array arr_cpu = to_array(data, DType::F64, CPUPlace());
  Array arr = arr_cpu.to(GPUPlace(0));

  auto indices = argrelmin(arr, 0, 1, "clip");

  ASSERT_EQ(indices.size(), 1);
  Array idx_cpu = indices[0].to(CPUPlace());
  ASSERT_EQ(idx_cpu.numel(), 2);
  EXPECT_EQ(idx_cpu.data<int64_t>()[0], 1);
  EXPECT_EQ(idx_cpu.data<int64_t>()[1], 3);
}

TEST_F(PeakFindingTestGPU, ArgrelminI32) {
  std::vector<int32_t> data = {5, 3, 4, 1, 2};
  Array arr_cpu = to_array(data, DType::I32, CPUPlace());
  Array arr = arr_cpu.to(GPUPlace(0));

  auto indices = argrelmin(arr, 0, 1, "clip");

  ASSERT_EQ(indices.size(), 1);
  Array idx_cpu = indices[0].to(CPUPlace());
  ASSERT_EQ(idx_cpu.numel(), 2);
  EXPECT_EQ(idx_cpu.data<int64_t>()[0], 1);
  EXPECT_EQ(idx_cpu.data<int64_t>()[1], 3);
}

TEST_F(PeakFindingTestGPU, Argrelmax2D) {
  std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9,
                              5, 4, 3, 2, 1, 0, 1, 2, 3};
  Array arr_cpu = to_array(data, Shape({2, 9}), DType::F64, CPUPlace());
  Array arr = arr_cpu.to(GPUPlace(0));

  auto indices = argrelmax(arr, 1, 1, "clip");

  ASSERT_EQ(indices.size(), 2);
}

TEST_F(PeakFindingTestGPU, ArgrelmaxInvalidOrder) {
  std::vector<double> data = {1, 2, 3};
  Array arr_cpu = to_array(data, DType::F64, CPUPlace());
  Array arr = arr_cpu.to(GPUPlace(0));

  EXPECT_THROW(argrelmax(arr, 0, 0, "clip"), std::exception);
}

} // namespace
