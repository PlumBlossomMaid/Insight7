// tests/cuda/test_signal_demod.cpp
#include "insight/insight.h"
#include <cmath>
#include <complex>
#include <gtest/gtest.h>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ins;
using namespace ins::signal;

namespace {

class DemodTestGPU : public ::testing::Test {
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

// ========== fm_demod ==========

TEST_F(DemodTestGPU, FmDemodBasic) {
  int N = 100;
  double fs = 1000.0;
  double freq = 100.0;
  std::vector<std::complex<double>> signal_data(N);
  for (int i = 0; i < N; ++i) {
    double t = i / fs;
    signal_data[i] = std::exp(std::complex<double>(0.0, 2.0 * M_PI * freq * t));
  }
  Array arr_cpu = to_array(signal_data, DType::C64, CPUPlace());
  Array arr = arr_cpu.to(GPUPlace(0));

  Array result = fm_demod(arr, -1);

  EXPECT_EQ(result.numel(), N - 1);
  EXPECT_EQ(result.dtype(), DType::F64);

  // Copy back to CPU for value checks
  Array result_cpu = result.to(CPUPlace());
  const double *rd = result_cpu.data<double>();
  double expected = 2.0 * M_PI * freq / fs;
  for (int i = 10; i < N - 10; ++i) {
    EXPECT_NEAR(rd[i], expected, 0.1);
  }
}

TEST_F(DemodTestGPU, FmDemodC32) {
  int N = 50;
  std::vector<std::complex<float>> signal_data(N);
  for (int i = 0; i < N; ++i) {
    float t = i / 1000.0f;
    signal_data[i] =
        std::exp(std::complex<float>(0.0f, 2.0f * 3.14159f * 50.0f * t));
  }
  Array arr_cpu = to_array(signal_data, DType::C32, CPUPlace());
  Array arr = arr_cpu.to(GPUPlace(0));

  Array result = fm_demod(arr, -1);

  EXPECT_EQ(result.numel(), N - 1);
  EXPECT_EQ(result.dtype(), DType::F32);
}

TEST_F(DemodTestGPU, FmDemodInvalidInput) {
  std::vector<double> data = {1.0, 2.0, 3.0};
  Array arr_cpu = to_array(data, DType::F64, CPUPlace());
  Array arr = arr_cpu.to(GPUPlace(0));

  EXPECT_THROW(fm_demod(arr), std::exception);
}

TEST_F(DemodTestGPU, FmDemodVaryingFrequency) {
  int N = 200;
  double fs = 1000.0;
  std::vector<std::complex<double>> signal_data(N);
  for (int i = 0; i < N; ++i) {
    double t = i / fs;
    double freq = 100.0 + 50.0 * std::sin(2.0 * M_PI * 10.0 * t);
    signal_data[i] = std::exp(std::complex<double>(0.0, 2.0 * M_PI * freq * t));
  }
  Array arr_cpu = to_array(signal_data, DType::C64, CPUPlace());
  Array arr = arr_cpu.to(GPUPlace(0));

  Array result = fm_demod(arr, -1);

  EXPECT_EQ(result.numel(), N - 1);

  Array result_cpu = result.to(CPUPlace());
  const double *rd = result_cpu.data<double>();
  double first = rd[10];
  double mid = rd[100];
  EXPECT_TRUE(std::isfinite(first));
  EXPECT_TRUE(std::isfinite(mid));
}

} // namespace
