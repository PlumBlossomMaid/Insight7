// tests/cuda/test_signal_radar.cpp
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

class RadarTestGPU : public ::testing::Test {
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

// ========== cfar_alpha ==========

TEST_F(RadarTestGPU, CfarAlphaBasic) {
  double alpha = cfar_alpha(1e-3, 20);
  EXPECT_GT(alpha, 5.0);
  EXPECT_LT(alpha, 15.0);
}

TEST_F(RadarTestGPU, CfarAlphaSmallPfa) {
  double alpha = cfar_alpha(1e-6, 100);
  EXPECT_GT(alpha, 0.0);
}

// ========== ca_cfar 1D ==========

TEST_F(RadarTestGPU, CaCfar1DBasic) {
  int N = 100;
  std::vector<double> data(N, 1.0);
  data[50] = 100.0;

  Array arr_cpu = to_array(data, DType::F64, CPUPlace());
  Array arr = arr_cpu.to(GPUPlace(0));

  auto [threshold, detections] = ca_cfar(arr, {2}, {5}, 1e-3);

  EXPECT_EQ(threshold.numel(), N);
  EXPECT_EQ(detections.numel(), N);

  // Copy detections back to CPU for value checks
  Array det_cpu = detections.to(CPUPlace());
  const bool *det = det_cpu.data<bool>();
  EXPECT_TRUE(det[50]);

  int false_alarms = 0;
  for (int i = 0; i < N; ++i) {
    if (i >= 40 && i <= 60)
      continue;
    if (det[i])
      false_alarms++;
  }
  EXPECT_LT(false_alarms, 10);
}

TEST_F(RadarTestGPU, CaCfar1DF32) {
  std::vector<float> data(100, 1.0f);
  data[50] = 100.0f;
  Array arr_cpu = to_array(data, DType::F32, CPUPlace());
  Array arr = arr_cpu.to(GPUPlace(0));

  auto [threshold, detections] = ca_cfar(arr, {2}, {5}, 1e-3);

  EXPECT_EQ(threshold.dtype(), DType::F32);
  Array det_cpu = detections.to(CPUPlace());
  EXPECT_TRUE(det_cpu.data<bool>()[50]);
}

TEST_F(RadarTestGPU, CaCfar1DNoTarget) {
  std::vector<double> data(100, 1.0);
  Array arr_cpu = to_array(data, DType::F64, CPUPlace());
  Array arr = arr_cpu.to(GPUPlace(0));

  auto [threshold, detections] = ca_cfar(arr, {2}, {5}, 1e-3);

  Array det_cpu = detections.to(CPUPlace());
  const bool *det = det_cpu.data<bool>();
  int count = 0;
  for (int i = 0; i < 100; ++i) {
    if (det[i])
      count++;
  }
  EXPECT_LT(count, 10);
}

// ========== ca_cfar 2D ==========

TEST_F(RadarTestGPU, CaCfar2DBasic) {
  int R = 20, C = 20;
  std::vector<double> data(R * C, 1.0);
  data[10 * C + 10] = 100.0;

  Array arr_cpu = to_array(data, Shape({R, C}), DType::F64, CPUPlace());
  Array arr = arr_cpu.to(GPUPlace(0));

  auto [threshold, detections] = ca_cfar(arr, {1, 1}, {2, 2}, 1e-3);

  EXPECT_EQ(threshold.numel(), R * C);
  Array det_cpu = detections.to(CPUPlace());
  const bool *det = det_cpu.data<bool>();
  EXPECT_TRUE(det[10 * C + 10]);
}

// ========== mvdr ==========

TEST_F(RadarTestGPU, MvdrBasic) {
#ifndef INSIGHT_USE_FFTW3
  GTEST_SKIP() << "FFTW3 not available";
#endif
#ifndef INSIGHT_USE_OPENBLAS
  GTEST_SKIP() << "OpenBLAS not available";
#endif
  int M = 2, N = 10;
  std::vector<double> x_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                3, 1, 4, 1, 5, 9, 2, 6, 5, 3};
  Array x_cpu = to_array(x_data, Shape({M, N}), DType::F64, CPUPlace());
  Array x = x_cpu.to(GPUPlace(0));

  std::vector<double> sv_data = {1.0, 0.5};
  Array sv_cpu = to_array(sv_data, DType::F64, CPUPlace());
  Array sv = sv_cpu.to(GPUPlace(0));

  Array w = mvdr(x, sv, true);

  Array w_cpu = w.to(CPUPlace());
  EXPECT_EQ(w_cpu.numel(), M);
  EXPECT_TRUE(std::isfinite(w_cpu.data<double>()[0]));
  EXPECT_TRUE(std::isfinite(w_cpu.data<double>()[1]));
}

TEST_F(RadarTestGPU, MvdrInvalidShape) {
#ifndef INSIGHT_USE_FFTW3
  GTEST_SKIP() << "FFTW3 not available";
#endif
  std::vector<double> x_data = {1, 2, 3, 4, 5, 6};
  Array x_cpu = to_array(x_data, Shape({3, 2}), DType::F64, CPUPlace());
  Array x = x_cpu.to(GPUPlace(0));
  std::vector<double> sv_data = {1.0, 0.5, 0.3};
  Array sv_cpu = to_array(sv_data, DType::F64, CPUPlace());
  Array sv = sv_cpu.to(GPUPlace(0));

  EXPECT_THROW(mvdr(x, sv, true), std::exception);
}

// ========== pulse_compression ==========

TEST_F(RadarTestGPU, PulseCompressionBasic) {
#ifndef INSIGHT_USE_FFTW3
  GTEST_SKIP() << "FFTW3 not available";
#endif
  int num_pulses = 3, samples = 8;
  std::vector<double> x_data(num_pulses * samples);
  for (int i = 0; i < num_pulses * samples; ++i)
    x_data[i] = std::sin(2.0 * M_PI * i / 8.0);

  std::vector<double> tpl_data = {1.0, 0.5, -0.5, -1.0};

  Array x_cpu =
      to_array(x_data, Shape({num_pulses, samples}), DType::F64, CPUPlace());
  Array x = x_cpu.to(GPUPlace(0));
  Array tpl_cpu = to_array(tpl_data, DType::F64, CPUPlace());
  Array tpl = tpl_cpu.to(GPUPlace(0));

  Array result = pulse_compression(x, tpl, false, "", 0);

  EXPECT_EQ(result.shape().dim(0), num_pulses);
  EXPECT_EQ(result.shape().dim(1), samples);
  EXPECT_EQ(result.dtype(), DType::C64);
}

TEST_F(RadarTestGPU, PulseCompressionNormalize) {
#ifndef INSIGHT_USE_FFTW3
  GTEST_SKIP() << "FFTW3 not available";
#endif
  int num_pulses = 2, samples = 8;
  std::vector<double> x_data(num_pulses * samples, 1.0);
  std::vector<double> tpl_data = {1.0, 1.0, 1.0, 1.0};

  Array x_cpu =
      to_array(x_data, Shape({num_pulses, samples}), DType::F64, CPUPlace());
  Array x = x_cpu.to(GPUPlace(0));
  Array tpl_cpu = to_array(tpl_data, DType::F64, CPUPlace());
  Array tpl = tpl_cpu.to(GPUPlace(0));

  Array result = pulse_compression(x, tpl, true);

  EXPECT_EQ(result.shape().dim(0), num_pulses);
}

// ========== pulse_doppler ==========

TEST_F(RadarTestGPU, PulseDopplerBasic) {
#ifndef INSIGHT_USE_FFTW3
  GTEST_SKIP() << "FFTW3 not available";
#endif
  int num_pulses = 8, samples = 4;
  std::vector<double> x_data(num_pulses * samples);
  for (int i = 0; i < num_pulses * samples; ++i)
    x_data[i] = std::sin(2.0 * M_PI * i / 8.0);

  Array x_cpu =
      to_array(x_data, Shape({num_pulses, samples}), DType::F64, CPUPlace());
  Array x = x_cpu.to(GPUPlace(0));

  Array result = pulse_doppler(x, "", 0);

  EXPECT_EQ(result.shape().dim(0), num_pulses);
  EXPECT_EQ(result.shape().dim(1), samples);
  EXPECT_EQ(result.dtype(), DType::C64);
}

} // namespace
