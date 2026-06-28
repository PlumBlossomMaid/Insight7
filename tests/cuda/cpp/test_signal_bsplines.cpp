// tests/cuda/test_signal_bsplines.cpp
#include "insight/insight.h"
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ins;

namespace {

class SignalBsplinesTestGPU : public ::testing::Test {
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

// ========== gauss_spline ==========

TEST_F(SignalBsplinesTestGPU, GaussSplineBasic) {
  std::vector<double> x_data = {0.0};
  Array x_cpu = to_array(x_data);
  Array x = x_cpu.to(GPUPlace(0));
  Array y = signal::gauss_spline(x, 3);
  Array y_cpu = y.to(CPUPlace());
  double sigma_sq = 4.0 / 12.0;
  double expected = 1.0 / std::sqrt(2.0 * M_PI * sigma_sq);
  EXPECT_NEAR(y_cpu.data<double>()[0], expected, 1e-10);
}

TEST_F(SignalBsplinesTestGPU, GaussSplineSymmetry) {
  std::vector<double> x_data = {-2.0, -1.0, 0.0, 1.0, 2.0};
  Array x_cpu = to_array(x_data);
  Array x = x_cpu.to(GPUPlace(0));
  Array y = signal::gauss_spline(x, 2);
  Array y_cpu = y.to(CPUPlace());
  const double *d = y_cpu.data<double>();
  EXPECT_NEAR(d[0], d[4], 1e-10);
  EXPECT_NEAR(d[1], d[3], 1e-10);
  EXPECT_GT(d[2], d[0]);
  EXPECT_GT(d[2], d[1]);
}

TEST_F(SignalBsplinesTestGPU, GaussSplineDecay) {
  std::vector<double> x_data = {0.0, 1.0, 2.0, 3.0, 5.0};
  Array x_cpu = to_array(x_data);
  Array x = x_cpu.to(GPUPlace(0));
  Array y = signal::gauss_spline(x, 3);
  Array y_cpu = y.to(CPUPlace());
  const double *d = y_cpu.data<double>();
  for (int i = 1; i < 5; ++i) {
    EXPECT_LT(d[i], d[i - 1]);
  }
}

// ========== cubic ==========

TEST_F(SignalBsplinesTestGPU, CubicBasic) {
  std::vector<double> x_data = {0.0};
  Array x_cpu = to_array(x_data);
  Array x = x_cpu.to(GPUPlace(0));
  Array y = signal::cubic(x);
  Array y_cpu = y.to(CPUPlace());
  EXPECT_NEAR(y_cpu.data<double>()[0], 2.0 / 3.0, 1e-10);
}

TEST_F(SignalBsplinesTestGPU, CubicSymmetry) {
  std::vector<double> x_data = {-1.5, -0.5, 0.0, 0.5, 1.5};
  Array x_cpu = to_array(x_data);
  Array x = x_cpu.to(GPUPlace(0));
  Array y = signal::cubic(x);
  Array y_cpu = y.to(CPUPlace());
  const double *d = y_cpu.data<double>();
  EXPECT_NEAR(d[0], d[4], 1e-10);
  EXPECT_NEAR(d[1], d[3], 1e-10);
}

TEST_F(SignalBsplinesTestGPU, CubicZeroOutside) {
  std::vector<double> x_data = {-2.5, -2.0, 2.0, 2.5};
  Array x_cpu = to_array(x_data);
  Array x = x_cpu.to(GPUPlace(0));
  Array y = signal::cubic(x);
  Array y_cpu = y.to(CPUPlace());
  const double *d = y_cpu.data<double>();
  EXPECT_NEAR(d[0], 0.0, 1e-10);
  EXPECT_NEAR(d[1], 0.0, 1e-10);
  EXPECT_NEAR(d[2], 0.0, 1e-10);
  EXPECT_NEAR(d[3], 0.0, 1e-10);
}

TEST_F(SignalBsplinesTestGPU, CubicRegion1) {
  std::vector<double> x_data = {0.5};
  Array x_cpu = to_array(x_data);
  Array x = x_cpu.to(GPUPlace(0));
  Array y = signal::cubic(x);
  Array y_cpu = y.to(CPUPlace());
  double expected = 2.0 / 3.0 - 0.5 * 0.25 * 1.5;
  EXPECT_NEAR(y_cpu.data<double>()[0], expected, 1e-10);
}

TEST_F(SignalBsplinesTestGPU, CubicRegion2) {
  std::vector<double> x_data = {1.5};
  Array x_cpu = to_array(x_data);
  Array x = x_cpu.to(GPUPlace(0));
  Array y = signal::cubic(x);
  Array y_cpu = y.to(CPUPlace());
  double expected = (1.0 / 6.0) * 0.125;
  EXPECT_NEAR(y_cpu.data<double>()[0], expected, 1e-10);
}

// ========== quadratic ==========

TEST_F(SignalBsplinesTestGPU, QuadraticBasic) {
  std::vector<double> x_data = {0.0};
  Array x_cpu = to_array(x_data);
  Array x = x_cpu.to(GPUPlace(0));
  Array y = signal::quadratic(x);
  Array y_cpu = y.to(CPUPlace());
  EXPECT_NEAR(y_cpu.data<double>()[0], 0.75, 1e-10);
}

TEST_F(SignalBsplinesTestGPU, QuadraticSymmetry) {
  std::vector<double> x_data = {-1.0, -0.25, 0.0, 0.25, 1.0};
  Array x_cpu = to_array(x_data);
  Array x = x_cpu.to(GPUPlace(0));
  Array y = signal::quadratic(x);
  Array y_cpu = y.to(CPUPlace());
  const double *d = y_cpu.data<double>();
  EXPECT_NEAR(d[0], d[4], 1e-10);
  EXPECT_NEAR(d[1], d[3], 1e-10);
}

TEST_F(SignalBsplinesTestGPU, QuadraticZeroOutside) {
  std::vector<double> x_data = {-2.0, -1.5, 1.5, 2.0};
  Array x_cpu = to_array(x_data);
  Array x = x_cpu.to(GPUPlace(0));
  Array y = signal::quadratic(x);
  Array y_cpu = y.to(CPUPlace());
  const double *d = y_cpu.data<double>();
  EXPECT_NEAR(d[0], 0.0, 1e-10);
  EXPECT_NEAR(d[1], 0.0, 1e-10);
  EXPECT_NEAR(d[2], 0.0, 1e-10);
  EXPECT_NEAR(d[3], 0.0, 1e-10);
}

TEST_F(SignalBsplinesTestGPU, QuadraticRegion1) {
  std::vector<double> x_data = {0.3};
  Array x_cpu = to_array(x_data);
  Array x = x_cpu.to(GPUPlace(0));
  Array y = signal::quadratic(x);
  Array y_cpu = y.to(CPUPlace());
  EXPECT_NEAR(y_cpu.data<double>()[0], 0.75 - 0.09, 1e-10);
}

TEST_F(SignalBsplinesTestGPU, QuadraticRegion2) {
  std::vector<double> x_data = {1.0};
  Array x_cpu = to_array(x_data);
  Array x = x_cpu.to(GPUPlace(0));
  Array y = signal::quadratic(x);
  Array y_cpu = y.to(CPUPlace());
  EXPECT_NEAR(y_cpu.data<double>()[0], 0.5 * 0.25, 1e-10);
}

} // namespace
