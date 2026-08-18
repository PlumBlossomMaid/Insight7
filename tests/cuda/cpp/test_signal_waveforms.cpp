// tests/cuda/test_signal_waveforms.cpp
#include "insight/insight.h"
#include <cmath>
#include <complex>
#include <gtest/gtest.h>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ins;

namespace {

class SignalWaveformsTestGPU : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    ins::init();
    try {
      set_device(GPUPlace(0));
      // IXUCA-only: signal API uses F64
      if (active_gpu_backend_name() == "ixuca") GTEST_SKIP() << "IXUCA: signal API uses F64 (hardware limit)";
    } catch (...) {
      GTEST_SKIP() << "GPU not available";
    }
  }
};

// ========== sawtooth ==========

TEST_F(SignalWaveformsTestGPU, SawtoothBasic) {
  std::vector<double> t_data = {0.0, M_PI, 2.0 * M_PI - 1e-10};
  Array t_cpu = to_array(t_data);
  Array t = t_cpu.to(GPUPlace(0));
  Array y = signal::sawtooth(t, 1.0);
  Array y_cpu = y.to(CPUPlace());
  const double *d = y_cpu.data<double>();
  EXPECT_NEAR(d[0], -1.0, 1e-6);
  EXPECT_NEAR(d[1], 0.0, 1e-6);
  EXPECT_NEAR(d[2], 1.0, 1e-6);
}

TEST_F(SignalWaveformsTestGPU, SawtoothTriangle) {
  std::vector<double> t_data = {0.0, M_PI / 2.0, M_PI};
  Array t_cpu = to_array(t_data);
  Array t = t_cpu.to(GPUPlace(0));
  Array y = signal::sawtooth(t, 0.5);
  Array y_cpu = y.to(CPUPlace());
  const double *d = y_cpu.data<double>();
  EXPECT_NEAR(d[0], -1.0, 1e-6);
  EXPECT_NEAR(d[1], 0.0, 1e-6);
  EXPECT_NEAR(d[2], 1.0, 1e-6);
}

TEST_F(SignalWaveformsTestGPU, SawtoothPeriodicity) {
  std::vector<double> t_data = {0.5, 0.5 + 2.0 * M_PI, 0.5 + 4.0 * M_PI};
  Array t_cpu = to_array(t_data);
  Array t = t_cpu.to(GPUPlace(0));
  Array y = signal::sawtooth(t, 1.0);
  Array y_cpu = y.to(CPUPlace());
  const double *d = y_cpu.data<double>();
  EXPECT_NEAR(d[0], d[1], 1e-6);
  EXPECT_NEAR(d[0], d[2], 1e-6);
}

// ========== square ==========

TEST_F(SignalWaveformsTestGPU, SquareBasic) {
  std::vector<double> t_data = {0.0, M_PI, 2.0 * M_PI - 1e-10};
  Array t_cpu = to_array(t_data);
  Array t = t_cpu.to(GPUPlace(0));
  Array y = signal::square(t, 0.5);
  Array y_cpu = y.to(CPUPlace());
  const double *d = y_cpu.data<double>();
  EXPECT_NEAR(d[0], 1.0, 1e-6);
  EXPECT_NEAR(d[1], -1.0, 1e-6);
  EXPECT_NEAR(d[2], -1.0, 1e-6);
}

TEST_F(SignalWaveformsTestGPU, SquareDuty) {
  std::vector<double> t_data = {0.0, M_PI / 2.0 - 1e-10, M_PI / 2.0 + 1e-10};
  Array t_cpu = to_array(t_data);
  Array t = t_cpu.to(GPUPlace(0));
  Array y = signal::square(t, 0.25);
  Array y_cpu = y.to(CPUPlace());
  const double *d = y_cpu.data<double>();
  EXPECT_NEAR(d[0], 1.0, 1e-6);
  EXPECT_NEAR(d[1], 1.0, 1e-6);
  EXPECT_NEAR(d[2], -1.0, 1e-6);
}

TEST_F(SignalWaveformsTestGPU, SquarePeriodicity) {
  std::vector<double> t_data = {1.0, 1.0 + 2.0 * M_PI, 1.0 + 4.0 * M_PI};
  Array t_cpu = to_array(t_data);
  Array t = t_cpu.to(GPUPlace(0));
  Array y = signal::square(t, 0.5);
  Array y_cpu = y.to(CPUPlace());
  const double *d = y_cpu.data<double>();
  EXPECT_NEAR(d[0], d[1], 1e-6);
  EXPECT_NEAR(d[0], d[2], 1e-6);
}

// ========== gausspulse ==========

TEST_F(SignalWaveformsTestGPU, GaussPulsePeak) {
  std::vector<double> t_data = {0.0};
  Array t_cpu = to_array(t_data);
  Array t = t_cpu.to(GPUPlace(0));
  Array y = signal::gausspulse(t, 1000.0);
  Array y_cpu = y.to(CPUPlace());
  EXPECT_NEAR(y_cpu.data<double>()[0], 1.0, 1e-6);
}

TEST_F(SignalWaveformsTestGPU, GaussPulseDecay) {
  std::vector<double> t_data = {0.0, 0.001, 0.01};
  Array t_cpu = to_array(t_data);
  Array t = t_cpu.to(GPUPlace(0));
  Array y = signal::gausspulse(t, 1000.0, 0.5);
  Array y_cpu = y.to(CPUPlace());
  const double *d = y_cpu.data<double>();
  EXPECT_NEAR(d[0], 1.0, 1e-6);
  EXPECT_GT(std::abs(d[0]), std::abs(d[2]));
}

TEST_F(SignalWaveformsTestGPU, GaussPulseFull) {
  std::vector<double> t_data = {0.0, 0.0005};
  Array t_cpu = to_array(t_data);
  Array t = t_cpu.to(GPUPlace(0));
  auto result = signal::gausspulse_full(t, 1000.0, 0.5);
  Array yi_cpu = result.yi.to(CPUPlace());
  Array yq_cpu = result.yq.to(CPUPlace());
  Array ye_cpu = result.ye.to(CPUPlace());
  EXPECT_NEAR(yi_cpu.data<double>()[0], 1.0, 1e-6);
  EXPECT_NEAR(yq_cpu.data<double>()[0], 0.0, 1e-6);
  EXPECT_NEAR(ye_cpu.data<double>()[0], 1.0, 1e-6);
}

// ========== chirp ==========

TEST_F(SignalWaveformsTestGPU, ChirpLinear) {
  std::vector<double> t_data = {0.0};
  Array t_cpu = to_array(t_data);
  Array t = t_cpu.to(GPUPlace(0));
  Array y = signal::chirp(t, 6.0, 1.0, 10.0, signal::ChirpMethod::Linear);
  Array y_cpu = y.to(CPUPlace());
  EXPECT_NEAR(y_cpu.data<double>()[0], 1.0, 1e-6);
}

TEST_F(SignalWaveformsTestGPU, ChirpQuadratic) {
  std::vector<double> t_data = {0.0};
  Array t_cpu = to_array(t_data);
  Array t = t_cpu.to(GPUPlace(0));
  Array y = signal::chirp(t, 6.0, 1.0, 10.0, signal::ChirpMethod::Quadratic,
                          0.0, true);
  Array y_cpu = y.to(CPUPlace());
  EXPECT_NEAR(y_cpu.data<double>()[0], 1.0, 1e-6);
}

TEST_F(SignalWaveformsTestGPU, ChirpLogarithmic) {
  std::vector<double> t_data = {0.0};
  Array t_cpu = to_array(t_data);
  Array t = t_cpu.to(GPUPlace(0));
  Array y = signal::chirp(t, 6.0, 1.0, 10.0, signal::ChirpMethod::Logarithmic);
  Array y_cpu = y.to(CPUPlace());
  EXPECT_NEAR(y_cpu.data<double>()[0], 1.0, 1e-6);
}

TEST_F(SignalWaveformsTestGPU, ChirpHyperbolic) {
  std::vector<double> t_data = {0.0};
  Array t_cpu = to_array(t_data);
  Array t = t_cpu.to(GPUPlace(0));
  Array y = signal::chirp(t, 6.0, 1.0, 10.0, signal::ChirpMethod::Hyperbolic);
  Array y_cpu = y.to(CPUPlace());
  EXPECT_NEAR(y_cpu.data<double>()[0], 1.0, 1e-6);
}

TEST_F(SignalWaveformsTestGPU, ChirpLinearEndFrequency) {
  std::vector<double> t_data = {1.0};
  Array t_cpu = to_array(t_data);
  Array t = t_cpu.to(GPUPlace(0));
  Array y = signal::chirp(t, 6.0, 1.0, 10.0, signal::ChirpMethod::Linear);
  Array y_cpu = y.to(CPUPlace());
  EXPECT_NEAR(y_cpu.data<double>()[0], 1.0, 1e-6);
}

TEST_F(SignalWaveformsTestGPU, ChirpWithPhaseOffset) {
  std::vector<double> t_data = {0.0};
  Array t_cpu = to_array(t_data);
  Array t = t_cpu.to(GPUPlace(0));
  Array y =
      signal::chirp(t, 6.0, 1.0, 10.0, signal::ChirpMethod::Linear, M_PI / 2.0);
  Array y_cpu = y.to(CPUPlace());
  EXPECT_NEAR(y_cpu.data<double>()[0], 0.0, 1e-6);
}

// ========== unit_impulse ==========

TEST_F(SignalWaveformsTestGPU, UnitImpulseCenter) {
  Array y = signal::unit_impulse(Shape({11}));
  Array y_cpu = y.to(CPUPlace());
  const double *d = y_cpu.data<double>();
  for (int64_t i = 0; i < 11; ++i) {
    if (i == 5) {
      EXPECT_NEAR(d[i], 1.0, 1e-10);
    } else {
      EXPECT_NEAR(d[i], 0.0, 1e-10);
    }
  }
}

TEST_F(SignalWaveformsTestGPU, UnitImpulseAtIdx) {
  Array y = signal::unit_impulse(Shape({7}), 3);
  Array y_cpu = y.to(CPUPlace());
  const double *d = y_cpu.data<double>();
  for (int64_t i = 0; i < 7; ++i) {
    if (i == 3) {
      EXPECT_NEAR(d[i], 1.0, 1e-10);
    } else {
      EXPECT_NEAR(d[i], 0.0, 1e-10);
    }
  }
}

TEST_F(SignalWaveformsTestGPU, UnitImpulseFloat32) {
  Array y = signal::unit_impulse(Shape({5}), 2, DType::F32);
  EXPECT_EQ(y.dtype(), DType::F32);
  Array y_cpu = y.to(CPUPlace());
  const float *d = y_cpu.data<float>();
  for (int64_t i = 0; i < 5; ++i) {
    if (i == 2) {
      EXPECT_FLOAT_EQ(d[i], 1.0f);
    } else {
      EXPECT_FLOAT_EQ(d[i], 0.0f);
    }
  }
}

} // namespace
