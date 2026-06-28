// tests/cuda/test_signal_filtering.cpp
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

class SignalFilteringTestGPU : public ::testing::Test {
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

template <typename T>
void expect_float_equal(const Array &arr, const std::vector<T> &expected,
                        T tol = 1e-6) {
  ASSERT_EQ(arr.numel(), static_cast<int64_t>(expected.size()));
  const T *data = arr.data<T>();
  for (int64_t i = 0; i < arr.numel(); ++i) {
    EXPECT_NEAR(data[i], expected[i], tol) << "mismatch at index " << i;
  }
}

// ========== hilbert ==========

TEST_F(SignalFilteringTestGPU, HilbertBasic) {
  int64_t N = 64;
  std::vector<double> x_data(N);
  for (int64_t i = 0; i < N; ++i) {
    x_data[i] = std::cos(2.0 * M_PI * i / N);
  }
  Array x_cpu = to_array(x_data);
  Array x = x_cpu.to(GPUPlace(0));
  Array y = signal::hilbert(x, N);
  Array y_cpu = y.to(CPUPlace());
  ASSERT_EQ(y_cpu.dtype(), DType::C64);
  const std::complex<double> *data =
      reinterpret_cast<const std::complex<double> *>(y_cpu.data<char>());
  for (int64_t i = 0; i < N; ++i) {
    EXPECT_GT(std::abs(data[i]), 0.01);
  }
}

TEST_F(SignalFilteringTestGPU, HilbertDC) {
  std::vector<double> x_data = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  Array x_cpu = to_array(x_data);
  Array x = x_cpu.to(GPUPlace(0));
  Array y = signal::hilbert(x, 8);
  Array y_cpu = y.to(CPUPlace());
  const std::complex<double> *data =
      reinterpret_cast<const std::complex<double> *>(y_cpu.data<char>());
  for (int64_t i = 0; i < 8; ++i) {
    EXPECT_NEAR(data[i].real(), 1.0, 1e-10);
    EXPECT_NEAR(data[i].imag(), 0.0, 1e-10);
  }
}

// ========== detrend ==========

TEST_F(SignalFilteringTestGPU, DetrendConstant) {
  std::vector<double> x_data = {1.0, 2.0, 3.0, 4.0, 5.0};
  Array x_cpu = to_array(x_data);
  Array x = x_cpu.to(GPUPlace(0));
  Array y = signal::detrend(x, -1, "constant");
  Array y_cpu = y.to(CPUPlace());
  const double *d = y_cpu.data<double>();
  EXPECT_NEAR(d[0], -2.0, 1e-10);
  EXPECT_NEAR(d[4], 2.0, 1e-10);
}

TEST_F(SignalFilteringTestGPU, DetrendLinear) {
  std::vector<double> x_data = {0.0, 3.0, 6.0, 9.0, 12.0};
  Array x_cpu = to_array(x_data);
  Array x = x_cpu.to(GPUPlace(0));
  Array y = signal::detrend(x, -1, "linear");
  Array y_cpu = y.to(CPUPlace());
  const double *d = y_cpu.data<double>();
  for (int64_t i = 0; i < 5; ++i) {
    EXPECT_NEAR(d[i], 0.0, 1e-10);
  }
}

// ========== lfilter (FIR) ==========

TEST_F(SignalFilteringTestGPU, LfilterFIRBasic) {
  std::vector<double> b_data = {1.0, 1.0};
  std::vector<double> a_data = {1.0};
  std::vector<double> x_data = {1.0, 0.0, 0.0, 0.0, 0.0};
  Array b = to_array(b_data);
  Array a = to_array(a_data);
  Array x_cpu = to_array(x_data);
  Array x = x_cpu.to(GPUPlace(0));
  Array y = signal::lfilter(b, a, x);
  Array y_cpu = y.to(CPUPlace());
  EXPECT_EQ(y_cpu.numel(), 6);
  const double *d = y_cpu.data<double>();
  EXPECT_NEAR(d[0], 1.0, 1e-10);
  EXPECT_NEAR(d[1], 1.0, 1e-10);
}

// ========== lfilter (IIR) ==========

TEST_F(SignalFilteringTestGPU, LfilterIIRBasic) {
  std::vector<double> b_data = {1.0};
  std::vector<double> a_data = {1.0, -0.5};
  std::vector<double> x_data = {1.0, 0.0, 0.0, 0.0, 0.0};
  Array b = to_array(b_data);
  Array a = to_array(a_data);
  Array x_cpu = to_array(x_data);
  Array x = x_cpu.to(GPUPlace(0));
  Array y = signal::lfilter(b, a, x);
  Array y_cpu = y.to(CPUPlace());
  const double *d = y_cpu.data<double>();
  EXPECT_NEAR(d[0], 1.0, 1e-10);
  EXPECT_NEAR(d[1], 0.5, 1e-10);
  EXPECT_NEAR(d[2], 0.25, 1e-10);
  EXPECT_NEAR(d[3], 0.125, 1e-10);
  EXPECT_NEAR(d[4], 0.0625, 1e-10);
}

// ========== lfilter_zi ==========

TEST_F(SignalFilteringTestGPU, LfilterZiIIR) {
  std::vector<double> b_data = {1.0};
  std::vector<double> a_data = {1.0, -0.5};
  Array b = to_array(b_data);
  Array a = to_array(a_data);
  Array zi = signal::lfilter_zi(b, a);
  EXPECT_EQ(zi.numel(), 1);
  EXPECT_NEAR(zi.data<double>()[0], 1.0, 1e-10);
}

// ========== filtfilt ==========

TEST_F(SignalFilteringTestGPU, FiltfiltFIR) {
  std::vector<double> b_data = {1.0, 1.0};
  std::vector<double> a_data = {1.0};
  std::vector<double> x_data = {1.0, 2.0, 3.0, 4.0, 5.0};
  Array b = to_array(b_data);
  Array a = to_array(a_data);
  Array x_cpu = to_array(x_data);
  Array x = x_cpu.to(GPUPlace(0));
  Array y = signal::filtfilt(b, a, x);
  EXPECT_EQ(y.shape().dim(0), x.shape().dim(0));
}

// ========== freq_shift ==========

TEST_F(SignalFilteringTestGPU, FreqShiftBasic) {
  int64_t N = 64;
  std::vector<double> x_data(N, 1.0);
  Array x_cpu = to_array(x_data);
  Array x = x_cpu.to(GPUPlace(0));
  Array y = signal::freq_shift(x, 0.0, 1.0);
  Array y_cpu = y.to(CPUPlace());
  ASSERT_EQ(y_cpu.dtype(), DType::C64);
  const std::complex<double> *data =
      reinterpret_cast<const std::complex<double> *>(y_cpu.data<char>());
  for (int64_t i = 0; i < N; ++i) {
    EXPECT_NEAR(std::abs(data[i]), 1.0, 1e-10);
  }
}

TEST_F(SignalFilteringTestGPU, FreqShiftPositive) {
  int64_t N = 16;
  std::vector<double> x_data(N, 1.0);
  Array x_cpu = to_array(x_data);
  Array x = x_cpu.to(GPUPlace(0));
  Array y = signal::freq_shift(x, 0.25, 1.0);
  Array y_cpu = y.to(CPUPlace());
  const std::complex<double> *data =
      reinterpret_cast<const std::complex<double> *>(y_cpu.data<char>());
  for (int64_t i = 0; i < N; ++i) {
    double expected_real = std::cos(2.0 * M_PI * 0.25 * i);
    double expected_imag = std::sin(2.0 * M_PI * 0.25 * i);
    EXPECT_NEAR(data[i].real(), expected_real, 1e-10);
    EXPECT_NEAR(data[i].imag(), expected_imag, 1e-10);
  }
}

// ========== decimate ==========

TEST_F(SignalFilteringTestGPU, DecimateBasic) {
  int64_t N = 100;
  std::vector<double> x_data(N);
  for (int64_t i = 0; i < N; ++i) {
    x_data[i] = std::sin(2.0 * M_PI * i / N);
  }
  Array x_cpu = to_array(x_data);
  Array x = x_cpu.to(GPUPlace(0));
  Array y = signal::decimate(x, 2, -1, false);
  EXPECT_EQ(y.numel(), 50);
}

// ========== resample ==========

TEST_F(SignalFilteringTestGPU, ResampleIdentity) {
  std::vector<double> x_data = {1.0, 2.0, 3.0, 4.0, 5.0};
  Array x_cpu = to_array(x_data);
  Array x = x_cpu.to(GPUPlace(0));
  Array y = signal::resample(x, 5);
  Array y_cpu = y.to(CPUPlace());
  EXPECT_EQ(y_cpu.numel(), 5);
  const double *d = y_cpu.data<double>();
  for (int64_t i = 0; i < 5; ++i) {
    EXPECT_NEAR(d[i], x_data[i], 1e-6);
  }
}

TEST_F(SignalFilteringTestGPU, ResampleUpsample) {
  std::vector<double> x_data = {1.0, 2.0, 3.0, 4.0};
  Array x_cpu = to_array(x_data);
  Array x = x_cpu.to(GPUPlace(0));
  Array y = signal::resample(x, 8);
  EXPECT_EQ(y.numel(), 8);
}

// ========== resample_poly ==========

TEST_F(SignalFilteringTestGPU, ResamplePolyIdentity) {
  std::vector<double> x_data = {1.0, 2.0, 3.0, 4.0, 5.0};
  Array x_cpu = to_array(x_data);
  Array x = x_cpu.to(GPUPlace(0));
  Array y = signal::resample_poly(x, 1, 1);
  Array y_cpu = y.to(CPUPlace());
  EXPECT_EQ(y_cpu.numel(), 5);
}

// ========== firfilter ==========

TEST_F(SignalFilteringTestGPU, FirfilterBasic) {
  std::vector<double> b_data = {1.0, 2.0, 1.0};
  std::vector<double> x_data = {1.0, 0.0, 0.0, 0.0, 0.0};
  Array b = to_array(b_data);
  Array x_cpu = to_array(x_data);
  Array x = x_cpu.to(GPUPlace(0));
  Array y = signal::firfilter(b, x);
  Array y_cpu = y.to(CPUPlace());
  EXPECT_EQ(y_cpu.numel(), 7);
  const double *d = y_cpu.data<double>();
  EXPECT_NEAR(d[0], 1.0, 1e-10);
  EXPECT_NEAR(d[1], 2.0, 1e-10);
  EXPECT_NEAR(d[2], 1.0, 1e-10);
}

} // namespace
