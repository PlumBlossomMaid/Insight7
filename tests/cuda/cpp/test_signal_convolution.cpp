// tests/cuda/test_signal_convolution.cpp
#include "insight/insight.h"
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

using namespace ins;

namespace {

class SignalConvolutionTestGPU : public ::testing::Test {
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

template <typename T>
void expect_float_equal(const Array &arr, const std::vector<T> &expected,
                        T tol = 1e-6) {
  ASSERT_EQ(arr.numel(), static_cast<int64_t>(expected.size()));
  const T *data = arr.data<T>();
  for (int64_t i = 0; i < arr.numel(); ++i) {
    EXPECT_NEAR(data[i], expected[i], tol) << "mismatch at index " << i;
  }
}

// ========== fftconvolve ==========

TEST_F(SignalConvolutionTestGPU, FftconvolveBasic) {
  std::vector<double> a = {1, 2, 3};
  std::vector<double> b = {1, 1};
  Array A = to_array(a);
  Array B = to_array(b);
  Array A_gpu = A.to(GPUPlace(0));
  Array B_gpu = B.to(GPUPlace(0));
  Array c_gpu = signal::fftconvolve(A_gpu, B_gpu, "full");
  Array c_cpu = c_gpu.to(CPUPlace());
  expect_float_equal(c_cpu, std::vector<double>{1, 3, 5, 3}, 1e-10);
}

TEST_F(SignalConvolutionTestGPU, FftconvolveSame) {
  std::vector<double> a = {1, 2, 3, 4, 5};
  std::vector<double> b = {1, 1, 1};
  Array A = to_array(a);
  Array B = to_array(b);
  Array c = signal::fftconvolve(A.to(GPUPlace(0)), B.to(GPUPlace(0)), "same");
  Array c_cpu = c.to(CPUPlace());
  expect_float_equal(c_cpu, std::vector<double>{3, 6, 9, 12, 9}, 1e-10);
}

TEST_F(SignalConvolutionTestGPU, FftconvolveValid) {
  std::vector<double> a = {1, 2, 3, 4, 5};
  std::vector<double> b = {1, 1, 1};
  Array A = to_array(a);
  Array B = to_array(b);
  Array c = signal::fftconvolve(A.to(GPUPlace(0)), B.to(GPUPlace(0)), "valid");
  Array c_cpu = c.to(CPUPlace());
  expect_float_equal(c_cpu, std::vector<double>{6, 9, 12}, 1e-10);
}

TEST_F(SignalConvolutionTestGPU, FftconvolveCommutative) {
  std::vector<double> a = {1, 2, 3};
  std::vector<double> b = {4, 5};
  Array A = to_array(a);
  Array B = to_array(b);
  Array c1 = signal::fftconvolve(A.to(GPUPlace(0)), B.to(GPUPlace(0)), "full");
  Array c2 = signal::fftconvolve(B.to(GPUPlace(0)), A.to(GPUPlace(0)), "full");
  Array c1_cpu = c1.to(CPUPlace());
  Array c2_cpu = c2.to(CPUPlace());
  const double *d1 = c1_cpu.data<double>();
  const double *d2 = c2_cpu.data<double>();
  ASSERT_EQ(c1_cpu.numel(), c2_cpu.numel());
  for (int64_t i = 0; i < c1_cpu.numel(); ++i) {
    EXPECT_NEAR(d1[i], d2[i], 1e-10);
  }
}

TEST_F(SignalConvolutionTestGPU, FftconvolveImpulse) {
  std::vector<double> a = {1, 2, 3, 4, 5};
  std::vector<double> b = {1};
  Array A = to_array(a);
  Array B = to_array(b);
  Array c = signal::fftconvolve(A.to(GPUPlace(0)), B.to(GPUPlace(0)), "full");
  Array c_cpu = c.to(CPUPlace());
  expect_float_equal(c_cpu, std::vector<double>{1, 2, 3, 4, 5}, 1e-10);
}

TEST_F(SignalConvolutionTestGPU, Fftconvolve2D) {
  std::vector<double> a_data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<double> b_data = {1, 0, 0, 1};
  Array A = to_array(a_data, {3, 3});
  Array B = to_array(b_data, {2, 2});
  Array c = signal::fftconvolve(A.to(GPUPlace(0)), B.to(GPUPlace(0)), "full");
  EXPECT_EQ(c.shape().dim(0), 4);
  EXPECT_EQ(c.shape().dim(1), 4);
  Array c_cpu = c.to(CPUPlace());
  EXPECT_NEAR(c_cpu.data<double>()[0], 1.0, 1e-10);
  EXPECT_NEAR(c_cpu.data<double>()[5], 6.0, 1e-10);
}

// ========== correlate ==========

TEST_F(SignalConvolutionTestGPU, CorrelateBasic) {
  std::vector<double> a = {1, 2, 3};
  std::vector<double> b = {1, 1};
  Array A = to_array(a);
  Array B = to_array(b);
  Array c = signal::correlate(A.to(GPUPlace(0)), B.to(GPUPlace(0)), "full");
  Array c_cpu = c.to(CPUPlace());
  expect_float_equal(c_cpu, std::vector<double>{1, 3, 5, 3}, 1e-10);
}

TEST_F(SignalConvolutionTestGPU, CorrelateAutoCorrelation) {
  std::vector<double> a = {1, 0, -1};
  Array A = to_array(a);
  Array c = signal::correlate(A.to(GPUPlace(0)), A.to(GPUPlace(0)), "full");
  Array c_cpu = c.to(CPUPlace());
  const double *d = c_cpu.data<double>();
  ASSERT_EQ(c_cpu.numel(), 5);
  EXPECT_NEAR(d[0], -1.0, 1e-10);
  EXPECT_NEAR(d[2], 2.0, 1e-10);
  EXPECT_NEAR(d[4], -1.0, 1e-10);
}

// ========== convolve2d ==========

TEST_F(SignalConvolutionTestGPU, Convolve2dBasic) {
  std::vector<double> a_data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<double> b_data = {1, 0, 0, 1};
  Array A = to_array(a_data, {3, 3});
  Array B = to_array(b_data, {2, 2});
  Array c = signal::convolve2d(A.to(GPUPlace(0)), B.to(GPUPlace(0)), "full");
  EXPECT_EQ(c.shape().dim(0), 4);
  EXPECT_EQ(c.shape().dim(1), 4);
}

TEST_F(SignalConvolutionTestGPU, Convolve2dValid) {
  // Use simpler case matching CPU test
  std::vector<double> a_data = {1, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<double> b_data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  Array A = to_array(a_data, {3, 3});
  Array B = to_array(b_data, {3, 3});
  Array c = signal::convolve2d(A.to(GPUPlace(0)), B.to(GPUPlace(0)), "full");
  EXPECT_EQ(c.shape().dim(0), 5);
  EXPECT_EQ(c.shape().dim(1), 5);
  Array c_cpu = c.to(CPUPlace());
  EXPECT_NEAR(c_cpu.data<double>()[0], 1.0, 1e-10);

  std::vector<double> k_data = {1, 1, 1, 1};
  Array K = to_array(k_data, {2, 2});
  Array cv = signal::convolve2d(A.to(GPUPlace(0)), K.to(GPUPlace(0)), "valid");
  EXPECT_EQ(cv.shape().dim(0), 2);
  EXPECT_EQ(cv.shape().dim(1), 2);
  Array cv_cpu = cv.to(CPUPlace());
  EXPECT_NEAR(cv_cpu.data<double>()[0], 1.0, 1e-10);
  EXPECT_NEAR(cv_cpu.data<double>()[1], 0.0, 1e-10);
  EXPECT_NEAR(cv_cpu.data<double>()[2], 0.0, 1e-10);
  EXPECT_NEAR(cv_cpu.data<double>()[3], 0.0, 1e-10);
}

// ========== correlate2d ==========

TEST_F(SignalConvolutionTestGPU, Correlate2dBasic) {
  std::vector<double> a_data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<double> b_data = {1, 1, 1, 1};
  Array A = to_array(a_data, {3, 3});
  Array B = to_array(b_data, {2, 2});
  Array c = signal::correlate2d(A.to(GPUPlace(0)), B.to(GPUPlace(0)), "valid");
  EXPECT_EQ(c.shape().dim(0), 2);
  EXPECT_EQ(c.shape().dim(1), 2);
}

TEST_F(SignalConvolutionTestGPU, Correlate2dFull) {
  std::vector<double> a_data = {1, 2, 3, 4};
  std::vector<double> b_data = {1, 1, 1, 1};
  Array A = to_array(a_data, {2, 2});
  Array B = to_array(b_data, {2, 2});
  Array c = signal::correlate2d(A.to(GPUPlace(0)), B.to(GPUPlace(0)), "full");
  EXPECT_EQ(c.shape().dim(0), 3);
  EXPECT_EQ(c.shape().dim(1), 3);
}

// ========== choose_conv_method ==========

TEST_F(SignalConvolutionTestGPU, ChooseConvMethodSmall) {
  std::vector<double> a = {1, 2, 3};
  std::vector<double> b = {1, 1};
  Array A = to_array(a);
  Array B = to_array(b);
  std::string method = signal::choose_conv_method(A, B, "full");
  EXPECT_EQ(method, "direct");
}

TEST_F(SignalConvolutionTestGPU, ChooseConvMethodLarge) {
  std::vector<double> a(1000, 1.0);
  std::vector<double> b(1000, 1.0);
  Array A = to_array(a);
  Array B = to_array(b);
  std::string method = signal::choose_conv_method(A, B, "full");
  EXPECT_EQ(method, "fft");
}

// ========== correlation_lags ==========

TEST_F(SignalConvolutionTestGPU, CorrelationLagsFull) {
  Array lags_gpu = signal::correlation_lags(5, 3, "full");
  Array lags = lags_gpu.to(CPUPlace());
  ASSERT_EQ(lags.numel(), 7);
  const int64_t *d = lags.data<int64_t>();
  EXPECT_EQ(d[0], -2);
  EXPECT_EQ(d[1], -1);
  EXPECT_EQ(d[2], 0);
  EXPECT_EQ(d[3], 1);
  EXPECT_EQ(d[4], 2);
  EXPECT_EQ(d[5], 3);
  EXPECT_EQ(d[6], 4);
}

TEST_F(SignalConvolutionTestGPU, CorrelationLagsSame) {
  Array lags_gpu = signal::correlation_lags(5, 3, "same");
  Array lags = lags_gpu.to(CPUPlace());
  ASSERT_EQ(lags.numel(), 5);
  const int64_t *d = lags.data<int64_t>();
  EXPECT_EQ(d[0], -1);
  EXPECT_EQ(d[4], 3);
}

TEST_F(SignalConvolutionTestGPU, CorrelationLagsValid) {
  Array lags_gpu = signal::correlation_lags(5, 3, "valid");
  Array lags = lags_gpu.to(CPUPlace());
  ASSERT_EQ(lags.numel(), 3);
  const int64_t *d = lags.data<int64_t>();
  EXPECT_EQ(d[0], 0);
  EXPECT_EQ(d[2], 2);
}

} // namespace
