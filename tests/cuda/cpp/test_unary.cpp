// tests/cuda/test_unary.cpp
#include "insight/insight.h"
#include "insight/ops/elementwise.h"
#include <cmath>
#include <complex>
#include <gtest/gtest.h>

using namespace ins;

namespace {

class UnaryTestGPU : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    ins::init();
    try {
      set_device(GPUPlace(0));
    } catch (...) {
      GTEST_SKIP() << "GPU not available";
    }
  }
};

template <typename T> void fill_sequential_gpu(Array &gpu_arr) {
  Array cpu_arr(gpu_arr.shape(), gpu_arr.dtype(), CPUPlace());
  T *data = cpu_arr.data<T>();
  int64_t n = cpu_arr.numel();
  for (int64_t i = 0; i < n; ++i) {
    data[i] = static_cast<T>(i);
  }
  gpu_arr = cpu_arr.to(GPUPlace(0));
}

template <typename T>
void fill_float_range_gpu(Array &gpu_arr, T start, T step) {
  Array cpu_arr(gpu_arr.shape(), gpu_arr.dtype(), CPUPlace());
  T *data = cpu_arr.data<T>();
  int64_t n = cpu_arr.numel();
  for (int64_t i = 0; i < n; ++i) {
    data[i] = start + static_cast<T>(i) * step;
  }
  gpu_arr = cpu_arr.to(GPUPlace(0));
}

template <typename T>
void expect_float_equal_gpu(const Array &gpu_arr,
                            const std::vector<T> &expected, T tol = 1e-5) {
  Array cpu_arr = gpu_arr.to(CPUPlace());
  ASSERT_EQ(cpu_arr.numel(), static_cast<int64_t>(expected.size()));
  const T *data = cpu_arr.data<T>();
  for (int64_t i = 0; i < cpu_arr.numel(); ++i) {
    EXPECT_NEAR(data[i], expected[i], tol);
  }
}

void expect_bool_equal_gpu(const Array &gpu_arr,
                           const std::vector<bool> &expected) {
  Array cpu_arr = gpu_arr.to(CPUPlace());
  ASSERT_EQ(cpu_arr.numel(), static_cast<int64_t>(expected.size()));
  const bool *data = cpu_arr.data<bool>();
  for (int64_t i = 0; i < cpu_arr.numel(); ++i) {
    EXPECT_EQ(data[i], expected[i]);
  }
}

} // namespace

TEST_F(UnaryTestGPU, Abs) {
  Array a_cpu({2, 3}, DType::F32, CPUPlace());
  float *a_data = a_cpu.data<float>();
  for (int64_t i = 0; i < 6; ++i)
    a_data[i] = static_cast<float>(i - 3);
  Array a = a_cpu.to(GPUPlace(0));
  Array c = abs(a);
  expect_float_equal_gpu<float>(c, {3.0f, 2.0f, 1.0f, 0.0f, 1.0f, 2.0f});
}

TEST_F(UnaryTestGPU, Negative) {
  Array a({2, 3}, DType::F32, GPUPlace(0));
  fill_sequential_gpu<float>(a);
  Array c = negative(a);
  expect_float_equal_gpu<float>(c, {0.0f, -1.0f, -2.0f, -3.0f, -4.0f, -5.0f});
}

TEST_F(UnaryTestGPU, Square) {
  Array a({2, 3}, DType::F32, GPUPlace(0));
  fill_sequential_gpu<float>(a);
  Array c = square(a);
  expect_float_equal_gpu<float>(c, {0.0f, 1.0f, 4.0f, 9.0f, 16.0f, 25.0f});
}

TEST_F(UnaryTestGPU, Exp) {
  Array a_cpu({6}, DType::F32, CPUPlace());
  float *a_data = a_cpu.data<float>();
  for (int64_t i = 0; i < 6; ++i)
    a_data[i] = i * 0.5f;
  Array a = a_cpu.to(GPUPlace(0));
  Array c = exp(a);
  Array cpu_c = c.to(CPUPlace());
  const float *data = cpu_c.data<float>();
  for (int64_t i = 0; i < 6; ++i)
    EXPECT_NEAR(data[i], std::exp(i * 0.5f), 1e-5);
}

TEST_F(UnaryTestGPU, Log) {
  Array a({2, 3}, DType::F32, GPUPlace(0));
  fill_float_range_gpu<float>(a, 1.0f, 1.0f);
  Array c = log(a);
  Array cpu_c = c.to(CPUPlace());
  const float *data = cpu_c.data<float>();
  for (int64_t i = 0; i < 6; ++i)
    EXPECT_NEAR(data[i], std::log(1.0f + i), 1e-5);
}

TEST_F(UnaryTestGPU, Sin) {
  Array a({2, 3}, DType::F32, GPUPlace(0));
  fill_float_range_gpu<float>(a, 0.0f, 0.5f);
  Array c = sin(a);
  Array cpu_c = c.to(CPUPlace());
  const float *data = cpu_c.data<float>();
  for (int64_t i = 0; i < 6; ++i)
    EXPECT_NEAR(data[i], std::sin(i * 0.5f), 1e-5);
}

TEST_F(UnaryTestGPU, Cos) {
  Array a({2, 3}, DType::F32, GPUPlace(0));
  fill_float_range_gpu<float>(a, 0.0f, 0.5f);
  Array c = cos(a);
  Array cpu_c = c.to(CPUPlace());
  const float *data = cpu_c.data<float>();
  for (int64_t i = 0; i < 6; ++i)
    EXPECT_NEAR(data[i], std::cos(i * 0.5f), 1e-5);
}

TEST_F(UnaryTestGPU, Tan) {
  Array a({2, 3}, DType::F32, GPUPlace(0));
  fill_float_range_gpu<float>(a, 0.1f, 0.2f);
  Array c = tan(a);
  Array cpu_c = c.to(CPUPlace());
  const float *data = cpu_c.data<float>();
  for (int64_t i = 0; i < 6; ++i)
    EXPECT_NEAR(data[i], std::tan(0.1f + i * 0.2f), 1e-5);
}

TEST_F(UnaryTestGPU, Tanh) {
  Array a({2, 3}, DType::F32, GPUPlace(0));
  fill_float_range_gpu<float>(a, -2.0f, 1.0f);
  Array c = tanh(a);
  Array cpu_c = c.to(CPUPlace());
  const float *data = cpu_c.data<float>();
  for (int64_t i = 0; i < 6; ++i)
    EXPECT_NEAR(data[i], std::tanh(-2.0f + i), 1e-5);
}

TEST_F(UnaryTestGPU, Floor) {
  Array a_cpu({2, 3}, DType::F32, CPUPlace());
  float *a_data = a_cpu.data<float>();
  a_data[0] = 1.2f;
  a_data[1] = 1.8f;
  a_data[2] = -1.2f;
  a_data[3] = -1.8f;
  a_data[4] = 2.0f;
  a_data[5] = -2.0f;
  Array a = a_cpu.to(GPUPlace(0));
  Array c = floor(a);
  expect_float_equal_gpu<float>(c, {1.0f, 1.0f, -2.0f, -2.0f, 2.0f, -2.0f});
}

TEST_F(UnaryTestGPU, Ceil) {
  Array a_cpu({2, 3}, DType::F32, CPUPlace());
  float *a_data = a_cpu.data<float>();
  a_data[0] = 1.2f;
  a_data[1] = 1.8f;
  a_data[2] = -1.2f;
  a_data[3] = -1.8f;
  a_data[4] = 2.0f;
  a_data[5] = -2.0f;
  Array a = a_cpu.to(GPUPlace(0));
  Array c = ceil(a);
  expect_float_equal_gpu<float>(c, {2.0f, 2.0f, -1.0f, -1.0f, 2.0f, -2.0f});
}

TEST_F(UnaryTestGPU, Trunc) {
  Array a_cpu({2, 3}, DType::F32, CPUPlace());
  float *a_data = a_cpu.data<float>();
  a_data[0] = 1.2f;
  a_data[1] = 1.8f;
  a_data[2] = -1.2f;
  a_data[3] = -1.8f;
  a_data[4] = 2.0f;
  a_data[5] = -2.0f;
  Array a = a_cpu.to(GPUPlace(0));
  Array c = trunc(a);
  expect_float_equal_gpu<float>(c, {1.0f, 1.0f, -1.0f, -1.0f, 2.0f, -2.0f});
}

TEST_F(UnaryTestGPU, Rint) {
  Array a_cpu({2, 3}, DType::F32, CPUPlace());
  float *a_data = a_cpu.data<float>();
  a_data[0] = 1.2f;
  a_data[1] = 1.8f;
  a_data[2] = -1.2f;
  a_data[3] = -1.8f;
  a_data[4] = 2.0f;
  a_data[5] = -2.0f;
  Array a = a_cpu.to(GPUPlace(0));
  Array c = rint(a);
  Array cpu_c = c.to(CPUPlace());
  const float *data = cpu_c.data<float>();
  EXPECT_FLOAT_EQ(data[0], 1.0f);
  EXPECT_FLOAT_EQ(data[1], 2.0f);
  EXPECT_FLOAT_EQ(data[2], -1.0f);
  EXPECT_FLOAT_EQ(data[3], -2.0f);
  EXPECT_FLOAT_EQ(data[4], 2.0f);
  EXPECT_FLOAT_EQ(data[5], -2.0f);
}

TEST_F(UnaryTestGPU, Sign) {
  Array a_cpu({2, 3}, DType::F32, CPUPlace());
  float *a_data = a_cpu.data<float>();
  a_data[0] = -5.0f;
  a_data[1] = 0.0f;
  a_data[2] = 3.0f;
  a_data[3] = -0.0f;
  a_data[4] = 0.5f;
  a_data[5] = -0.5f;
  Array a = a_cpu.to(GPUPlace(0));
  Array c = sign(a);
  expect_float_equal_gpu<float>(c, {-1.0f, 0.0f, 1.0f, 0.0f, 1.0f, -1.0f});
}

TEST_F(UnaryTestGPU, Conj) {
  Array a_cpu({2, 3}, DType::C32, CPUPlace());
  std::complex<float> *a_data = a_cpu.data<std::complex<float>>();
  for (int64_t i = 0; i < 6; ++i)
    a_data[i] =
        std::complex<float>(static_cast<float>(i), static_cast<float>(i + 1));
  Array a = a_cpu.to(GPUPlace(0));
  Array c = conj(a);
  Array cpu_c = c.to(CPUPlace());
  const std::complex<float> *data = cpu_c.data<std::complex<float>>();
  for (int64_t i = 0; i < 6; ++i) {
    EXPECT_FLOAT_EQ(data[i].real(), static_cast<float>(i));
    EXPECT_FLOAT_EQ(data[i].imag(), -static_cast<float>(i + 1));
  }
}

TEST_F(UnaryTestGPU, ConjReal) {
  Array a({2, 3}, DType::F32, GPUPlace(0));
  fill_sequential_gpu<float>(a);
  Array c = conj(a);
  Array cpu_c = c.to(CPUPlace());
  const float *data = cpu_c.data<float>();
  for (int64_t i = 0; i < 6; ++i)
    EXPECT_FLOAT_EQ(data[i], static_cast<float>(i));
}

TEST_F(UnaryTestGPU, Deg2Rad) {
  Array a_cpu({2, 3}, DType::F32, CPUPlace());
  float *a_data = a_cpu.data<float>();
  for (int64_t i = 0; i < 6; ++i)
    a_data[i] = static_cast<float>(i * 45);
  Array a = a_cpu.to(GPUPlace(0));
  Array c = deg2rad(a);
  Array cpu_c = c.to(CPUPlace());
  const float *data = cpu_c.data<float>();
  for (int64_t i = 0; i < 6; ++i)
    EXPECT_NEAR(data[i], i * 45 * 3.1415926535f / 180.0f, 1e-5);
}

TEST_F(UnaryTestGPU, Rad2Deg) {
  Array a({2, 3}, DType::F32, GPUPlace(0));
  fill_float_range_gpu<float>(a, 0.0f, 0.5f);
  Array c = rad2deg(a);
  Array cpu_c = c.to(CPUPlace());
  const float *data = cpu_c.data<float>();
  for (int64_t i = 0; i < 6; ++i)
    EXPECT_NEAR(data[i], i * 0.5f * 180.0f / 3.1415926535f, 2e-5);
}

TEST_F(UnaryTestGPU, LogicalNot) {
  Array a_cpu({2, 3}, DType::F32, CPUPlace());
  float *a_data = a_cpu.data<float>();
  a_data[0] = 0.0f;
  a_data[1] = 1.0f;
  a_data[2] = -1.0f;
  a_data[3] = 0.0f;
  a_data[4] = 0.5f;
  a_data[5] = 0.0f;
  Array a = a_cpu.to(GPUPlace(0));
  Array c = logical_not(a);
  std::vector<bool> expected = {true, false, false, true, false, true};
  expect_bool_equal_gpu(c, expected);
}

TEST_F(UnaryTestGPU, BitwiseNot) {
  Array a_cpu({2, 3}, DType::I32, CPUPlace());
  int32_t *a_data = a_cpu.data<int32_t>();
  for (int64_t i = 0; i < 6; ++i)
    a_data[i] = static_cast<int32_t>(i);
  Array a = a_cpu.to(GPUPlace(0));
  Array c = bitwise_not(a);
  Array cpu_c = c.to(CPUPlace());
  const int32_t *data = cpu_c.data<int32_t>();
  for (int64_t i = 0; i < 6; ++i)
    EXPECT_EQ(data[i], ~static_cast<int32_t>(i));
}

TEST_F(UnaryTestGPU, ViewAbs) {
  Array a({3, 4}, DType::F32, GPUPlace(0));
  fill_float_range_gpu<float>(a, -5.0f, 1.0f);
  Array view = a.slice(0, 0, 3, 2);
  EXPECT_FALSE(view.is_contiguous());
  Array c = abs(view);
  Array cpu_c = c.to(CPUPlace());
  const float *data = cpu_c.data<float>();
  EXPECT_FLOAT_EQ(data[0], 5.0f);
  EXPECT_FLOAT_EQ(data[1], 4.0f);
  EXPECT_FLOAT_EQ(data[2], 3.0f);
  EXPECT_FLOAT_EQ(data[3], 2.0f);
  EXPECT_FLOAT_EQ(data[4], 3.0f);
  EXPECT_FLOAT_EQ(data[5], 4.0f);
  EXPECT_FLOAT_EQ(data[6], 5.0f);
  EXPECT_FLOAT_EQ(data[7], 6.0f);
}

TEST_F(UnaryTestGPU, IsNan) {
  Array a_cpu({2, 4}, DType::F32, CPUPlace());
  float *a_data = a_cpu.data<float>();
  a_data[0] = 1.0f;
  a_data[1] = std::nanf("");
  a_data[2] = 2.0f;
  a_data[3] = std::nanf("");
  a_data[4] = 3.0f;
  a_data[5] = 4.0f;
  a_data[6] = std::nanf("");
  a_data[7] = 5.0f;
  Array a = a_cpu.to(GPUPlace(0));
  Array c = isnan(a);
  std::vector<bool> expected = {false, true,  false, true,
                                false, false, true,  false};
  expect_bool_equal_gpu(c, expected);
}

TEST_F(UnaryTestGPU, IsInf) {
  Array a_cpu({2, 4}, DType::F32, CPUPlace());
  float *a_data = a_cpu.data<float>();
  a_data[0] = 1.0f;
  a_data[1] = std::numeric_limits<float>::infinity();
  a_data[2] = 2.0f;
  a_data[3] = -std::numeric_limits<float>::infinity();
  a_data[4] = 3.0f;
  a_data[5] = std::numeric_limits<float>::infinity();
  a_data[6] = 4.0f;
  a_data[7] = 5.0f;
  Array a = a_cpu.to(GPUPlace(0));
  Array c = isinf(a);
  std::vector<bool> expected = {false, true, false, true,
                                false, true, false, false};
  expect_bool_equal_gpu(c, expected);
}

TEST_F(UnaryTestGPU, IsFinite) {
  Array a_cpu({2, 5}, DType::F32, CPUPlace());
  float *a_data = a_cpu.data<float>();
  a_data[0] = 1.0f;
  a_data[1] = std::nanf("");
  a_data[2] = std::numeric_limits<float>::infinity();
  a_data[3] = -std::numeric_limits<float>::infinity();
  a_data[4] = 5.0f;
  a_data[5] = 6.0f;
  a_data[6] = std::nanf("");
  a_data[7] = 7.0f;
  a_data[8] = std::numeric_limits<float>::infinity();
  a_data[9] = 8.0f;
  Array a = a_cpu.to(GPUPlace(0));
  Array c = isfinite(a);
  std::vector<bool> expected = {true, false, false, false, true,
                                true, false, true,  false, true};
  expect_bool_equal_gpu(c, expected);
}

TEST_F(UnaryTestGPU, IsNanComplex) {
  Array a_cpu({2, 3}, DType::C32, CPUPlace());
  std::complex<float> *a_data = a_cpu.data<std::complex<float>>();
  a_data[0] = std::complex<float>(1.0f, 2.0f);
  a_data[1] = std::complex<float>(std::nanf(""), 3.0f);
  a_data[2] = std::complex<float>(4.0f, std::nanf(""));
  a_data[3] = std::complex<float>(std::nanf(""), std::nanf(""));
  a_data[4] = std::complex<float>(5.0f, 6.0f);
  a_data[5] = std::complex<float>(7.0f, 8.0f);
  Array a = a_cpu.to(GPUPlace(0));
  Array c = isnan(a);
  std::vector<bool> expected = {false, true, true, true, false, false};
  expect_bool_equal_gpu(c, expected);
}

TEST_F(UnaryTestGPU, IsInfComplex) {
  Array a_cpu({2, 3}, DType::C32, CPUPlace());
  std::complex<float> *a_data = a_cpu.data<std::complex<float>>();
  a_data[0] = std::complex<float>(1.0f, 2.0f);
  a_data[1] = std::complex<float>(std::numeric_limits<float>::infinity(), 3.0f);
  a_data[2] = std::complex<float>(4.0f, std::numeric_limits<float>::infinity());
  a_data[3] = std::complex<float>(std::numeric_limits<float>::infinity(),
                                  std::numeric_limits<float>::infinity());
  a_data[4] = std::complex<float>(5.0f, 6.0f);
  a_data[5] = std::complex<float>(7.0f, 8.0f);
  Array a = a_cpu.to(GPUPlace(0));
  Array c = isinf(a);
  std::vector<bool> expected = {false, true, true, true, false, false};
  expect_bool_equal_gpu(c, expected);
}

TEST_F(UnaryTestGPU, IsFiniteComplex) {
  Array a_cpu({2, 4}, DType::C32, CPUPlace());
  std::complex<float> *a_data = a_cpu.data<std::complex<float>>();
  a_data[0] = std::complex<float>(1.0f, 2.0f);
  a_data[1] = std::complex<float>(std::nanf(""), 3.0f);
  a_data[2] = std::complex<float>(4.0f, std::numeric_limits<float>::infinity());
  a_data[3] = std::complex<float>(5.0f, 6.0f);
  a_data[4] = std::complex<float>(std::numeric_limits<float>::infinity(), 7.0f);
  a_data[5] = std::complex<float>(8.0f, 9.0f);
  a_data[6] = std::complex<float>(10.0f, std::nanf(""));
  a_data[7] = std::complex<float>(11.0f, 12.0f);
  Array a = a_cpu.to(GPUPlace(0));
  Array c = isfinite(a);
  std::vector<bool> expected = {true,  false, false, true,
                                false, true,  false, true};
  expect_bool_equal_gpu(c, expected);
}

// ============================================================================
// Angle tests
// ============================================================================

TEST_F(UnaryTestGPU, AngleReal) {
  Array a_cpu({5}, DType::F32, CPUPlace());
  float *a_data = a_cpu.data<float>();
  a_data[0] = -2.0f;
  a_data[1] = -1.0f;
  a_data[2] = 0.0f;
  a_data[3] = 1.0f;
  a_data[4] = 2.0f;
  Array a = a_cpu.to(GPUPlace(0));
  Array c = angle(a);
  ASSERT_EQ(c.dtype(), DType::F32);
  expect_float_equal_gpu<float>(
      c, {static_cast<float>(M_PI), static_cast<float>(M_PI), 0.0f, 0.0f, 0.0f},
      1e-5f);
}

TEST_F(UnaryTestGPU, AngleComplex) {
  Array a_cpu({4}, DType::C32, CPUPlace());
  std::complex<float> *a_data = a_cpu.data<std::complex<float>>();
  a_data[0] = std::complex<float>(1.0f, 0.0f);  // angle = 0
  a_data[1] = std::complex<float>(0.0f, 1.0f);  // angle = pi/2
  a_data[2] = std::complex<float>(-1.0f, 0.0f); // angle = pi
  a_data[3] = std::complex<float>(0.0f, -1.0f); // angle = -pi/2
  Array a = a_cpu.to(GPUPlace(0));
  Array c = angle(a);
  ASSERT_EQ(c.dtype(), DType::F32);
  expect_float_equal_gpu<float>(c,
                                {0.0f, static_cast<float>(M_PI / 2.0),
                                 static_cast<float>(M_PI),
                                 static_cast<float>(-M_PI / 2.0)},
                                1e-5f);
}
