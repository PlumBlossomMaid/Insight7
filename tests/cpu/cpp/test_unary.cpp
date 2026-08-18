// tests/cpu/test_unary.cpp
#include "insight/insight.h"
#include "insight/ops/elementwise.h"
#include <cmath>
#include <complex>
#include <gtest/gtest.h>

using namespace ins;

namespace {

// ============================================================================
// Test Fixture for Unary CPU Tests
// ============================================================================

class UnaryTestCPU : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    ins::init({"cpu"});
    set_device(CPUPlace());
  }

  static void TearDownTestSuite() {
    // Optional: cleanup if needed
  }
};

// Helper: fill array with sequential values
template <typename T> void fill_sequential(Array &arr) {
  T *data = arr.data<T>();
  int64_t n = arr.numel();
  for (int64_t i = 0; i < n; ++i) {
    data[i] = static_cast<T>(i);
  }
}

template <typename T> void fill_float_range(Array &arr, T start, T step) {
  T *data = arr.data<T>();
  int64_t n = arr.numel();
  for (int64_t i = 0; i < n; ++i) {
    data[i] = start + static_cast<T>(i) * step;
  }
}

template <typename T>
void expect_equal(const Array &arr, const std::vector<T> &expected) {
  ASSERT_EQ(arr.numel(), static_cast<int64_t>(expected.size()));
  const T *data = arr.data<T>();
  for (int64_t i = 0; i < arr.numel(); ++i) {
    EXPECT_EQ(data[i], expected[i]);
  }
}

template <typename T>
void expect_float_equal(const Array &arr, const std::vector<T> &expected,
                        T tol = 1e-6) {
  ASSERT_EQ(arr.numel(), static_cast<int64_t>(expected.size()));
  const T *data = arr.data<T>();
  for (int64_t i = 0; i < arr.numel(); ++i) {
    EXPECT_NEAR(data[i], expected[i], tol);
  }
}

void expect_bool_equal(const Array &arr, const std::vector<bool> &expected) {
  ASSERT_EQ(arr.numel(), static_cast<int64_t>(expected.size()));
  const bool *data = arr.data<bool>();
  for (int64_t i = 0; i < arr.numel(); ++i) {
    EXPECT_EQ(data[i], expected[i]);
  }
}

} // namespace

// ============================================================================
// Basic math tests
// ============================================================================

TEST_F(UnaryTestCPU, Abs) {
  Array a({2, 3}, DType::I32);
  int32_t *a_data = a.data<int32_t>();
  for (int64_t i = 0; i < 6; ++i) {
    a_data[i] = static_cast<int32_t>(i - 3);
  }

  Array c = abs(a);

  std::vector<int32_t> expected = {3, 2, 1, 0, 1, 2};
  expect_equal<int32_t>(c, expected);
}

TEST_F(UnaryTestCPU, Negative) {
  Array a({2, 3}, DType::I32);
  fill_sequential<int32_t>(a);

  Array c = negative(a);

  std::vector<int32_t> expected = {0, -1, -2, -3, -4, -5};
  expect_equal<int32_t>(c, expected);
}

TEST_F(UnaryTestCPU, Square) {
  Array a({2, 3}, DType::I32);
  fill_sequential<int32_t>(a);

  Array c = square(a);

  std::vector<int32_t> expected = {0, 1, 4, 9, 16, 25};
  expect_equal<int32_t>(c, expected);
}

// ============================================================================
// Exponential and logarithmic tests
// ============================================================================

TEST_F(UnaryTestCPU, Exp) {
  Array a({2, 3}, DType::F32);
  fill_float_range<float>(a, 0.0f, 0.5f);

  Array c = exp(a);

  const float *data = c.data<float>();
  for (int64_t i = 0; i < 6; ++i) {
    EXPECT_NEAR(data[i], std::exp(i * 0.5f), 1e-6);
  }
}

TEST_F(UnaryTestCPU, Log) {
  Array a({2, 3}, DType::F32);
  fill_float_range<float>(a, 1.0f, 1.0f);

  Array c = log(a);

  const float *data = c.data<float>();
  for (int64_t i = 0; i < 6; ++i) {
    EXPECT_NEAR(data[i], std::log(1.0f + i), 1e-6);
  }
}

// ============================================================================
// Trigonometric tests
// ============================================================================

TEST_F(UnaryTestCPU, Sin) {
  Array a({2, 3}, DType::F32);
  fill_float_range<float>(a, 0.0f, 0.5f);

  Array c = sin(a);

  const float *data = c.data<float>();
  for (int64_t i = 0; i < 6; ++i) {
    EXPECT_NEAR(data[i], std::sin(i * 0.5f), 1e-6);
  }
}

TEST_F(UnaryTestCPU, Cos) {
  Array a({2, 3}, DType::F32);
  fill_float_range<float>(a, 0.0f, 0.5f);

  Array c = cos(a);

  const float *data = c.data<float>();
  for (int64_t i = 0; i < 6; ++i) {
    EXPECT_NEAR(data[i], std::cos(i * 0.5f), 1e-6);
  }
}

TEST_F(UnaryTestCPU, Tan) {
  Array a({2, 3}, DType::F32);
  fill_float_range<float>(a, 0.1f, 0.2f);

  Array c = tan(a);

  const float *data = c.data<float>();
  for (int64_t i = 0; i < 6; ++i) {
    EXPECT_NEAR(data[i], std::tan(0.1f + i * 0.2f), 1e-5);
  }
}

TEST_F(UnaryTestCPU, Tanh) {
  Array a({2, 3}, DType::F32);
  fill_float_range<float>(a, -2.0f, 1.0f);

  Array c = tanh(a);

  const float *data = c.data<float>();
  for (int64_t i = 0; i < 6; ++i) {
    EXPECT_NEAR(data[i], std::tanh(-2.0f + i), 1e-6);
  }
}

// ============================================================================
// Rounding tests
// ============================================================================

TEST_F(UnaryTestCPU, Floor) {
  Array a({2, 3}, DType::F32);
  float *a_data = a.data<float>();
  a_data[0] = 1.2f;
  a_data[1] = 1.8f;
  a_data[2] = -1.2f;
  a_data[3] = -1.8f;
  a_data[4] = 2.0f;
  a_data[5] = -2.0f;

  Array c = floor(a);

  std::vector<float> expected = {1.0f, 1.0f, -2.0f, -2.0f, 2.0f, -2.0f};
  expect_float_equal<float>(c, expected);
}

TEST_F(UnaryTestCPU, Ceil) {
  Array a({2, 3}, DType::F32);
  float *a_data = a.data<float>();
  a_data[0] = 1.2f;
  a_data[1] = 1.8f;
  a_data[2] = -1.2f;
  a_data[3] = -1.8f;
  a_data[4] = 2.0f;
  a_data[5] = -2.0f;

  Array c = ceil(a);

  std::vector<float> expected = {2.0f, 2.0f, -1.0f, -1.0f, 2.0f, -2.0f};
  expect_float_equal<float>(c, expected);
}

TEST_F(UnaryTestCPU, Trunc) {
  Array a({2, 3}, DType::F32);
  float *a_data = a.data<float>();
  a_data[0] = 1.2f;
  a_data[1] = 1.8f;
  a_data[2] = -1.2f;
  a_data[3] = -1.8f;
  a_data[4] = 2.0f;
  a_data[5] = -2.0f;

  Array c = trunc(a);

  std::vector<float> expected = {1.0f, 1.0f, -1.0f, -1.0f, 2.0f, -2.0f};
  expect_float_equal<float>(c, expected);
}

TEST_F(UnaryTestCPU, Rint) {
  Array a({2, 3}, DType::F32);
  float *a_data = a.data<float>();
  a_data[0] = 1.2f;
  a_data[1] = 1.8f;
  a_data[2] = -1.2f;
  a_data[3] = -1.8f;
  a_data[4] = 2.0f;
  a_data[5] = -2.0f;

  Array c = rint(a);

  const float *data = c.data<float>();
  EXPECT_FLOAT_EQ(data[0], 1.0f);
  EXPECT_FLOAT_EQ(data[1], 2.0f);
  EXPECT_FLOAT_EQ(data[2], -1.0f);
  EXPECT_FLOAT_EQ(data[3], -2.0f);
  EXPECT_FLOAT_EQ(data[4], 2.0f);
  EXPECT_FLOAT_EQ(data[5], -2.0f);
}

// ============================================================================
// Sign test
// ============================================================================

TEST_F(UnaryTestCPU, Sign) {
  Array a({2, 3}, DType::F32);
  float *a_data = a.data<float>();
  a_data[0] = -5.0f;
  a_data[1] = 0.0f;
  a_data[2] = 3.0f;
  a_data[3] = -0.0f;
  a_data[4] = 0.5f;
  a_data[5] = -0.5f;

  Array c = sign(a);

  std::vector<float> expected = {-1.0f, 0.0f, 1.0f, 0.0f, 1.0f, -1.0f};
  expect_float_equal<float>(c, expected);
}

// ============================================================================
// Complex conjugate test
// ============================================================================

TEST_F(UnaryTestCPU, Conj) {
  Array a({2, 3}, DType::C32);
  std::complex<float> *a_data = a.data<std::complex<float>>();
  for (int64_t i = 0; i < 6; ++i) {
    a_data[i] =
        std::complex<float>(static_cast<float>(i), static_cast<float>(i + 1));
  }

  Array c = conj(a);

  const std::complex<float> *data = c.data<std::complex<float>>();
  for (int64_t i = 0; i < 6; ++i) {
    EXPECT_FLOAT_EQ(data[i].real(), static_cast<float>(i));
    EXPECT_FLOAT_EQ(data[i].imag(), -static_cast<float>(i + 1));
  }
}

TEST_F(UnaryTestCPU, ConjReal) {
  Array a({2, 3}, DType::F32);
  fill_sequential<float>(a);

  Array c = conj(a);

  const float *data = c.data<float>();
  for (int64_t i = 0; i < 6; ++i) {
    EXPECT_FLOAT_EQ(data[i], static_cast<float>(i));
  }
}

// ============================================================================
// Degree/radian conversion tests
// ============================================================================

TEST_F(UnaryTestCPU, Deg2Rad) {
  Array a({2, 3}, DType::F32);
  float *a_data = a.data<float>();
  for (int64_t i = 0; i < 6; ++i) {
    a_data[i] = static_cast<float>(i * 45);
  }

  Array c = deg2rad(a);

  const float *data = c.data<float>();
  for (int64_t i = 0; i < 6; ++i) {
    EXPECT_NEAR(data[i], i * 45 * 3.1415926535f / 180.0f, 1e-6);
  }
}

TEST_F(UnaryTestCPU, Rad2Deg) {
  Array a({2, 3}, DType::F32);
  fill_float_range<float>(a, 0.0f, 0.5f);

  Array c = rad2deg(a);

  const float *data = c.data<float>();
  for (int64_t i = 0; i < 6; ++i) {
    EXPECT_NEAR(data[i], i * 0.5f * 180.0f / 3.1415926535f, 1e-6);
  }
}

// ============================================================================
// Logical not test
// ============================================================================

TEST_F(UnaryTestCPU, LogicalNot) {
  Array a({2, 3}, DType::F32);
  float *a_data = a.data<float>();
  a_data[0] = 0.0f;
  a_data[1] = 1.0f;
  a_data[2] = -1.0f;
  a_data[3] = 0.0f;
  a_data[4] = 0.5f;
  a_data[5] = 0.0f;

  Array c = logical_not(a);

  std::vector<bool> expected = {true, false, false, true, false, true};
  expect_bool_equal(c, expected);
}

// ============================================================================
// Bitwise not test
// ============================================================================

TEST_F(UnaryTestCPU, BitwiseNot) {
  Array a({2, 3}, DType::I32);
  int32_t *a_data = a.data<int32_t>();
  for (int64_t i = 0; i < 6; ++i) {
    a_data[i] = static_cast<int32_t>(i);
  }

  Array c = bitwise_not(a);

  const int32_t *data = c.data<int32_t>();
  for (int64_t i = 0; i < 6; ++i) {
    EXPECT_EQ(data[i], ~static_cast<int32_t>(i));
  }
}

// ============================================================================
// View (non-contiguous) test
// ============================================================================

TEST_F(UnaryTestCPU, ViewAbs) {
  Array a({3, 4}, DType::F32);
  fill_float_range<float>(a, -5.0f, 1.0f);

  Array view = a.slice(0, 0, 3, 2);
  EXPECT_FALSE(view.is_contiguous());

  Array c = abs(view);

  const float *data = c.data<float>();
  EXPECT_FLOAT_EQ(data[0], 5.0f);
  EXPECT_FLOAT_EQ(data[1], 4.0f);
  EXPECT_FLOAT_EQ(data[2], 3.0f);
  EXPECT_FLOAT_EQ(data[3], 2.0f);
  EXPECT_FLOAT_EQ(data[4], 3.0f);
  EXPECT_FLOAT_EQ(data[5], 4.0f);
  EXPECT_FLOAT_EQ(data[6], 5.0f);
  EXPECT_FLOAT_EQ(data[7], 6.0f);
}

TEST_F(UnaryTestCPU, ViewAngle) {
  Array a({4}, DType::F32);
  float *data = a.data<float>();
  data[0] = -2.0f;
  data[1] = 3.0f;
  data[2] = -4.0f;
  data[3] = 5.0f;

  Array view = a.slice(0, 0, 4, 2);
  Array c = angle(view);

  ASSERT_EQ(c.dtype(), DType::F32);
  expect_float_equal<float>(c, {static_cast<float>(M_PI),
                                static_cast<float>(M_PI)});
}

// ============================================================================
// IsNaN / IsInf / IsFinite tests
// ============================================================================

TEST_F(UnaryTestCPU, IsNan) {
  Array a({2, 4}, DType::F32);
  float *a_data = a.data<float>();
  a_data[0] = 1.0f;
  a_data[1] = std::nanf("");
  a_data[2] = 2.0f;
  a_data[3] = std::nanf("");
  a_data[4] = 3.0f;
  a_data[5] = 4.0f;
  a_data[6] = std::nanf("");
  a_data[7] = 5.0f;

  Array c = isnan(a);

  std::vector<bool> expected = {false, true,  false, true,
                                false, false, true,  false};
  expect_bool_equal(c, expected);

  Array b({2, 4}, DType::F64);
  double *b_data = b.data<double>();
  b_data[0] = 1.0;
  b_data[1] = std::nan("");
  b_data[2] = 2.0;
  b_data[3] = std::nan("");
  b_data[4] = 3.0;
  b_data[5] = 4.0;
  b_data[6] = std::nan("");
  b_data[7] = 5.0;

  Array d = isnan(b);
  expect_bool_equal(d, expected);

  Array e({2, 4}, DType::I32);
  int32_t *e_data = e.data<int32_t>();
  for (int64_t i = 0; i < 8; ++i) {
    e_data[i] = static_cast<int32_t>(i);
  }

  Array f = isnan(e);
  std::vector<bool> int_expected(8, false);
  expect_bool_equal(f, int_expected);
}

TEST_F(UnaryTestCPU, IsInf) {
  Array a({2, 4}, DType::F32);
  float *a_data = a.data<float>();
  a_data[0] = 1.0f;
  a_data[1] = std::numeric_limits<float>::infinity();
  a_data[2] = 2.0f;
  a_data[3] = -std::numeric_limits<float>::infinity();
  a_data[4] = 3.0f;
  a_data[5] = std::numeric_limits<float>::infinity();
  a_data[6] = 4.0f;
  a_data[7] = 5.0f;

  Array c = isinf(a);

  std::vector<bool> expected = {false, true, false, true,
                                false, true, false, false};
  expect_bool_equal(c, expected);

  Array b({2, 4}, DType::F64);
  double *b_data = b.data<double>();
  b_data[0] = 1.0;
  b_data[1] = std::numeric_limits<double>::infinity();
  b_data[2] = 2.0;
  b_data[3] = -std::numeric_limits<double>::infinity();
  b_data[4] = 3.0;
  b_data[5] = std::numeric_limits<double>::infinity();
  b_data[6] = 4.0;
  b_data[7] = 5.0;

  Array d = isinf(b);
  expect_bool_equal(d, expected);

  Array e({2, 4}, DType::I32);
  fill_sequential<int32_t>(e);

  Array f = isinf(e);
  std::vector<bool> int_expected(8, false);
  expect_bool_equal(f, int_expected);
}

TEST_F(UnaryTestCPU, IsFinite) {
  Array a({2, 5}, DType::F32);
  float *a_data = a.data<float>();
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

  Array c = isfinite(a);

  std::vector<bool> expected = {true, false, false, false, true,
                                true, false, true,  false, true};
  expect_bool_equal(c, expected);

  Array b({2, 5}, DType::F64);
  double *b_data = b.data<double>();
  b_data[0] = 1.0;
  b_data[1] = std::nan("");
  b_data[2] = std::numeric_limits<double>::infinity();
  b_data[3] = -std::numeric_limits<double>::infinity();
  b_data[4] = 5.0;
  b_data[5] = 6.0;
  b_data[6] = std::nan("");
  b_data[7] = 7.0;
  b_data[8] = std::numeric_limits<double>::infinity();
  b_data[9] = 8.0;

  Array d = isfinite(b);
  expect_bool_equal(d, expected);

  Array e({2, 5}, DType::I32);
  int32_t *e_data = e.data<int32_t>();
  for (int64_t i = 0; i < 10; ++i) {
    e_data[i] = static_cast<int32_t>(i - 5);
  }

  Array f = isfinite(e);
  std::vector<bool> int_expected(10, true);
  expect_bool_equal(f, int_expected);
}

// ============================================================================
// IsNan with complex numbers
// ============================================================================

TEST_F(UnaryTestCPU, IsNanComplex) {
  Array a({2, 3}, DType::C32);
  std::complex<float> *a_data = a.data<std::complex<float>>();
  a_data[0] = std::complex<float>(1.0f, 2.0f);
  a_data[1] = std::complex<float>(std::nanf(""), 3.0f);
  a_data[2] = std::complex<float>(4.0f, std::nanf(""));
  a_data[3] = std::complex<float>(std::nanf(""), std::nanf(""));
  a_data[4] = std::complex<float>(5.0f, 6.0f);
  a_data[5] = std::complex<float>(7.0f, 8.0f);

  Array c = isnan(a);

  std::vector<bool> expected = {false, true, true, true, false, false};
  expect_bool_equal(c, expected);

  Array b({2, 3}, DType::C64);
  std::complex<double> *b_data = b.data<std::complex<double>>();
  b_data[0] = std::complex<double>(1.0, 2.0);
  b_data[1] = std::complex<double>(std::nan(""), 3.0);
  b_data[2] = std::complex<double>(4.0, std::nan(""));
  b_data[3] = std::complex<double>(std::nan(""), std::nan(""));
  b_data[4] = std::complex<double>(5.0, 6.0);
  b_data[5] = std::complex<double>(7.0, 8.0);

  Array d = isnan(b);
  expect_bool_equal(d, expected);
}

TEST_F(UnaryTestCPU, IsInfComplex) {
  Array a({2, 3}, DType::C32);
  std::complex<float> *a_data = a.data<std::complex<float>>();
  a_data[0] = std::complex<float>(1.0f, 2.0f);
  a_data[1] = std::complex<float>(std::numeric_limits<float>::infinity(), 3.0f);
  a_data[2] = std::complex<float>(4.0f, std::numeric_limits<float>::infinity());
  a_data[3] = std::complex<float>(std::numeric_limits<float>::infinity(),
                                  std::numeric_limits<float>::infinity());
  a_data[4] = std::complex<float>(5.0f, 6.0f);
  a_data[5] = std::complex<float>(7.0f, 8.0f);

  Array c = isinf(a);

  std::vector<bool> expected = {false, true, true, true, false, false};
  expect_bool_equal(c, expected);
}

TEST_F(UnaryTestCPU, IsFiniteComplex) {
  Array a({2, 4}, DType::C32);
  std::complex<float> *a_data = a.data<std::complex<float>>();
  a_data[0] = std::complex<float>(1.0f, 2.0f);
  a_data[1] = std::complex<float>(std::nanf(""), 3.0f);
  a_data[2] = std::complex<float>(4.0f, std::numeric_limits<float>::infinity());
  a_data[3] = std::complex<float>(5.0f, 6.0f);
  a_data[4] = std::complex<float>(std::numeric_limits<float>::infinity(), 7.0f);
  a_data[5] = std::complex<float>(8.0f, 9.0f);
  a_data[6] = std::complex<float>(10.0f, std::nanf(""));
  a_data[7] = std::complex<float>(11.0f, 12.0f);

  Array c = isfinite(a);

  std::vector<bool> expected = {true,  false, false, true,
                                false, true,  false, true};
  expect_bool_equal(c, expected);
}

// ============================================================================
// Angle tests
// ============================================================================

TEST_F(UnaryTestCPU, AngleReal) {
  Array a({5}, DType::F32);
  float *a_data = a.data<float>();
  a_data[0] = -2.0f;
  a_data[1] = -1.0f;
  a_data[2] = 0.0f;
  a_data[3] = 1.0f;
  a_data[4] = 2.0f;

  Array c = angle(a);

  ASSERT_EQ(c.dtype(), DType::F32);
  std::vector<float> expected = {static_cast<float>(M_PI),
                                 static_cast<float>(M_PI), 0.0f, 0.0f, 0.0f};
  expect_float_equal<float>(c, expected, 1e-6f);
}

TEST_F(UnaryTestCPU, AngleComplex) {
  Array a({4}, DType::C32);
  std::complex<float> *a_data = a.data<std::complex<float>>();
  a_data[0] = std::complex<float>(1.0f, 0.0f);  // angle = 0
  a_data[1] = std::complex<float>(0.0f, 1.0f);  // angle = pi/2
  a_data[2] = std::complex<float>(-1.0f, 0.0f); // angle = pi
  a_data[3] = std::complex<float>(0.0f, -1.0f); // angle = -pi/2

  Array c = angle(a);

  ASSERT_EQ(c.dtype(), DType::F32);
  std::vector<float> expected = {0.0f, static_cast<float>(M_PI / 2.0),
                                 static_cast<float>(M_PI),
                                 static_cast<float>(-M_PI / 2.0)};
  expect_float_equal<float>(c, expected, 1e-6f);
}
