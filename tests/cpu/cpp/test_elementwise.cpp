// tests/cpu/test_elementwise.cpp
#include "insight/insight.h"
#include <cmath>
#include <complex>
#include <gtest/gtest.h>

using namespace ins;

// ============================================================================
// Helper functions
// ============================================================================
namespace {
template <typename T> void fill_sequential(Array &arr) {
  T *data = arr.data<T>();
  int64_t n = arr.numel();
  for (int64_t i = 0; i < n; ++i) {
    data[i] = static_cast<T>(i);
  }
}

template <typename T> void fill_constant(Array &arr, T val) {
  T *data = arr.data<T>();
  int64_t n = arr.numel();
  for (int64_t i = 0; i < n; ++i) {
    data[i] = val;
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

template <typename T>
void expect_complex_equal(const Array &arr,
                          const std::vector<std::complex<T>> &expected,
                          T tol = 1e-6) {
  ASSERT_EQ(arr.numel(), static_cast<int64_t>(expected.size()));
  const std::complex<T> *data = arr.data<std::complex<T>>();
  for (int64_t i = 0; i < arr.numel(); ++i) {
    EXPECT_NEAR(data[i].real(), expected[i].real(), tol);
    EXPECT_NEAR(data[i].imag(), expected[i].imag(), tol);
  }
}
} // namespace

class ElementwiseTestCPU : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    ins::init({"cpu"});
    set_device(ins::CPUPlace(0));
  }
};

// ============================================================================
// Subtraction Tests (sub)
// ============================================================================

TEST_F(ElementwiseTestCPU, SubSameShape) {
  ins::init();
  Array a({2, 3}, DType::F32);
  Array b({2, 3}, DType::F32);
  fill_sequential<float>(a);
  fill_sequential<float>(b);

  Array c = sub(a, b);

  std::vector<float> expected = {0, 0, 0, 0, 0, 0};
  expect_float_equal<float>(c, expected);
}

TEST_F(ElementwiseTestCPU, SubBroadcast) {
  Array a({2, 3}, DType::F32);
  Array b({3}, DType::F32);
  fill_sequential<float>(a);
  fill_sequential<float>(b);

  Array c = sub(a, b);

  // a: [0,1,2,3,4,5], b: [0,1,2] -> broadcast to [0,1,2,0,1,2]
  // result: [0-0,1-1,2-2,3-0,4-1,5-2] = [0,0,0,3,3,3]
  std::vector<float> expected = {0, 0, 0, 3, 3, 3};
  expect_float_equal<float>(c, expected);
}

// ============================================================================
// Multiplication Tests (mul)
// ============================================================================

TEST_F(ElementwiseTestCPU, MulSameShape) {
  Array a({2, 3}, DType::F32);
  Array b({2, 3}, DType::F32);
  fill_sequential<float>(a);
  fill_sequential<float>(b);

  Array c = mul(a, b);

  // 0*0=0, 1*1=1, 2*2=4, 3*3=9, 4*4=16, 5*5=25
  std::vector<float> expected = {0, 1, 4, 9, 16, 25};
  expect_float_equal<float>(c, expected);
}

// ============================================================================
// Division Tests (div)
// ============================================================================

TEST_F(ElementwiseTestCPU, DivSameShape) {
  Array a({2, 3}, DType::F32);
  Array b({2, 3}, DType::F32);
  fill_sequential<float>(a);
  float *b_data = b.data<float>();
  for (int64_t i = 0; i < 6; ++i) {
    b_data[i] = static_cast<float>(i + 1);
  }

  Array c = div(a, b);

  // 0/1=0, 1/2=0.5, 2/3≈0.6667, 3/4=0.75, 4/5=0.8, 5/6≈0.8333
  const float *data = c.data<float>();
  EXPECT_FLOAT_EQ(data[0], 0.0f);
  EXPECT_FLOAT_EQ(data[1], 0.5f);
  EXPECT_NEAR(data[2], 2.0f / 3.0f, 1e-6);
  EXPECT_FLOAT_EQ(data[3], 0.75f);
  EXPECT_FLOAT_EQ(data[4], 0.8f);
  EXPECT_NEAR(data[5], 5.0f / 6.0f, 1e-6);
}

// ============================================================================
// Power Tests (pow)
// ============================================================================

TEST_F(ElementwiseTestCPU, PowSameShape) {
  Array a({2, 3}, DType::F32);
  Array b({2, 3}, DType::F32);
  fill_sequential<float>(a);
  fill_sequential<float>(b);

  Array c = pow(a, b);

  // 0^0=1, 1^1=1, 2^2=4, 3^3=27, 4^4=256, 5^5=3125
  const float *data = c.data<float>();
  EXPECT_FLOAT_EQ(data[0], 1.0f);
  EXPECT_FLOAT_EQ(data[1], 1.0f);
  EXPECT_FLOAT_EQ(data[2], 4.0f);
  EXPECT_FLOAT_EQ(data[3], 27.0f);
  EXPECT_FLOAT_EQ(data[4], 256.0f);
  EXPECT_FLOAT_EQ(data[5], 3125.0f);
}

TEST_F(ElementwiseTestCPU, PowBroadcast) {
  Array a({2, 3}, DType::F32);
  Array b({3}, DType::F32);
  fill_sequential<float>(a);
  float *b_data = b.data<float>();
  b_data[0] = 0.0f;
  b_data[1] = 1.0f;
  b_data[2] = 2.0f;

  Array c = pow(a, b);

  expect_float_equal<float>(c, {1, 1, 4, 1, 4, 25});
}

TEST_F(ElementwiseTestCPU, PowFloatExponentPromotesIntegerBase) {
  Array a({3}, DType::I32);
  Array b({3}, DType::F32);
  int32_t *a_data = a.data<int32_t>();
  float *b_data = b.data<float>();
  a_data[0] = 4;
  a_data[1] = 9;
  a_data[2] = 16;
  b_data[0] = 0.5f;
  b_data[1] = 0.5f;
  b_data[2] = 0.5f;

  Array c = pow(a, b);

  EXPECT_EQ(c.dtype(), DType::F32);
  expect_float_equal<float>(c, {2.0f, 3.0f, 4.0f});
}

// ============================================================================
// Modulo Tests (mod)
// ============================================================================

TEST_F(ElementwiseTestCPU, ModSameShape) {
  Array a({2, 3}, DType::I32);
  Array b({2, 3}, DType::I32);
  int32_t *a_data = a.data<int32_t>();
  int32_t *b_data = b.data<int32_t>();
  for (int64_t i = 0; i < 6; ++i) {
    a_data[i] = static_cast<int32_t>(i + 10);
    b_data[i] = static_cast<int32_t>(i + 3);
  }

  Array c = mod(a, b);

  std::vector<int32_t> expected = {10 % 3, 11 % 4, 12 % 5,
                                   13 % 6, 14 % 7, 15 % 8};
  expect_equal<int32_t>(c, expected);
}

TEST_F(ElementwiseTestCPU, ModFloat) {
  Array a({2, 3}, DType::F32);
  Array b({2, 3}, DType::F32);
  float *a_data = a.data<float>();
  float *b_data = b.data<float>();
  for (int64_t i = 0; i < 6; ++i) {
    a_data[i] = static_cast<float>(i + 10);
    b_data[i] = static_cast<float>(i + 3);
  }

  Array c = mod(a, b);

  const float *data = c.data<float>();
  EXPECT_FLOAT_EQ(data[0], fmodf(10, 3));
  EXPECT_FLOAT_EQ(data[1], fmodf(11, 4));
  EXPECT_FLOAT_EQ(data[2], fmodf(12, 5));
  EXPECT_FLOAT_EQ(data[3], fmodf(13, 6));
  EXPECT_FLOAT_EQ(data[4], fmodf(14, 7));
  EXPECT_FLOAT_EQ(data[5], fmodf(15, 8));
}

// ============================================================================
// Comparison Tests (equal, not_equal, greater, less, greater_equal, less_equal)
// ============================================================================

TEST_F(ElementwiseTestCPU, Equal) {
  Array a({2, 3}, DType::F32);
  Array b({2, 3}, DType::F32);
  float *a_data = a.data<float>();
  float *b_data = b.data<float>();
  for (int64_t i = 0; i < 6; ++i) {
    a_data[i] = static_cast<float>(i);
    b_data[i] = static_cast<float>(i); // exactly the same
  }

  Array c = equal(a, b);

  std::vector<bool> expected = {true, true, true, true, true, true};
  expect_bool_equal(c, expected);
}

TEST_F(ElementwiseTestCPU, EqualBroadcast) {
  Array a({2, 3}, DType::F32);
  Array b({3}, DType::F32);
  fill_sequential<float>(a);
  fill_sequential<float>(b);

  Array c = equal(a, b);

  std::vector<bool> expected = {true, true, true, false, false, false};
  expect_bool_equal(c, expected);
}

TEST_F(ElementwiseTestCPU, NotEqual) {
  Array a({2, 3}, DType::F32);
  Array b({2, 3}, DType::F32);
  float *a_data = a.data<float>();
  float *b_data = b.data<float>();
  for (int64_t i = 0; i < 6; ++i) {
    a_data[i] = static_cast<float>(i);
    b_data[i] = static_cast<float>(i + 1); // completely different
  }

  Array c = not_equal(a, b);
  std::vector<bool> expected = {true, true, true, true, true, true};
  expect_bool_equal(c, expected);
}

TEST_F(ElementwiseTestCPU, Greater) {
  Array a({2, 3}, DType::F32);
  Array b({2, 3}, DType::F32);
  fill_sequential<float>(a);
  fill_sequential<float>(b);

  Array c = greater(a, b);

  std::vector<bool> expected = {false, false, false, false, false, false};
  expect_bool_equal(c, expected);
}

TEST_F(ElementwiseTestCPU, Less) {
  Array a({2, 3}, DType::F32);
  Array b({2, 3}, DType::F32);
  fill_sequential<float>(a);
  fill_sequential<float>(b);

  Array c = less(a, b);

  std::vector<bool> expected = {false, false, false, false, false, false};
  expect_bool_equal(c, expected);
}

TEST_F(ElementwiseTestCPU, GreaterEqual) {
  Array a({2, 3}, DType::F32);
  Array b({2, 3}, DType::F32);
  float *a_data = a.data<float>();
  float *b_data = b.data<float>();
  for (int64_t i = 0; i < 6; ++i) {
    a_data[i] = static_cast<float>(i);
    b_data[i] = static_cast<float>(i); // a >= b all true
  }

  Array c = greater_equal(a, b);

  std::vector<bool> expected = {true, true, true, true, true, true};
  expect_bool_equal(c, expected);
}

TEST_F(ElementwiseTestCPU, LessEqual) {
  Array a({2, 3}, DType::F32);
  Array b({2, 3}, DType::F32);
  float *a_data = a.data<float>();
  float *b_data = b.data<float>();
  for (int64_t i = 0; i < 6; ++i) {
    a_data[i] = static_cast<float>(i);
    b_data[i] = static_cast<float>(i); // a <= b all true
  }

  Array c = less_equal(a, b);

  std::vector<bool> expected = {true, true, true, true, true, true};
  expect_bool_equal(c, expected);
}

// ============================================================================
// Logical Operations Tests (logical_and, logical_or, logical_xor)
// ============================================================================

TEST_F(ElementwiseTestCPU, LogicalAnd) {
  Array a({2, 3}, DType::BOOL);
  Array b({2, 3}, DType::BOOL);
  bool *a_data = a.data<bool>();
  bool *b_data = b.data<bool>();

  // Simple pattern
  for (int64_t i = 0; i < 6; ++i) {
    a_data[i] = (i % 2 == 0); // true, false, true, false, true, false
    b_data[i] = (i < 3);      // true, true, true, false, false, false
  }

  Array c = logical_and(a, b);

  // T&T=T, F&T=F, T&T=T, F&F=F, T&F=F, F&F=F
  std::vector<bool> expected = {true, false, true, false, false, false};
  expect_bool_equal(c, expected);
}

TEST_F(ElementwiseTestCPU, LogicalOr) {
  Array a({2, 3}, DType::BOOL);
  Array b({2, 3}, DType::BOOL);
  bool *a_data = a.data<bool>();
  bool *b_data = b.data<bool>();

  for (int64_t i = 0; i < 6; ++i) {
    a_data[i] = (i % 2 == 0);
    b_data[i] = (i % 3 != 0);
  }

  Array c = logical_or(a, b);

  // T|T=T, F|T=T, T|F=T, F|F=F, T|T=T, F|T=T
  std::vector<bool> expected = {true, true, true, false, true, true};
  expect_bool_equal(c, expected);
}

TEST_F(ElementwiseTestCPU, LogicalXor) {
  Array a({2, 3}, DType::BOOL);
  Array b({2, 3}, DType::BOOL);
  bool *a_data = a.data<bool>();
  bool *b_data = b.data<bool>();

  for (int64_t i = 0; i < 6; ++i) {
    a_data[i] = (i % 2 == 0);
    b_data[i] = (i < 3);
  }

  Array c = logical_xor(a, b);

  // T^T=F, F^T=T, T^T=F, F^F=F, T^F=T, F^F=F
  std::vector<bool> expected = {false, true, false, false, true, false};
  expect_bool_equal(c, expected);
}

TEST_F(ElementwiseTestCPU, LogicalBroadcast) {
  Array a({2, 3}, DType::BOOL);
  Array b({3}, DType::BOOL);
  bool *a_data = a.data<bool>();
  bool *b_data = b.data<bool>();

  for (int64_t i = 0; i < 6; ++i) {
    a_data[i] = (i % 2 == 0);
  }
  b_data[0] = true;
  b_data[1] = false;
  b_data[2] = true;

  expect_bool_equal(logical_and(a, b), {true, false, true, false, false, false});
  expect_bool_equal(logical_or(a, b), {true, false, true, true, true, true});
  expect_bool_equal(logical_xor(a, b), {false, false, false, true, true, true});
}

// ============================================================================
// Bitwise Operations Tests
// ============================================================================

TEST_F(ElementwiseTestCPU, BitwiseAnd) {
  Array a({2, 3}, DType::I32);
  Array b({2, 3}, DType::I32);
  int32_t *a_data = a.data<int32_t>();
  int32_t *b_data = b.data<int32_t>();
  for (int64_t i = 0; i < 6; ++i) {
    a_data[i] = static_cast<int32_t>(i + 1);
    b_data[i] = static_cast<int32_t>(i + 5);
  }

  Array c = bitwise_and(a, b);

  const int32_t *data = c.data<int32_t>();
  for (int64_t i = 0; i < 6; ++i) {
    EXPECT_EQ(data[i], (i + 1) & (i + 5));
  }
}

TEST_F(ElementwiseTestCPU, BitwiseOr) {
  Array a({2, 3}, DType::I32);
  Array b({2, 3}, DType::I32);
  int32_t *a_data = a.data<int32_t>();
  int32_t *b_data = b.data<int32_t>();
  for (int64_t i = 0; i < 6; ++i) {
    a_data[i] = static_cast<int32_t>(i + 1);
    b_data[i] = static_cast<int32_t>(i + 5);
  }

  Array c = bitwise_or(a, b);

  const int32_t *data = c.data<int32_t>();
  for (int64_t i = 0; i < 6; ++i) {
    EXPECT_EQ(data[i], (i + 1) | (i + 5));
  }
}

TEST_F(ElementwiseTestCPU, BitwiseXor) {
  Array a({2, 3}, DType::I32);
  Array b({2, 3}, DType::I32);
  int32_t *a_data = a.data<int32_t>();
  int32_t *b_data = b.data<int32_t>();
  for (int64_t i = 0; i < 6; ++i) {
    a_data[i] = static_cast<int32_t>(i + 1);
    b_data[i] = static_cast<int32_t>(i + 5);
  }

  Array c = bitwise_xor(a, b);

  const int32_t *data = c.data<int32_t>();
  for (int64_t i = 0; i < 6; ++i) {
    EXPECT_EQ(data[i], (i + 1) ^ (i + 5));
  }
}

TEST_F(ElementwiseTestCPU, BitwiseLeftShift) {
  Array a({2, 3}, DType::I32);
  Array b({2, 3}, DType::I32);
  int32_t *a_data = a.data<int32_t>();
  int32_t *b_data = b.data<int32_t>();
  for (int64_t i = 0; i < 6; ++i) {
    a_data[i] = static_cast<int32_t>(i + 1);
    b_data[i] = static_cast<int32_t>(i % 3);
  }

  Array c = bitwise_left_shift(a, b);

  const int32_t *data = c.data<int32_t>();
  for (int64_t i = 0; i < 6; ++i) {
    EXPECT_EQ(data[i], (i + 1) << (i % 3));
  }
}

TEST_F(ElementwiseTestCPU, BitwiseRightShift) {
  Array a({2, 3}, DType::I32);
  Array b({2, 3}, DType::I32);
  int32_t *a_data = a.data<int32_t>();
  int32_t *b_data = b.data<int32_t>();
  for (int64_t i = 0; i < 6; ++i) {
    a_data[i] = static_cast<int32_t>((i + 1) * 8);
    b_data[i] = static_cast<int32_t>(i % 4);
  }

  Array c = bitwise_right_shift(a, b);

  const int32_t *data = c.data<int32_t>();
  for (int64_t i = 0; i < 6; ++i) {
    EXPECT_EQ(data[i], (i + 1) * 8 >> (i % 4));
  }
}

TEST_F(ElementwiseTestCPU, BitwiseBroadcast) {
  Array a({2, 3}, DType::I32);
  Array b({3}, DType::I32);
  int32_t *a_data = a.data<int32_t>();
  int32_t *b_data = b.data<int32_t>();
  for (int64_t i = 0; i < 6; ++i) {
    a_data[i] = static_cast<int32_t>(i + 1);
  }
  b_data[0] = 1;
  b_data[1] = 2;
  b_data[2] = 3;

  expect_equal<int32_t>(bitwise_and(a, b), {1, 2, 3, 0, 0, 2});
  expect_equal<int32_t>(bitwise_or(a, b), {1, 2, 3, 5, 7, 7});
  expect_equal<int32_t>(bitwise_xor(a, b), {0, 0, 0, 5, 7, 5});
}

// ============================================================================
// Maximum / Minimum Tests
// ============================================================================

TEST_F(ElementwiseTestCPU, Maximum) {
  Array a({2, 3}, DType::F32);
  Array b({2, 3}, DType::F32);
  float *a_data = a.data<float>();
  float *b_data = b.data<float>();
  for (int64_t i = 0; i < 6; ++i) {
    a_data[i] = static_cast<float>(i);
    b_data[i] = static_cast<float>(5 - i);
  }

  Array c = maximum(a, b);

  std::vector<float> expected = {5, 4, 3, 3, 4, 5};
  expect_float_equal<float>(c, expected);
}

TEST_F(ElementwiseTestCPU, Minimum) {
  Array a({2, 3}, DType::F32);
  Array b({2, 3}, DType::F32);
  float *a_data = a.data<float>();
  float *b_data = b.data<float>();
  for (int64_t i = 0; i < 6; ++i) {
    a_data[i] = static_cast<float>(i);
    b_data[i] = static_cast<float>(5 - i);
  }

  Array c = minimum(a, b);

  std::vector<float> expected = {0, 1, 2, 2, 1, 0};
  expect_float_equal<float>(c, expected);
}

// ============================================================================
// Broadcasting Tests for all ops
// ============================================================================

TEST_F(ElementwiseTestCPU, Broadcast2D1D) {
  Array a({2, 3}, DType::F32);
  Array b({3}, DType::F32);
  fill_sequential<float>(a);
  fill_sequential<float>(b);

  Array add_c = add(a, b);
  Array sub_c = sub(a, b);
  Array mul_c = mul(a, b);

  // Expected: a: [0,1,2,3,4,5], b: [0,1,2] broadcast -> [0,1,2,0,1,2]
  // add: [0,2,4,3,5,7]
  std::vector<float> add_expected = {0, 2, 4, 3, 5, 7};
  // sub: [0,0,0,3,3,3]
  std::vector<float> sub_expected = {0, 0, 0, 3, 3, 3};
  // mul: [0,1,4,0,4,10]
  std::vector<float> mul_expected = {0, 1, 4, 0, 4, 10};

  expect_float_equal<float>(add_c, add_expected);
  expect_float_equal<float>(sub_c, sub_expected);
  expect_float_equal<float>(mul_c, mul_expected);
}

TEST_F(ElementwiseTestCPU, BroadcastDivMaxMin) {
  Array a({2, 3}, DType::F32);
  Array b({3}, DType::F32);
  float *a_data = a.data<float>();
  float *b_data = b.data<float>();
  for (int64_t i = 0; i < 6; ++i) {
    a_data[i] = static_cast<float>(i + 2);
  }
  b_data[0] = 1.0f;
  b_data[1] = 2.0f;
  b_data[2] = 4.0f;

  expect_float_equal<float>(div(a, b), {2.0f, 1.5f, 1.0f, 5.0f, 3.0f, 1.75f});
  expect_float_equal<float>(maximum(a, b), {2, 3, 4, 5, 6, 7});
  expect_float_equal<float>(minimum(a, b), {1, 2, 4, 1, 2, 4});
}

TEST_F(ElementwiseTestCPU, BroadcastComparisons) {
  Array a({2, 3}, DType::F32);
  Array b({3}, DType::F32);
  fill_sequential<float>(a);
  fill_sequential<float>(b);

  expect_bool_equal(not_equal(a, b), {false, false, false, true, true, true});
  expect_bool_equal(greater(a, b), {false, false, false, true, true, true});
  expect_bool_equal(less(a, b), {false, false, false, false, false, false});
  expect_bool_equal(greater_equal(a, b), {true, true, true, true, true, true});
  expect_bool_equal(less_equal(a, b), {true, true, true, false, false, false});
}

// ============================================================================
// Complex Number Tests
// ============================================================================

TEST_F(ElementwiseTestCPU, ComplexAdd) {
  Array a({2, 3}, DType::C32);
  Array b({2, 3}, DType::C32);

  std::complex<float> *a_data = a.data<std::complex<float>>();
  std::complex<float> *b_data = b.data<std::complex<float>>();
  for (int64_t i = 0; i < 6; ++i) {
    a_data[i] =
        std::complex<float>(static_cast<float>(i), static_cast<float>(i * 2));
    b_data[i] =
        std::complex<float>(static_cast<float>(i), static_cast<float>(i * 3));
  }

  Array c = add(a, b);

  std::vector<std::complex<float>> expected(6);
  for (int64_t i = 0; i < 6; ++i) {
    expected[i] = std::complex<float>(static_cast<float>(i + i),
                                      static_cast<float>(i * 2 + i * 3));
  }
  expect_complex_equal<float>(c, expected);
}

// ============================================================================
// Scalar Broadcast Tests
// ============================================================================

TEST_F(ElementwiseTestCPU, ScalarAdd) {
  Array a({2, 3}, DType::F32);
  fill_sequential<float>(a);
  Array b(5.0f);

  Array c = add(a, b);

  std::vector<float> expected = {5, 6, 7, 8, 9, 10};
  expect_float_equal<float>(c, expected);
}

// ============================================================================
// View (Non-Contiguous) Tests
// ============================================================================

TEST_F(ElementwiseTestCPU, ViewAdd) {
  Array a({3, 4}, DType::F32);
  fill_sequential<float>(a);

  // Use slicing with stride 2, resulting in a non-contiguous view
  Array view = a.slice(0, 0, 3, 2); // Take row 0 and row 2, skipping row 1
  EXPECT_FALSE(view.is_contiguous());

  Array b({2, 4}, DType::F32);
  fill_constant<float>(b, 1.0f);

  Array c = add(view, b);

  // view row 0 = a row 0 = [0,1,2,3] + 1 = [1,2,3,4]
  // view row 1 = a row 2 = [8,9,10,11] + 1 = [9,10,11,12]
  const float *data = c.data<float>();
  EXPECT_FLOAT_EQ(data[0], 1);
  EXPECT_FLOAT_EQ(data[1], 2);
  EXPECT_FLOAT_EQ(data[2], 3);
  EXPECT_FLOAT_EQ(data[3], 4);
  EXPECT_FLOAT_EQ(data[4], 9);
  EXPECT_FLOAT_EQ(data[5], 10);
  EXPECT_FLOAT_EQ(data[6], 11);
  EXPECT_FLOAT_EQ(data[7], 12);
}

// ============================================================================
// Type Promotion Tests
// ============================================================================

TEST_F(ElementwiseTestCPU, TypePromotionIntToFloat) {
  Array a({2, 3}, DType::I32);
  Array b({2, 3}, DType::F32);

  int32_t *a_data = a.data<int32_t>();
  float *b_data = b.data<float>();
  for (int64_t i = 0; i < 6; ++i) {
    a_data[i] = static_cast<int32_t>(i);
    b_data[i] = static_cast<float>(i) * 1.5f;
  }

  Array c = add(a, b);

  EXPECT_EQ(c.dtype(), DType::F32);
  const float *data = c.data<float>();
  for (int64_t i = 0; i < 6; ++i) {
    EXPECT_FLOAT_EQ(data[i], static_cast<float>(i) + i * 1.5f);
  }
}