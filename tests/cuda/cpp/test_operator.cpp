// tests/cuda/test_operator.cpp
#include "insight/insight.h"
#include "insight/ops/operator.h"
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>

using namespace ins;

class OperatorTestGPU : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    ins::init({"cpu", "iluvatar"});
    try {
      set_device(GPUPlace(0));
    } catch (...) {
      GTEST_SKIP() << "GPU not available";
    }
  }
};

// ========== Helper Functions ==========

static Array gpu_1d() {
  return to_array({1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, Shape({5}), DType::F32,
                  CPUPlace())
      .to(GPUPlace(0));
}

static Array gpu_2d() {
  return to_array({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, Shape({2, 3}),
                  DType::F32, CPUPlace())
      .to(GPUPlace(0));
}

static bool approx_equal(double a, double b, double rtol = 1e-6,
                         double atol = 1e-8) {
  return std::abs(a - b) <= atol + rtol * std::abs(b);
}

static void expect_array_eq(const Array &gpu_a,
                            const std::vector<double> &expected) {
  Array a = gpu_a.to(CPUPlace());
  ASSERT_EQ(a.numel(), static_cast<int64_t>(expected.size()));
  for (int64_t i = 0; i < a.numel(); ++i) {
    EXPECT_TRUE(
        approx_equal(static_cast<double>(a.at(i).item<float>()), expected[i]));
  }
}

// ========== Unary Operator Tests ==========

TEST_F(OperatorTestGPU, UnaryMinus) {
  Array a = gpu_1d();
  Array b = -a;
  expect_array_eq(b, {-1.0, -2.0, -3.0, -4.0, -5.0});
}

TEST_F(OperatorTestGPU, UnaryPlus) {
  Array a = gpu_1d();
  Array b = +a;
  expect_array_eq(b, {1.0, 2.0, 3.0, 4.0, 5.0});
}

TEST_F(OperatorTestGPU, BitwiseNot) {
  Array cpu_a = to_array<bool>({1, 0, 1, 0}, Shape({4}), CPUPlace());
  Array a = cpu_a.to(GPUPlace(0));
  Array b = ~a;
  Array cpu_b = b.to(CPUPlace());
  const bool *result = cpu_b.data<bool>();
  EXPECT_FALSE(result[0]);
  EXPECT_TRUE(result[1]);
  EXPECT_FALSE(result[2]);
  EXPECT_TRUE(result[3]);
}

TEST_F(OperatorTestGPU, LogicalNot) {
  Array cpu_a =
      to_array({1.0, 0.0, 3.0, 0.0, 5.0}, Shape({5}), DType::F32, CPUPlace());
  Array a = cpu_a.to(GPUPlace(0));
  Array b = !a;
  Array cpu_b = b.to(CPUPlace());
  const bool *result = cpu_b.data<bool>();
  EXPECT_FALSE(result[0]);
  EXPECT_TRUE(result[1]);
  EXPECT_FALSE(result[2]);
  EXPECT_TRUE(result[3]);
  EXPECT_FALSE(result[4]);
}

// ========== Arithmetic Operators (Array-Array) ==========

TEST_F(OperatorTestGPU, AddArrayArray) {
  Array a = gpu_1d();
  Array b = gpu_1d();
  Array c = a + b;
  expect_array_eq(c, {2.0, 4.0, 6.0, 8.0, 10.0});
}

TEST_F(OperatorTestGPU, SubArrayArray) {
  Array a = gpu_1d();
  Array b = gpu_1d();
  Array c = a - b;
  expect_array_eq(c, {0.0, 0.0, 0.0, 0.0, 0.0});
}

TEST_F(OperatorTestGPU, MulArrayArray) {
  Array a = gpu_1d();
  Array b = gpu_1d();
  Array c = a * b;
  expect_array_eq(c, {1.0, 4.0, 9.0, 16.0, 25.0});
}

TEST_F(OperatorTestGPU, DivArrayArray) {
  Array a = gpu_1d();
  Array b =
      to_array({1.0, 2.0, 3.0, 4.0, 5.0}, Shape({5}), DType::F32, CPUPlace())
          .to(GPUPlace(0));
  Array c = a / b;
  expect_array_eq(c, {1.0, 1.0, 1.0, 1.0, 1.0});
}

TEST_F(OperatorTestGPU, ModArrayArray) {
  Array cpu_a =
      to_array({5LL, 7LL, 9LL, 11LL}, Shape({4}), DType::I64, CPUPlace());
  Array cpu_b =
      to_array({2LL, 3LL, 4LL, 5LL}, Shape({4}), DType::I64, CPUPlace());
  Array a = cpu_a.to(GPUPlace(0));
  Array b = cpu_b.to(GPUPlace(0));
  Array c = a % b;
  Array cpu_c = c.to(CPUPlace());
  const int64_t *result = cpu_c.data<int64_t>();
  EXPECT_EQ(result[0], 1);
  EXPECT_EQ(result[1], 1);
  EXPECT_EQ(result[2], 1);
  EXPECT_EQ(result[3], 1);
}

// ========== Arithmetic Operators (Array-Scalar) ==========

TEST_F(OperatorTestGPU, AddArrayScalar) {
  Array a = gpu_1d();
  Array b = a + 10.0;
  expect_array_eq(b, {11.0, 12.0, 13.0, 14.0, 15.0});
}

TEST_F(OperatorTestGPU, AddScalarArray) {
  Array a = gpu_1d();
  Array b = 10.0 + a;
  expect_array_eq(b, {11.0, 12.0, 13.0, 14.0, 15.0});
}

TEST_F(OperatorTestGPU, SubArrayScalar) {
  Array a = gpu_1d();
  Array b = a - 1.0;
  expect_array_eq(b, {0.0, 1.0, 2.0, 3.0, 4.0});
}

TEST_F(OperatorTestGPU, SubScalarArray) {
  Array a = gpu_1d();
  Array b = 10.0 - a;
  expect_array_eq(b, {9.0, 8.0, 7.0, 6.0, 5.0});
}

TEST_F(OperatorTestGPU, MulArrayScalar) {
  Array a = gpu_1d();
  Array b = a * 2.0;
  expect_array_eq(b, {2.0, 4.0, 6.0, 8.0, 10.0});
}

TEST_F(OperatorTestGPU, MulScalarArray) {
  Array a = gpu_1d();
  Array b = 2.0 * a;
  expect_array_eq(b, {2.0, 4.0, 6.0, 8.0, 10.0});
}

TEST_F(OperatorTestGPU, DivArrayScalar) {
  Array a = gpu_1d();
  Array b = a / 2.0;
  expect_array_eq(b, {0.5, 1.0, 1.5, 2.0, 2.5});
}

TEST_F(OperatorTestGPU, DivScalarArray) {
  Array a = gpu_1d();
  Array b = 10.0 / a;
  expect_array_eq(b, {10.0, 5.0, 10.0 / 3.0, 2.5, 2.0});
}

// ========== Comparison Operators (Array-Array) ==========

TEST_F(OperatorTestGPU, EqualArrayArray) {
  Array a = gpu_1d();
  Array b = gpu_1d();
  Array c = a == b;
  Array cpu_c = c.to(CPUPlace());
  const bool *data = cpu_c.data<bool>();
  for (int i = 0; i < 5; ++i)
    EXPECT_TRUE(data[i]);
}

TEST_F(OperatorTestGPU, NotEqualArrayArray) {
  Array a = gpu_1d();
  Array b = a + 1.0;
  Array c = a != b;
  Array cpu_c = c.to(CPUPlace());
  const bool *data = cpu_c.data<bool>();
  for (int i = 0; i < 5; ++i)
    EXPECT_TRUE(data[i]);
}

TEST_F(OperatorTestGPU, LessArrayArray) {
  Array a = gpu_1d();
  Array b = a + 1.0;
  Array c = a < b;
  Array cpu_c = c.to(CPUPlace());
  const bool *data = cpu_c.data<bool>();
  for (int i = 0; i < 5; ++i)
    EXPECT_TRUE(data[i]);
}

TEST_F(OperatorTestGPU, GreaterArrayArray) {
  Array a = gpu_1d();
  Array b = a + 1.0;
  Array c = a > b;
  Array cpu_c = c.to(CPUPlace());
  const bool *data = cpu_c.data<bool>();
  for (int i = 0; i < 5; ++i)
    EXPECT_FALSE(data[i]);
}

TEST_F(OperatorTestGPU, LessEqualArrayArray) {
  Array a = gpu_1d();
  Array b = a.copy();
  Array c = a <= b;
  Array cpu_c = c.to(CPUPlace());
  const bool *data = cpu_c.data<bool>();
  for (int i = 0; i < 5; ++i)
    EXPECT_TRUE(data[i]);
}

TEST_F(OperatorTestGPU, GreaterEqualArrayArray) {
  Array a = gpu_1d();
  Array b = a.copy();
  Array c = a >= b;
  Array cpu_c = c.to(CPUPlace());
  const bool *data = cpu_c.data<bool>();
  for (int i = 0; i < 5; ++i)
    EXPECT_TRUE(data[i]);
}

// ========== Comparison Operators (Array-Scalar) ==========

TEST_F(OperatorTestGPU, EqualArrayScalar) {
  Array a =
      to_array({3.0, 4.0, 5.0, 4.0, 3.0}, Shape({5}), DType::F32, CPUPlace())
          .to(GPUPlace(0));
  Array c = a == 4.0;
  Array cpu_c = c.to(CPUPlace());
  const bool *result = cpu_c.data<bool>();
  EXPECT_FALSE(result[0]);
  EXPECT_TRUE(result[1]);
  EXPECT_FALSE(result[2]);
  EXPECT_TRUE(result[3]);
  EXPECT_FALSE(result[4]);
}

TEST_F(OperatorTestGPU, EqualScalarArray) {
  Array a =
      to_array({3.0, 4.0, 5.0, 4.0, 3.0}, Shape({5}), DType::F32, CPUPlace())
          .to(GPUPlace(0));
  Array c = 4.0 == a;
  Array cpu_c = c.to(CPUPlace());
  const bool *result = cpu_c.data<bool>();
  EXPECT_FALSE(result[0]);
  EXPECT_TRUE(result[1]);
  EXPECT_FALSE(result[2]);
  EXPECT_TRUE(result[3]);
  EXPECT_FALSE(result[4]);
}

TEST_F(OperatorTestGPU, LessArrayScalar) {
  Array a = gpu_1d();
  Array c = a < 3.0;
  Array cpu_c = c.to(CPUPlace());
  const bool *data = cpu_c.data<bool>();
  EXPECT_TRUE(data[0]);
  EXPECT_TRUE(data[1]);
  EXPECT_FALSE(data[2]);
  EXPECT_FALSE(data[3]);
  EXPECT_FALSE(data[4]);
}

TEST_F(OperatorTestGPU, LessScalarArray) {
  Array a = gpu_1d();
  Array c = 3.0 < a;
  Array cpu_c = c.to(CPUPlace());
  const bool *data = cpu_c.data<bool>();
  EXPECT_FALSE(data[0]);
  EXPECT_FALSE(data[1]);
  EXPECT_FALSE(data[2]);
  EXPECT_TRUE(data[3]);
  EXPECT_TRUE(data[4]);
}

// ========== Compound Assignment Operators ==========

TEST_F(OperatorTestGPU, AddAssignArray) {
  Array a = gpu_1d();
  Array b = gpu_1d();
  a += b;
  expect_array_eq(a, {2.0, 4.0, 6.0, 8.0, 10.0});
}

TEST_F(OperatorTestGPU, SubAssignArray) {
  Array a = gpu_1d();
  Array b = gpu_1d();
  a -= b;
  expect_array_eq(a, {0.0, 0.0, 0.0, 0.0, 0.0});
}

TEST_F(OperatorTestGPU, MulAssignArray) {
  Array a = gpu_1d();
  Array b = gpu_1d();
  a *= b;
  expect_array_eq(a, {1.0, 4.0, 9.0, 16.0, 25.0});
}

TEST_F(OperatorTestGPU, DivAssignArray) {
  Array a = gpu_1d();
  Array b = gpu_1d();
  a /= b;
  expect_array_eq(a, {1.0, 1.0, 1.0, 1.0, 1.0});
}

TEST_F(OperatorTestGPU, AddAssignScalar) {
  Array a = gpu_1d();
  a += 10.0;
  expect_array_eq(a, {11.0, 12.0, 13.0, 14.0, 15.0});
}

TEST_F(OperatorTestGPU, SubAssignScalar) {
  Array a = gpu_1d();
  a -= 1.0;
  expect_array_eq(a, {0.0, 1.0, 2.0, 3.0, 4.0});
}

TEST_F(OperatorTestGPU, MulAssignScalar) {
  Array a = gpu_1d();
  a *= 2.0;
  expect_array_eq(a, {2.0, 4.0, 6.0, 8.0, 10.0});
}

TEST_F(OperatorTestGPU, DivAssignScalar) {
  Array a = gpu_1d();
  a /= 2.0;
  expect_array_eq(a, {0.5, 1.0, 1.5, 2.0, 2.5});
}

// ========== Increment/Decrement Operators ==========

TEST_F(OperatorTestGPU, PrefixIncrement) {
  Array a = gpu_1d();
  ++a;
  expect_array_eq(a, {2.0, 3.0, 4.0, 5.0, 6.0});
}

TEST_F(OperatorTestGPU, PostfixIncrement) {
  Array a = gpu_1d();
  Array b = a++;
  expect_array_eq(b, {1.0, 2.0, 3.0, 4.0, 5.0});
  expect_array_eq(a, {2.0, 3.0, 4.0, 5.0, 6.0});
}

TEST_F(OperatorTestGPU, PrefixDecrement) {
  Array a = gpu_1d();
  --a;
  expect_array_eq(a, {0.0, 1.0, 2.0, 3.0, 4.0});
}

TEST_F(OperatorTestGPU, PostfixDecrement) {
  Array a = gpu_1d();
  Array b = a--;
  expect_array_eq(b, {1.0, 2.0, 3.0, 4.0, 5.0});
  expect_array_eq(a, {0.0, 1.0, 2.0, 3.0, 4.0});
}

// ========== Chain Operations ==========

TEST_F(OperatorTestGPU, ChainArithmetic) {
  Array a = gpu_1d();
  Array b = gpu_1d();
  Array c = a + b * 2.0 - 3.0;
  expect_array_eq(c, {0.0, 3.0, 6.0, 9.0, 12.0});
}

TEST_F(OperatorTestGPU, ChainComparison) {
  Array a =
      to_array({1.0, 2.0, 3.0, 4.0, 5.0}, Shape({5}), DType::F32, CPUPlace())
          .to(GPUPlace(0));
  Array mask = (a > 1) & (a < 5);
  Array cpu_mask = mask.to(CPUPlace());
  const bool *result = cpu_mask.data<bool>();
  EXPECT_FALSE(result[0]);
  EXPECT_TRUE(result[1]);
  EXPECT_TRUE(result[2]);
  EXPECT_TRUE(result[3]);
  EXPECT_FALSE(result[4]);
}

// ========== Array::bool() Tests ==========

TEST_F(OperatorTestGPU, ScalarBoolTrue) {
  Array a(3.14);
  EXPECT_TRUE(static_cast<bool>(a));
}

TEST_F(OperatorTestGPU, ScalarBoolFalse) {
  Array a(0.0);
  EXPECT_FALSE(static_cast<bool>(a));
}

TEST_F(OperatorTestGPU, ScalarIntTrue) {
  Array a(42);
  EXPECT_TRUE(static_cast<bool>(a));
}

TEST_F(OperatorTestGPU, ScalarIntFalse) {
  Array a(0);
  EXPECT_FALSE(static_cast<bool>(a));
}

TEST_F(OperatorTestGPU, MultiElementBoolThrows) {
  Array a = gpu_1d();
  EXPECT_ANY_THROW(static_cast<bool>(a));
}

TEST_F(OperatorTestGPU, AnyAllOnMultiElement) {
  Array a = gpu_1d();
  EXPECT_TRUE(a.any());
  EXPECT_TRUE(a.all());

  Array b =
      to_array({0.0, 1.0, 0.0, 2.0, 0.0}, Shape({5}), DType::F32, CPUPlace())
          .to(GPUPlace(0));
  EXPECT_TRUE(b.any());
  EXPECT_FALSE(b.all());

  Array c = zeros({3, 3}, DType::F32, GPUPlace(0));
  EXPECT_FALSE(c.any());
  EXPECT_FALSE(c.all());
}

// ========== Edge Cases ==========

TEST_F(OperatorTestGPU, EmptyArray) {
  Array a;
  EXPECT_FALSE(static_cast<bool>(a));
}

TEST_F(OperatorTestGPU, BroadcastingInOperations) {
  Array a = gpu_2d();
  Array b = to_array({10.0f, 20.0f, 30.0f}, Shape({3}), DType::F32, CPUPlace())
                .to(GPUPlace(0));
  Array c = a + b;
  Array cpu_c = c.to(CPUPlace());
  const float *result = cpu_c.data<float>();
  EXPECT_NEAR(result[0], 1.0f + 10.0f, 1e-4f);
  EXPECT_NEAR(result[1], 2.0f + 20.0f, 1e-4f);
  EXPECT_NEAR(result[2], 3.0f + 30.0f, 1e-4f);
  EXPECT_NEAR(result[3], 4.0f + 10.0f, 1e-4f);
  EXPECT_NEAR(result[4], 5.0f + 20.0f, 1e-4f);
  EXPECT_NEAR(result[5], 6.0f + 30.0f, 1e-4f);
}

TEST_F(OperatorTestGPU, DivisionByScalarZero) {
  Array a = gpu_1d();
  Array b = a / 0.0f;
  Array cpu_b = b.to(CPUPlace());
  const float *data = cpu_b.data<float>();
  EXPECT_TRUE(std::isinf(data[0]));
}
