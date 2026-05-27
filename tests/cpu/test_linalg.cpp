// tests/cpu/test_linalg.cpp
#include "insight/insight.h"
#include "insight/ops/linalg.h"
#include "insight/utils/features.h"
#include <cmath>
#include <complex>
#include <gtest/gtest.h>

using namespace ins;

class LinalgTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    ins::init();
    if (!ins::is_compiled_with_openblas()) {
      GTEST_SKIP() << "Skipping linear algebra tests: OpenBLAS not available";
    }
    set_device(CPUPlace());
  }
};
// ============================================================================
// Helper functions
// ============================================================================

static Array create_matrix_f64(int rows, int cols,
                               const std::vector<double> &data) {
  Array result(Shape({rows, cols}), DType::F64);
  std::memcpy(result.data<double>(), data.data(), data.size() * sizeof(double));
  return result;
}

static Array create_matrix_f32(int rows, int cols,
                               const std::vector<float> &data) {
  Array result(Shape({rows, cols}), DType::F32);
  std::memcpy(result.data<float>(), data.data(), data.size() * sizeof(float));
  return result;
}

static Array create_vector_f64(const std::vector<double> &data) {
  Array result(Shape({static_cast<int64_t>(data.size())}), DType::F64);
  std::memcpy(result.data<double>(), data.data(), data.size() * sizeof(double));
  return result;
}

static Array create_vector_f32(const std::vector<float> &data) {
  Array result(Shape({static_cast<int64_t>(data.size())}), DType::F32);
  std::memcpy(result.data<float>(), data.data(), data.size() * sizeof(float));
  return result;
}

static bool approx_equal(double a, double b, double rtol = 1e-6,
                         double atol = 1e-8) {
  return std::abs(a - b) <= atol + rtol * std::abs(b);
}

static bool approx_equal(float a, float b, float rtol = 1e-5f,
                         float atol = 1e-6f) {
  return std::abs(a - b) <= atol + rtol * std::abs(b);
}

static bool check_matrix_equal(const Array &A, const Array &B,
                               double rtol = 1e-5) {
  if (A.shape() != B.shape())
    return false;
  if (A.dtype() != B.dtype())
    return false;
  int64_t n = A.numel();
  if (A.dtype() == DType::F64) {
    const double *a = A.data<double>();
    const double *b = B.data<double>();
    for (int64_t i = 0; i < n; ++i) {
      if (!approx_equal(a[i], b[i], rtol))
        return false;
    }
  } else if (A.dtype() == DType::F32) {
    const float *a = A.data<float>();
    const float *b = B.data<float>();
    for (int64_t i = 0; i < n; ++i) {
      if (!approx_equal(a[i], b[i], static_cast<float>(rtol)))
        return false;
    }
  } else {
    return false;
  }
  return true;
}

// ============================================================================
// matmul tests
// ============================================================================

TEST_F(LinalgTest, MatMul2x2F64) {
  Array A = create_matrix_f64(2, 2, {1, 2, 3, 4});
  Array B = create_matrix_f64(2, 2, {5, 6, 7, 8});
  Array C = matmul(A, B);

  const double *c = C.data<double>();
  EXPECT_NEAR(c[0], 19, 1e-6);
  EXPECT_NEAR(c[1], 22, 1e-6);
  EXPECT_NEAR(c[2], 43, 1e-6);
  EXPECT_NEAR(c[3], 50, 1e-6);
}

TEST_F(LinalgTest, MatMul2x2F32) {
  Array A = create_matrix_f32(2, 2, {1.0f, 2.0f, 3.0f, 4.0f});
  Array B = create_matrix_f32(2, 2, {5.0f, 6.0f, 7.0f, 8.0f});
  Array C = matmul(A, B);

  const float *c = C.data<float>();
  EXPECT_NEAR(c[0], 19.0f, 1e-5f);
  EXPECT_NEAR(c[1], 22.0f, 1e-5f);
  EXPECT_NEAR(c[2], 43.0f, 1e-5f);
  EXPECT_NEAR(c[3], 50.0f, 1e-5f);
}

TEST_F(LinalgTest, MatMulNonSquare) {
  {
    Array A = create_matrix_f64(2, 3, {1, 2, 3, 4, 5, 6});
    Array B = create_matrix_f64(3, 2, {7, 8, 9, 10, 11, 12});
    Array C = matmul(A, B);

    const double *c = C.data<double>();
    // Expected: [[58, 64], [139, 154]]
    EXPECT_NEAR(c[0], 58, 1e-6);
    EXPECT_NEAR(c[1], 64, 1e-6);
    EXPECT_NEAR(c[2], 139, 1e-6);
    EXPECT_NEAR(c[3], 154, 1e-6);
  }
  {
    Array A = create_matrix_f32(2, 3, {1, 2, 3, 4, 5, 6});
    Array B = create_matrix_f32(3, 2, {7, 8, 9, 10, 11, 12});
    Array C = matmul(A, B);

    const float *c = C.data<float>();
    // Expected: [[58, 64], [139, 154]]
    EXPECT_NEAR(c[0], 58, 1e-5f);
    EXPECT_NEAR(c[1], 64, 1e-5f);
    EXPECT_NEAR(c[2], 139, 1e-5f);
    EXPECT_NEAR(c[3], 154, 1e-5f);
  }
}

// ============================================================================
// det tests
// ============================================================================

TEST_F(LinalgTest, Det2x2) {
  {
    Array A = create_matrix_f64(2, 2, {1, 2, 3, 4});
    Array d = det(A);
    EXPECT_NEAR(d.item<double>(), -2.0, 1e-6);
  }
  {
    Array A = create_matrix_f32(2, 2, {1, 2, 3, 4});
    Array d = det(A);
    EXPECT_NEAR(d.item<float>(), -2.0f, 1e-5f);
  }
}

TEST_F(LinalgTest, Det3x3) {
  {
    Array A = create_matrix_f64(3, 3, {1, 2, 3, 2, 5, 3, 1, 0, 8});
    Array d = det(A);
    EXPECT_NEAR(d.item<double>(), -1.0, 1e-6);
  }
  {
    Array A = create_matrix_f32(3, 3, {1, 2, 3, 2, 5, 3, 1, 0, 8});
    Array d = det(A);
    EXPECT_NEAR(d.item<float>(), -1.0f, 1e-5f);
  }
}

TEST_F(LinalgTest, DetIdentity) {
  {
    Array A = create_matrix_f64(3, 3, {1, 0, 0, 0, 1, 0, 0, 0, 1});
    Array d = det(A);
    EXPECT_NEAR(d.item<double>(), 1.0, 1e-6);
  }
  {
    Array A = create_matrix_f32(3, 3, {1, 0, 0, 0, 1, 0, 0, 0, 1});
    Array d = det(A);
    EXPECT_NEAR(d.item<float>(), 1.0f, 1e-5f);
  }
}

// ============================================================================
// slogdet tests
// ============================================================================

TEST_F(LinalgTest, Slogdet2x2) {
  {
    Array A = create_matrix_f64(2, 2, {1, 2, 3, 4});
    auto [sign, logdet] = slogdet(A);
    EXPECT_NEAR(sign.item<double>(), -1.0, 1e-6);
    EXPECT_NEAR(logdet.item<double>(), std::log(2.0), 1e-6);
  }
  {
    Array A = create_matrix_f32(2, 2, {1, 2, 3, 4});
    auto [sign, logdet] = slogdet(A);
    EXPECT_NEAR(sign.item<float>(), -1.0f, 1e-5f);
    EXPECT_NEAR(logdet.item<double>(), std::log(2.0), 1e-6);
  }
}

// ============================================================================
// inv tests
// ============================================================================

TEST_F(LinalgTest, Inv2x2) {
  {
    Array A = create_matrix_f64(2, 2, {1, 2, 3, 4});
    Array invA = inv(A);
    const double *data = invA.data<double>();
    EXPECT_NEAR(data[0], -2.0, 1e-6);
    EXPECT_NEAR(data[1], 1.0, 1e-6);
    EXPECT_NEAR(data[2], 1.5, 1e-6);
    EXPECT_NEAR(data[3], -0.5, 1e-6);

    // Verify A * invA = I
    Array I = matmul(A, invA);
    const double *i = I.data<double>();
    EXPECT_NEAR(i[0], 1.0, 1e-6);
    EXPECT_NEAR(i[1], 0.0, 1e-6);
    EXPECT_NEAR(i[2], 0.0, 1e-6);
    EXPECT_NEAR(i[3], 1.0, 1e-6);
  }
  {
    Array A = create_matrix_f32(2, 2, {1, 2, 3, 4});
    Array invA = inv(A);
    const float *data = invA.data<float>();
    EXPECT_NEAR(data[0], -2.0f, 1e-5f);
    EXPECT_NEAR(data[1], 1.0f, 1e-5f);
    EXPECT_NEAR(data[2], 1.5f, 1e-5f);
    EXPECT_NEAR(data[3], -0.5f, 1e-5f);

    // Verify A * invA = I
    Array I = matmul(A, invA);
    const float *i = I.data<float>();
    EXPECT_NEAR(i[0], 1.0f, 1e-5f);
    EXPECT_NEAR(i[1], 0.0f, 1e-5f);
    EXPECT_NEAR(i[2], 0.0f, 1e-5f);
    EXPECT_NEAR(i[3], 1.0f, 1e-5f);
  }
}

TEST_F(LinalgTest, Inv3x3) {
  {
    Array A = create_matrix_f64(3, 3, {1, 2, 3, 2, 5, 3, 1, 0, 8});
    Array invA = inv(A);
    Array I = matmul(A, invA);

    const double *i = I.data<double>();
    EXPECT_NEAR(i[0], 1.0, 1e-5);
    EXPECT_NEAR(i[1], 0.0, 1e-5);
    EXPECT_NEAR(i[2], 0.0, 1e-5);
    EXPECT_NEAR(i[3], 0.0, 1e-5);
    EXPECT_NEAR(i[4], 1.0, 1e-5);
    EXPECT_NEAR(i[5], 0.0, 1e-5);
    EXPECT_NEAR(i[6], 0.0, 1e-5);
    EXPECT_NEAR(i[7], 0.0, 1e-5);
    EXPECT_NEAR(i[8], 1.0, 1e-5);
  }
  {
    Array A = create_matrix_f32(3, 3, {1, 2, 3, 2, 5, 3, 1, 0, 8});
    Array invA = inv(A);
    Array I = matmul(A, invA);

    const float *i = I.data<float>();
    EXPECT_NEAR(i[0], 1.0f, 1e-5f);
    EXPECT_NEAR(i[1], 0.0f, 1e-5f);
    EXPECT_NEAR(i[2], 0.0f, 1e-5f);
    EXPECT_NEAR(i[3], 0.0f, 1e-5f);
    EXPECT_NEAR(i[4], 1.0f, 1e-5f);
    EXPECT_NEAR(i[5], 0.0f, 1e-5f);
    EXPECT_NEAR(i[6], 0.0f, 1e-5f);
    EXPECT_NEAR(i[7], 0.0f, 1e-5f);
    EXPECT_NEAR(i[8], 1.0f, 1e-5f);
  }
}

// ============================================================================
// solve tests
// ============================================================================

TEST_F(LinalgTest, Solve3x3) {
  {
    // Double precision: use 1e-12
    Array A = create_matrix_f64(3, 3, {3, 2, -1, 2, -2, 4, -1, 0.5, -1});
    Array b = create_vector_f64({1.0, -2.0, 0.0});
    Array x = solve(A, b);
    const double *data = x.data<double>();
    EXPECT_NEAR(data[0], 1.0, 1e-10);
    EXPECT_NEAR(data[1], -2.0, 1e-10);
    EXPECT_NEAR(data[2], -2.0, 1e-10);
  }
  {
    // Single precision: use 1e-5 (float precision is ~1e-6)
    Array A = create_matrix_f32(3, 3, {3, 2, -1, 2, -2, 4, -1, 0.5, -1});
    Array b = create_vector_f32({1.0f, -2.0f, 0.0f});
    Array x = solve(A, b);
    const float *data = x.data<float>();
    EXPECT_NEAR(data[0], 1.0f, 1e-5f);
    EXPECT_NEAR(data[1], -2.0f, 1e-5f);
    EXPECT_NEAR(data[2], -2.0f, 1e-5f);
  }
}

TEST_F(LinalgTest, SolveMultipleRHS) {
  {
    Array A = create_matrix_f64(3, 3, {3, 2, -1, 2, -2, 4, -1, 0.5, -1});
    Array B = create_matrix_f64(3, 2, {1, 0, -2, 1, 0, 2});
    Array X = solve(A, B);

    EXPECT_EQ(X.shape(), Shape({3, 2}));

    const double *x = X.data<double>();
    // Row-major layout:
    // row0: index 0, 1
    // row1: index 2, 3
    // row2: index 4, 5

    // First column (solution for B[:,0] = [1, -2, 0]) should be [1, -2, -2]
    EXPECT_NEAR(x[0], 1.0, 1e-6);
    EXPECT_NEAR(x[2], -2.0, 1e-6);
    EXPECT_NEAR(x[4], -2.0, 1e-6);

    // Second column: verify A * X[:,1] == B[:,1]
    Array X_col1 = create_matrix_f64(3, 1, {x[1], x[3], x[5]});
    Array B_col1 = create_matrix_f64(3, 1, {0.0, 1.0, 2.0});
    Array check = matmul(A, X_col1);
    EXPECT_NEAR(check.data<double>()[0], B_col1.data<double>()[0], 1e-5);
    EXPECT_NEAR(check.data<double>()[1], B_col1.data<double>()[1], 1e-5);
    EXPECT_NEAR(check.data<double>()[2], B_col1.data<double>()[2], 1e-5);
  }
  {
    Array A = create_matrix_f32(3, 3, {3, 2, -1, 2, -2, 4, -1, 0.5, -1});
    Array B = create_matrix_f32(3, 2, {1, 0, -2, 1, 0, 2});
    Array X = solve(A, B);

    EXPECT_EQ(X.shape(), Shape({3, 2}));

    const float *x = X.data<float>();
    // Row-major layout:
    // row0: index 0, 1
    // row1: index 2, 3
    // row2: index 4, 5

    // First column (solution for B[:,0] = [1, -2, 0]) should be [1, -2, -2]
    EXPECT_NEAR(x[0], 1.0f, 1e-5f);
    EXPECT_NEAR(x[2], -2.0f, 1e-5f);
    EXPECT_NEAR(x[4], -2.0f, 1e-5f);

    // Second column: verify A * X[:,1] == B[:,1]
    Array X_col1 = create_matrix_f32(3, 1, {x[1], x[3], x[5]});
    Array B_col1 = create_matrix_f32(3, 1, {0.0f, 1.0f, 2.0f});
    Array check = matmul(A, X_col1);
    EXPECT_NEAR(check.data<float>()[0], B_col1.data<float>()[0], 1e-5f);
    EXPECT_NEAR(check.data<float>()[1], B_col1.data<float>()[1], 1e-5f);
    EXPECT_NEAR(check.data<float>()[2], B_col1.data<float>()[2], 1e-5f);
  }
}

// ============================================================================
// cholesky tests
// ============================================================================

TEST_F(LinalgTest, Cholesky3x3) {
  {
    Array A = create_matrix_f64(3, 3, {4, 12, -16, 12, 37, -43, -16, -43, 98});
    Array L = cholesky(A, true);
    Array LLT = matmul(L, transpose(L));

    EXPECT_TRUE(check_matrix_equal(A, LLT, 1e-5));
  }
  {
    Array A = create_matrix_f32(
        3, 3,
        {4.0f, 12.0f, -16.0f, 12.0f, 37.0f, -43.0f, -16.0f, -43.0f, 98.0f});
    Array L = cholesky(A, true);
    Array LLT = matmul(L, transpose(L));

    EXPECT_TRUE(check_matrix_equal(A, LLT, 1e-4f));
  }
}

// ============================================================================
// qr tests
// ============================================================================

TEST_F(LinalgTest, QR3x3) {
  // F64 测试
  {
    Array A = create_matrix_f64(3, 3, {12, -51, 4, 6, 167, -68, -4, 24, -41});
    auto [Q, R] = qr(A, "reduced");
    Array QR = matmul(Q, R);

    EXPECT_TRUE(check_matrix_equal(A, QR, 1e-5));

    // Check Q is orthogonal: Q^T * Q = I
    Array QTQ = matmul(transpose(Q), Q);
    const double *qtq_data = QTQ.data<double>();
    EXPECT_NEAR(qtq_data[0], 1.0, 1e-5);
    EXPECT_NEAR(qtq_data[4], 1.0, 1e-5);
    EXPECT_NEAR(qtq_data[8], 1.0, 1e-5);
  }

  // F32 测试
  {
    Array A = create_matrix_f32(
        3, 3,
        {12.0f, -51.0f, 4.0f, 6.0f, 167.0f, -68.0f, -4.0f, 24.0f, -41.0f});
    auto [Q, R] = qr(A, "reduced");
    Array QR = matmul(Q, R);

    EXPECT_TRUE(check_matrix_equal(A, QR, 1e-4f));

    // Check Q is orthogonal: Q^T * Q = I
    Array QTQ = matmul(transpose(Q), Q);
    const float *qtq_data = QTQ.data<float>();
    EXPECT_NEAR(qtq_data[0], 1.0f, 1e-4f);
    EXPECT_NEAR(qtq_data[4], 1.0f, 1e-4f);
    EXPECT_NEAR(qtq_data[8], 1.0f, 1e-4f);
  }
}

TEST_F(LinalgTest, QR3x2) {
  // F64 测试
  {
    Array A = create_matrix_f64(3, 2, {1, 2, 3, 4, 5, 6});
    auto [Q, R] = qr(A, "reduced");
    EXPECT_EQ(Q.shape(), Shape({3, 2}));
    EXPECT_EQ(R.shape(), Shape({2, 2}));

    Array QR = matmul(Q, R);
    EXPECT_TRUE(check_matrix_equal(A, QR, 1e-5));
  }

  // F32 测试
  {
    Array A = create_matrix_f32(3, 2, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto [Q, R] = qr(A, "reduced");
    EXPECT_EQ(Q.shape(), Shape({3, 2}));
    EXPECT_EQ(R.shape(), Shape({2, 2}));

    Array QR = matmul(Q, R);
    EXPECT_TRUE(check_matrix_equal(A, QR, 1e-4f));
  }
}

// 可选：随机矩阵测试（同时测试两种类型）
TEST_F(LinalgTest, QRSquareRandom) {
  // F64 测试
  {
    Array A = randn({4, 4}, DType::F64);
    auto [Q, R] = qr(A, "complete");

    // Check Q^T * Q = I
    Array QTQ = matmul(transpose(Q), Q);
    const double *qtq_data = QTQ.data<double>();
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        double expected = (i == j) ? 1.0 : 0.0;
        EXPECT_NEAR(qtq_data[i * 4 + j], expected, 1e-5);
      }
    }

    // Check A = Q * R
    Array QR = matmul(Q, R);
    EXPECT_TRUE(check_matrix_equal(A, QR, 1e-5));
  }

  // F32 测试
  {
    Array A = randn({4, 4}, DType::F32);
    auto [Q, R] = qr(A, "complete");

    Array QTQ = matmul(transpose(Q), Q);
    const float *qtq_data = QTQ.data<float>();
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        float expected = (i == j) ? 1.0f : 0.0f;
        EXPECT_NEAR(qtq_data[i * 4 + j], expected, 1e-4f);
      }
    }

    Array QR = matmul(Q, R);
    EXPECT_TRUE(check_matrix_equal(A, QR, 1e-4f));
  }
}

// ============================================================================
// svd tests
// ============================================================================

TEST_F(LinalgTest, SVD3x3) {
  {
    Array A = create_matrix_f64(3, 3, {1, 0, 0, 0, 2, 0, 0, 0, 3});
    auto [U, S, VT] = svd(A, false);

    const double *s = S.data<double>();
    EXPECT_NEAR(s[0], 3.0, 1e-6);
    EXPECT_NEAR(s[1], 2.0, 1e-6);
    EXPECT_NEAR(s[2], 1.0, 1e-6);
  }
  {
    Array A = create_matrix_f32(
        3, 3, {1.0f, 0.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f, 0.0f, 3.0f});
    auto [U, S, VT] = svd(A, false);

    const float *s = S.data<float>();
    EXPECT_NEAR(s[0], 3.0f, 1e-6f);
    EXPECT_NEAR(s[1], 2.0f, 1e-6f);
    EXPECT_NEAR(s[2], 1.0f, 1e-6f);
  }
}

TEST_F(LinalgTest, SVDvals) {
  {
    Array A = create_matrix_f64(3, 3, {1, 0, 0, 0, 2, 0, 0, 0, 3});
    Array S = svdvals(A);

    const double *s = S.data<double>();
    EXPECT_NEAR(s[0], 3.0, 1e-6);
    EXPECT_NEAR(s[1], 2.0, 1e-6);
    EXPECT_NEAR(s[2], 1.0, 1e-6);
  }
  {
    Array A = create_matrix_f32(
        3, 3, {1.0f, 0.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f, 0.0f, 3.0f});
    Array S = svdvals(A);

    const float *s = S.data<float>();
    EXPECT_NEAR(s[0], 3.0f, 1e-6f);
    EXPECT_NEAR(s[1], 2.0f, 1e-6f);
    EXPECT_NEAR(s[2], 1.0f, 1e-6f);
  }
}

// ============================================================================
// eigh / eigvalsh tests
// ============================================================================

TEST_F(LinalgTest, Eigh3x3) {
  {
    Array A = create_matrix_f64(3, 3, {2, -1, 0, -1, 2, -1, 0, -1, 2});
    auto [vals, vecs] = eigh(A, "L");

    const double *v = vals.data<double>();
    EXPECT_NEAR(v[0], 0.5858, 1e-3);
    EXPECT_NEAR(v[1], 2.0, 1e-3);
    EXPECT_NEAR(v[2], 3.4142, 1e-3);
  }
  {
    Array A = create_matrix_f32(
        3, 3, {2.0f, -1.0f, 0.0f, -1.0f, 2.0f, -1.0f, 0.0f, -1.0f, 2.0f});
    auto [vals, vecs] = eigh(A, "L");

    const float *v = vals.data<float>();
    EXPECT_NEAR(v[0], 0.5858f, 1e-3f);
    EXPECT_NEAR(v[1], 2.0f, 1e-3f);
    EXPECT_NEAR(v[2], 3.4142f, 1e-3f);
  }
}

TEST_F(LinalgTest, Eigvalsh) {
  {
    Array A = create_matrix_f64(3, 3, {2, -1, 0, -1, 2, -1, 0, -1, 2});
    Array vals = eigvalsh(A, "L");

    const double *v = vals.data<double>();
    EXPECT_NEAR(v[0], 0.5858, 1e-3);
    EXPECT_NEAR(v[1], 2.0, 1e-3);
    EXPECT_NEAR(v[2], 3.4142, 1e-3);
  }
  {
    Array A = create_matrix_f32(3, 3, {2, -1, 0, -1, 2, -1, 0, -1, 2});
    Array vals = eigvalsh(A, "L");

    const float *v = vals.data<float>();
    EXPECT_NEAR(v[0], 0.5858f, 1e-3f);
    EXPECT_NEAR(v[1], 2.0f, 1e-3f);
    EXPECT_NEAR(v[2], 3.4142f, 1e-3f);
  }
}

// ============================================================================
// pinv tests
// ============================================================================

TEST_F(LinalgTest, Pinv2x2) {
  // Double precision
  Array A_f64 = create_matrix_f64(2, 2, {1, 2, 3, 4});
  Array pinvA_f64 = pinv(A_f64);

  Array I1_f64 = matmul(A_f64, pinvA_f64);
  Array I2_f64 = matmul(pinvA_f64, A_f64);

  EXPECT_NEAR(I1_f64.data<double>()[0], 1.0, 1e-5);
  EXPECT_NEAR(I1_f64.data<double>()[3], 1.0, 1e-5);
  EXPECT_NEAR(I2_f64.data<double>()[0], 1.0, 1e-5);
  EXPECT_NEAR(I2_f64.data<double>()[3], 1.0, 1e-5);

  // Single precision
  Array A_f32 = create_matrix_f32(2, 2, {1.0f, 2.0f, 3.0f, 4.0f});
  Array pinvA_f32 = pinv(A_f32);

  Array I1_f32 = matmul(A_f32, pinvA_f32);
  Array I2_f32 = matmul(pinvA_f32, A_f32);

  EXPECT_NEAR(I1_f32.data<float>()[0], 1.0f, 1e-5f);
  EXPECT_NEAR(I1_f32.data<float>()[3], 1.0f, 1e-5f);
  EXPECT_NEAR(I2_f32.data<float>()[0], 1.0f, 1e-5f);
  EXPECT_NEAR(I2_f32.data<float>()[3], 1.0f, 1e-5f);
}

TEST_F(LinalgTest, PinvRectangular) {
  // Double precision
  Array A_f64 = create_matrix_f64(3, 2, {1, 2, 3, 4, 5, 6});
  Array pinvA_f64 = pinv(A_f64);
  EXPECT_EQ(pinvA_f64.shape(), Shape({2, 3}));

  Array I_f64 = matmul(pinvA_f64, A_f64);
  EXPECT_NEAR(I_f64.data<double>()[0], 1.0, 1e-5);
  EXPECT_NEAR(I_f64.data<double>()[3], 1.0, 1e-5);

  // Single precision
  Array A_f32 = create_matrix_f32(3, 2, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  Array pinvA_f32 = pinv(A_f32);
  EXPECT_EQ(pinvA_f32.shape(), Shape({2, 3}));

  Array I_f32 = matmul(pinvA_f32, A_f32);
  EXPECT_NEAR(I_f32.data<float>()[0], 1.0f, 1e-5f);
  EXPECT_NEAR(I_f32.data<float>()[3], 1.0f, 1e-5f);
}

// ============================================================================
// dot / outer tests
// ============================================================================

TEST_F(LinalgTest, Dot) {
  {
    Array a = create_vector_f64({1.0, 2.0, 3.0});
    Array b = create_vector_f64({4.0, 5.0, 6.0});
    Array d = dot(a, b);
    EXPECT_NEAR(d.item<double>(), 32.0, 1e-6);
  }
  {
    Array a = create_vector_f32({1.0f, 2.0f, 3.0f});
    Array b = create_vector_f32({4.0f, 5.0f, 6.0f});
    Array d = dot(a, b);
    EXPECT_NEAR(d.item<float>(), 32.0f, 1e-6f);
  }
}

TEST_F(LinalgTest, Outer) {
  {
    Array a = create_vector_f64({1.0, 2.0, 3.0});
    Array b = create_vector_f64({4.0, 5.0, 6.0});
    Array o = outer(a, b);

    const double *data = o.data<double>();
    EXPECT_NEAR(data[0], 4.0, 1e-6);
    EXPECT_NEAR(data[1], 5.0, 1e-6);
    EXPECT_NEAR(data[2], 6.0, 1e-6);
    EXPECT_NEAR(data[3], 8.0, 1e-6);
    EXPECT_NEAR(data[4], 10.0, 1e-6);
    EXPECT_NEAR(data[5], 12.0, 1e-6);
    EXPECT_NEAR(data[6], 12.0, 1e-6);
    EXPECT_NEAR(data[7], 15.0, 1e-6);
    EXPECT_NEAR(data[8], 18.0, 1e-6);
  }
  {
    Array a = create_vector_f32({1.0f, 2.0f, 3.0f});
    Array b = create_vector_f32({4.0f, 5.0f, 6.0f});
    Array o = outer(a, b);

    const float *data = o.data<float>();
    EXPECT_NEAR(data[0], 4.0f, 1e-6f);
    EXPECT_NEAR(data[1], 5.0f, 1e-6f);
    EXPECT_NEAR(data[2], 6.0f, 1e-6f);
    EXPECT_NEAR(data[3], 8.0f, 1e-6f);
    EXPECT_NEAR(data[4], 10.0f, 1e-6f);
    EXPECT_NEAR(data[5], 12.0f, 1e-6f);
    EXPECT_NEAR(data[6], 12.0f, 1e-6f);
    EXPECT_NEAR(data[7], 15.0f, 1e-6f);
    EXPECT_NEAR(data[8], 18.0f, 1e-6f);
  }
}

// ============================================================================
// matrix_power tests
// ============================================================================

TEST_F(LinalgTest, MatrixPower2x2) {
  {
    Array A = create_matrix_f64(2, 2, {1, 2, 3, 4});
    Array A2 = matrix_power(A, 2);

    const double *data = A2.data<double>();
    EXPECT_NEAR(data[0], 7.0, 1e-6);
    EXPECT_NEAR(data[1], 10.0, 1e-6);
    EXPECT_NEAR(data[2], 15.0, 1e-6);
    EXPECT_NEAR(data[3], 22.0, 1e-6);
  }
  {
    Array A = create_matrix_f32(2, 2, {1.0f, 2.0f, 3.0f, 4.0f});
    Array A2 = matrix_power(A, 2);

    const float *data = A2.data<float>();
    EXPECT_NEAR(data[0], 7.0f, 1e-5f);
    EXPECT_NEAR(data[1], 10.0f, 1e-5f);
    EXPECT_NEAR(data[2], 15.0f, 1e-5f);
    EXPECT_NEAR(data[3], 22.0f, 1e-5f);
  }
}

TEST_F(LinalgTest, MatrixPowerZero) {
  {
    Array A = create_matrix_f64(2, 2, {1, 2, 3, 4});
    Array A0 = matrix_power(A, 0);

    const double *data = A0.data<double>();
    EXPECT_NEAR(data[0], 1.0, 1e-6);
    EXPECT_NEAR(data[1], 0.0, 1e-6);
    EXPECT_NEAR(data[2], 0.0, 1e-6);
    EXPECT_NEAR(data[3], 1.0, 1e-6);
  }
  {
    Array A = create_matrix_f32(2, 2, {1.0f, 2.0f, 3.0f, 4.0f});
    Array A0 = matrix_power(A, 0);

    const float *data = A0.data<float>();
    EXPECT_NEAR(data[0], 1.0f, 1e-5f);
    EXPECT_NEAR(data[1], 0.0f, 1e-5f);
    EXPECT_NEAR(data[2], 0.0f, 1e-5f);
    EXPECT_NEAR(data[3], 1.0f, 1e-5f);
  }
}

// ============================================================================
// matrix_rank tests
// ============================================================================

TEST_F(LinalgTest, MatrixRankFull) {
  {
    Array A = create_matrix_f64(3, 3, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    Array r = matrix_rank(A);
    EXPECT_EQ(r.item<int64_t>(), 2);
  }
  {
    Array A = create_matrix_f32(
        3, 3, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
    Array r = matrix_rank(A);
    EXPECT_EQ(r.item<int64_t>(), 2);
  }
}

TEST_F(LinalgTest, MatrixRankFullRank) {
  {
    Array A = create_matrix_f64(3, 3, {1, 2, 3, 2, 5, 3, 1, 0, 8});
    Array r = matrix_rank(A);
    EXPECT_EQ(r.item<int64_t>(), 3);
  }
  {
    Array A = create_matrix_f32(
        3, 3, {1.0f, 2.0f, 3.0f, 2.0f, 5.0f, 3.0f, 1.0f, 0.0f, 8.0f});
    Array r = matrix_rank(A);
    EXPECT_EQ(r.item<int64_t>(), 3);
  }
}

// ============================================================================
// trace tests
// ============================================================================

TEST_F(LinalgTest, Trace) {
  {
    Array A = create_matrix_f64(3, 3, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    Array t = trace(A);
    EXPECT_NEAR(t.item<double>(), 15.0, 1e-6);
  }
  {
    Array A = create_matrix_f32(
        3, 3, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
    Array t = trace(A);
    EXPECT_NEAR(t.item<float>(), 15.0f, 1e-5f);
  }
}

// ============================================================================
// norm tests
// ============================================================================

TEST_F(LinalgTest, NormVector2) {
  {
    Array v = create_vector_f64({3.0, 4.0});
    Array n = norm(v);
    EXPECT_NEAR(n.item<double>(), 5.0, 1e-6);
  }
  {
    Array v = create_vector_f32({3.0f, 4.0f});
    Array n = norm(v);
    EXPECT_NEAR(n.item<float>(), 5.0f, 1e-5f);
  }
}

TEST_F(LinalgTest, NormMatrixFrobenius) {
  {
    Array A = create_matrix_f64(2, 2, {1, 2, 3, 4});
    Array n = norm(A, 2);
    double expected = std::sqrt(1.0 + 4.0 + 9.0 + 16.0);
    EXPECT_NEAR(n.item<double>(), expected, 1e-6);
  }
  {
    Array A = create_matrix_f32(2, 2, {1.0f, 2.0f, 3.0f, 4.0f});
    Array n = norm(A, 2);
    float expected = std::sqrt(1.0f + 4.0f + 9.0f + 16.0f);
    EXPECT_NEAR(n.item<float>(), expected, 1e-5f);
  }
}

TEST_F(LinalgTest, NormMatrix1) {
  {
    Array A = create_matrix_f64(2, 2, {1, 2, 3, 4});
    Array n = norm(A, 1);
    EXPECT_NEAR(n.item<double>(), 6.0, 1e-6);
  }
  {
    Array A = create_matrix_f32(2, 2, {1.0f, 2.0f, 3.0f, 4.0f});
    Array n = norm(A, 1);
    EXPECT_NEAR(n.item<float>(), 6.0f, 1e-5f);
  }
}

TEST_F(LinalgTest, NormMatrixInf) {
  {
    Array A = create_matrix_f64(2, 2, {1, 2, 3, 4});
    Array n = norm(A, std::numeric_limits<double>::infinity());
    EXPECT_NEAR(n.item<double>(), 7.0, 1e-6);
  }
  {
    Array A = create_matrix_f32(2, 2, {1.0f, 2.0f, 3.0f, 4.0f});
    Array n = norm(A, std::numeric_limits<float>::infinity());
    EXPECT_NEAR(n.item<float>(), 7.0f, 1e-5f);
  }
}

// ============================================================================
// cond tests
// ============================================================================

TEST_F(LinalgTest, Cond) {
  {
    Array A = create_matrix_f64(2, 2, {1, 2, 3, 4});
    Array c = cond(A, 1);
    EXPECT_NEAR(c.item<double>(), 21.0, 1.0);
  }
  {
    Array A = create_matrix_f32(2, 2, {1.0f, 2.0f, 3.0f, 4.0f});
    Array c = cond(A, 1);
    EXPECT_NEAR(c.item<float>(), 21.0f, 1.0f);
  }
}

// ============================================================================
// solve_triangular tests
// ============================================================================

TEST_F(LinalgTest, SolveTriangularUpper) {
  {
    Array U = create_matrix_f64(3, 3, {1, 2, 3, 0, 4, 5, 0, 0, 6});
    Array b = create_vector_f64({1.0, 2.0, 3.0});
    Array x = solve_triangular(U, b, false, false);

    const double *data = x.data<double>();
    EXPECT_NEAR(data[0], -0.25, 1e-6);
    EXPECT_NEAR(data[1], -0.125, 1e-6);
    EXPECT_NEAR(data[2], 0.5, 1e-6);
  }
  {
    Array U = create_matrix_f32(
        3, 3, {1.0f, 2.0f, 3.0f, 0.0f, 4.0f, 5.0f, 0.0f, 0.0f, 6.0f});
    Array b = create_vector_f32({1.0f, 2.0f, 3.0f});
    Array x = solve_triangular(U, b, false, false);

    const float *data = x.data<float>();
    EXPECT_NEAR(data[0], -0.25f, 1e-5f);
    EXPECT_NEAR(data[1], -0.125f, 1e-5f);
    EXPECT_NEAR(data[2], 0.5f, 1e-5f);
  }
}

TEST_F(LinalgTest, SolveTriangularLower) {
  {
    Array L = create_matrix_f64(3, 3, {2, 0, 0, 1, 2, 0, 1, 1, 2});
    Array b = create_vector_f64({4.0, 5.0, 6.0});
    Array x = solve_triangular(L, b, true, false);

    const double *data = x.data<double>();
    EXPECT_NEAR(data[0], 2.0, 1e-6);
    EXPECT_NEAR(data[1], 1.5, 1e-6);
    EXPECT_NEAR(data[2], 1.25, 1e-6);
  }
  {
    Array L = create_matrix_f32(
        3, 3, {2.0f, 0.0f, 0.0f, 1.0f, 2.0f, 0.0f, 1.0f, 1.0f, 2.0f});
    Array b = create_vector_f32({4.0f, 5.0f, 6.0f});
    Array x = solve_triangular(L, b, true, false);

    const float *data = x.data<float>();
    EXPECT_NEAR(data[0], 2.0f, 1e-5f);
    EXPECT_NEAR(data[1], 1.5f, 1e-5f);
    EXPECT_NEAR(data[2], 1.25f, 1e-5f);
  }
}

// ============================================================================
// lstsq tests
// ============================================================================

TEST_F(LinalgTest, LstsqOverdetermined) {
  {
    Array A = create_matrix_f64(3, 2, {1, 1, 1, 2, 1, 3});
    Array b = create_vector_f64({2.0, 3.0, 4.0});
    Array x = lstsq(A, b);

    const double *data = x.data<double>();
    EXPECT_NEAR(data[0], 1.0, 1e-5);
    EXPECT_NEAR(data[1], 1.0, 1e-5);
  }
  {
    Array A = create_matrix_f32(3, 2, {1.0f, 1.0f, 1.0f, 2.0f, 1.0f, 3.0f});
    Array b = create_vector_f32({2.0f, 3.0f, 4.0f});
    Array x = lstsq(A, b);

    const float *data = x.data<float>();
    EXPECT_NEAR(data[0], 1.0f, 1e-4f);
    EXPECT_NEAR(data[1], 1.0f, 1e-4f);
  }
}

TEST_F(LinalgTest, LstsqUnderdetermined) {
  {
    Array A = create_matrix_f64(2, 3, {1, 0, 1, 0, 1, 1});
    Array b = create_vector_f64({2.0, 3.0});
    Array x = lstsq(A, b);

    EXPECT_EQ(x.shape(), Shape({3}));

    Array Ax = matmul(A, x);
    const double *ax_data = Ax.data<double>();
    EXPECT_NEAR(ax_data[0], 2.0, 1e-5);
    EXPECT_NEAR(ax_data[1], 3.0, 1e-5);
  }
  {
    Array A = create_matrix_f32(2, 3, {1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f});
    Array b = create_vector_f32({2.0f, 3.0f});
    Array x = lstsq(A, b);

    EXPECT_EQ(x.shape(), Shape({3}));

    Array Ax = matmul(A, x);
    const float *ax_data = Ax.data<float>();
    EXPECT_NEAR(ax_data[0], 2.0f, 1e-4f);
    EXPECT_NEAR(ax_data[1], 3.0f, 1e-4f);
  }
}

// ============================================================================
// lu tests
// ============================================================================

TEST_F(LinalgTest, LuDecomposition) {
  {
    Array A = create_matrix_f64(3, 3, {1, 2, 3, 2, 5, 3, 1, 0, 8});
    auto [LU, pivots] = lu(A, true);
    auto [P, L, U] = lu_unpack(LU, pivots);

    Array PA = matmul(P, A);
    Array LU_mat = matmul(L, U);

    EXPECT_TRUE(check_matrix_equal(PA, LU_mat, 1e-5));
  }
  {
    Array A = create_matrix_f32(
        3, 3, {1.0f, 2.0f, 3.0f, 2.0f, 5.0f, 3.0f, 1.0f, 0.0f, 8.0f});
    auto [LU, pivots] = lu(A, true);
    auto [P, L, U] = lu_unpack(LU, pivots);

    Array PA = matmul(P, A);
    Array LU_mat = matmul(L, U);

    EXPECT_TRUE(check_matrix_equal(PA, LU_mat, 1e-4f));
  }
}

// ============================================================================
// lq tests
// ============================================================================

TEST_F(LinalgTest, LqDecomposition) {
  {
    Array A = create_matrix_f64(3, 3, {1, 2, 3, 2, 5, 3, 1, 0, 8});
    auto [L, Q] = lq(A, "reduced");
    Array LQ = matmul(L, Q);
    EXPECT_TRUE(check_matrix_equal(A, LQ, 1e-5));
  }
  {
    Array A = create_matrix_f32(
        3, 3, {1.0f, 2.0f, 3.0f, 2.0f, 5.0f, 3.0f, 1.0f, 0.0f, 8.0f});
    auto [L, Q] = lq(A, "reduced");
    Array LQ = matmul(L, Q);
    EXPECT_TRUE(check_matrix_equal(A, LQ, 1e-4f));
  }
}

// ============================================================================
// Additional data type tests
// ============================================================================

TEST_F(LinalgTest, DetF32) {
  Array A = create_matrix_f32(2, 2, {1.0f, 2.0f, 3.0f, 4.0f});
  Array d = det(A);
  EXPECT_NEAR(d.item<float>(), -2.0f, 1e-5f);
}

TEST_F(LinalgTest, InvF32) {
  Array A = create_matrix_f32(2, 2, {1.0f, 2.0f, 3.0f, 4.0f});
  Array invA = inv(A);
  const float *data = invA.data<float>();
  EXPECT_NEAR(data[0], -2.0f, 1e-5f);
  EXPECT_NEAR(data[1], 1.0f, 1e-5f);
  EXPECT_NEAR(data[2], 1.5f, 1e-5f);
  EXPECT_NEAR(data[3], -0.5f, 1e-5f);
}

TEST_F(LinalgTest, SvdF32) {
  Array A = create_matrix_f32(
      3, 3, {1.0f, 0.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f, 0.0f, 3.0f});
  auto [U, S, VT] = svd(A, false);
  const float *s = S.data<float>();
  EXPECT_NEAR(s[0], 3.0f, 1e-4f);
  EXPECT_NEAR(s[1], 2.0f, 1e-4f);
  EXPECT_NEAR(s[2], 1.0f, 1e-4f);
}

TEST_F(LinalgTest, Cov) {
  std::vector<double> data = {1.0, 2.0, 3.0, 4.0,  5.0,  6.0,
                              7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
  {
    Array x = to_array(data, Shape({3, 4}));
    Array c = ins::cov(x);
    EXPECT_EQ(c.shape(), Shape({3, 3}));
    const double *c_data = c.data<double>();
    EXPECT_NEAR(c_data[0], 5.0 / 3.0, 1e-6);
    EXPECT_NEAR(c_data[1], 5.0 / 3.0, 1e-6);
  }
  {
    std::vector<float> data_f32 = {1.0f, 2.0f, 3.0f, 4.0f,  5.0f,  6.0f,
                                   7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    Array x = to_array(data_f32, Shape({3, 4}));
    Array c = ins::cov(x);
    EXPECT_EQ(c.shape(), Shape({3, 3}));
    const float *c_data = c.data<float>();
    EXPECT_NEAR(c_data[0], 5.0f / 3.0f, 1e-4f);
    EXPECT_NEAR(c_data[1], 5.0f / 3.0f, 1e-4f);
  }
}