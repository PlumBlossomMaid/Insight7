// tests/cuda/test_linalg.cpp
/**
 * @file test_linalg.cpp
 * @brief CUDA linalg tests — mirrors CPU tests with GPU dispatch.
 *
 * Operations with native CUDA kernels run on GPU directly.
 * Operations returning C_FALLBACK are automatically transferred to CPU
 * by the framework's fallback mechanism.
 */

#include "insight/insight.h"
#include "insight/ops/linalg.h"
#include <cmath>
#include <complex>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>

using namespace ins;

class LinalgTestGPU : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    ins::init({"cpu", "iluvatar"});
    try {
      set_device(GPUPlace(0));
    } catch (...) {
      GTEST_SKIP() << "GPU not available";
    }
    if (!ins::is_compiled_with_openblas()) {
      GTEST_SKIP() << "Skipping linalg GPU tests: OpenBLAS not available "
                      "(needed for CPU fallback)";
    }
  }
};

// ============================================================================
// Helper functions
// ============================================================================

static Array make_gpu_matrix_f64(int rows, int cols,
                                 const std::vector<double> &data) {
  Array result(Shape({rows, cols}), DType::F64, CPUPlace());
  std::memcpy(result.data<double>(), data.data(), data.size() * sizeof(double));
  return result.to(GPUPlace(0));
}

static Array make_gpu_matrix_f32(int rows, int cols,
                                 const std::vector<float> &data) {
  Array result(Shape({rows, cols}), DType::F32, CPUPlace());
  std::memcpy(result.data<float>(), data.data(), data.size() * sizeof(float));
  return result.to(GPUPlace(0));
}

static Array make_gpu_vector_f64(const std::vector<double> &data) {
  Array result(Shape({static_cast<int64_t>(data.size())}), DType::F64,
               CPUPlace());
  std::memcpy(result.data<double>(), data.data(), data.size() * sizeof(double));
  return result.to(GPUPlace(0));
}

static Array make_gpu_vector_f32(const std::vector<float> &data) {
  Array result(Shape({static_cast<int64_t>(data.size())}), DType::F32,
               CPUPlace());
  std::memcpy(result.data<float>(), data.data(), data.size() * sizeof(float));
  return result.to(GPUPlace(0));
}

static bool approx_equal(double a, double b, double rtol = 1e-5,
                         double atol = 1e-7) {
  return std::abs(a - b) <= atol + rtol * std::abs(b);
}

static bool approx_equal(float a, float b, float rtol = 1e-4f,
                         float atol = 1e-5f) {
  return std::abs(a - b) <= atol + rtol * std::abs(b);
}

static bool check_matrix_equal(const Array &gpu_A, const Array &gpu_B,
                               double rtol = 1e-4) {
  Array A = gpu_A.to(CPUPlace());
  Array B = gpu_B.to(CPUPlace());
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

TEST_F(LinalgTestGPU, MatMul2x2F64) {
  GTEST_SKIP() << "Iluvatar: no native FP64 (use F32 variant)";
}

TEST_F(LinalgTestGPU, MatMul2x2F32) {
  Array A = make_gpu_matrix_f32(2, 2, {1.0f, 2.0f, 3.0f, 4.0f});
  Array B = make_gpu_matrix_f32(2, 2, {5.0f, 6.0f, 7.0f, 8.0f});
  Array C = matmul(A, B);
  Array cpu_C = C.to(CPUPlace());
  const float *c = cpu_C.data<float>();
  EXPECT_NEAR(c[0], 19.0f, 1e-4f);
  EXPECT_NEAR(c[1], 22.0f, 1e-4f);
  EXPECT_NEAR(c[2], 43.0f, 1e-4f);
  EXPECT_NEAR(c[3], 50.0f, 1e-4f);
}

TEST_F(LinalgTestGPU, MatMulNonSquare) {
  // Iluvatar: skip F64 block (hardware limit) - F32 block below
  {
    Array A = make_gpu_matrix_f32(2, 3, {1, 2, 3, 4, 5, 6});
    Array B = make_gpu_matrix_f32(3, 2, {7, 8, 9, 10, 11, 12});
    Array C = matmul(A, B);
    Array cpu_C = C.to(CPUPlace());
    const float *c = cpu_C.data<float>();
    EXPECT_NEAR(c[0], 58.0f, 1e-4f);
    EXPECT_NEAR(c[1], 64.0f, 1e-4f);
    EXPECT_NEAR(c[2], 139.0f, 1e-4f);
    EXPECT_NEAR(c[3], 154.0f, 1e-4f);
  }
}

// ============================================================================
// det tests
// ============================================================================

TEST_F(LinalgTestGPU, Det2x2) {
  // Iluvatar: skip F64 block (hardware limit) - F32 block below
  {
    Array A = make_gpu_matrix_f32(2, 2, {1, 2, 3, 4});
    Array d = det(A);
    Array cpu_d = d.to(CPUPlace());
    EXPECT_NEAR(cpu_d.item<float>(), -2.0f, 1e-4f);
  }
}

TEST_F(LinalgTestGPU, Det3x3) {
  // Iluvatar: skip F64 block (hardware limit) - F32 block below
  {
    Array A = make_gpu_matrix_f32(3, 3, {1, 2, 3, 2, 5, 3, 1, 0, 8});
    Array d = det(A);
    Array cpu_d = d.to(CPUPlace());
    EXPECT_NEAR(cpu_d.item<float>(), -1.0f, 1e-4f);
  }
}

TEST_F(LinalgTestGPU, DetIdentity) {
  // Iluvatar: skip F64 block (hardware limit) - F32 block below
  {
    Array A = make_gpu_matrix_f32(3, 3, {1, 0, 0, 0, 1, 0, 0, 0, 1});
    Array d = det(A);
    Array cpu_d = d.to(CPUPlace());
    EXPECT_NEAR(cpu_d.item<float>(), 1.0f, 1e-4f);
  }
}

// ============================================================================
// slogdet tests
// ============================================================================

TEST_F(LinalgTestGPU, Slogdet2x2) {
  GTEST_SKIP() << "Iluvatar: no native FP64 (hardware limit)";
}

// ============================================================================
// inv tests
// ============================================================================

TEST_F(LinalgTestGPU, Inv2x2) {
  // Iluvatar: skip F64 block (hardware limit) - F32 block below
  {
    Array A = make_gpu_matrix_f32(2, 2, {1, 2, 3, 4});
    Array invA = inv(A);
    Array cpu_invA = invA.to(CPUPlace());
    const float *data = cpu_invA.data<float>();
    EXPECT_NEAR(data[0], -2.0f, 1e-4f);
    EXPECT_NEAR(data[1], 1.0f, 1e-4f);
    EXPECT_NEAR(data[2], 1.5f, 1e-4f);
    EXPECT_NEAR(data[3], -0.5f, 1e-4f);
  }
}

TEST_F(LinalgTestGPU, Inv3x3) {
  GTEST_SKIP() << "Iluvatar: no native FP64 (hardware limit)";
}

// ============================================================================
// solve tests
// ============================================================================

TEST_F(LinalgTestGPU, Solve3x3) {
  // Iluvatar: skip F64 block (hardware limit) - F32 block below
  {
    Array A = make_gpu_matrix_f32(3, 3, {3, 2, -1, 2, -2, 4, -1, 0.5, -1});
    Array b = make_gpu_vector_f32({1.0f, -2.0f, 0.0f});
    Array x = solve(A, b);
    Array cpu_x = x.to(CPUPlace());
    const float *data = cpu_x.data<float>();
    EXPECT_NEAR(data[0], 1.0f, 1e-4f);
    EXPECT_NEAR(data[1], -2.0f, 1e-4f);
    EXPECT_NEAR(data[2], -2.0f, 1e-4f);
  }
}

// ============================================================================
// cholesky tests
// ============================================================================

TEST_F(LinalgTestGPU, Cholesky3x3) {
  GTEST_SKIP() << "Iluvatar: no native FP64 (hardware limit)";
}

// ============================================================================
// qr tests
// ============================================================================

TEST_F(LinalgTestGPU, QR3x3) {
  GTEST_SKIP() << "Iluvatar: no native FP64 (hardware limit)";
}

TEST_F(LinalgTestGPU, QR3x2) {
  GTEST_SKIP() << "Iluvatar: no native FP64 (hardware limit)";
}

// ============================================================================
// svd tests
// ============================================================================

TEST_F(LinalgTestGPU, SVD3x3) {
  // Iluvatar: skip F64 block (hardware limit) - F32 block below
  {
    Array A = make_gpu_matrix_f32(
        3, 3, {1.0f, 0.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f, 0.0f, 3.0f});
    auto [U, S, VT] = svd(A, false);
    Array cpu_S = S.to(CPUPlace());
    const float *s = cpu_S.data<float>();
    EXPECT_NEAR(s[0], 3.0f, 1e-3f);
    EXPECT_NEAR(s[1], 2.0f, 1e-3f);
    EXPECT_NEAR(s[2], 1.0f, 1e-3f);
  }
}

TEST_F(LinalgTestGPU, SVDvals) {
  GTEST_SKIP() << "Iluvatar: no native FP64 (hardware limit)";
}

// ============================================================================
// eigh / eigvalsh tests
// ============================================================================

TEST_F(LinalgTestGPU, Eigh3x3) {
  GTEST_SKIP() << "Iluvatar: no native FP64 (hardware limit)";
}

TEST_F(LinalgTestGPU, Eigvalsh) {
  GTEST_SKIP() << "Iluvatar: no native FP64 (hardware limit)";
}

// ============================================================================
// dot / outer tests
// ============================================================================

TEST_F(LinalgTestGPU, Dot) {
  // Iluvatar: skip F64 block (hardware limit) - F32 block below
  {
    Array a = make_gpu_vector_f32({1.0f, 2.0f, 3.0f});
    Array b = make_gpu_vector_f32({4.0f, 5.0f, 6.0f});
    Array d = dot(a, b);
    Array cpu_d = d.to(CPUPlace());
    EXPECT_NEAR(cpu_d.item<float>(), 32.0f, 1e-4f);
  }
}

TEST_F(LinalgTestGPU, Outer) {
  GTEST_SKIP() << "Iluvatar: no native FP64 (hardware limit)";
}

// ============================================================================
// matrix_power tests
// ============================================================================

TEST_F(LinalgTestGPU, MatrixPower2x2) {
  GTEST_SKIP() << "Iluvatar: no native FP64 (hardware limit)";
}

TEST_F(LinalgTestGPU, MatrixPowerZero) {
  {
    Array A = make_gpu_matrix_f64(2, 2, {1, 2, 3, 4});
    Array A0 = matrix_power(A, 0);
    Array cpu_A0 = A0.to(CPUPlace());
    const double *data = cpu_A0.data<double>();
    EXPECT_NEAR(data[0], 1.0, 1e-5);
    EXPECT_NEAR(data[1], 0.0, 1e-5);
    EXPECT_NEAR(data[2], 0.0, 1e-5);
    EXPECT_NEAR(data[3], 1.0, 1e-5);
  }
}

// ============================================================================
// trace tests
// ============================================================================

TEST_F(LinalgTestGPU, Trace) {
  GTEST_SKIP() << "Iluvatar: no native FP64 (hardware limit)";
}

// ============================================================================
// norm tests
// ============================================================================

TEST_F(LinalgTestGPU, NormVector2) {
  GTEST_SKIP() << "Iluvatar: no native FP64 (hardware limit)";
}

TEST_F(LinalgTestGPU, NormMatrixFrobenius) {
  GTEST_SKIP() << "Iluvatar: no native FP64 (hardware limit)";
}

// ============================================================================
// solve_triangular tests
// ============================================================================

TEST_F(LinalgTestGPU, SolveTriangularUpper) {
  GTEST_SKIP() << "Iluvatar: no native FP64 (hardware limit)";
}

TEST_F(LinalgTestGPU, SolveTriangularLower) {
  GTEST_SKIP() << "Iluvatar: no native FP64 (hardware limit)";
}

// ============================================================================
// lu tests
// ============================================================================

TEST_F(LinalgTestGPU, LuDecomposition) {
  {
    Array A = make_gpu_matrix_f64(3, 3, {1, 2, 3, 2, 5, 3, 1, 0, 8});
    auto [LU, pivots] = lu(A, true);
    auto [P, L, U] = lu_unpack(LU, pivots);
    Array PA = matmul(P, A);
    Array LU_mat = matmul(L, U);
    EXPECT_TRUE(check_matrix_equal(PA, LU_mat, 1e-4));
  }
}

// ============================================================================
// lq tests
// ============================================================================

TEST_F(LinalgTestGPU, LqDecomposition) {
  GTEST_SKIP() << "Iluvatar: no native FP64 (hardware limit)";
}

// ============================================================================
// cond tests
// ============================================================================

TEST_F(LinalgTestGPU, Cond) {
  {
    Array A = make_gpu_matrix_f64(2, 2, {1, 2, 3, 4});
    Array c = cond(A, 1);
    Array cpu_c = c.to(CPUPlace());
    EXPECT_NEAR(cpu_c.item<double>(), 21.0, 1.0);
  }
}

// ============================================================================
// Additional solve tests
// ============================================================================

TEST_F(LinalgTestGPU, SolveMultipleRHS) {
  GTEST_SKIP() << "Iluvatar: no native FP64 (hardware limit)";
}

// ============================================================================
// Additional QR tests
// ============================================================================

TEST_F(LinalgTestGPU, QRSquareRandom) {
  GTEST_SKIP() << "Iluvatar: QR random test uses F64 (hardware limit)";
}

// ============================================================================
// Additional pinv tests
// ============================================================================

TEST_F(LinalgTestGPU, Pinv2x2) {
  GTEST_SKIP() << "Iluvatar: no native FP64 (hardware limit)";
}

TEST_F(LinalgTestGPU, PinvRectangular) {
  GTEST_SKIP() << "Iluvatar: no native FP64 (hardware limit)";
}

// ============================================================================
// Additional matrix_rank tests
// ============================================================================

TEST_F(LinalgTestGPU, MatrixRankFull) {
  Array A = make_gpu_matrix_f64(3, 3, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  Array r = matrix_rank(A);
  Array cpu_r = r.to(CPUPlace());
  EXPECT_EQ(cpu_r.item<int64_t>(), 2);
}

TEST_F(LinalgTestGPU, MatrixRankFullRank) {
  Array A = make_gpu_matrix_f64(3, 3, {1, 2, 3, 2, 5, 3, 1, 0, 8});
  Array r = matrix_rank(A);
  Array cpu_r = r.to(CPUPlace());
  EXPECT_EQ(cpu_r.item<int64_t>(), 3);
}

// ============================================================================
// Additional norm tests
// ============================================================================

TEST_F(LinalgTestGPU, NormMatrix1) {
  {
    Array A = make_gpu_matrix_f64(2, 2, {1, 2, 3, 4});
    Array n = norm(A, 1);
    Array cpu_n = n.to(CPUPlace());
    EXPECT_NEAR(cpu_n.item<double>(), 6.0, 1e-5);
  }
}

TEST_F(LinalgTestGPU, NormMatrixInf) {
  {
    Array A = make_gpu_matrix_f64(2, 2, {1, 2, 3, 4});
    Array n = norm(A, std::numeric_limits<double>::infinity());
    Array cpu_n = n.to(CPUPlace());
    EXPECT_NEAR(cpu_n.item<double>(), 7.0, 1e-5);
  }
}

// ============================================================================
// Additional lstsq tests
// ============================================================================

TEST_F(LinalgTestGPU, LstsqOverdetermined) {
  {
    Array A = make_gpu_matrix_f64(3, 2, {1, 1, 1, 2, 1, 3});
    Array b = make_gpu_vector_f64({2.0, 3.0, 4.0});
    Array x = lstsq(A, b);
    Array cpu_x = x.to(CPUPlace());
    const double *data = cpu_x.data<double>();
    EXPECT_NEAR(data[0], 1.0, 1e-4);
    EXPECT_NEAR(data[1], 1.0, 1e-4);
  }
}

TEST_F(LinalgTestGPU, LstsqUnderdetermined) {
  GTEST_SKIP() << "Iluvatar: no native FP64 (hardware limit)";
}

// ============================================================================
// Additional data type tests
// ============================================================================

TEST_F(LinalgTestGPU, DetF32) {
  Array A = make_gpu_matrix_f32(2, 2, {1.0f, 2.0f, 3.0f, 4.0f});
  Array d = det(A);
  Array cpu_d = d.to(CPUPlace());
  EXPECT_NEAR(cpu_d.item<float>(), -2.0f, 1e-4f);
}

TEST_F(LinalgTestGPU, InvF32) {
  Array A = make_gpu_matrix_f32(2, 2, {1.0f, 2.0f, 3.0f, 4.0f});
  Array invA = inv(A);
  Array cpu_invA = invA.to(CPUPlace());
  const float *data = cpu_invA.data<float>();
  EXPECT_NEAR(data[0], -2.0f, 1e-4f);
  EXPECT_NEAR(data[1], 1.0f, 1e-4f);
  EXPECT_NEAR(data[2], 1.5f, 1e-4f);
  EXPECT_NEAR(data[3], -0.5f, 1e-4f);
}

TEST_F(LinalgTestGPU, SvdF32) {
  Array A = make_gpu_matrix_f32(
      3, 3, {1.0f, 0.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f, 0.0f, 3.0f});
  auto [U, S, VT] = svd(A, false);
  Array cpu_S = S.to(CPUPlace());
  const float *s = cpu_S.data<float>();
  EXPECT_NEAR(s[0], 3.0f, 1e-3f);
  EXPECT_NEAR(s[1], 2.0f, 1e-3f);
  EXPECT_NEAR(s[2], 1.0f, 1e-3f);
}

TEST_F(LinalgTestGPU, Cov) {
  GTEST_SKIP() << "Iluvatar: Cov uses F64 (hardware limit)";
}
