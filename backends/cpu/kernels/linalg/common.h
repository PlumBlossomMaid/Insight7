// backends/cpu/kernels/linalg/common.h
/**
 * @file common.h
 * @brief Common utilities for CPU linear algebra kernels.
 */

#ifndef CPU_LINALG_COMMON_H
#define CPU_LINALG_COMMON_H

#include "../../registry/cpu_registry.h"
#include "insight/c_api/array.h"
#include <cstdio>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// 平台/库配置
// ============================================================================

// OpenBLAS 可选，用于加速 BLAS 运算
#ifdef INSIGHT_USE_OPENBLAS
#include <openblas/cblas.h>
// LAPACK Fortran interfaces
void dgemm_(char *transa, char *transb, int *m, int *n, int *k, double *alpha,
            double *a, int *lda, double *b, int *ldb, double *beta, double *c,
            int *ldc);
void sgemm_(char *transa, char *transb, int *m, int *n, int *k, float *alpha,
            float *a, int *lda, float *b, int *ldb, float *beta, float *c,
            int *ldc);
void dger_(int *m, int *n, double *alpha, double *x, int *incx, double *y,
           int *incy, double *a, int *lda);
void sger_(int *m, int *n, float *alpha, float *x, int *incx, float *y,
           int *incy, float *a, int *lda);
double ddot_(int *n, double *x, int *incx, double *y, int *incy);
float sdot_(int *n, float *x, int *incx, float *y, int *incy);

// LAPACK
void dgetrf_(int *m, int *n, double *a, int *lda, int *ipiv, int *info);
void sgetrf_(int *m, int *n, float *a, int *lda, int *ipiv, int *info);
void dgetri_(int *n, double *a, int *lda, int *ipiv, double *work, int *lwork,
             int *info);
void sgetri_(int *n, float *a, int *lda, int *ipiv, float *work, int *lwork,
             int *info);
void dgesv_(int *n, int *nrhs, double *a, int *lda, int *ipiv, double *b,
            int *ldb, int *info);
void sgesv_(int *n, int *nrhs, float *a, int *lda, int *ipiv, float *b,
            int *ldb, int *info);
void dgecon_(char *norm, int *n, double *a, int *lda, double *anorm,
             double *rcond, double *work, int *iwork, int *info);
void sgecon_(char *norm, int *n, float *a, int *lda, float *anorm, float *rcond,
             float *work, int *iwork, int *info);
double dlange_(char *norm, int *m, int *n, double *a, int *lda, double *work);
float slange_(char *norm, int *m, int *n, float *a, int *lda, float *work);
void dgeqrf_(int *m, int *n, double *a, int *lda, double *tau, double *work,
             int *lwork, int *info);
void sgeqrf_(int *m, int *n, float *a, int *lda, float *tau, float *work,
             int *lwork, int *info);
void dorgqr_(int *m, int *n, int *k, double *a, int *lda, double *tau,
             double *work, int *lwork, int *info);
void sorgqr_(int *m, int *n, int *k, float *a, int *lda, float *tau,
             float *work, int *lwork, int *info);
void dgelqf_(int *m, int *n, double *a, int *lda, double *tau, double *work,
             int *lwork, int *info);
void sgelqf_(int *m, int *n, float *a, int *lda, float *tau, float *work,
             int *lwork, int *info);
void dorglq_(int *m, int *n, int *k, double *a, int *lda, double *tau,
             double *work, int *lwork, int *info);
void sorglq_(int *m, int *n, int *k, float *a, int *lda, float *tau,
             float *work, int *lwork, int *info);
void dpotrf_(char *uplo, int *n, double *a, int *lda, int *info);
void spotrf_(char *uplo, int *n, float *a, int *lda, int *info);
void dpotrs_(char *uplo, int *n, int *nrhs, double *a, int *lda, double *b,
             int *ldb, int *info);
void spotrs_(char *uplo, int *n, int *nrhs, float *a, int *lda, float *b,
             int *ldb, int *info);
void dgesvd_(char *jobu, char *jobvt, int *m, int *n, double *a, int *lda,
             double *s, double *u, int *ldu, double *vt, int *ldvt,
             double *work, int *lwork, int *info);
void sgesvd_(char *jobu, char *jobvt, int *m, int *n, float *a, int *lda,
             float *s, float *u, int *ldu, float *vt, int *ldvt, float *work,
             int *lwork, int *info);
void dgeev_(char *jobvl, char *jobvr, int *n, double *a, int *lda, double *wr,
            double *wi, double *vl, int *ldvl, double *vr, int *ldvr,
            double *work, int *lwork, int *info);
void sgeev_(char *jobvl, char *jobvr, int *n, float *a, int *lda, float *wr,
            float *wi, float *vl, int *ldvl, float *vr, int *ldvr, float *work,
            int *lwork, int *info);
void dsyev_(char *jobz, char *uplo, int *n, double *a, int *lda, double *w,
            double *work, int *lwork, int *info);
void ssyev_(char *jobz, char *uplo, int *n, float *a, int *lda, float *w,
            float *work, int *lwork, int *info);
void dgels_(char *trans, int *m, int *n, int *nrhs, double *a, int *lda,
            double *b, int *ldb, double *work, int *lwork, int *info);
void sgels_(char *trans, int *m, int *n, int *nrhs, float *a, int *lda,
            float *b, int *ldb, float *work, int *lwork, int *info);
void dtrtrs_(char *uplo, char *trans, char *diag, int *n, int *nrhs, double *a,
             int *lda, double *b, int *ldb, int *info);
void strtrs_(char *uplo, char *trans, char *diag, int *n, int *nrhs, float *a,
             int *lda, float *b, int *ldb, int *info);

#define HAVE_BLAS_ACCEL 1
#else
#define HAVE_BLAS_ACCEL 0
#endif

// ============================================================================
// 内存布局转换 (行主序 ? 列主序)
// ============================================================================

/**
 * @brief Convert row-major to column-major (float) with multi-threading
 */
static inline void cpu_rowmajor_to_colmajor_f32(const float *src, float *dst,
                                                int rows, int cols) {
#pragma omp parallel for
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      dst[i + j * rows] = src[i * cols + j];
    }
  }
}

/**
 * @brief Convert row-major to column-major (double) with multi-threading
 */
static inline void cpu_rowmajor_to_colmajor_f64(const double *src, double *dst,
                                                int rows, int cols) {
#pragma omp parallel for
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      dst[i + j * rows] = src[i * cols + j];
    }
  }
}

/**
 * @brief Convert column-major to row-major (float) with multi-threading
 */
static inline void cpu_colmajor_to_rowmajor_f32(const float *src, float *dst,
                                                int rows, int cols) {
#pragma omp parallel for
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      dst[i * cols + j] = src[i + j * rows];
    }
  }
}

/**
 * @brief Convert column-major to row-major (double) with multi-threading
 */
static inline void cpu_colmajor_to_rowmajor_f64(const double *src, double *dst,
                                                int rows, int cols) {
#pragma omp parallel for
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      dst[i * cols + j] = src[i + j * rows];
    }
  }
}

// ============================================================================
// LAPACK 错误处理
// ============================================================================

/**
 * @brief Check LAPACK info code and set error message if needed
 */
static inline void cpu_check_lapack_info(int info, const char *func_name) {
  if (info < 0) {
    char msg[256];
    snprintf(msg, sizeof(msg), "%s: illegal value at argument %d", func_name,
             -info);
    cpu_set_last_error(msg);
  } else if (info > 0) {
    char msg[256];
    snprintf(msg, sizeof(msg),
             "%s: singular/not positive definite (U(%d,%d) is zero)", func_name,
             info, info);
    cpu_set_last_error(msg);
  }
}

// ============================================================================
// 确保数组连续（简单实现，仅检查）
// ============================================================================

/**
 * @brief Check if array is contiguous; if not, set error and return 0.
 */
static inline int cpu_ensure_contiguous(InsightArray *arr) {
  if (insight_array_is_contiguous(arr))
    return 1;
  cpu_set_last_error("non-contiguous array not supported for this operation");
  return 0;
}

#ifdef __cplusplus
}
#endif

#endif // CPU_LINALG_COMMON_H
