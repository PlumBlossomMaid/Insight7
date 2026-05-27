// backends/cpu/kernels/linalg/lq.cpp
/**
 * @file lq.cpp
 * @brief CPU kernel for LQ decomposition using LAPACK.
 */

#ifdef INSIGHT_USE_OPENBLAS

#include "common.h"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif
/**
 * @brief LQ decomposition for double precision (column-major LAPACK interface)
 *
 * Computes LQ decomposition A = L * Q, where L is lower triangular and Q is
 * orthogonal.
 *
 * @param src Input matrix (row-major)
 * @param l Output L matrix (row-major, m x n, lower triangular)
 * @param q Output Q matrix (row-major, n x n for complete, n x m for reduced)
 * @param m Number of rows in input matrix
 * @param n Number of columns in input matrix
 * @param mode 0 = complete Q (n x n), 1 = reduced Q (n x m)
 */
static void lq_f64(const double *src, double *l, double *q, int m, int n,
                   int mode) {
  int k = m < n ? m : n;
  int lda = m;

  // Convert to column-major for LAPACK
  double *a = (double *)malloc(m * n * sizeof(double));
  cpu_rowmajor_to_colmajor_f64(src, a, m, n);

  double *tau = (double *)malloc(k * sizeof(double));
  double *work = (double *)malloc(1 * sizeof(double));
  int lwork = -1;
  int info;

  // Query and compute LQ decomposition
  dgelqf_(&m, &n, a, &lda, tau, work, &lwork, &info);
  lwork = (int)work[0];
  work = (double *)realloc(work, lwork * sizeof(double));
  dgelqf_(&m, &n, a, &lda, tau, work, &lwork, &info);

  // Extract L matrix (lower triangular part)
  memset(l, 0, m * n * sizeof(double));
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j <= i && j < n; ++j) {
      l[i * n + j] = a[i + j * m];
    }
  }

  // Build Q matrix from Householder reflectors (column-major)
  double *qmat = (double *)malloc(n * n * sizeof(double));
  memset(qmat, 0, n * n * sizeof(double));

  // Copy Householder vectors from A (each row contains the reflector)
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      qmat[i + j * n] = a[i + j * m];
    }
  }
  // Set diagonal to 1 for rows beyond m
  for (int i = m; i < n; ++i) {
    qmat[i + i * n] = 1.0;
  }

  // Generate Q matrix
  lwork = -1;
  dorglq_(&n, &n, &k, qmat, &n, tau, work, &lwork, &info);
  lwork = (int)work[0];
  work = (double *)realloc(work, lwork * sizeof(double));
  dorglq_(&n, &n, &k, qmat, &n, tau, work, &lwork, &info);

  // Convert Q from column-major to row-major
  if (mode == 0) { // Complete Q (n x n)
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        q[i * n + j] = qmat[i + j * n];
      }
    }
  } else { // Reduced Q (n x m) - take first m columns
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < m; ++j) {
        q[i * m + j] = qmat[i + j * n];
      }
    }
  }

  free(a);
  free(tau);
  free(work);
  free(qmat);
}

/**
 * @brief LQ decomposition for single precision (column-major LAPACK interface)
 *
 * Computes LQ decomposition A = L * Q, where L is lower triangular and Q is
 * orthogonal.
 *
 * @param src Input matrix (row-major)
 * @param l Output L matrix (row-major, m x n, lower triangular)
 * @param q Output Q matrix (row-major, n x n for complete, n x m for reduced)
 * @param m Number of rows in input matrix
 * @param n Number of columns in input matrix
 * @param mode 0 = complete Q (n x n), 1 = reduced Q (n x m)
 */
static void lq_f32(const float *src, float *l, float *q, int m, int n,
                   int mode) {
  double *src_f64 = (double *)malloc(m * n * sizeof(double));
  double *l_f64 = (double *)malloc(m * n * sizeof(double));
  double *q_f64 = (double *)malloc(n * n * sizeof(double));

  for (int i = 0; i < m * n; ++i) {
    src_f64[i] = src[i];
  }

  lq_f64(src_f64, l_f64, q_f64, m, n, mode);

  for (int i = 0; i < m * n; ++i) {
    l[i] = (float)l_f64[i];
  }

  int q_size = (mode == 0) ? n * n : n * m;
  for (int i = 0; i < q_size; ++i) {
    q[i] = (float)q_f64[i];
  }

  free(src_f64);
  free(l_f64);
  free(q_f64);
}

C_Status lq_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *L = (InsightArray *)outputs[0];
  InsightArray *Q = (InsightArray *)outputs[1];
  InsightArray *x = (InsightArray *)inputs[2];
  int mode = *(int *)inputs[3];

  if (!L || !Q || !x) {
    cpu_set_last_error("lq: null array pointer");
    return C_FAILED;
  }
  if (!cpu_ensure_contiguous(x) || !cpu_ensure_contiguous(L) ||
      !cpu_ensure_contiguous(Q))
    return C_FAILED;

  int m = (int)x->dims[0];
  int n = (int)x->dims[1];

  if (x->dtype == INSIGHT_DTYPE_F32) {
    lq_f32((float *)x->data, (float *)L->data, (float *)Q->data, m, n, mode);
  } else {
    lq_f64((double *)x->data, (double *)L->data, (double *)Q->data, m, n, mode);
  }
  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(lq, INSIGHT_DTYPE_F32, lq_kernel_cpu);
REGISTER_CPU_KERNEL(lq, INSIGHT_DTYPE_F64, lq_kernel_cpu);

#endif
