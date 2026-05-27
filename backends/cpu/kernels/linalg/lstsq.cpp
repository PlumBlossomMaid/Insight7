// backends/cpu/kernels/linalg/lstsq.cpp
/**
 * @file lstsq.cpp
 * @brief CPU kernel for least squares solution using LAPACK.
 */

#ifdef INSIGHT_USE_OPENBLAS

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

static void lstsq_f64(const double *A, const double *B, double *X, int m, int n,
                      int nrhs) {
  double *a = (double *)malloc(m * n * sizeof(double));
  cpu_rowmajor_to_colmajor_f64(A, a, m, n);

  int b_rows = (m > n) ? m : n;
  double *b = (double *)malloc(b_rows * nrhs * sizeof(double));
  memset(b, 0, b_rows * nrhs * sizeof(double));
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < nrhs; ++j) {
      b[i + j * b_rows] = B[i * nrhs + j];
    }
  }

  char trans = 'N';
  double *work = (double *)malloc(1 * sizeof(double));
  int lwork = -1;
  int info;

  // Query workspace
  dgels_(&trans, &m, &n, &nrhs, a, &m, b, &b_rows, work, &lwork, &info);
  lwork = (int)work[0];
  work = (double *)realloc(work, lwork * sizeof(double));

  // Solve least squares
  dgels_(&trans, &m, &n, &nrhs, a, &m, b, &b_rows, work, &lwork, &info);

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < nrhs; ++j) {
      X[i * nrhs + j] = b[i + j * b_rows];
    }
  }

  free(a);
  free(b);
  free(work);

  if (info != 0) {
    cpu_set_last_error("lstsq: LAPACK dgels failed");
  }
}

static void lstsq_f32(const float *A, const float *B, float *X, int m, int n,
                      int nrhs) {
  double *A_f64 = (double *)malloc(m * n * sizeof(double));
  double *B_f64 = (double *)malloc(m * nrhs * sizeof(double));
  double *X_f64 = (double *)malloc(n * nrhs * sizeof(double));

  for (int i = 0; i < m * n; ++i)
    A_f64[i] = A[i];
  for (int i = 0; i < m * nrhs; ++i)
    B_f64[i] = B[i];

  lstsq_f64(A_f64, B_f64, X_f64, m, n, nrhs);

  for (int i = 0; i < n * nrhs; ++i)
    X[i] = (float)X_f64[i];

  free(A_f64);
  free(B_f64);
  free(X_f64);
}

C_Status lstsq_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];
  InsightArray *A = (InsightArray *)inputs[1];
  InsightArray *B = (InsightArray *)inputs[2];
  double rcond = *(double *)inputs[3]; // not used, kept for compatibility

  if (!out || !A || !B) {
    cpu_set_last_error("lstsq: null array pointer");
    return C_FAILED;
  }
  if (!cpu_ensure_contiguous(A) || !cpu_ensure_contiguous(B) ||
      !cpu_ensure_contiguous(out))
    return C_FAILED;

  int m = (int)A->dims[0];
  int n = (int)A->dims[1];
  int nrhs = (B->ndim == 1) ? 1 : (int)B->dims[1];

  if (A->dtype == INSIGHT_DTYPE_F32) {
    lstsq_f32((float *)A->data, (float *)B->data, (float *)out->data, m, n,
              nrhs);
  } else {
    lstsq_f64((double *)A->data, (double *)B->data, (double *)out->data, m, n,
              nrhs);
  }
  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(lstsq, INSIGHT_DTYPE_F32, lstsq_kernel_cpu);
REGISTER_CPU_KERNEL(lstsq, INSIGHT_DTYPE_F64, lstsq_kernel_cpu);

#endif
