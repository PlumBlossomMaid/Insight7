// backends/cpu/kernels/linalg/cholesky_solve.cpp
/**
 * @file cholesky_solve.cpp
 * @brief CPU kernel for solving linear systems using Cholesky factor.
 */

#ifdef INSIGHT_USE_OPENBLAS

#include "common.h"

static void cholesky_solve_f32(const float *A, const float *B, float *X, int n,
                               int nrhs, int lower) {
  float *a = (float *)malloc(n * n * sizeof(float));
  memcpy(a, A, n * n * sizeof(float));
  float *b = (float *)malloc(n * nrhs * sizeof(float));
  memcpy(b, B, n * nrhs * sizeof(float));
  char uplo = lower ? 'L' : 'U';
  int info;
  spotrs_(&uplo, &n, &nrhs, a, &n, b, &n, &info);
  memcpy(X, b, n * nrhs * sizeof(float));
  free(a);
  free(b);
  if (info != 0) {
    cpu_set_last_error("cholesky_solve: LAPACK spotrs failed");
  }
}

static void cholesky_solve_f64(const double *A, const double *B, double *X,
                               int n, int nrhs, int lower) {
  double *a = (double *)malloc(n * n * sizeof(double));
  memcpy(a, A, n * n * sizeof(double));
  double *b = (double *)malloc(n * nrhs * sizeof(double));
  memcpy(b, B, n * nrhs * sizeof(double));
  char uplo = lower ? 'L' : 'U';
  int info;
  dpotrs_(&uplo, &n, &nrhs, a, &n, b, &n, &info);
  memcpy(X, b, n * nrhs * sizeof(double));
  free(a);
  free(b);
  if (info != 0) {
    cpu_set_last_error("cholesky_solve: LAPACK dpotrs failed");
  }
}

#ifdef __cplusplus
extern "C" {
#endif

C_Status cholesky_solve_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];
  InsightArray *A = (InsightArray *)inputs[1];
  InsightArray *B = (InsightArray *)inputs[2];
  int lower = *(int *)inputs[3];

  if (!out || !A || !B) {
    cpu_set_last_error("cholesky_solve: null array pointer");
    return C_FAILED;
  }
  if (!cpu_ensure_contiguous(A) || !cpu_ensure_contiguous(B) ||
      !cpu_ensure_contiguous(out))
    return C_FAILED;

  int n = (int)A->dims[0];
  int nrhs = (B->ndim == 1) ? 1 : (int)B->dims[1];
  if (A->dims[1] != n) {
    cpu_set_last_error("cholesky_solve: A must be square");
    return C_FAILED;
  }

  if (A->dtype == INSIGHT_DTYPE_F32) {
    cholesky_solve_f32((float *)A->data, (float *)B->data, (float *)out->data,
                       n, nrhs, lower);
  } else {
    cholesky_solve_f64((double *)A->data, (double *)B->data,
                       (double *)out->data, n, nrhs, lower);
  }
  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(cholesky_solve, INSIGHT_DTYPE_F32,
                    cholesky_solve_kernel_cpu);
REGISTER_CPU_KERNEL(cholesky_solve, INSIGHT_DTYPE_F64,
                    cholesky_solve_kernel_cpu);

#endif
