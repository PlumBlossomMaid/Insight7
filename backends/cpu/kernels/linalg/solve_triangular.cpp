// backends/cpu/kernels/linalg/solve_triangular.cpp
/**
 * @file solve_triangular.cpp
 * @brief CPU kernel for solving triangular systems.
 */

#ifdef INSIGHT_USE_OPENBLAS

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

static void solve_triangular_f32(const float *A, const float *B, float *X,
                                 int n, int nrhs, int lower, int unit_diag) {
  float *a = (float *)malloc(n * n * sizeof(float));
  cpu_rowmajor_to_colmajor_f32(A, a, n, n);

  float *b = (float *)malloc(n * nrhs * sizeof(float));
  cpu_rowmajor_to_colmajor_f32(B, b, n, nrhs);

  char uplo = lower ? 'L' : 'U';
  char trans = 'N';
  char diag = unit_diag ? 'U' : 'N';
  int info;

  strtrs_(&uplo, &trans, &diag, &n, &nrhs, a, &n, b, &n, &info);

  cpu_colmajor_to_rowmajor_f32(b, X, n, nrhs);

  free(a);
  free(b);

  if (info != 0) {
    cpu_set_last_error("solve_triangular: LAPACK strtrs failed");
  }
}

static void solve_triangular_f64(const double *A, const double *B, double *X,
                                 int n, int nrhs, int lower, int unit_diag) {
  double *a = (double *)malloc(n * n * sizeof(double));
  cpu_rowmajor_to_colmajor_f64(A, a, n, n);

  double *b = (double *)malloc(n * nrhs * sizeof(double));
  cpu_rowmajor_to_colmajor_f64(B, b, n, nrhs);

  char uplo = lower ? 'L' : 'U';
  char trans = 'N';
  char diag = unit_diag ? 'U' : 'N';
  int info;

  dtrtrs_(&uplo, &trans, &diag, &n, &nrhs, a, &n, b, &n, &info);

  cpu_colmajor_to_rowmajor_f64(b, X, n, nrhs);

  free(a);
  free(b);

  if (info != 0) {
    cpu_set_last_error("solve_triangular: LAPACK dtrtrs failed");
  }
}

C_Status solve_triangular_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];
  InsightArray *A = (InsightArray *)inputs[1];
  InsightArray *B = (InsightArray *)inputs[2];
  int lower = *(int *)inputs[3];
  int unit_diag = *(int *)inputs[4];

  if (!out || !A || !B) {
    cpu_set_last_error("solve_triangular: null array pointer");
    return C_FAILED;
  }
  if (!cpu_ensure_contiguous(A) || !cpu_ensure_contiguous(B) ||
      !cpu_ensure_contiguous(out))
    return C_FAILED;

  int n = (int)A->dims[0];
  int nrhs = (B->ndim == 1) ? 1 : (int)B->dims[1];
  if (A->dims[1] != n) {
    cpu_set_last_error("solve_triangular: A must be square");
    return C_FAILED;
  }

  if (A->dtype == INSIGHT_DTYPE_F32) {
    solve_triangular_f32((float *)A->data, (float *)B->data, (float *)out->data,
                         n, nrhs, lower, unit_diag);
  } else {
    solve_triangular_f64((double *)A->data, (double *)B->data,
                         (double *)out->data, n, nrhs, lower, unit_diag);
  }
  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(solve_triangular, INSIGHT_DTYPE_F32,
                    solve_triangular_kernel_cpu);
REGISTER_CPU_KERNEL(solve_triangular, INSIGHT_DTYPE_F64,
                    solve_triangular_kernel_cpu);

#endif
