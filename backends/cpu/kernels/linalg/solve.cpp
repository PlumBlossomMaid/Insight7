// backends/cpu/kernels/linalg/solve.cpp
/**
 * @file solve.cpp
 * @brief CPU kernel for solving linear systems AX = B using LAPACK.
 */

#ifdef INSIGHT_USE_OPENBLAS

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

static void solve_f32(const float *A, const float *B, float *X, int n,
                      int nrhs) {
  // 将 A 从行主序转换为列主序（LAPACK 期望列主序）
  float *a = (float *)malloc(n * n * sizeof(float));
  cpu_rowmajor_to_colmajor_f32(A, a, n, n);

  // 将 B 从行主序转换为列主序
  float *b = (float *)malloc(n * nrhs * sizeof(float));
  cpu_rowmajor_to_colmajor_f32(B, b, n, nrhs);

  int *ipiv = (int *)malloc(n * sizeof(int));
  int info;
  sgesv_(&n, &nrhs, a, &n, ipiv, b, &n, &info);

  // 将结果从列主序转换回行主序
  cpu_colmajor_to_rowmajor_f32(b, X, n, nrhs);

  free(a);
  free(b);
  free(ipiv);
  if (info != 0) {
    cpu_set_last_error("solve: LAPACK sgesv failed (singular matrix)");
  }
}

static void solve_f64(const double *A, const double *B, double *X, int n,
                      int nrhs) {
  // 将 A 从行主序转换为列主序（LAPACK 期望列主序）
  double *a = (double *)malloc(n * n * sizeof(double));
  cpu_rowmajor_to_colmajor_f64(A, a, n, n);

  // 将 B 从行主序转换为列主序
  double *b = (double *)malloc(n * nrhs * sizeof(double));
  cpu_rowmajor_to_colmajor_f64(B, b, n, nrhs);

  int *ipiv = (int *)malloc(n * sizeof(int));
  int info;
  dgesv_(&n, &nrhs, a, &n, ipiv, b, &n, &info);

  // 将结果从列主序转换回行主序
  cpu_colmajor_to_rowmajor_f64(b, X, n, nrhs);

  free(a);
  free(b);
  free(ipiv);
  if (info != 0) {
    cpu_set_last_error("solve: LAPACK dgesv failed (singular matrix)");
  }
}

C_Status solve_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];
  InsightArray *A = (InsightArray *)inputs[1];
  InsightArray *B = (InsightArray *)inputs[2];

  if (!out || !A || !B) {
    cpu_set_last_error("solve: null array pointer");
    return C_FAILED;
  }
  if (!cpu_ensure_contiguous(A) || !cpu_ensure_contiguous(B) ||
      !cpu_ensure_contiguous(out))
    return C_FAILED;

  int n = (int)A->dims[0];
  int nrhs = (B->ndim == 1) ? 1 : (int)B->dims[1];
  if (A->dims[1] != n) {
    cpu_set_last_error("solve: A must be square");
    return C_FAILED;
  }

  if (A->dtype == INSIGHT_DTYPE_F32) {
    solve_f32((float *)A->data, (float *)B->data, (float *)out->data, n, nrhs);
  } else {
    solve_f64((double *)A->data, (double *)B->data, (double *)out->data, n,
              nrhs);
  }
  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(solve, INSIGHT_DTYPE_F32, solve_kernel_cpu);
REGISTER_CPU_KERNEL(solve, INSIGHT_DTYPE_F64, solve_kernel_cpu);

#endif
