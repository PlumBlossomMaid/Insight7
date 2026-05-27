// backends/cpu/kernels/linalg/inv.cpp
/**
 * @file inv.cpp
 * @brief CPU kernel for matrix inverse using LAPACK.
 */

#ifdef INSIGHT_USE_OPENBLAS

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

void inv_f32(const float *src, float *dst, int n) {
  float *a = (float *)malloc(n * n * sizeof(float));
  memcpy(a, src, n * n * sizeof(float));
  int *ipiv = (int *)malloc(n * sizeof(int));
  int info;
  sgetrf_(&n, &n, a, &n, ipiv, &info);
  if (info != 0) {
    free(a);
    free(ipiv);
    cpu_set_last_error("inv: LAPACK sgetrf failed (singular matrix)");
    return;
  }
  float *work = (float *)malloc(4 * n * sizeof(float));
  int lwork = 4 * n;
  sgetri_(&n, a, &n, ipiv, work, &lwork, &info);
  memcpy(dst, a, n * n * sizeof(float));
  free(a);
  free(ipiv);
  free(work);
  if (info != 0) {
    cpu_set_last_error("inv: LAPACK sgetri failed");
  }
}

void inv_f64(const double *src, double *dst, int n) {
  double *a = (double *)malloc(n * n * sizeof(double));
  memcpy(a, src, n * n * sizeof(double));
  int *ipiv = (int *)malloc(n * sizeof(int));
  int info;
  dgetrf_(&n, &n, a, &n, ipiv, &info);
  if (info != 0) {
    free(a);
    free(ipiv);
    cpu_set_last_error("inv: LAPACK dgetrf failed (singular matrix)");
    return;
  }
  double *work = (double *)malloc(4 * n * sizeof(double));
  int lwork = 4 * n;
  dgetri_(&n, a, &n, ipiv, work, &lwork, &info);
  memcpy(dst, a, n * n * sizeof(double));
  free(a);
  free(ipiv);
  free(work);
  if (info != 0) {
    cpu_set_last_error("inv: LAPACK dgetri failed");
  }
}

C_Status inv_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];
  InsightArray *x = (InsightArray *)inputs[1];

  if (!out || !x) {
    cpu_set_last_error("inv: null array pointer");
    return C_FAILED;
  }
  if (!cpu_ensure_contiguous(x) || !cpu_ensure_contiguous(out))
    return C_FAILED;

  int n = (int)x->dims[0];
  if (x->dims[1] != n) {
    cpu_set_last_error("inv: matrix must be square");
    return C_FAILED;
  }

  if (x->dtype == INSIGHT_DTYPE_F32) {
    inv_f32((float *)x->data, (float *)out->data, n);
  } else {
    inv_f64((double *)x->data, (double *)out->data, n);
  }
  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(inv, INSIGHT_DTYPE_F32, inv_kernel_cpu);
REGISTER_CPU_KERNEL(inv, INSIGHT_DTYPE_F64, inv_kernel_cpu);

#endif
