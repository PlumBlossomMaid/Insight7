// backends/cpu/kernels/linalg/det.cpp
/**
 * @file det.cpp
 * @brief CPU kernel for determinant of a square matrix using LAPACK (CLAPACK).
 */

#ifdef INSIGHT_USE_OPENBLAS

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

static float det_f32(const float *src, int n) {
  float *a = (float *)malloc(n * n * sizeof(float));
  memcpy(a, src, n * n * sizeof(float));
  int *ipiv = (int *)malloc(n * sizeof(int));
  int info;
  sgetrf_(&n, &n, a, &n, ipiv, &info);
  float det = 1.0f;
  int sign = 1;
  for (int i = 0; i < n; ++i) {
    det *= a[i * n + i];
    if (ipiv[i] != i + 1)
      sign = -sign;
  }
  if (sign == -1)
    det = -det;
  free(a);
  free(ipiv);
  if (info != 0) {
    cpu_set_last_error("det: LAPACK sgetrf failed");
    return 0.0f;
  }
  return det;
}

static double det_f64(const double *src, int n) {
  double *a = (double *)malloc(n * n * sizeof(double));
  memcpy(a, src, n * n * sizeof(double));
  int *ipiv = (int *)malloc(n * sizeof(int));
  int info;
  dgetrf_(&n, &n, a, &n, ipiv, &info);
  double det = 1.0;
  int sign = 1;
  for (int i = 0; i < n; ++i) {
    det *= a[i * n + i];
    if (ipiv[i] != i + 1)
      sign = -sign;
  }
  if (sign == -1)
    det = -det;
  free(a);
  free(ipiv);
  if (info != 0) {
    cpu_set_last_error("det: LAPACK dgetrf failed");
    return 0.0;
  }
  return det;
}

C_Status det_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];
  InsightArray *x = (InsightArray *)inputs[1];

  if (!out || !x) {
    cpu_set_last_error("det: null array pointer");
    return C_FAILED;
  }
  if (!cpu_ensure_contiguous(x))
    return C_FAILED;

  int n = (int)x->dims[0];
  if (x->dims[1] != n) {
    cpu_set_last_error("det: matrix must be square");
    return C_FAILED;
  }

  if (x->dtype == INSIGHT_DTYPE_F32) {
    float *dst = (float *)out->data;
    *dst = det_f32((float *)x->data, n);
  } else {
    double *dst = (double *)out->data;
    *dst = det_f64((double *)x->data, n);
  }
  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(det, INSIGHT_DTYPE_F32, det_kernel_cpu);
REGISTER_CPU_KERNEL(det, INSIGHT_DTYPE_F64, det_kernel_cpu);

#endif
