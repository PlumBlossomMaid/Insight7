// backends/cpu/kernels/linalg/slogdet.cpp
/**
 * @file slogdet.cpp
 * @brief CPU kernel for sign and log determinant.
 */

#ifdef INSIGHT_USE_OPENBLAS

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

static void slogdet_f32(const float *src, int n, float *sign, double *logdet) {
  float *a = (float *)malloc(n * n * sizeof(float));
  memcpy(a, src, n * n * sizeof(float));
  int *ipiv = (int *)malloc(n * sizeof(int));
  int info;
  sgetrf_(&n, &n, a, &n, ipiv, &info);
  double det = 1.0;
  int s = 1;
  for (int i = 0; i < n; ++i) {
    det *= fabs(a[i * n + i]);
    if (ipiv[i] != i + 1)
      s = -s;
  }
  *sign = (float)s;
  *logdet = log(det);
  free(a);
  free(ipiv);
  if (info != 0) {
    cpu_set_last_error("slogdet: LAPACK sgetrf failed");
    *sign = 0.0f;
    *logdet = -INFINITY;
  }
}

static void slogdet_f64(const double *src, int n, double *sign,
                        double *logdet) {
  double *a = (double *)malloc(n * n * sizeof(double));
  memcpy(a, src, n * n * sizeof(double));
  int *ipiv = (int *)malloc(n * sizeof(int));
  int info;
  dgetrf_(&n, &n, a, &n, ipiv, &info);
  double det = 1.0;
  int s = 1;
  for (int i = 0; i < n; ++i) {
    det *= fabs(a[i * n + i]);
    if (ipiv[i] != i + 1)
      s = -s;
  }
  *sign = (double)s;
  *logdet = log(det);
  free(a);
  free(ipiv);
  if (info != 0) {
    cpu_set_last_error("slogdet: LAPACK dgetrf failed");
    *sign = 0.0;
    *logdet = -INFINITY;
  }
}

C_Status slogdet_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *sign_arr = (InsightArray *)outputs[0];
  InsightArray *logdet_arr = (InsightArray *)outputs[1];
  InsightArray *x = (InsightArray *)inputs[2];

  if (!sign_arr || !logdet_arr || !x) {
    cpu_set_last_error("slogdet: null array pointer");
    return C_FAILED;
  }
  if (!cpu_ensure_contiguous(x))
    return C_FAILED;

  int n = (int)x->dims[0];
  if (x->dims[1] != n) {
    cpu_set_last_error("slogdet: matrix must be square");
    return C_FAILED;
  }

  if (x->dtype == INSIGHT_DTYPE_F32) {
    float *s = (float *)sign_arr->data;
    double *ld = (double *)logdet_arr->data;
    slogdet_f32((float *)x->data, n, s, ld);
  } else {
    double *s = (double *)sign_arr->data;
    double *ld = (double *)logdet_arr->data;
    slogdet_f64((double *)x->data, n, s, ld);
  }
  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(slogdet, INSIGHT_DTYPE_F32, slogdet_kernel_cpu);
REGISTER_CPU_KERNEL(slogdet, INSIGHT_DTYPE_F64, slogdet_kernel_cpu);

#endif
