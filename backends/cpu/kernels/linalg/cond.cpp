// backends/cpu/kernels/linalg/cond.cpp
/**
 * @file cond.cpp
 * @brief CPU kernel for condition number of a matrix.
 */

#ifdef INSIGHT_USE_OPENBLAS

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

static double cond_f64(const double *src, int m, int n) {
  double *a = (double *)malloc(m * n * sizeof(double));
  memcpy(a, src, m * n * sizeof(double));
  int *ipiv = (int *)malloc(n * sizeof(int));
  int info;
  dgetrf_(&m, &n, a, &m, ipiv, &info);
  if (info != 0) {
    free(a);
    free(ipiv);
    cpu_set_last_error("cond: LAPACK dgetrf failed");
    return 0.0;
  }
  char norm = '1';
  double anorm = dlange_(&norm, &m, &n, (double *)src, &m, NULL);
  double rcond;
  double *work = (double *)malloc(4 * n * sizeof(double));
  int *iwork = (int *)malloc(n * sizeof(int));
  dgecon_(&norm, &n, a, &m, &anorm, &rcond, work, iwork, &info);
  free(a);
  free(ipiv);
  free(work);
  free(iwork);
  if (info != 0)
    return 0.0;
  return 1.0 / rcond;
}

static float cond_f32(const float *src, int m, int n) {
  double *src_f64 = (double *)malloc(m * n * sizeof(double));
  for (int i = 0; i < m * n; ++i) {
    src_f64[i] = src[i];
  }

  double result_f64 = cond_f64(src_f64, m, n);

  free(src_f64);
  return (float)result_f64;
}

C_Status cond_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];
  InsightArray *x = (InsightArray *)inputs[1];
  double p = *(double *)inputs[2]; // unused for 1-norm

  if (!out || !x) {
    cpu_set_last_error("cond: null array pointer");
    return C_FAILED;
  }
  if (!cpu_ensure_contiguous(x))
    return C_FAILED;

  int m = (int)x->dims[0];
  int n = (int)x->dims[1];

  if (x->dtype == INSIGHT_DTYPE_F32) {
    float *dst = (float *)out->data;
    *dst = cond_f32((float *)x->data, m, n);
  } else {
    double *dst = (double *)out->data;
    *dst = cond_f64((double *)x->data, m, n);
  }
  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(cond, INSIGHT_DTYPE_F32, cond_kernel_cpu);
REGISTER_CPU_KERNEL(cond, INSIGHT_DTYPE_F64, cond_kernel_cpu);

#endif
