// backends/cpu/kernels/linalg/svdvals.cpp
/**
 * @file svdvals.cpp
 * @brief CPU kernel for singular values only.
 */

#ifdef INSIGHT_USE_OPENBLAS

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

static void svdvals_f32(const float *src, float *s, int m, int n) {
  float *a = (float *)malloc(m * n * sizeof(float));
  memcpy(a, src, m * n * sizeof(float));
  int min_mn = m < n ? m : n;
  float *work = (float *)malloc(1 * sizeof(float));
  int lwork = -1;
  int info;
  char jobu[2] = "N";
  char jobvt[2] = "N";
  sgesvd_(jobu, jobvt, &m, &n, a, &m, s, NULL, &m, NULL, &n, work, &lwork,
          &info);
  lwork = (int)work[0];
  work = (float *)realloc(work, lwork * sizeof(float));
  sgesvd_(jobu, jobvt, &m, &n, a, &m, s, NULL, &m, NULL, &n, work, &lwork,
          &info);
  free(a);
  free(work);
}

static void svdvals_f64(const double *src, double *s, int m, int n) {
  double *a = (double *)malloc(m * n * sizeof(double));
  memcpy(a, src, m * n * sizeof(double));
  int min_mn = m < n ? m : n;
  double *work = (double *)malloc(1 * sizeof(double));
  int lwork = -1;
  int info;
  char jobu[2] = "N";
  char jobvt[2] = "N";
  dgesvd_(jobu, jobvt, &m, &n, a, &m, s, NULL, &m, NULL, &n, work, &lwork,
          &info);
  lwork = (int)work[0];
  work = (double *)realloc(work, lwork * sizeof(double));
  dgesvd_(jobu, jobvt, &m, &n, a, &m, s, NULL, &m, NULL, &n, work, &lwork,
          &info);
  free(a);
  free(work);
}

C_Status svdvals_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];
  InsightArray *x = (InsightArray *)inputs[1];

  if (!out || !x) {
    cpu_set_last_error("svdvals: null array pointer");
    return C_FAILED;
  }
  if (!cpu_ensure_contiguous(x) || !cpu_ensure_contiguous(out))
    return C_FAILED;

  int m = (int)x->dims[0];
  int n = (int)x->dims[1];
  int min_mn = m < n ? m : n;

  if (x->dtype == INSIGHT_DTYPE_F32) {
    svdvals_f32((float *)x->data, (float *)out->data, m, n);
  } else {
    svdvals_f64((double *)x->data, (double *)out->data, m, n);
  }
  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(svdvals, INSIGHT_DTYPE_F32, svdvals_kernel_cpu);
REGISTER_CPU_KERNEL(svdvals, INSIGHT_DTYPE_F64, svdvals_kernel_cpu);

#endif
