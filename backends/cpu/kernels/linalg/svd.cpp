// backends/cpu/kernels/linalg/svd.cpp
/**
 * @file svd.cpp
 * @brief CPU kernel for Singular Value Decomposition using LAPACK.
 */

#ifdef INSIGHT_USE_OPENBLAS

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

static void svd_f32(const float *src, float *u, float *s, float *vt, int m,
                    int n, int full) {
  float *a = (float *)malloc(m * n * sizeof(float));
  memcpy(a, src, m * n * sizeof(float));
  int min_mn = m < n ? m : n;
  float *work = (float *)malloc(1 * sizeof(float));
  int lwork = -1;
  int info;
  char jobu = full ? 'A' : 'S';
  char jobvt = full ? 'A' : 'S';
  int ldu = m;
  int ldvt = full ? n : min_mn;
  sgesvd_(&jobu, &jobvt, &m, &n, a, &m, s, u, &ldu, vt, &ldvt, work, &lwork,
          &info);
  lwork = (int)work[0];
  work = (float *)realloc(work, lwork * sizeof(float));
  sgesvd_(&jobu, &jobvt, &m, &n, a, &m, s, u, &ldu, vt, &ldvt, work, &lwork,
          &info);
  free(a);
  free(work);
}

static void svd_f64(const double *src, double *u, double *s, double *vt, int m,
                    int n, int full) {
  double *a = (double *)malloc(m * n * sizeof(double));
  memcpy(a, src, m * n * sizeof(double));
  int min_mn = m < n ? m : n;
  double *work = (double *)malloc(1 * sizeof(double));
  int lwork = -1;
  int info;
  char jobu = full ? 'A' : 'S';
  char jobvt = full ? 'A' : 'S';
  int ldu = m;
  int ldvt = full ? n : min_mn;
  dgesvd_(&jobu, &jobvt, &m, &n, a, &m, s, u, &ldu, vt, &ldvt, work, &lwork,
          &info);
  lwork = (int)work[0];
  work = (double *)realloc(work, lwork * sizeof(double));
  dgesvd_(&jobu, &jobvt, &m, &n, a, &m, s, u, &ldu, vt, &ldvt, work, &lwork,
          &info);
  free(a);
  free(work);
}

C_Status svd_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *U = (InsightArray *)outputs[0];
  InsightArray *S = (InsightArray *)outputs[1];
  InsightArray *VT = (InsightArray *)outputs[2];
  InsightArray *x = (InsightArray *)inputs[3];
  int full = *(int *)inputs[4];

  if (!U || !S || !VT || !x) {
    cpu_set_last_error("svd: null array pointer");
    return C_FAILED;
  }
  if (!cpu_ensure_contiguous(x) || !cpu_ensure_contiguous(U) ||
      !cpu_ensure_contiguous(S) || !cpu_ensure_contiguous(VT))
    return C_FAILED;

  int m = (int)x->dims[0];
  int n = (int)x->dims[1];

  if (x->dtype == INSIGHT_DTYPE_F32) {
    svd_f32((float *)x->data, (float *)U->data, (float *)S->data,
            (float *)VT->data, m, n, full);
  } else {
    svd_f64((double *)x->data, (double *)U->data, (double *)S->data,
            (double *)VT->data, m, n, full);
  }
  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(svd, INSIGHT_DTYPE_F32, svd_kernel_cpu);
REGISTER_CPU_KERNEL(svd, INSIGHT_DTYPE_F64, svd_kernel_cpu);

#endif
