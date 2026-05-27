// backends/cpu/kernels/linalg/matrix_rank.cpp
/**
 * @file matrix_rank.cpp
 * @brief CPU kernel for matrix rank using SVD.
 */

#include "common.h"
#ifdef INSIGHT_USE_OPENBLAS
#ifdef __cplusplus
extern "C" {
#endif

static int rank_f64(const double *src, int m, int n, double tol) {
  int min_mn = m < n ? m : n;
  double *a = (double *)malloc(m * n * sizeof(double));
  memcpy(a, src, m * n * sizeof(double));
  double *s = (double *)malloc(min_mn * sizeof(double));
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
  double max_s = s[0];
  double actual_tol = (tol < 0) ? max_s * (m > n ? m : n) * 1e-14 : tol;
  int rank = 0;
  for (int i = 0; i < min_mn; ++i) {
    if (s[i] > actual_tol)
      ++rank;
  }
  free(a);
  free(s);
  free(work);
  return rank;
}

static int rank_f32(const float *src, int m, int n, double tol) {
  double *src_f64 = (double *)malloc(m * n * sizeof(double));
  for (int i = 0; i < m * n; ++i) {
    src_f64[i] = src[i];
  }

  int rank = rank_f64(src_f64, m, n, tol);

  free(src_f64);
  return rank;
}

C_Status matrix_rank_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];
  InsightArray *x = (InsightArray *)inputs[1];
  double tol = *(double *)inputs[2];

  if (!out || !x) {
    cpu_set_last_error("matrix_rank: null array pointer");
    return C_FAILED;
  }
  if (!cpu_ensure_contiguous(x))
    return C_FAILED;

  int m = (int)x->dims[0];
  int n = (int)x->dims[1];

  int r;
  if (x->dtype == INSIGHT_DTYPE_F32) {
    r = rank_f32((float *)x->data, m, n, tol);
  } else {
    r = rank_f64((double *)x->data, m, n, tol);
  }
  *(int64_t *)out->data = r;
  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(matrix_rank, INSIGHT_DTYPE_F32, matrix_rank_kernel_cpu);
REGISTER_CPU_KERNEL(matrix_rank, INSIGHT_DTYPE_F64, matrix_rank_kernel_cpu);
#endif
