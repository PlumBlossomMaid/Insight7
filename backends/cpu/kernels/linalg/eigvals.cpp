// backends/cpu/kernels/linalg/eigvals.cpp
/**
 * @file eigvals.cpp
 * @brief CPU kernel for eigenvalues only (complex).
 */

#include "common.h"
#ifdef INSIGHT_USE_OPENBLAS
#ifdef __cplusplus
extern "C" {
#endif

static void eigvals_f32(const float *src, float *vals_real, float *vals_imag,
                        int n) {
  float *a = (float *)malloc(n * n * sizeof(float));
  memcpy(a, src, n * n * sizeof(float));
  float *wr = (float *)malloc(n * sizeof(float));
  float *wi = (float *)malloc(n * sizeof(float));
  float *work = (float *)malloc(1 * sizeof(float));
  int lwork = -1;
  int info;
  char jobvl[2] = "N";
  char jobvr[2] = "N";
  sgeev_(jobvl, jobvr, &n, a, &n, wr, wi, NULL, &n, NULL, &n, work, &lwork,
         &info);
  lwork = (int)work[0];
  work = (float *)realloc(work, lwork * sizeof(float));
  sgeev_(jobvl, jobvr, &n, a, &n, wr, wi, NULL, &n, NULL, &n, work, &lwork,
         &info);
  for (int i = 0; i < n; ++i) {
    vals_real[i] = wr[i];
    vals_imag[i] = wi[i];
  }
  free(a);
  free(wr);
  free(wi);
  free(work);
}

static void eigvals_f64(const double *src, double *vals_real, double *vals_imag,
                        int n) {
  double *a = (double *)malloc(n * n * sizeof(double));
  memcpy(a, src, n * n * sizeof(double));
  double *wr = (double *)malloc(n * sizeof(double));
  double *wi = (double *)malloc(n * sizeof(double));
  double *work = (double *)malloc(1 * sizeof(double));
  int lwork = -1;
  int info;
  char jobvl[2] = "N";
  char jobvr[2] = "N";
  dgeev_(jobvl, jobvr, &n, a, &n, wr, wi, NULL, &n, NULL, &n, work, &lwork,
         &info);
  lwork = (int)work[0];
  work = (double *)realloc(work, lwork * sizeof(double));
  dgeev_(jobvl, jobvr, &n, a, &n, wr, wi, NULL, &n, NULL, &n, work, &lwork,
         &info);
  for (int i = 0; i < n; ++i) {
    vals_real[i] = wr[i];
    vals_imag[i] = wi[i];
  }
  free(a);
  free(wr);
  free(wi);
  free(work);
}

C_Status eigvals_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out_real = (InsightArray *)outputs[0];
  InsightArray *out_imag = (InsightArray *)outputs[1];
  InsightArray *x = (InsightArray *)inputs[2];

  if (!out_real || !out_imag || !x) {
    cpu_set_last_error("eigvals: null array pointer");
    return C_FAILED;
  }
  if (!cpu_ensure_contiguous(x) || !cpu_ensure_contiguous(out_real) ||
      !cpu_ensure_contiguous(out_imag))
    return C_FAILED;

  int n = (int)x->dims[0];
  if (x->dims[1] != n) {
    cpu_set_last_error("eigvals: matrix must be square");
    return C_FAILED;
  }

  if (x->dtype == INSIGHT_DTYPE_F32) {
    eigvals_f32((float *)x->data, (float *)out_real->data,
                (float *)out_imag->data, n);
  } else {
    eigvals_f64((double *)x->data, (double *)out_real->data,
                (double *)out_imag->data, n);
  }
  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(eigvals, INSIGHT_DTYPE_F32, eigvals_kernel_cpu);
REGISTER_CPU_KERNEL(eigvals, INSIGHT_DTYPE_F64, eigvals_kernel_cpu);
#endif
