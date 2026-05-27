// backends/cpu/kernels/linalg/eig.cpp
/**
 * @file eig.cpp
 * @brief CPU kernel for eigenvalues and eigenvectors (complex).
 */

#ifdef INSIGHT_USE_OPENBLAS

#include "common.h"
#include <complex.h>

#ifdef __cplusplus
extern "C" {
#endif

static void eig_f32(const float *src, float *vals_real, float *vals_imag,
                    float *vecs_real, float *vecs_imag, int n) {
  float *a = (float *)malloc(n * n * sizeof(float));
  memcpy(a, src, n * n * sizeof(float));
  float *wr = (float *)malloc(n * sizeof(float));
  float *wi = (float *)malloc(n * sizeof(float));
  float *vr = (float *)malloc(n * n * sizeof(float));
  float *work = (float *)malloc(1 * sizeof(float));
  int lwork = -1;
  int info;
  char jobvl[2] = "N";
  char jobvr[2] = "V";
  sgeev_(jobvl, jobvr, &n, a, &n, wr, wi, NULL, &n, vr, &n, work, &lwork,
         &info);
  lwork = (int)work[0];
  work = (float *)realloc(work, lwork * sizeof(float));
  sgeev_(jobvl, jobvr, &n, a, &n, wr, wi, NULL, &n, vr, &n, work, &lwork,
         &info);
  // eigenvalues
  for (int i = 0; i < n; ++i) {
    vals_real[i] = wr[i];
    vals_imag[i] = wi[i];
  }
  // eigenvectors (complex: real+imag pairs)
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      vecs_real[i * n + j] = vr[j * n + i];
      vecs_imag[i * n + j] = (wi[j] != 0.0f) ? vr[j * n + i + n] : 0.0f;
    }
  }
  free(a);
  free(wr);
  free(wi);
  free(vr);
  free(work);
}

static void eig_f64(const double *src, double *vals_real, double *vals_imag,
                    double *vecs_real, double *vecs_imag, int n) {
  double *a = (double *)malloc(n * n * sizeof(double));
  memcpy(a, src, n * n * sizeof(double));
  double *wr = (double *)malloc(n * sizeof(double));
  double *wi = (double *)malloc(n * sizeof(double));
  double *vr = (double *)malloc(n * n * sizeof(double));
  double *work = (double *)malloc(1 * sizeof(double));
  int lwork = -1;
  int info;
  char jobvl[2] = "N";
  char jobvr[2] = "V";
  dgeev_(jobvl, jobvr, &n, a, &n, wr, wi, NULL, &n, vr, &n, work, &lwork,
         &info);
  lwork = (int)work[0];
  work = (double *)realloc(work, lwork * sizeof(double));
  dgeev_(jobvl, jobvr, &n, a, &n, wr, wi, NULL, &n, vr, &n, work, &lwork,
         &info);
  for (int i = 0; i < n; ++i) {
    vals_real[i] = wr[i];
    vals_imag[i] = wi[i];
  }
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      vecs_real[i * n + j] = vr[j * n + i];
      vecs_imag[i * n + j] = (wi[j] != 0.0) ? vr[j * n + i + n] : 0.0;
    }
  }
  free(a);
  free(wr);
  free(wi);
  free(vr);
  free(work);
}

C_Status eig_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *vals_real = (InsightArray *)outputs[0];
  InsightArray *vals_imag = (InsightArray *)outputs[1];
  InsightArray *vecs_real = (InsightArray *)outputs[2];
  InsightArray *vecs_imag = (InsightArray *)outputs[3];
  InsightArray *x = (InsightArray *)inputs[4];

  if (!vals_real || !vals_imag || !vecs_real || !vecs_imag || !x) {
    cpu_set_last_error("eig: null array pointer");
    return C_FAILED;
  }
  if (!cpu_ensure_contiguous(x) || !cpu_ensure_contiguous(vals_real) ||
      !cpu_ensure_contiguous(vals_imag) || !cpu_ensure_contiguous(vecs_real) ||
      !cpu_ensure_contiguous(vecs_imag))
    return C_FAILED;

  int n = (int)x->dims[0];
  if (x->dims[1] != n) {
    cpu_set_last_error("eig: matrix must be square");
    return C_FAILED;
  }

  if (x->dtype == INSIGHT_DTYPE_F32) {
    eig_f32((float *)x->data, (float *)vals_real->data,
            (float *)vals_imag->data, (float *)vecs_real->data,
            (float *)vecs_imag->data, n);
  } else {
    eig_f64((double *)x->data, (double *)vals_real->data,
            (double *)vals_imag->data, (double *)vecs_real->data,
            (double *)vecs_imag->data, n);
  }
  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(eig, INSIGHT_DTYPE_F32, eig_kernel_cpu);
REGISTER_CPU_KERNEL(eig, INSIGHT_DTYPE_F64, eig_kernel_cpu);

#endif
