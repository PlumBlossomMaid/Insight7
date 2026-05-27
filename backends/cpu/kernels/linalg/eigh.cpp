// backends/cpu/kernels/linalg/eigh.cpp
/**
 * @file eigh.cpp
 * @brief CPU kernel for symmetric/Hermitian eigenvalue decomposition.
 */

#ifdef INSIGHT_USE_OPENBLAS

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

static void eigh_f32(const float *src, float *vals, float *vecs, int n,
                     int uplo) {
  memcpy(vecs, src, n * n * sizeof(float));
  char uplo_char[2];
  uplo_char[0] = uplo ? 'U' : 'L';
  uplo_char[1] = '\0';
  char jobz[2] = "V";
  float *work = (float *)malloc(1 * sizeof(float));
  int lwork = -1;
  int info;
  ssyev_(jobz, uplo_char, &n, vecs, &n, vals, work, &lwork, &info);
  lwork = (int)work[0];
  work = (float *)realloc(work, lwork * sizeof(float));
  ssyev_(jobz, uplo_char, &n, vecs, &n, vals, work, &lwork, &info);
  free(work);
  if (info != 0) {
    cpu_set_last_error("eigh: LAPACK ssyev failed");
  }
}

static void eigh_f64(const double *src, double *vals, double *vecs, int n,
                     int uplo) {
  memcpy(vecs, src, n * n * sizeof(double));
  char uplo_char[2];
  uplo_char[0] = uplo ? 'U' : 'L';
  uplo_char[1] = '\0';
  char jobz[2] = "V";
  double *work = (double *)malloc(1 * sizeof(double));
  int lwork = -1;
  int info;
  dsyev_(jobz, uplo_char, &n, vecs, &n, vals, work, &lwork, &info);
  lwork = (int)work[0];
  work = (double *)realloc(work, lwork * sizeof(double));
  dsyev_(jobz, uplo_char, &n, vecs, &n, vals, work, &lwork, &info);
  free(work);
  if (info != 0) {
    cpu_set_last_error("eigh: LAPACK dsyev failed");
  }
}

C_Status eigh_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *vals = (InsightArray *)outputs[0];
  InsightArray *vecs = (InsightArray *)outputs[1];
  InsightArray *x = (InsightArray *)inputs[2];
  int uplo = *(int *)inputs[3];

  if (!vals || !vecs || !x) {
    cpu_set_last_error("eigh: null array pointer");
    return C_FAILED;
  }
  if (!cpu_ensure_contiguous(x) || !cpu_ensure_contiguous(vals) ||
      !cpu_ensure_contiguous(vecs))
    return C_FAILED;

  int n = (int)x->dims[0];
  if (x->dims[1] != n) {
    cpu_set_last_error("eigh: matrix must be square");
    return C_FAILED;
  }

  if (x->dtype == INSIGHT_DTYPE_F32) {
    eigh_f32((float *)x->data, (float *)vals->data, (float *)vecs->data, n,
             uplo);
  } else {
    eigh_f64((double *)x->data, (double *)vals->data, (double *)vecs->data, n,
             uplo);
  }
  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(eigh, INSIGHT_DTYPE_F32, eigh_kernel_cpu);
REGISTER_CPU_KERNEL(eigh, INSIGHT_DTYPE_F64, eigh_kernel_cpu);

#endif
