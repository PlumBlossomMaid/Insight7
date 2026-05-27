// backends/cpu/kernels/linalg/eigvalsh.cpp
/**
 * @file eigvalsh.cpp
 * @brief CPU kernel for eigenvalues of symmetric matrix.
 */

#ifdef INSIGHT_USE_OPENBLAS

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

static void eigvalsh_f32(const float *src, float *vals, int n, int uplo) {
  float *a = (float *)malloc(n * n * sizeof(float));
  memcpy(a, src, n * n * sizeof(float));
  char uplo_char[2];
  uplo_char[0] = uplo ? 'U' : 'L';
  uplo_char[1] = '\0';
  char jobz[2] = "N";
  float *work = (float *)malloc(1 * sizeof(float));
  int lwork = -1;
  int info;
  ssyev_(jobz, uplo_char, &n, a, &n, vals, work, &lwork, &info);
  lwork = (int)work[0];
  work = (float *)realloc(work, lwork * sizeof(float));
  ssyev_(jobz, uplo_char, &n, a, &n, vals, work, &lwork, &info);
  free(a);
  free(work);
  if (info != 0) {
    cpu_set_last_error("eigvalsh: LAPACK ssyev failed");
  }
}

static void eigvalsh_f64(const double *src, double *vals, int n, int uplo) {
  double *a = (double *)malloc(n * n * sizeof(double));
  memcpy(a, src, n * n * sizeof(double));
  char uplo_char[2];
  uplo_char[0] = uplo ? 'U' : 'L';
  uplo_char[1] = '\0';
  char jobz[2] = "N";
  double *work = (double *)malloc(1 * sizeof(double));
  int lwork = -1;
  int info;
  dsyev_(jobz, uplo_char, &n, a, &n, vals, work, &lwork, &info);
  lwork = (int)work[0];
  work = (double *)realloc(work, lwork * sizeof(double));
  dsyev_(jobz, uplo_char, &n, a, &n, vals, work, &lwork, &info);
  free(a);
  free(work);
  if (info != 0) {
    cpu_set_last_error("eigvalsh: LAPACK dsyev failed");
  }
}

C_Status eigvalsh_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];
  InsightArray *x = (InsightArray *)inputs[1];
  int uplo = *(int *)inputs[2];

  if (!out || !x) {
    cpu_set_last_error("eigvalsh: null array pointer");
    return C_FAILED;
  }
  if (!cpu_ensure_contiguous(x) || !cpu_ensure_contiguous(out))
    return C_FAILED;

  int n = (int)x->dims[0];
  if (x->dims[1] != n) {
    cpu_set_last_error("eigvalsh: matrix must be square");
    return C_FAILED;
  }

  if (x->dtype == INSIGHT_DTYPE_F32) {
    eigvalsh_f32((float *)x->data, (float *)out->data, n, uplo);
  } else {
    eigvalsh_f64((double *)x->data, (double *)out->data, n, uplo);
  }
  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(eigvalsh, INSIGHT_DTYPE_F32, eigvalsh_kernel_cpu);
REGISTER_CPU_KERNEL(eigvalsh, INSIGHT_DTYPE_F64, eigvalsh_kernel_cpu);

#endif
