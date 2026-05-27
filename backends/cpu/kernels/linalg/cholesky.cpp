// backends/cpu/kernels/linalg/cholesky.cpp
/**
 * @file cholesky.cpp
 * @brief CPU kernel for Cholesky decomposition using LAPACK.
 */

#include "common.h"
#include <cstdio>

#ifdef INSIGHT_USE_OPENBLAS

#ifdef __cplusplus
extern "C" {
#endif

static void cholesky_f64(const double *src, double *dst, int n, int lower) {
  double *a = (double *)malloc(n * n * sizeof(double));
  cpu_rowmajor_to_colmajor_f64(src, a, n, n);

  char uplo = lower ? 'L' : 'U';
  int info;
  dpotrf_(&uplo, &n, a, &n, &info);

  if (info != 0) {
    cpu_set_last_error("cholesky: matrix not positive definite");
    free(a);
    return;
  }

  if (lower) {
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        if (i >= j) {
          dst[i * n + j] = a[i + j * n];
        } else {
          dst[i * n + j] = 0.0;
        }
      }
    }
  } else {
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        if (i <= j) {
          dst[i * n + j] = a[i + j * n];
        } else {
          dst[i * n + j] = 0.0;
        }
      }
    }
  }

  free(a);
}

static void cholesky_f32(const float *src, float *dst, int n, int lower) {
  double *src_f64 = (double *)malloc(n * n * sizeof(double));
  double *dst_f64 = (double *)malloc(n * n * sizeof(double));

  for (int i = 0; i < n * n; ++i)
    src_f64[i] = src[i];
  cholesky_f64(src_f64, dst_f64, n, lower);
  for (int i = 0; i < n * n; ++i)
    dst[i] = (float)dst_f64[i];

  free(src_f64);
  free(dst_f64);
}

C_Status cholesky_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];
  InsightArray *x = (InsightArray *)inputs[1];
  int lower = *(int *)inputs[2];

  if (!out || !x) {
    cpu_set_last_error("cholesky: null array pointer");
    return C_FAILED;
  }
  if (!cpu_ensure_contiguous(x) || !cpu_ensure_contiguous(out))
    return C_FAILED;

  int n = (int)x->dims[0];
  if (x->dims[1] != n) {
    cpu_set_last_error("cholesky: matrix must be square");
    return C_FAILED;
  }

  if (x->dtype == INSIGHT_DTYPE_F32) {
    cholesky_f32((float *)x->data, (float *)out->data, n, lower);
  } else {
    cholesky_f64((double *)x->data, (double *)out->data, n, lower);
  }
  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(cholesky, INSIGHT_DTYPE_F32, cholesky_kernel_cpu);
REGISTER_CPU_KERNEL(cholesky, INSIGHT_DTYPE_F64, cholesky_kernel_cpu);
#endif
