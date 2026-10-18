// backends/cpu/kernels/linalg/svd.cpp
/**
 * @file svd.cpp
 * @brief CPU kernel for Singular Value Decomposition using LAPACK.
 */

#ifdef INSIGHT_USE_OPENBLAS

#include "../common/half_utils.h"
#include "common.h"
#include <cstdlib>
#include <cstring>
#include <vector>

static void svd_f32(const float *src, float *u, float *s, float *vt, int m,
                    int n, int full) {

  int min_mn = m < n ? m : n;

  // Define variables
  char jobu = full ? 'A' : 'S';
  char jobvt = full ? 'A' : 'S';
  int ldu = m;
  int ldvt = full ? n : min_mn;

  float *a = (float *)malloc(m * n * sizeof(float));
  if (!a) {
    cpu_set_last_error("svd: memory allocation failed");
    return;
  }

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      a[i + j * m] = src[i * n + j];
    }
  }

  float *superb = NULL;
  if (min_mn > 1) {
    superb = (float *)malloc((min_mn - 1) * sizeof(float));
    if (!superb) {
      cpu_set_last_error("svd: memory allocation failed");
      free(a);
      return;
    }
  }

  // LAPACK writes U and VT in column-major order; the output buffers are
  // row-major, so gather into temporary column-major buffers and transpose.
  int u_cols = full ? m : min_mn;
  int vt_rows = full ? n : min_mn;
  float *u_cm = (float *)malloc((size_t)m * u_cols * sizeof(float));
  float *vt_cm = (float *)malloc((size_t)vt_rows * n * sizeof(float));
  if (!u_cm || !vt_cm) {
    cpu_set_last_error("svd: memory allocation failed");
    free(a);
    free(u_cm);
    free(vt_cm);
    if (superb)
      free(superb);
    return;
  }

  int info = LAPACKE_sgesvd(LAPACK_COL_MAJOR, jobu, jobvt, m, n, a, m, s,
                            u_cm, ldu, vt_cm, ldvt, superb);
  if (info != 0) {
    cpu_set_last_error("svd: LAPACK sgesvd failed");
  } else {
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < u_cols; ++j) {
        u[i * u_cols + j] = u_cm[i + j * ldu];
      }
    }
    for (int i = 0; i < vt_rows; ++i) {
      for (int j = 0; j < n; ++j) {
        vt[i * n + j] = vt_cm[i + j * ldvt];
      }
    }
  }

  free(a);
  free(u_cm);
  free(vt_cm);
  if (superb) {
    free(superb);
  }
}

static void svd_f64(const double *src, double *u, double *s, double *vt, int m,
                    int n, int full) {

  int min_mn = m < n ? m : n;
  char jobu = full ? 'A' : 'S';
  char jobvt = full ? 'A' : 'S';
  int ldu = m;
  int ldvt = full ? n : min_mn;

  double *a = (double *)malloc(m * n * sizeof(double));
  if (!a) {
    cpu_set_last_error("svd: memory allocation failed");
    return;
  }
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      a[i + j * m] = src[i * n + j];
    }
  }

  double *superb = NULL;
  if (min_mn > 1) {
    superb = (double *)malloc((min_mn - 1) * sizeof(double));
    if (!superb) {
      cpu_set_last_error("svd: memory allocation failed");
      free(a);
      return;
    }
  }

  // LAPACK writes U and VT in column-major order; the output buffers are
  // row-major, so gather into temporary column-major buffers and transpose.
  int u_cols = full ? m : min_mn;
  int vt_rows = full ? n : min_mn;
  double *u_cm = (double *)malloc((size_t)m * u_cols * sizeof(double));
  double *vt_cm = (double *)malloc((size_t)vt_rows * n * sizeof(double));
  if (!u_cm || !vt_cm) {
    cpu_set_last_error("svd: memory allocation failed");
    free(a);
    free(u_cm);
    free(vt_cm);
    if (superb)
      free(superb);
    return;
  }

  int info = LAPACKE_dgesvd(LAPACK_COL_MAJOR, jobu, jobvt, m, n, a, m, s,
                            u_cm, ldu, vt_cm, ldvt, superb);

  if (info != 0) {
    cpu_set_last_error("svd: LAPACK dgesvd failed");
  } else {
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < u_cols; ++j) {
        u[i * u_cols + j] = u_cm[i + j * ldu];
      }
    }
    for (int i = 0; i < vt_rows; ++i) {
      for (int j = 0; j < n; ++j) {
        vt[i * n + j] = vt_cm[i + j * ldvt];
      }
    }
  }

  free(a);
  free(u_cm);
  free(vt_cm);
  if (superb)
    free(superb);
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
      !cpu_ensure_contiguous(S) || !cpu_ensure_contiguous(VT)) {
    return C_FAILED;
  }

  int m = (int)x->dims[0];
  int n = (int)x->dims[1];

  cpu_set_last_error("");

  if (x->dtype == INSIGHT_DTYPE_F32) {
    svd_f32((float *)x->data, (float *)U->data, (float *)S->data,
            (float *)VT->data, m, n, full);
  } else if (x->dtype == INSIGHT_DTYPE_F16 || x->dtype == INSIGHT_DTYPE_BF16) {
    const uint16_t *x_src = (const uint16_t *)x->data;
    uint16_t *u_dst = (uint16_t *)U->data;
    uint16_t *s_dst = (uint16_t *)S->data;
    uint16_t *vt_dst = (uint16_t *)VT->data;
    bool is_f16 = (x->dtype == INSIGHT_DTYPE_F16);
    int min_mn = m < n ? m : n;
    int u_cols = full ? n : min_mn;
    int vt_rows = full ? n : min_mn;
    std::vector<float> x_f32((int64_t)m * n);
    std::vector<float> u_f32((int64_t)m * u_cols);
    std::vector<float> s_f32(min_mn);
    std::vector<float> vt_f32((int64_t)vt_rows * n);
    if (is_f16) {
      for (int64_t i = 0; i < (int64_t)m * n; ++i)
        x_f32[i] = insight::f16_to_f32(x_src[i]);
    } else {
      for (int64_t i = 0; i < (int64_t)m * n; ++i)
        x_f32[i] = insight::bf16_to_f32(x_src[i]);
    }
    svd_f32(x_f32.data(), u_f32.data(), s_f32.data(), vt_f32.data(), m, n,
            full);
    if (is_f16) {
      for (int64_t i = 0; i < (int64_t)m * u_cols; ++i)
        u_dst[i] = insight::f32_to_f16(u_f32[i]);
      for (int i = 0; i < min_mn; ++i)
        s_dst[i] = insight::f32_to_f16(s_f32[i]);
      for (int64_t i = 0; i < (int64_t)vt_rows * n; ++i)
        vt_dst[i] = insight::f32_to_f16(vt_f32[i]);
    } else {
      for (int64_t i = 0; i < (int64_t)m * u_cols; ++i)
        u_dst[i] = insight::f32_to_bf16(u_f32[i]);
      for (int i = 0; i < min_mn; ++i)
        s_dst[i] = insight::f32_to_bf16(s_f32[i]);
      for (int64_t i = 0; i < (int64_t)vt_rows * n; ++i)
        vt_dst[i] = insight::f32_to_bf16(vt_f32[i]);
    }
  } else {
    svd_f64((double *)x->data, (double *)U->data, (double *)S->data,
            (double *)VT->data, m, n, full);
  }

  const char *err = cpu_get_last_error();
  if (err && err[0] != '\0') {
    return C_FAILED;
  }

  return C_SUCCESS;
}

REGISTER_CPU_KERNEL(svd, INSIGHT_DTYPE_F32, svd_kernel_cpu);
REGISTER_CPU_KERNEL(svd, INSIGHT_DTYPE_F64, svd_kernel_cpu);
REGISTER_CPU_KERNEL(svd, INSIGHT_DTYPE_F16, svd_kernel_cpu);
REGISTER_CPU_KERNEL(svd, INSIGHT_DTYPE_BF16, svd_kernel_cpu);

#else // !INSIGHT_USE_OPENBLAS

#include "../../registry/cpu_registry.h"
#include "insight/c_api/array.h"

C_Status svd_kernel_cpu(void **inputs, void **outputs) {
  cpu_set_last_error("svd: not compiled with OpenBLAS support");
  return C_FAILED;
}

REGISTER_CPU_KERNEL(svd, INSIGHT_DTYPE_F32, svd_kernel_cpu);
REGISTER_CPU_KERNEL(svd, INSIGHT_DTYPE_F64, svd_kernel_cpu);

#endif // INSIGHT_USE_OPENBLAS
