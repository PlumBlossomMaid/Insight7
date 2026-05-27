// backends/cpu/kernels/linalg/norm.cpp
/**
 * @file norm.cpp
 * @brief CPU kernel for vector and matrix norms.
 */

#ifdef INSIGHT_USE_OPENBLAS

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

static float norm_vec_f32(const float *data, int n, double ord) {
  if (ord == 2.0) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i)
      sum += data[i] * data[i];
    return sqrtf(sum);
  } else if (ord == 1.0) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i)
      sum += fabsf(data[i]);
    return sum;
  } else if (ord == INFINITY) {
    float max_val = 0.0f;
    for (int i = 0; i < n; ++i) {
      float a = fabsf(data[i]);
      if (a > max_val)
        max_val = a;
    }
    return max_val;
  } else {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i)
      sum += powf(fabsf(data[i]), (float)ord);
    return powf(sum, 1.0f / (float)ord);
  }
}

static double norm_vec_f64(const double *data, int n, double ord) {
  if (ord == 2.0) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i)
      sum += data[i] * data[i];
    return sqrt(sum);
  } else if (ord == 1.0) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i)
      sum += fabs(data[i]);
    return sum;
  } else if (ord == INFINITY) {
    double max_val = 0.0;
    for (int i = 0; i < n; ++i) {
      double a = fabs(data[i]);
      if (a > max_val)
        max_val = a;
    }
    return max_val;
  } else {
    double sum = 0.0;
    for (int i = 0; i < n; ++i)
      sum += pow(fabs(data[i]), ord);
    return pow(sum, 1.0 / ord);
  }
}

static double norm_mat_f64(const double *data, int m, int n, double ord) {
  double *a = (double *)malloc(m * n * sizeof(double));
  cpu_rowmajor_to_colmajor_f64(data, a, m, n);

  char norm_char[2];
  double result;
  double *work = NULL;

  if (ord == 1.0) {
    norm_char[0] = '1';
    norm_char[1] = '\0';
    result = dlange_(norm_char, &m, &n, a, &m, NULL);
  } else if (ord == INFINITY) {
    norm_char[0] = 'I';
    norm_char[1] = '\0';
    work = (double *)malloc(m * sizeof(double));
    result = dlange_(norm_char, &m, &n, a, &m, work);
    free(work);
  } else {
    norm_char[0] = 'F';
    norm_char[1] = '\0';
    result = dlange_(norm_char, &m, &n, a, &m, NULL);
  }

  free(a);
  return result;
}

static float norm_mat_f32(const float *data, int m, int n, double ord) {
  double *data_f64 = (double *)malloc(m * n * sizeof(double));
  for (int i = 0; i < m * n; ++i) {
    data_f64[i] = data[i];
  }

  double result_f64 = norm_mat_f64(data_f64, m, n, ord);

  free(data_f64);
  return (float)result_f64;
}

C_Status norm_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];
  InsightArray *x = (InsightArray *)inputs[1];
  double ord = *(double *)inputs[2];

  if (!out || !x) {
    cpu_set_last_error("norm: null array pointer");
    return C_FAILED;
  }
  if (!cpu_ensure_contiguous(x))
    return C_FAILED;

  if (x->ndim == 1) {
    int n = (int)x->numel;
    if (x->dtype == INSIGHT_DTYPE_F32) {
      float *dst = (float *)out->data;
      *dst = norm_vec_f32((float *)x->data, n, ord);
    } else {
      double *dst = (double *)out->data;
      *dst = norm_vec_f64((double *)x->data, n, ord);
    }
  } else if (x->ndim == 2) {
    int m = (int)x->dims[0];
    int n = (int)x->dims[1];
    if (x->dtype == INSIGHT_DTYPE_F32) {
      float *dst = (float *)out->data;
      *dst = norm_mat_f32((float *)x->data, m, n, ord);
    } else {
      double *dst = (double *)out->data;
      *dst = norm_mat_f64((double *)x->data, m, n, ord);
    }
  } else {
    cpu_set_last_error("norm: unsupported dimension");
    return C_FAILED;
  }
  return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(norm, INSIGHT_DTYPE_F32, norm_kernel_cpu);
REGISTER_CPU_KERNEL(norm, INSIGHT_DTYPE_F64, norm_kernel_cpu);

#endif
