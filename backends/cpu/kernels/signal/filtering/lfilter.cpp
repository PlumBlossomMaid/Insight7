// backends/cpu/kernels/signal/filtering/lfilter.cpp
// CPU kernel for IIR filter (Direct Form II Transposed)
#include "../../../registry/cpu_registry.h"
#include "insight/c_api/array.h"
#include <algorithm>
#include <cstring>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

extern "C" {

// lfilter kernel
// inputs: [0]=b (numerator), [1]=a (denominator, normalized), [2]=x (signal)
// outputs: [0]=y (filtered signal)
//
// The frontend normalizes coefficients so a[0]=1 and passes them as
// contiguous arrays. Dtype dispatch based on output array.
C_Status lfilter_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *b_arr = (InsightArray *)inputs[0];
  InsightArray *a_arr = (InsightArray *)inputs[1];
  InsightArray *x_arr = (InsightArray *)inputs[2];
  InsightArray *y_arr = (InsightArray *)outputs[0];

  if (!b_arr || !a_arr || !x_arr || !y_arr) {
    cpu_set_last_error("lfilter: null array pointer");
    return C_FAILED;
  }

  int64_t nb = b_arr->numel;
  int64_t na = a_arr->numel;
  int64_t n = x_arr->dims[x_arr->ndim - 1]; // signal length (last dim)
  int64_t batch = x_arr->numel / n;
  int64_t nmax = std::max(nb, na);

  if (y_arr->dtype == INSIGHT_DTYPE_F64) {
    const double *bp = (const double *)b_arr->data;
    const double *ap = (const double *)a_arr->data;
    double *out = (double *)y_arr->data;

    if (!bp || !ap || !out) {
      cpu_set_last_error("lfilter: null data pointer");
      return C_FAILED;
    }

    for (int64_t b_idx = 0; b_idx < batch; ++b_idx) {
      const double *xi = (const double *)x_arr->data + b_idx * n;
      double *yi = out + b_idx * n;
      std::vector<double> z(nmax - 1, 0.0);

      for (int64_t i = 0; i < n; ++i) {
        double val = bp[0] * xi[i];
        if (nmax >= 2)
          val += z[0];
        yi[i] = val;
        for (int64_t j = 0; j < nmax - 2; ++j) {
          double bj1 = (j + 1 < nb) ? bp[j + 1] : 0.0;
          double aj1 = (j + 1 < na) ? ap[j + 1] : 0.0;
          z[j] = z[j + 1] + bj1 * xi[i] - aj1 * val;
        }
        if (nmax >= 2) {
          double bn = (nmax - 1 < nb) ? bp[nmax - 1] : 0.0;
          double an = (nmax - 1 < na) ? ap[nmax - 1] : 0.0;
          z[nmax - 2] = bn * xi[i] - an * val;
        }
      }
    }
  } else if (y_arr->dtype == INSIGHT_DTYPE_F32) {
    const float *bp = (const float *)b_arr->data;
    const float *ap = (const float *)a_arr->data;
    float *out = (float *)y_arr->data;

    if (!bp || !ap || !out) {
      cpu_set_last_error("lfilter: null data pointer");
      return C_FAILED;
    }

    for (int64_t b_idx = 0; b_idx < batch; ++b_idx) {
      const float *xi = (const float *)x_arr->data + b_idx * n;
      float *yi = out + b_idx * n;
      std::vector<float> z(nmax - 1, 0.0f);

      for (int64_t i = 0; i < n; ++i) {
        float val = bp[0] * xi[i];
        if (nmax >= 2)
          val += z[0];
        yi[i] = val;
        for (int64_t j = 0; j < nmax - 2; ++j) {
          float bj1 = (j + 1 < nb) ? bp[j + 1] : 0.0f;
          float aj1 = (j + 1 < na) ? ap[j + 1] : 0.0f;
          z[j] = z[j + 1] + bj1 * xi[i] - aj1 * val;
        }
        if (nmax >= 2) {
          float bn = (nmax - 1 < nb) ? bp[nmax - 1] : 0.0f;
          float an = (nmax - 1 < na) ? ap[nmax - 1] : 0.0f;
          z[nmax - 2] = bn * xi[i] - an * val;
        }
      }
    }
  } else {
    cpu_set_last_error("lfilter: unsupported dtype");
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

// Register for F64 (frontend passes normalized double coefficients)
REGISTER_CPU_KERNEL(lfilter, INSIGHT_DTYPE_F64, lfilter_kernel_cpu);
REGISTER_CPU_KERNEL(lfilter, INSIGHT_DTYPE_F32, lfilter_kernel_cpu);
