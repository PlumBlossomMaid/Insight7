// backends/cuda/kernels/signal/filtering/lfilter.cu
// CUDA kernel for IIR filter (Direct Form II Transposed)
// Each thread handles one batch element, sequential within batch.
#include "../../../registry/ixuca_registry.h"
#include "insight/c_api/array.h"
#include <cuda_runtime.h>

// Max filter order supported in constant memory
#define MAX_FILTER_ORDER 64

__device__ void lfilter_f64_impl(const double *b, int nb, const double *a,
                                 int na, const double *x, int n, double *y) {
  int nmax = nb > na ? nb : na;
  double z[MAX_FILTER_ORDER];
  for (int j = 0; j < nmax - 1 && j < MAX_FILTER_ORDER; ++j)
    z[j] = 0.0;

  for (int i = 0; i < n; ++i) {
    double val = x[i] + z[0];
    y[i] = val;
    for (int j = 0; j < nmax - 2 && j < MAX_FILTER_ORDER - 1; ++j) {
      double bj1 = (j + 1 < nb) ? b[j + 1] : 0.0;
      double aj1 = (j + 1 < na) ? a[j + 1] : 0.0;
      z[j] = z[j + 1] + bj1 * x[i] - aj1 * val;
    }
    if (nmax >= 2 && nmax - 2 < MAX_FILTER_ORDER) {
      double bn = (nmax - 1 < nb) ? b[nmax - 1] : 0.0;
      double an = (nmax - 1 < na) ? a[nmax - 1] : 0.0;
      z[nmax - 2] = bn * x[i] - an * val;
    }
  }
}

__device__ void lfilter_f32_impl(const float *b, int nb, const float *a, int na,
                                 const float *x, int n, float *y) {
  int nmax = nb > na ? nb : na;
  float z[MAX_FILTER_ORDER];
  for (int j = 0; j < nmax - 1 && j < MAX_FILTER_ORDER; ++j)
    z[j] = 0.0f;

  for (int i = 0; i < n; ++i) {
    float val = x[i] + z[0];
    y[i] = val;
    for (int j = 0; j < nmax - 2 && j < MAX_FILTER_ORDER - 1; ++j) {
      float bj1 = (j + 1 < nb) ? b[j + 1] : 0.0f;
      float aj1 = (j + 1 < na) ? a[j + 1] : 0.0f;
      z[j] = z[j + 1] + bj1 * x[i] - aj1 * val;
    }
    if (nmax >= 2 && nmax - 2 < MAX_FILTER_ORDER) {
      float bn = (nmax - 1 < nb) ? b[nmax - 1] : 0.0f;
      float an = (nmax - 1 < na) ? a[nmax - 1] : 0.0f;
      z[nmax - 2] = bn * x[i] - an * val;
    }
  }
}

__global__ void lfilter_kernel_f64(const double *b, int nb, const double *a,
                                   int na, const double *x, int n, double *y,
                                   int batch) {
  int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (batch_idx >= batch)
    return;
  lfilter_f64_impl(b, nb, a, na, x + batch_idx * n, n, y + batch_idx * n);
}

__global__ void lfilter_kernel_f32(const float *b, int nb, const float *a,
                                   int na, const float *x, int n, float *y,
                                   int batch) {
  int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (batch_idx >= batch)
    return;
  lfilter_f32_impl(b, nb, a, na, x + batch_idx * n, n, y + batch_idx * n);
}

extern "C" {

C_Status lfilter_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *b_arr = (InsightArray *)inputs[0];
  InsightArray *a_arr = (InsightArray *)inputs[1];
  InsightArray *x_arr = (InsightArray *)inputs[2];
  InsightArray *y_arr = (InsightArray *)outputs[0];

  if (!b_arr || !a_arr || !x_arr || !y_arr) {
    gpu_set_last_error("lfilter: null array pointer");
    return C_FAILED;
  }

  int nb = (int)b_arr->numel;
  int na = (int)a_arr->numel;
  int n = (int)x_arr->dims[x_arr->ndim - 1];
  int batch = (int)(x_arr->numel / n);

  if (nb > MAX_FILTER_ORDER || na > MAX_FILTER_ORDER) {
    gpu_set_last_error("lfilter: filter order exceeds MAX_FILTER_ORDER (64)");
    return C_FAILED;
  }

  int threads = 256;
  int blocks = (batch + threads - 1) / threads;

  if (x_arr->dtype == INSIGHT_DTYPE_F64) {
    lfilter_kernel_f64<<<blocks, threads>>>(
        (const double *)b_arr->data, nb, (const double *)a_arr->data, na,
        (const double *)x_arr->data, n, (double *)y_arr->data, batch);
  } else if (x_arr->dtype == INSIGHT_DTYPE_F32) {
    lfilter_kernel_f32<<<blocks, threads>>>(
        (const float *)b_arr->data, nb, (const float *)a_arr->data, na,
        (const float *)x_arr->data, n, (float *)y_arr->data, batch);
  } else {
    gpu_set_last_error("lfilter: unsupported dtype, need F32 or F64");
    return C_FAILED;
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }
  return C_SUCCESS;
}

} // extern "C"

REGISTER_IXUCA_KERNEL(lfilter, INSIGHT_DTYPE_F32, lfilter_kernel_gpu);
REGISTER_IXUCA_KERNEL(lfilter, INSIGHT_DTYPE_F64, lfilter_kernel_gpu);
