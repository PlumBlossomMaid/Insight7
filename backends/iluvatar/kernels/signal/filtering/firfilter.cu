// NOTE: This kernel is currently not called from the frontend.
// The frontend uses composite ops for this operation.
// This kernel is preserved for future backend dispatch optimization.
//
// backends/cuda/kernels/signal/filtering/firfilter.cu
// CUDA kernel for FIR filter — each thread computes one output element.
#include "../../../registry/iluvatar_registry.h"
#include "insight/c_api/array.h"
#include <cuda_runtime.h>

__global__ void firfilter_kernel_f64(const double *x, const double *b,
                                     double *y, int batch, int signal_len,
                                     int nb, int out_len) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch * out_len;
  if (idx >= total)
    return;

  int row = idx / out_len;
  int col = idx % out_len;
  const double *xi = x + row * signal_len;
  double sum = 0.0;
  for (int j = 0; j < nb; ++j) {
    int k = col - j;
    if (k >= 0 && k < signal_len) {
      sum += xi[k] * b[j];
    }
  }
  y[idx] = sum;
}

__global__ void firfilter_kernel_f32(const float *x, const float *b, float *y,
                                     int batch, int signal_len, int nb,
                                     int out_len) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch * out_len;
  if (idx >= total)
    return;

  int row = idx / out_len;
  int col = idx % out_len;
  const float *xi = x + row * signal_len;
  float sum = 0.0f;
  for (int j = 0; j < nb; ++j) {
    int k = col - j;
    if (k >= 0 && k < signal_len) {
      sum += xi[k] * b[j];
    }
  }
  y[idx] = sum;
}

extern "C" {

C_Status firfilter_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *x_arr = (InsightArray *)inputs[0];
  InsightArray *b_arr = (InsightArray *)inputs[1];
  InsightArray *y_arr = (InsightArray *)outputs[0];

  if (!x_arr || !b_arr || !y_arr) {
    gpu_set_last_error("firfilter: null array pointer");
    return C_FAILED;
  }

  int batch = (int)x_arr->dims[0];
  int signal_len = (int)x_arr->dims[1];
  int nb = (int)b_arr->numel;
  int out_len = signal_len + nb - 1;
  int total = batch * out_len;

  int threads = 256;
  int blocks = (total + threads - 1) / threads;

  if (x_arr->dtype == INSIGHT_DTYPE_F64) {
    firfilter_kernel_f64<<<blocks, threads>>>(
        (const double *)x_arr->data, (const double *)b_arr->data,
        (double *)y_arr->data, batch, signal_len, nb, out_len);
  } else if (x_arr->dtype == INSIGHT_DTYPE_F32) {
    firfilter_kernel_f32<<<blocks, threads>>>(
        (const float *)x_arr->data, (const float *)b_arr->data,
        (float *)y_arr->data, batch, signal_len, nb, out_len);
  } else {
    gpu_set_last_error("firfilter: unsupported dtype");
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

REGISTER_ILUVATAR_KERNEL(firfilter, INSIGHT_DTYPE_F32, firfilter_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(firfilter, INSIGHT_DTYPE_F64, firfilter_kernel_gpu);
