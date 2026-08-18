// backends/cuda/kernels/signal/wavelets/morlet2.cu
// CUDA kernel for Morlet wavelet (CWT-compatible).
// w[n] = exp(j*2*pi*s*t) * exp(-t^2/2) / sqrt(2*pi)
// where t = n - (M-1)/2, s = center frequency, w = width
// inputs[0]: s (center frequency, F64, 1-element)
// inputs[1]: w (width, F64, 1-element)
// outputs[0]: complex wavelet (C64)
#include "../../../registry/ixuca_registry.h"
#include "insight/c_api/array.h"
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

__global__ void signal_morlet2_kernel(cuDoubleComplex *out, int64_t M,
                                      double s_val, double w_val) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M)
    return;

  double half = (double)(M - 1) / 2.0;
  double t = (double)i - half;
  double inv_sqrt2pi = 1.0 / sqrt(2.0 * M_PI);
  double gauss = exp(-0.5 * t * t) * inv_sqrt2pi;
  double phase = 2.0 * M_PI * s_val * t;
  out[i] = make_cuDoubleComplex(gauss * cos(phase), gauss * sin(phase));
}

extern "C" {

C_Status signal_morlet2_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *s_arr = (InsightArray *)inputs[0];
  InsightArray *w_arr = (InsightArray *)inputs[1];
  InsightArray *out = (InsightArray *)outputs[0];

  if (!s_arr || !w_arr || !out) {
    gpu_set_last_error("signal_morlet2: null array pointer");
    return C_FAILED;
  }

  if (!s_arr->data || !w_arr->data || !out->data) {
    gpu_set_last_error("signal_morlet2: null data pointer");
    return C_FAILED;
  }

  double s_val, w_val;
  cudaMemcpy(&s_val, s_arr->data, sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(&w_val, w_arr->data, sizeof(double), cudaMemcpyDeviceToHost);

  int64_t M = out->numel;
  if (M == 0)
    return C_SUCCESS;

  int threads = 256;
  int blocks = (int)((M + threads - 1) / threads);

  switch (out->dtype) {
  case INSIGHT_DTYPE_C64:
    signal_morlet2_kernel<<<blocks, threads>>>((cuDoubleComplex *)out->data, M,
                                               s_val, w_val);
    break;
  default:
    gpu_set_last_error("signal_morlet2: only C64 dtype supported");
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

REGISTER_IXUCA_KERNEL(signal_morlet2, INSIGHT_DTYPE_C64,
                         signal_morlet2_kernel_gpu);
