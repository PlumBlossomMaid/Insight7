// backends/cuda/kernels/signal/wavelets/morlet.cu
// CUDA kernel for complex Morlet wavelet.
// w[n] = exp(j*2*pi*w0*t) * exp(-t^2/2) where t = n - (M-1)/2
// inputs[0]: w0 (center frequency, F64, 1-element)
// inputs[1]: s (scale, F64, 1-element)
// outputs[0]: complex wavelet (C64)
#include "../../../registry/iluvatar_registry.h"
#include "insight/c_api/array.h"
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

__global__ void signal_morlet_kernel(cuDoubleComplex *out, int64_t M, double w0,
                                     double s) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M)
    return;

  double half = (double)(M - 1) / 2.0;
  double t = ((double)i - half) / s;
  double gauss = exp(-0.5 * t * t);
  double phase = 2.0 * M_PI * w0 * t;
  out[i] = make_cuDoubleComplex(gauss * cos(phase), gauss * sin(phase));
}

extern "C" {

C_Status signal_morlet_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *w0_arr = (InsightArray *)inputs[0];
  InsightArray *s_arr = (InsightArray *)inputs[1];
  InsightArray *out = (InsightArray *)outputs[0];

  if (!w0_arr || !s_arr || !out) {
    gpu_set_last_error("signal_morlet: null array pointer");
    return C_FAILED;
  }

  if (!w0_arr->data || !s_arr->data || !out->data) {
    gpu_set_last_error("signal_morlet: null data pointer");
    return C_FAILED;
  }

  double w0, s;
  cudaMemcpy(&w0, w0_arr->data, sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(&s, s_arr->data, sizeof(double), cudaMemcpyDeviceToHost);

  int64_t M = out->numel;
  if (M == 0)
    return C_SUCCESS;

  int threads = 256;
  int blocks = (int)((M + threads - 1) / threads);

  switch (out->dtype) {
  case INSIGHT_DTYPE_C64:
    signal_morlet_kernel<<<blocks, threads>>>((cuDoubleComplex *)out->data, M,
                                              w0, s);
    break;
  default:
    gpu_set_last_error("signal_morlet: only C64 dtype supported");
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

REGISTER_ILUVATAR_KERNEL(signal_morlet, INSIGHT_DTYPE_C64,
                         signal_morlet_kernel_gpu);
