// backends/cuda/kernels/signal/radar/ambgfun.cu
// CUDA kernel for Ambiguity Function — 2D grid: each thread computes one
// (delay, doppler) cell.
#include "../../../registry/ixuca_registry.h"
#include "insight/c_api/array.h"
#include <cmath>
#include <complex>
#include <cuda_runtime.h>
#include <vector>

__global__ void ambgfun_kernel(const double *x, const double *y_real,
                               const double *y_imag, double *ambg, double fs,
                               int64_t N, int64_t M, int64_t delay_len,
                               int64_t doppler_len) {
  int64_t fd = (int64_t)blockIdx.y * blockDim.y + threadIdx.y;
  int64_t tau = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (fd >= doppler_len || tau >= delay_len)
    return;

  double freq = (fd - doppler_len / 2.0) * fs / doppler_len;
  double omega = 2.0 * M_PI * freq / fs;
  int64_t tau_shift = tau - (M - 1);

  // x is stored as interleaved real/imag pairs (complex128)
  // x[2*n] = real, x[2*n+1] = imag
  double sum_real = 0.0, sum_imag = 0.0;
  for (int64_t n = 0; n < N; ++n) {
    int64_t m = n - tau_shift;
    if (m >= 0 && m < M) {
      double xr = x[2 * n], xi = x[2 * n + 1];
      double yr = y_real[m], yi = y_imag[m];
      // conj(y) = (yr, -yi)
      // x * conj(y) = (xr*yr + xi*yi, xi*yr - xr*yi)
      double prod_r = xr * yr + xi * yi;
      double prod_i = xi * yr - xr * yi;
      // multiply by exp(j*omega*n) = (cos(omega*n), sin(omega*n))
      double phase = omega * n;
      double cp = cos(phase), sp = sin(phase);
      sum_real += prod_r * cp - prod_i * sp;
      sum_imag += prod_r * sp + prod_i * cp;
    }
  }
  ambg[fd * delay_len + tau] = sqrt(sum_real * sum_real + sum_imag * sum_imag);
}

extern "C" {

C_Status ambgfun_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *x_arr = (InsightArray *)inputs[0];
  InsightArray *y_arr = (InsightArray *)inputs[1];
  InsightArray *params_arr = (InsightArray *)inputs[2];
  InsightArray *out_arr = (InsightArray *)outputs[0];

  if (!x_arr || !y_arr || !params_arr || !out_arr) {
    gpu_set_last_error("ambgfun: null array pointer");
    return C_FAILED;
  }

  double *params = (double *)params_arr->data;
  double fs = params[0];
  int64_t N = (int64_t)params[1];
  int64_t M = (int64_t)params[2];
  int64_t delay_len = N + M - 1;
  int64_t doppler_len = N;

  // y is complex128 — extract real and imag parts on device
  // x is also complex128, stored as interleaved doubles
  // We'll pass x as raw doubles (interleaved) and y as separate real/imag
  double *y_real = nullptr, *y_imag = nullptr;
  cudaMalloc(&y_real, M * sizeof(double));
  cudaMalloc(&y_imag, M * sizeof(double));

  // Deinterleave y on host then copy to device
  std::vector<double> y_r_h(M), y_i_h(M);
  const double *y_raw = (const double *)y_arr->data;
  for (int64_t i = 0; i < M; ++i) {
    y_r_h[i] = y_raw[2 * i];
    y_i_h[i] = y_raw[2 * i + 1];
  }
  cudaMemcpy(y_real, y_r_h.data(), M * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(y_imag, y_i_h.data(), M * sizeof(double), cudaMemcpyHostToDevice);

  dim3 threads(16, 16);
  dim3 blocks((int)((delay_len + 15) / 16), (int)((doppler_len + 15) / 16));

  ambgfun_kernel<<<blocks, threads>>>((const double *)x_arr->data, y_real,
                                      y_imag, (double *)out_arr->data, fs, N, M,
                                      delay_len, doppler_len);

  cudaFree(y_real);
  cudaFree(y_imag);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }

  // Normalize peak to 1 — copy to host, find max, scale on device
  int64_t total = doppler_len * delay_len;
  std::vector<double> ambg_h(total);
  cudaMemcpy(ambg_h.data(), out_arr->data, total * sizeof(double),
             cudaMemcpyDeviceToHost);
  double peak = 0.0;
  for (int64_t i = 0; i < total; ++i) {
    if (ambg_h[i] > peak)
      peak = ambg_h[i];
  }
  if (peak > 0) {
    // Scale on device using a simple kernel
    // For simplicity, do it on host
    for (int64_t i = 0; i < total; ++i)
      ambg_h[i] /= peak;
    cudaMemcpy(out_arr->data, ambg_h.data(), total * sizeof(double),
               cudaMemcpyHostToDevice);
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_IXUCA_KERNEL(ambgfun, INSIGHT_DTYPE_F64, ambgfun_kernel_gpu);
