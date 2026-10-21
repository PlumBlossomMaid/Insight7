// backends/cuda/kernels/signal/windows/tukey.cu
// CUDA kernel for Tukey (tapered cosine) window generation.
// inputs[0]: alpha (1-element F64 array on device)
// outputs[0]: window output
#include "../../../registry/ixuca_registry.h"
#include "insight/c_api/array.h"
#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

template <typename T> __global__ void boxcar_fill_kernel(T *out, int64_t M) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M)
    return;
  out[i] = T(1.0);
}

template <typename T> __global__ void tukey_hann_kernel(T *out, int64_t M) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M)
    return;
  out[i] = T(0.5) * (T(1.0) - cos(T(2.0 * M_PI) * T(i) / T(M - 1)));
}

template <typename T>
__global__ void tukey_kernel(T *out, int64_t M, T alpha, T width, T boundary1,
                             T boundary2) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M)
    return;
  T n = T(i);
  if (n < boundary1) {
    out[i] = T(0.5) * (T(1.0) + cos(T(M_PI) * (T(2.0) * n / width - T(1.0))));
  } else if (n > boundary2) {
    out[i] = T(0.5) * (T(1.0) + cos(T(M_PI) * (T(2.0) * n / width -
                                               T(2.0) / alpha + T(1.0))));
  } else {
    out[i] = T(1.0);
  }
}

// FP16 wrapper kernels
__global__ void boxcar_fill_kernel_fp16(uint16_t *out, int64_t M) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M)
    return;
  out[i] = __float2half(1.0f);
}

__global__ void tukey_hann_kernel_fp16(uint16_t *out, int64_t M) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M)
    return;
  float val =
      0.5f * (1.0f - cosf(2.0f * (float)M_PI * (float)i / (float)(M - 1)));
  out[i] = __float2half(val);
}

__global__ void tukey_kernel_fp16(uint16_t *out, int64_t M, float alpha,
                                  float width, float boundary1,
                                  float boundary2) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M)
    return;
  float n = (float)i;
  if (n < boundary1) {
    float val = 0.5f * (1.0f + cosf((float)M_PI * (2.0f * n / width - 1.0f)));
    out[i] = __float2half(val);
  } else if (n > boundary2) {
    float val =
        0.5f *
        (1.0f + cosf((float)M_PI * (2.0f * n / width - 2.0f / alpha + 1.0f)));
    out[i] = __float2half(val);
  } else {
    out[i] = __float2half(1.0f);
  }
}

// BF16 wrapper kernels
__global__ void boxcar_fill_kernel_bf16(uint16_t *out, int64_t M) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M)
    return;
  out[i] = __float2bfloat16(1.0f);
}

__global__ void tukey_hann_kernel_bf16(uint16_t *out, int64_t M) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M)
    return;
  float val =
      0.5f * (1.0f - cosf(2.0f * (float)M_PI * (float)i / (float)(M - 1)));
  out[i] = __float2bfloat16(val);
}

__global__ void tukey_kernel_bf16(uint16_t *out, int64_t M, float alpha,
                                  float width, float boundary1,
                                  float boundary2) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M)
    return;
  float n = (float)i;
  if (n < boundary1) {
    float val = 0.5f * (1.0f + cosf((float)M_PI * (2.0f * n / width - 1.0f)));
    out[i] = __float2bfloat16(val);
  } else if (n > boundary2) {
    float val =
        0.5f *
        (1.0f + cosf((float)M_PI * (2.0f * n / width - 2.0f / alpha + 1.0f)));
    out[i] = __float2bfloat16(val);
  } else {
    out[i] = __float2bfloat16(1.0f);
  }
}

extern "C" {

C_Status tukey_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *alpha_arr = (InsightArray *)inputs[0];
  InsightArray *out = (InsightArray *)outputs[0];

  if (!alpha_arr || !out) {
    gpu_set_last_error("tukey: null array pointer");
    return C_FAILED;
  }

  double alpha_host;
  cudaMemcpy(&alpha_host, alpha_arr->data, sizeof(double),
             cudaMemcpyDeviceToHost);

  int64_t M = out->numel;
  if (M == 0)
    return C_SUCCESS;

  dim3 blocks((int)((M + 255) / 256));
  dim3 threads(256);

  if (out->dtype == INSIGHT_DTYPE_F64) {
    double *d_out = (double *)out->data;
    if (alpha_host <= 0.0) {
      boxcar_fill_kernel<double><<<blocks, threads>>>(d_out, M);
    } else if (alpha_host >= 1.0) {
      tukey_hann_kernel<double><<<blocks, threads>>>(d_out, M);
    } else {
      double width = alpha_host * (M - 1);
      double half_width = width / 2.0;
      double b1 = half_width;
      double b2 = (M - 1) - half_width;
      tukey_kernel<double>
          <<<blocks, threads>>>(d_out, M, alpha_host, width, b1, b2);
    }
  } else if (out->dtype == INSIGHT_DTYPE_F32) {
    float *d_out = (float *)out->data;
    float alpha_f = (float)alpha_host;
    if (alpha_host <= 0.0) {
      boxcar_fill_kernel<float><<<blocks, threads>>>(d_out, M);
    } else if (alpha_host >= 1.0) {
      tukey_hann_kernel<float><<<blocks, threads>>>(d_out, M);
    } else {
      float width = alpha_f * (float)(M - 1);
      float half_width = width / 2.0f;
      float b1 = half_width;
      float b2 = (float)(M - 1) - half_width;
      tukey_kernel<float>
          <<<blocks, threads>>>(d_out, M, alpha_f, width, b1, b2);
    }
  } else if (out->dtype == INSIGHT_DTYPE_F16) {
    uint16_t *d_out = (uint16_t *)out->data;
    float alpha_f = (float)alpha_host;
    if (alpha_host <= 0.0) {
      boxcar_fill_kernel_fp16<<<blocks, threads>>>(d_out, M);
    } else if (alpha_host >= 1.0) {
      tukey_hann_kernel_fp16<<<blocks, threads>>>(d_out, M);
    } else {
      float width = alpha_f * (float)(M - 1);
      float half_width = width / 2.0f;
      float b1 = half_width;
      float b2 = (float)(M - 1) - half_width;
      tukey_kernel_fp16<<<blocks, threads>>>(d_out, M, alpha_f, width, b1, b2);
    }
  } else if (out->dtype == INSIGHT_DTYPE_BF16) {
    uint16_t *d_out = (uint16_t *)out->data;
    float alpha_f = (float)alpha_host;
    if (alpha_host <= 0.0) {
      boxcar_fill_kernel_bf16<<<blocks, threads>>>(d_out, M);
    } else if (alpha_host >= 1.0) {
      tukey_hann_kernel_bf16<<<blocks, threads>>>(d_out, M);
    } else {
      float width = alpha_f * (float)(M - 1);
      float half_width = width / 2.0f;
      float b1 = half_width;
      float b2 = (float)(M - 1) - half_width;
      tukey_kernel_bf16<<<blocks, threads>>>(d_out, M, alpha_f, width, b1, b2);
    }
  } else {
    gpu_set_last_error("tukey: unsupported dtype");
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

REGISTER_IXUCA_KERNEL(tukey, INSIGHT_DTYPE_F64, tukey_kernel_gpu);
REGISTER_IXUCA_KERNEL(tukey, INSIGHT_DTYPE_F32, tukey_kernel_gpu);
REGISTER_IXUCA_KERNEL(tukey, INSIGHT_DTYPE_F16, tukey_kernel_gpu);
REGISTER_IXUCA_KERNEL(tukey, INSIGHT_DTYPE_BF16, tukey_kernel_gpu);
