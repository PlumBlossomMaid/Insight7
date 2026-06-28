// backends/cuda/kernels/cast/common.cu
#include "common.cuh"

// ============================================================================
// Non-template Kernel Definitions
// ============================================================================

__global__ void cast_f32_to_c32_kernel(const float *src, cuFloatComplex *dst,
                                       int64_t n) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    dst[idx] = make_cuFloatComplex(src[idx], 0.0f);
  }
}

__global__ void cast_f64_to_c64_kernel(const double *src, cuDoubleComplex *dst,
                                       int64_t n) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    dst[idx] = make_cuDoubleComplex(src[idx], 0.0);
  }
}

__global__ void cast_c32_to_f32_kernel(const cuFloatComplex *src, float *dst,
                                       int64_t n) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    dst[idx] = cuCrealf(src[idx]);
  }
}

__global__ void cast_c32_to_f64_kernel(const cuFloatComplex *src, double *dst,
                                       int64_t n) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    dst[idx] = static_cast<double>(cuCrealf(src[idx]));
  }
}

__global__ void cast_c64_to_f64_kernel(const cuDoubleComplex *src, double *dst,
                                       int64_t n) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    dst[idx] = cuCreal(src[idx]);
  }
}

__global__ void cast_c64_to_f32_kernel(const cuDoubleComplex *src, float *dst,
                                       int64_t n) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    dst[idx] = static_cast<float>(cuCreal(src[idx]));
  }
}

__global__ void cast_c32_to_bool_kernel(const cuFloatComplex *src, bool *dst,
                                        int64_t n) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    dst[idx] = (cuCrealf(src[idx]) != 0.0f || cuCimagf(src[idx]) != 0.0f);
  }
}

__global__ void cast_c64_to_bool_kernel(const cuDoubleComplex *src, bool *dst,
                                        int64_t n) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    dst[idx] = (cuCreal(src[idx]) != 0.0 || cuCimag(src[idx]) != 0.0);
  }
}

__global__ void cast_c32_to_c64_kernel(const cuFloatComplex *src,
                                       cuDoubleComplex *dst, int64_t n) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    dst[idx] = make_cuDoubleComplex(static_cast<double>(cuCrealf(src[idx])),
                                    static_cast<double>(cuCimagf(src[idx])));
  }
}

__global__ void cast_c64_to_c32_kernel(const cuDoubleComplex *src,
                                       cuFloatComplex *dst, int64_t n) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    dst[idx] = make_cuFloatComplex(static_cast<float>(cuCreal(src[idx])),
                                   static_cast<float>(cuCimag(src[idx])));
  }
}