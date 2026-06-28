// backends/cuda/kernels/cast/common.cuh
#pragma once
#include "insight/c_api/array.h"
#include <cuComplex.h>
#include <cuda_runtime.h>

// Copy layout information (host function - inline to avoid duplicate)
static inline void copy_layout(InsightArray *dst, const InsightArray *src) {
  dst->ndim = src->ndim;
  dst->numel = src->numel;
  dst->offset = src->offset;
  for (int i = 0; i < src->ndim; ++i) {
    dst->dims[i] = src->dims[i];
    dst->strides[i] = src->strides[i];
  }
}

// Helper: compute grid/block dimensions
static inline void cast_launch_config(int64_t n, int &blocks, int &threads) {
  threads = 256;
  blocks = (n + threads - 1) / threads;
}

// ============================================================================
// Template Kernels (defined in header)
// ============================================================================

template <typename T>
__device__ static inline cuFloatComplex to_float_complex(T v) {
  return make_cuFloatComplex(static_cast<float>(v), 0.0f);
}

template <typename T>
__device__ static inline cuDoubleComplex to_double_complex(T v) {
  return make_cuDoubleComplex(static_cast<double>(v), 0.0);
}

template <typename SrcT, typename DstT>
__global__ void cast_kernel(const SrcT *src, DstT *dst, int64_t n) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;
  if constexpr (std::is_same_v<DstT, cuFloatComplex>) {
    dst[idx] = to_float_complex(src[idx]);
  } else if constexpr (std::is_same_v<DstT, cuDoubleComplex>) {
    dst[idx] = to_double_complex(src[idx]);
  } else {
    dst[idx] = static_cast<DstT>(src[idx]);
  }
}

template <typename DstT>
__global__ void cast_bool_to_kernel(const bool *src, DstT *dst, int64_t n) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;

  if constexpr (std::is_same_v<DstT, cuFloatComplex>) {
    dst[idx] = src[idx] ? make_cuFloatComplex(1.0f, 0.0f)
                        : make_cuFloatComplex(0.0f, 0.0f);
  } else if constexpr (std::is_same_v<DstT, cuDoubleComplex>) {
    dst[idx] = src[idx] ? make_cuDoubleComplex(1.0, 0.0)
                        : make_cuDoubleComplex(0.0, 0.0);
  } else {
    dst[idx] = src[idx] ? DstT(1) : DstT(0);
  }
}

template <typename SrcT>
__global__ void cast_to_bool_kernel(const SrcT *src, bool *dst, int64_t n) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    dst[idx] = (src[idx] != SrcT(0));
  }
}

// ============================================================================
// Non-template Kernel Declarations (defined in common.cu)
// ============================================================================

__global__ void cast_f32_to_c32_kernel(const float *src, cuFloatComplex *dst,
                                       int64_t n);
__global__ void cast_f64_to_c64_kernel(const double *src, cuDoubleComplex *dst,
                                       int64_t n);
__global__ void cast_c32_to_f32_kernel(const cuFloatComplex *src, float *dst,
                                       int64_t n);
__global__ void cast_c32_to_f64_kernel(const cuFloatComplex *src, double *dst,
                                       int64_t n);
__global__ void cast_c64_to_f64_kernel(const cuDoubleComplex *src, double *dst,
                                       int64_t n);
__global__ void cast_c64_to_f32_kernel(const cuDoubleComplex *src, float *dst,
                                       int64_t n);
__global__ void cast_c32_to_bool_kernel(const cuFloatComplex *src, bool *dst,
                                        int64_t n);
__global__ void cast_c64_to_bool_kernel(const cuDoubleComplex *src, bool *dst,
                                        int64_t n);
__global__ void cast_c32_to_c64_kernel(const cuFloatComplex *src,
                                       cuDoubleComplex *dst, int64_t n);
__global__ void cast_c64_to_c32_kernel(const cuDoubleComplex *src,
                                       cuFloatComplex *dst, int64_t n);