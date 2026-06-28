// backends/cuda/kernels/elementwise/common.cuh
/**
 * @file common.cuh
 * @brief Common utilities for CUDA elementwise kernels.
 */

#pragma once
#include "insight/c_api/array.h"
#include <cuComplex.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

static inline dim3 elementwise_blocks(int64_t n) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  return dim3(blocks);
}

static inline dim3 elementwise_threads() { return dim3(256); }

struct ElementwiseMetadata {
  int64_t ndim;
  int64_t numel;
  int64_t dims[INSIGHT_MAX_NDIM];
  int64_t a_strides[INSIGHT_MAX_NDIM];
  int64_t b_strides[INSIGHT_MAX_NDIM];
  int64_t out_strides[INSIGHT_MAX_NDIM];
  int64_t a_offset;
  int64_t b_offset;
  int64_t out_offset;
};

static inline ElementwiseMetadata *
alloc_elementwise_metadata(const InsightArray *a, const InsightArray *b,
                           const InsightArray *out) {

  ElementwiseMetadata *d_meta;
  cudaMalloc(&d_meta, sizeof(ElementwiseMetadata));

  ElementwiseMetadata h_meta;
  h_meta.ndim = out->ndim;
  h_meta.numel = out->numel;
  h_meta.a_offset = a->offset;
  h_meta.b_offset = b->offset;
  h_meta.out_offset = out->offset;

  int b_ndim = b->ndim;
  int b_offset = h_meta.ndim - b_ndim;

  for (int i = 0; i < h_meta.ndim; ++i) {
    h_meta.dims[i] = out->dims[i];
    h_meta.a_strides[i] = a->strides[i];
    h_meta.out_strides[i] = out->strides[i];

    int b_i = i - b_offset;
    if (b_i >= 0 && b_i < b_ndim && b->dims[b_i] == out->dims[i]) {
      h_meta.b_strides[i] = b->strides[b_i];
    } else {
      h_meta.b_strides[i] = 0;
    }
  }

  cudaMemcpy(d_meta, &h_meta, sizeof(ElementwiseMetadata),
             cudaMemcpyHostToDevice);
  return d_meta;
}

static inline void free_elementwise_metadata(ElementwiseMetadata *meta) {
  if (meta)
    cudaFree(meta);
}

__device__ static inline int64_t
elementwise_offset(int64_t linear, const ElementwiseMetadata *meta,
                   const int64_t *strides) {

  int64_t offset = 0;
  int64_t tmp = linear;
  for (int d = meta->ndim - 1; d >= 0; --d) {
    int64_t idx = tmp % meta->dims[d];
    tmp /= meta->dims[d];
    offset += idx * strides[d];
  }
  return offset;
}

__device__ static inline int64_t
elementwise_offset_with_base(int64_t linear, const ElementwiseMetadata *meta,
                             const int64_t *strides, int64_t base_offset) {
  return base_offset + elementwise_offset(linear, meta, strides);
}

// Half-precision arithmetic helpers (avoid ambiguous conversion on CUDA 11.8)
// CUDA 11.8 only has __hadd for __half; others must cast through float
// CoreX provides hdiv(__half) via iluvatar_fp16.hpp; guard ours to avoid
// redefinition.
__device__ __forceinline__ __half hadd(__half a, __half b) {
  return __hadd(a, b);
}
__device__ __forceinline__ __half hsub(__half a, __half b) {
  return __float2half(__half2float(a) - __half2float(b));
}
__device__ __forceinline__ __half hmul(__half a, __half b) {
  return __float2half(__half2float(a) * __half2float(b));
}
#ifndef __ILUVATAR_FP16_HPP__
__device__ __forceinline__ __half hdiv(__half a, __half b) {
  return __float2half(__half2float(a) / __half2float(b));
}
#endif
__device__ __forceinline__ __half hneg(__half a) {
  return __float2half(-__half2float(a));
}

// BFloat16: cast through float for CUDA 11.8 compatibility
__device__ __forceinline__ __nv_bfloat16 hadd(__nv_bfloat16 a,
                                              __nv_bfloat16 b) {
  return __float2bfloat16(__bfloat162float(a) + __bfloat162float(b));
}
__device__ __forceinline__ __nv_bfloat16 hsub(__nv_bfloat16 a,
                                              __nv_bfloat16 b) {
  return __float2bfloat16(__bfloat162float(a) - __bfloat162float(b));
}
__device__ __forceinline__ __nv_bfloat16 hmul(__nv_bfloat16 a,
                                              __nv_bfloat16 b) {
  return __float2bfloat16(__bfloat162float(a) * __bfloat162float(b));
}
__device__ __forceinline__ __nv_bfloat16 hdiv(__nv_bfloat16 a,
                                              __nv_bfloat16 b) {
  return __float2bfloat16(__bfloat162float(a) / __bfloat162float(b));
}
__device__ __forceinline__ __nv_bfloat16 hneg(__nv_bfloat16 a) {
  return __float2bfloat16(-__bfloat162float(a));
}
