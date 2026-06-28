// backends/cuda/kernels/unary/common.cuh
/**
 * @file common.cuh
 * @brief Common utilities for CUDA unary kernels.
 *
 * Provides thread/block configuration and stride-aware offset calculation
 * for unary operations on arrays with arbitrary memory layouts.
 */

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

/**
 * @brief Get the number of threads per block for unary kernels.
 *
 * @return Fixed value of 256 threads per block
 */
inline int unary_threads() { return 256; }

/**
 * @brief Calculate the number of blocks needed for a given element count.
 *
 * Uses ceiling division: blocks = (n + threads - 1) / threads
 *
 * @param n Total number of elements to process
 * @return Number of blocks to launch
 */
inline int unary_blocks(int64_t n) {
  return static_cast<int>((n + unary_threads() - 1) / unary_threads());
}

/**
 * @brief Device function to compute element offset from linear index.
 *
 * Converts a linear index to a multi-dimensional offset using dimension
 * sizes and strides. Supports arbitrary memory layouts.
 *
 * @param linear Linear index
 * @param ndim   Number of dimensions
 * @param dims   Dimension sizes
 * @param strides Strides for each dimension
 * @return Element offset in memory
 */
__device__ inline int64_t unary_offset(int64_t linear, int ndim,
                                       const int64_t *dims,
                                       const int64_t *strides) {
  int64_t offset = 0;
  for (int d = ndim - 1; d >= 0; --d) {
    offset += (linear % dims[d]) * strides[d];
    linear /= dims[d];
  }
  return offset;
}

__device__ inline int64_t unary_offset_with_base(int64_t linear, int ndim,
                                                 const int64_t *dims,
                                                 const int64_t *strides,
                                                 int64_t base_offset) {
  return base_offset + unary_offset(linear, ndim, dims, strides);
}
