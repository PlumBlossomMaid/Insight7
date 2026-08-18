// backends/cuda/kernels/reduction/common.cuh
/**
 * @file common.cuh
 * @brief Common utilities for CUDA reduction kernels.
 */

#pragma once

#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

/**
 * @brief Check if value is NaN.
 */
template <typename T> __device__ bool is_nan_device(T val) {
  return val != val;
}

/**
 * @brief Get the number of threads per block for reduction kernels.
 */
inline int reduction_threads() { return 256; }

/**
 * @brief Calculate the number of blocks needed for reduction.
 */
inline int reduction_blocks(int64_t n) {
  return static_cast<int>((n + reduction_threads() - 1) / reduction_threads());
}
