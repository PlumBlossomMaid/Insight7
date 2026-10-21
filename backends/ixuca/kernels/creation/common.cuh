// backends/cuda/kernels/creation/common.cuh
/**
 * @file common.cuh
 * @brief Common utilities for CUDA creation kernels.
 *
 * Provides thread/block configuration helpers for 1D element-wise kernels.
 */
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

/**
 * @brief Get the number of threads per block for creation kernels.
 *
 * @return Fixed value of 256 threads per block
 */
inline int creation_threads() { return 256; }

/**
 * @brief Calculate the number of blocks needed for a given element count.
 *
 * Uses ceiling division: blocks = (n + threads - 1) / threads
 *
 * @param n Total number of elements to process
 * @return Number of blocks to launch
 */
inline int creation_blocks(int64_t n) {
  return static_cast<int>((n + creation_threads() - 1) / creation_threads());
}
