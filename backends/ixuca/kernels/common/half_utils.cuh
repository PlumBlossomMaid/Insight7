// backends/cuda/kernels/common/half_utils.cuh
/**
 * @file half_utils.cuh
 * @brief CUDA half-precision type utilities.
 *
 * Uses native CUDA __half and __nv_bfloat16 types with hardware support.
 * Requires compute capability 5.3+ for __half and 8.0+ for __nv_bfloat16.
 */

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

// CUDA natively supports __half and __nv_bfloat16 via hardware instructions.
// No software emulation needed — the GPU has dedicated tensor core support
// for these types on SM 7.0+ (half) and SM 8.0+ (bfloat16).
