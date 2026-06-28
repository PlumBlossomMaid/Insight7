// backends/cuda/kernels/common/atomic_helpers.cuh
/**
 * @file atomic_helpers.cuh
 * @brief Common atomic helpers for CUDA kernels.
 */

#pragma once

#include <cuda_runtime.h>

/**
 * @brief atomicAdd for double via CAS loop.
 *
 * On architectures without native atomicAdd(double*) (sm_52, sm_61),
 * use this helper instead of calling atomicAdd directly.
 */
__device__ __forceinline__ double atomicAddDouble(double *address, double val) {
  unsigned long long int *address_as_ull =
      reinterpret_cast<unsigned long long int *>(address);
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}
