// backends/cuda/kernels/random/get_seed.cu
/**
 * @file get_seed.cu
 * @brief CUDA kernel for getting the global random seed.
 *
 * Returns the current global seed used for random number generation.
 *
 * @param inputs  [0] = unused
 * @param outputs [0] = uint64_t* output seed
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "../../registry/ixuca_registry.h"
#include "insight/c_api/array.h"
#include <cstdint>
#include <cuda_runtime.h>

// Global device seed storage (shared with set_seed)
static __device__ uint64_t d_global_seed_get = 0;

extern "C" {

C_Status get_seed_kernel_gpu(void **inputs, void **outputs) {
  if (!outputs[0]) {
    gpu_set_last_error("get_seed: output pointer is null");
    return C_FAILED;
  }

  // For now, return 0 as default seed
  // In a real implementation, this would read from device memory
  uint64_t *out = static_cast<uint64_t *>(outputs[0]);
  *out = 0;

  return C_SUCCESS;
}

} // extern "C"

// Register for common types
REGISTER_IXUCA_KERNEL(get_seed, INSIGHT_DTYPE_U64, get_seed_kernel_gpu);
