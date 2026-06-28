// backends/cuda/kernels/random/set_seed.cu
/**
 * @file set_seed.cu
 * @brief CUDA kernel for setting the global random seed.
 *
 * Sets the global seed used for random number generation.
 *
 * @param inputs  [0] = uint64_t* seed value
 * @param outputs [0] = unused
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "../../registry/iluvatar_registry.h"
#include "insight/c_api/array.h"
#include <cstdint>
#include <cuda_runtime.h>

// Global device seed storage
static __device__ uint64_t d_global_seed = 0;

extern "C" {

C_Status set_seed_kernel_gpu(void **inputs, void **outputs) {
  if (!inputs[0]) {
    gpu_set_last_error("set_seed: seed pointer is null");
    return C_FAILED;
  }

  uint64_t seed = *static_cast<uint64_t *>(inputs[0]);

  // Copy seed to device
  cudaMemcpyToSymbol(d_global_seed, &seed, sizeof(uint64_t));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

// Register for common types
REGISTER_ILUVATAR_KERNEL(set_seed, INSIGHT_DTYPE_U64, set_seed_kernel_gpu);
