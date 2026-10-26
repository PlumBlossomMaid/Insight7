// backends/cuda/kernels/random/set_seed.cu
/**
 * @file set_seed.cu
 * @brief CUDA kernel for setting the global random seed.
 *
 * Seeds the per-device NumPy-compatible MT19937 stream used by every GPU
 * random kernel (which generate on the host and copy to the device), so
 * GPU output is bit-identical to CPU output for the same device seed.
 *
 * @param inputs  [0] = uint64_t* seed value
 * @param outputs [0] = unused
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "../../registry/cuda_registry.h"
#include "insight/c_api/array.h"
#include <cstdint>
#include <cuda_runtime.h>

#include "../../../cpu/kernels/random/numpy_rng_state.h"

extern "C" {

C_Status set_seed_kernel_gpu(void **inputs, void **outputs) {
  if (!inputs[0]) {
    gpu_set_last_error("set_seed: seed pointer is null");
    return C_FAILED;
  }

  uint64_t seed = *static_cast<uint64_t *>(inputs[0]);

  int dev = 0;
  cudaGetDevice(&dev);
  ins::numpy_rng::seed_device(1 /* INSIGHT_DEVICE_GPU */, dev, seed);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_GPU_KERNEL(set_seed, INSIGHT_DTYPE_U64, set_seed_kernel_gpu);