// backends/cuda/kernels/random/uniform.cu
/**
 * @file uniform.cu
 * @brief CUDA kernel for uniform distribution [low, high).
 *
 * Uses the same NumPy-compatible RNG stream as the CPU backend, generated
 * on the host and copied to the device, so GPU output is bit-identical to
 * CPU output and to `np.random.uniform(low, high, size)` for the same seed.
 *
 * @param inputs  [0] = InsightArray* output array
 *                [1] = double* low
 *                [2] = double* high
 *                [3] = uint64_t* device seed (unused)
 * @param outputs [0] = InsightArray* output array (same as inputs[0])
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "../../registry/cuda_registry.h"
#include "insight/c_api/array.h"
#include <cuda_runtime.h>
#include <vector>

#include "../../cpu/kernels/random/numpy_rng_state.h"

extern "C" {

C_Status uniform_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = static_cast<InsightArray *>(outputs[0]);

  if (!out) {
    gpu_set_last_error("uniform: output array is null");
    return C_FAILED;
  }

  double low = *static_cast<double *>(inputs[1]);
  double high = *static_cast<double *>(inputs[2]);
  double range = high - low;

  int64_t n = out->numel;
  if (n == 0)
    return C_SUCCESS;

  int32_t device_id = out->device_id;
  numpy_rk_state *st = ins::numpy_rng::lock_device(out->device_type, device_id);

  cudaError_t err = cudaSuccess;
  switch (out->dtype) {
  case INSIGHT_DTYPE_F32: {
    std::vector<float> host(n);
    for (int64_t i = 0; i < n; ++i) {
      host[i] = (float)(low + range * numpy_rk_double(st));
    }
    err = cudaMemcpy(out->data, host.data(), n * sizeof(float),
                     cudaMemcpyHostToDevice);
    break;
  }
  case INSIGHT_DTYPE_F64: {
    std::vector<double> host(n);
    for (int64_t i = 0; i < n; ++i) {
      host[i] = low + range * numpy_rk_double(st);
    }
    err = cudaMemcpy(out->data, host.data(), n * sizeof(double),
                     cudaMemcpyHostToDevice);
    break;
  }
  default:
    ins::numpy_rng::unlock_device();
    gpu_set_last_error("uniform: unsupported dtype");
    return C_FAILED;
  }

  ins::numpy_rng::unlock_device();

  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }
  return C_SUCCESS;
}

} // extern "C"

REGISTER_GPU_KERNEL(uniform, INSIGHT_DTYPE_F32, uniform_kernel_gpu);
REGISTER_GPU_KERNEL(uniform, INSIGHT_DTYPE_F64, uniform_kernel_gpu);