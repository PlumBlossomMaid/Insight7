// backends/cuda/kernels/random/chisquare.cu
/**
 * @file chisquare.cu
 * @brief CUDA kernel for chi-square distribution.
 *
 * Uses the same NumPy-compatible RNG stream as the CPU backend, generated
 * on the host and copied to the device, so GPU output is bit-identical to
 * CPU output and to `np.random.chisquare(df)` for the same seed.
 *
 * @param inputs  [0] = InsightArray* output array
 *                [1] = double* df (degrees of freedom)
 *                [2] = uint64_t* device seed (unused)
 * @param outputs [0] = InsightArray* output array (same as inputs[0])
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "../../registry/cuda_registry.h"
#include "insight/c_api/array.h"
#include <cuda_runtime.h>
#include <vector>

#include "../../cpu/kernels/random/numpy_rng_state.h"

extern "C" {

C_Status chisquare_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = static_cast<InsightArray *>(outputs[0]);

  if (!out) {
    gpu_set_last_error("chisquare: output array is null");
    return C_FAILED;
  }

  double df = *static_cast<double *>(inputs[1]);

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
      host[i] = (float)numpy_legacy_chisquare(st, df);
    }
    err = cudaMemcpy(out->data, host.data(), n * sizeof(float),
                     cudaMemcpyHostToDevice);
    break;
  }
  case INSIGHT_DTYPE_F64: {
    std::vector<double> host(n);
    for (int64_t i = 0; i < n; ++i) {
      host[i] = numpy_legacy_chisquare(st, df);
    }
    err = cudaMemcpy(out->data, host.data(), n * sizeof(double),
                     cudaMemcpyHostToDevice);
    break;
  }
  default:
    ins::numpy_rng::unlock_device();
    gpu_set_last_error("chisquare: unsupported dtype");
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

REGISTER_GPU_KERNEL(chisquare, INSIGHT_DTYPE_F32, chisquare_kernel_gpu);
REGISTER_GPU_KERNEL(chisquare, INSIGHT_DTYPE_F64, chisquare_kernel_gpu);