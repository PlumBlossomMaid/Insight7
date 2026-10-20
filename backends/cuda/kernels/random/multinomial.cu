// backends/cuda/kernels/random/multinomial.cu
/**
 * @file multinomial.cu
 * @brief CUDA kernel for multinomial sampling.
 *
 * Draws num_samples indices from categories 0..n-1 with probabilities
 * proportional to probs (inverse-CDF sampling). Generation happens on the
 * host (like randperm.cu) and the result is copied back to the device.
 *
 * @param inputs  [0] = InsightArray* output array (int64, shape [num_samples])
 *                [1] = InsightArray* probs array (1D, float)
 *                [2] = int64_t* num_samples
 *                [3] = int* replacement (0/1)
 *                [4] = uint64_t* device seed
 * @param outputs [0] = InsightArray* output array
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "../../registry/cuda_registry.h"
#include "insight/c_api/array.h"
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <random>
#include <vector>

template <typename P>
static C_Status multinomial_gpu_dispatch(InsightArray *out,
                                         InsightArray *probs,
                                         int64_t num_samples, int replacement,
                                         uint64_t seed) {
  int64_t n = probs->numel;

  // Copy probabilities to host.
  std::vector<double> w(n);
  std::vector<P> p_host(n);
  cudaError_t err = cudaMemcpy(p_host.data(), probs->data,
                               n * sizeof(P), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }
  for (int64_t i = 0; i < n; ++i) {
    w[i] = static_cast<double>(p_host[i]);
  }

  // Sample on host using inverse-CDF.
  std::vector<int64_t> idx(num_samples);
  std::mt19937 rng(static_cast<uint32_t>(seed));
  for (int64_t s = 0; s < num_samples; ++s) {
    double total = 0.0;
    for (int64_t i = 0; i < n; ++i)
      total += w[i];
    if (total <= 0.0) {
      gpu_set_last_error("multinomial: total probability must be positive");
      return C_FAILED;
    }
    double u = std::uniform_real_distribution<double>(0.0, 1.0)(rng) * total;
    double cum = 0.0;
    int64_t picked = n - 1;
    for (int64_t i = 0; i < n; ++i) {
      cum += w[i];
      if (cum >= u) {
        picked = i;
        break;
      }
    }
    idx[s] = picked;
    if (!replacement) {
      w[picked] = 0.0;
    }
  }

  // Copy indices back to device.
  err = cudaMemcpy(out->data, idx.data(), num_samples * sizeof(int64_t),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }
  return C_SUCCESS;
}

extern "C" {

C_Status multinomial_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = static_cast<InsightArray *>(outputs[0]);
  InsightArray *probs = static_cast<InsightArray *>(inputs[1]);

  if (!out || !probs) {
    gpu_set_last_error("multinomial: null array pointer");
    return C_FAILED;
  }
  if (!inputs[2] || !inputs[3] || !inputs[4]) {
    gpu_set_last_error("multinomial: missing scalar arguments");
    return C_FAILED;
  }

  int64_t num_samples = *static_cast<int64_t *>(inputs[2]);
  int replacement = *static_cast<int *>(inputs[3]);
  uint64_t seed = *static_cast<uint64_t *>(inputs[4]);

  switch (probs->dtype) {
  case INSIGHT_DTYPE_F32:
    return multinomial_gpu_dispatch<float>(out, probs, num_samples,
                                           replacement, seed);
  case INSIGHT_DTYPE_F64:
    return multinomial_gpu_dispatch<double>(out, probs, num_samples,
                                            replacement, seed);
  default:
    gpu_set_last_error("multinomial: unsupported dtype (need f32/f64)");
    return C_FAILED;
  }
}

} // extern "C"

REGISTER_GPU_KERNEL(multinomial, INSIGHT_DTYPE_I64, multinomial_kernel_gpu);