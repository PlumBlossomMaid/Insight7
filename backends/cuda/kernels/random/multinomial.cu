// backends/cuda/kernels/random/multinomial.cu
/**
 * @file multinomial.cu
 * @brief CUDA kernel for multinomial index sampling (Paddle semantics).
 *
 * Bit-exact with the CPU kernel: matches the index generation inside
 * `np.random.choice(np.arange(d), num_samples, replacement, p=probs)`
 * (cdf + searchsorted-right for replacement=true; the repeated
 * sample/rebuild/dedup loop for replacement=false). Generation happens on
 * the host via the shared NumPy-compatible RNG stream, then the result is
 * copied back to the device.
 *
 * @param inputs  [0] = InsightArray* output array (int64, shape [num_samples])
 *                [1] = InsightArray* probs array (1D, float)
 *                [2] = int64_t* num_samples
 *                [3] = int* replacement (0/1)
 *                [4] = uint64_t* device seed (unused)
 * @param outputs [0] = InsightArray* output array
 * @return C_SUCCESS on success, C_FAILED on error
 */

#include "../../registry/cuda_registry.h"
#include "insight/c_api/array.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <vector>

#include "../../../cpu/kernels/random/numpy_rng_state.h"

static int64_t npy_searchsorted_right(const std::vector<double> &cdf,
                                      double v) {
  return (int64_t)(std::upper_bound(cdf.begin(), cdf.end(), v) - cdf.begin());
}

template <typename P>
static C_Status multinomial_gpu_dispatch(InsightArray *out, InsightArray *probs,
                                         int64_t num_samples, int replacement,
                                         numpy_rk_state *st) {
  int64_t d = probs->numel;

  // Copy probabilities to host.
  std::vector<P> p_host(d);
  cudaError_t err = cudaMemcpy(p_host.data(), probs->data, d * sizeof(P),
                               cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }
  std::vector<double> p(d);
  for (int64_t i = 0; i < d; ++i) {
    p[i] = static_cast<double>(p_host[i]);
  }

  std::vector<int64_t> idx(num_samples);
  std::vector<double> cdf(d);

  if (replacement) {
    double acc = 0.0;
    bool any_positive = false;
    for (int64_t i = 0; i < d; ++i) {
      acc += p[i];
      cdf[i] = acc;
      if (p[i] > 0.0)
        any_positive = true;
    }
    if (!any_positive || cdf[d - 1] <= 0.0) {
      gpu_set_last_error("multinomial: total probability must be positive");
      return C_FAILED;
    }
    double total = cdf[d - 1];
    for (int64_t i = 0; i < d; ++i) {
      cdf[i] /= total;
    }
    std::vector<double> u(num_samples);
    for (int64_t s = 0; s < num_samples; ++s) {
      u[s] = numpy_rk_double(st);
    }
    for (int64_t s = 0; s < num_samples; ++s) {
      idx[s] = npy_searchsorted_right(cdf, u[s]);
    }
  } else {
    int64_t nz = 0;
    for (int64_t i = 0; i < d; ++i) {
      if (p[i] > 0.0)
        ++nz;
    }
    if (nz < num_samples) {
      gpu_set_last_error(
          "multinomial: fewer non-zero entries in probs than num_samples "
          "when replacement=false");
      return C_FAILED;
    }
    std::vector<double> p_work = p;
    int64_t n_uniq = 0;
    while (n_uniq < num_samples) {
      int64_t missing = num_samples - n_uniq;
      std::vector<double> x(missing);
      for (int64_t i = 0; i < missing; ++i) {
        x[i] = numpy_rk_double(st);
      }
      if (n_uniq > 0) {
        for (int64_t i = 0; i < n_uniq; ++i) {
          p_work[idx[i]] = 0.0;
        }
      }
      double acc = 0.0;
      bool any_positive = false;
      for (int64_t i = 0; i < d; ++i) {
        acc += p_work[i];
        cdf[i] = acc;
        if (p_work[i] > 0.0)
          any_positive = true;
      }
      if (!any_positive || cdf[d - 1] <= 0.0) {
        gpu_set_last_error("multinomial: total probability must be positive");
        return C_FAILED;
      }
      double total = cdf[d - 1];
      for (int64_t i = 0; i < d; ++i) {
        cdf[i] /= total;
      }
      std::vector<int64_t> fresh;
      fresh.reserve(missing);
      for (int64_t i = 0; i < missing; ++i) {
        fresh.push_back(npy_searchsorted_right(cdf, x[i]));
      }
      // NumPy: _, uniq_idx = np.unique(new, return_index=True);
      //        uniq_idx.sort(); new = new.take(uniq_idx)
      // This keeps first-occurrence order of `new`, NOT a sorted dedup.
      std::vector<int64_t> keep;
      keep.reserve(fresh.size());
      for (int64_t v : fresh) {
        if (std::find(keep.begin(), keep.end(), v) == keep.end()) {
          keep.push_back(v);
        }
      }
      for (size_t i = 0; i < keep.size() && n_uniq < num_samples; ++i) {
        idx[n_uniq++] = keep[i];
      }
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
  if (!inputs[2] || !inputs[3]) {
    gpu_set_last_error("multinomial: missing scalar arguments");
    return C_FAILED;
  }

  int64_t num_samples = *static_cast<int64_t *>(inputs[2]);
  int replacement = *static_cast<int *>(inputs[3]);

  int32_t device_id = out->device_id;
  numpy_rk_state *st = ins::numpy_rng::lock_device(out->device_type, device_id);

  C_Status rc;
  switch (probs->dtype) {
  case INSIGHT_DTYPE_F32:
    rc = multinomial_gpu_dispatch<float>(out, probs, num_samples, replacement,
                                         st);
    break;
  case INSIGHT_DTYPE_F64:
    rc = multinomial_gpu_dispatch<double>(out, probs, num_samples, replacement,
                                          st);
    break;
  default:
    ins::numpy_rng::unlock_device();
    gpu_set_last_error("multinomial: unsupported dtype (need f32/f64)");
    return C_FAILED;
  }

  ins::numpy_rng::unlock_device();
  return rc;
}

} // extern "C"

REGISTER_GPU_KERNEL(multinomial, INSIGHT_DTYPE_I64, multinomial_kernel_gpu);