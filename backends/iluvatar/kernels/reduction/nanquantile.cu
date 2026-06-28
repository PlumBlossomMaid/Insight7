// backends/cuda/kernels/reduction/nanquantile.cu
/**
 * @file nanquantile.cu
 * @brief CUDA kernel for nanquantile reduction (ignoring NaN values).
 *
 * Input layout (matches CPU kernel):
 *   inputs[0] = InsightArray* output (same dtype as input)
 *   inputs[1] = InsightArray* prepared
 *   inputs[2] = int64_t* batch_size
 *   inputs[3] = int64_t* reduce_size
 *   inputs[4] = double* q
 *
 * Note: Uses CPU fallback with sorting.
 */

#include "../../registry/iluvatar_registry.h"
#include "common.cuh"
#include "insight/c_api/array.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <vector>

template <typename T>
static void compute_nanquantile(const T *host_data, T *result,
                                int64_t batch_size, int64_t reduce_size,
                                double q) {
  for (int64_t i = 0; i < batch_size; ++i) {
    std::vector<T> valid;
    valid.reserve(reduce_size);
    for (int64_t j = 0; j < reduce_size; ++j) {
      T val = host_data[i * reduce_size + j];
      if (val == val) { // not NaN
        valid.push_back(val);
      }
    }

    if (valid.empty()) {
      result[i] = T(0);
      continue;
    }

    std::sort(valid.begin(), valid.end());
    int64_t n = static_cast<int64_t>(valid.size());

    if (q == 0.5) {
      if (n % 2 == 1) {
        result[i] = valid[n / 2];
      } else {
        result[i] = (valid[n / 2 - 1] + valid[n / 2]) * T(0.5);
      }
    } else {
      double idx = q * (n - 1);
      int64_t lo = static_cast<int64_t>(idx);
      int64_t hi = lo + 1;
      if (hi >= n) {
        result[i] = valid[lo];
      } else {
        double frac = idx - lo;
        result[i] = static_cast<T>(valid[lo] * (1 - frac) + valid[hi] * frac);
      }
    }
  }
}

extern "C" {

C_Status nanquantile_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = static_cast<InsightArray *>(outputs[0]);
  InsightArray *prepared = static_cast<InsightArray *>(inputs[1]);

  if (!out || !prepared) {
    gpu_set_last_error("nanquantile: null array pointer");
    return C_FAILED;
  }

  int64_t batch_size = *static_cast<int64_t *>(inputs[2]);
  int64_t reduce_size = *static_cast<int64_t *>(inputs[3]);
  double q = *static_cast<double *>(inputs[4]);

  switch (prepared->dtype) {
  case INSIGHT_DTYPE_F32: {
    float *host_data = new float[prepared->numel];
    cudaMemcpy(host_data, prepared->data, prepared->numel * sizeof(float),
               cudaMemcpyDeviceToHost);
    float *result = new float[batch_size];
    compute_nanquantile(host_data, result, batch_size, reduce_size, q);
    cudaMemcpy(out->data, result, batch_size * sizeof(float),
               cudaMemcpyHostToDevice);
    delete[] host_data;
    delete[] result;
    break;
  }
  case INSIGHT_DTYPE_F64: {
    double *host_data = new double[prepared->numel];
    cudaMemcpy(host_data, prepared->data, prepared->numel * sizeof(double),
               cudaMemcpyDeviceToHost);
    double *result = new double[batch_size];
    compute_nanquantile(host_data, result, batch_size, reduce_size, q);
    cudaMemcpy(out->data, result, batch_size * sizeof(double),
               cudaMemcpyHostToDevice);
    delete[] host_data;
    delete[] result;
    break;
  }
  default:
    gpu_set_last_error("nanquantile: unsupported dtype");
    return C_FAILED;
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_ILUVATAR_KERNEL(nanquantile, INSIGHT_DTYPE_F32,
                         nanquantile_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(nanquantile, INSIGHT_DTYPE_F64,
                         nanquantile_kernel_gpu);
