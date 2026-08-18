// backends/cuda/kernels/indexing/lexsort.cu
/**
 * @file lexsort.cu
 * @brief CUDA kernel for lexsort operation.
 *
 * Performs indirect stable sort using a sequence of keys.
 * Uses CPU fallback.
 *
 * Input layout (matches CPU kernel):
 *   inputs[0] = InsightArray* output (indices)
 *   inputs[1] = InsightArray* transposed (keys flattened, shape [batch, nkeys,
 * last_dim]) inputs[2] = int64_t* batch_size inputs[3] = int64_t* last_dim
 *   inputs[4] = int64_t* nkeys
 */

#include "../../registry/ixuca_registry.h"
#include "insight/c_api/array.h"
#include <algorithm>
#include <cstring>
#include <cuda_runtime.h>
#include <vector>

template <typename T>
static void lexsort_impl(const T *host_data, int64_t *result,
                         int64_t batch_size, int64_t last_dim, int64_t nkeys) {
  for (int64_t batch = 0; batch < batch_size; ++batch) {
    // Extract keys for this batch
    std::vector<std::vector<T>> keys(nkeys);
    for (int64_t k = 0; k < nkeys; ++k) {
      keys[k].resize(last_dim);
      const T *key_start = host_data + (batch * nkeys + k) * last_dim;
      for (int64_t i = 0; i < last_dim; ++i) {
        keys[k][i] = key_start[i];
      }
    }

    // Create index array
    std::vector<int64_t> indices(last_dim);
    for (int64_t i = 0; i < last_dim; ++i) {
      indices[i] = i;
    }

    // Sort using lexicographic ordering (key 0 has highest priority)
    std::sort(indices.begin(), indices.end(),
              [&keys, nkeys](int64_t a, int64_t b) {
                for (int64_t k = 0; k < nkeys; ++k) {
                  if (keys[k][a] < keys[k][b])
                    return true;
                  if (keys[k][a] > keys[k][b])
                    return false;
                }
                return false;
              });

    for (int64_t i = 0; i < last_dim; ++i) {
      result[batch * last_dim + i] = indices[i];
    }
  }
}

extern "C" {

C_Status lexsort_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = static_cast<InsightArray *>(outputs[0]);
  InsightArray *transposed = static_cast<InsightArray *>(inputs[1]);

  if (!out || !transposed) {
    gpu_set_last_error("lexsort: null array pointer");
    return C_FAILED;
  }

  int64_t batch_size = *static_cast<int64_t *>(inputs[2]);
  int64_t last_dim = *static_cast<int64_t *>(inputs[3]);
  int64_t nkeys = *static_cast<int64_t *>(inputs[4]);

  int64_t total = transposed->numel;
  int64_t *result = new int64_t[batch_size * last_dim];

  switch (transposed->dtype) {
  case INSIGHT_DTYPE_F32: {
    float *host_data = new float[total];
    cudaMemcpy(host_data, transposed->data, total * sizeof(float),
               cudaMemcpyDeviceToHost);
    lexsort_impl(host_data, result, batch_size, last_dim, nkeys);
    delete[] host_data;
    break;
  }
  case INSIGHT_DTYPE_F64: {
    double *host_data = new double[total];
    cudaMemcpy(host_data, transposed->data, total * sizeof(double),
               cudaMemcpyDeviceToHost);
    lexsort_impl(host_data, result, batch_size, last_dim, nkeys);
    delete[] host_data;
    break;
  }
  case INSIGHT_DTYPE_I32: {
    int32_t *host_data = new int32_t[total];
    cudaMemcpy(host_data, transposed->data, total * sizeof(int32_t),
               cudaMemcpyDeviceToHost);
    lexsort_impl(host_data, result, batch_size, last_dim, nkeys);
    delete[] host_data;
    break;
  }
  case INSIGHT_DTYPE_I64: {
    int64_t *host_data = new int64_t[total];
    cudaMemcpy(host_data, transposed->data, total * sizeof(int64_t),
               cudaMemcpyDeviceToHost);
    lexsort_impl(host_data, result, batch_size, last_dim, nkeys);
    delete[] host_data;
    break;
  }
  default:
    delete[] result;
    gpu_set_last_error("lexsort: unsupported dtype");
    return C_FAILED;
  }

  // Copy result back to GPU
  cudaMemcpy(out->data, result, batch_size * last_dim * sizeof(int64_t),
             cudaMemcpyHostToDevice);

  delete[] result;

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_IXUCA_KERNEL(lexsort, INSIGHT_DTYPE_F32, lexsort_kernel_gpu);
REGISTER_IXUCA_KERNEL(lexsort, INSIGHT_DTYPE_F64, lexsort_kernel_gpu);
REGISTER_IXUCA_KERNEL(lexsort, INSIGHT_DTYPE_I32, lexsort_kernel_gpu);
REGISTER_IXUCA_KERNEL(lexsort, INSIGHT_DTYPE_I64, lexsort_kernel_gpu);
