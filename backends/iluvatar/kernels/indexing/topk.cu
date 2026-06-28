// backends/cuda/kernels/indexing/topk.cu
/**
 * @file topk.cu
 * @brief CUDA kernel for topk operation (CPU fallback with dtype support).
 *
 * Input layout (matches CPU kernel):
 *   inputs[0] = InsightArray* values output
 *   inputs[1] = InsightArray* indices output
 *   inputs[2] = InsightArray* input (already transposed so axis is last)
 *   inputs[3] = int64_t* k
 *   inputs[4] = bool* largest
 *   inputs[5] = bool* sorted
 */

#include "../../registry/iluvatar_registry.h"
#include "insight/c_api/array.h"
#include <algorithm>
#include <cstring>
#include <cuda_runtime.h>
#include <utility>

extern "C" {

C_Status topk_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out_values = static_cast<InsightArray *>(outputs[0]);
  InsightArray *out_indices = static_cast<InsightArray *>(outputs[1]);
  InsightArray *x = static_cast<InsightArray *>(inputs[2]);

  if (!out_values || !out_indices || !x) {
    gpu_set_last_error("topk: null array pointer");
    return C_FAILED;
  }
  if (!out_values->data || !out_indices->data || !x->data) {
    gpu_set_last_error("topk: null data pointer");
    return C_FAILED;
  }

  int64_t k = *static_cast<int64_t *>(inputs[3]);
  bool largest = *static_cast<bool *>(inputs[4]);

  int64_t ndim = x->ndim;
  int64_t n = x->numel;
  int64_t axis_size = x->dims[ndim - 1];
  int64_t outer_size = n / axis_size;

  if (k > axis_size)
    k = axis_size;

  int64_t out_n = outer_size * k;

  switch (x->dtype) {
  case INSIGHT_DTYPE_F32: {
    float *host_data = new float[n];
    cudaMemcpy(host_data, x->data, n * sizeof(float), cudaMemcpyDeviceToHost);
    float *values_data = new float[out_n];
    int64_t *indices_data = new int64_t[out_n];
    for (int64_t batch = 0; batch < outer_size; ++batch) {
      std::pair<float, int64_t> *pairs =
          new std::pair<float, int64_t>[axis_size];
      for (int64_t i = 0; i < axis_size; ++i)
        pairs[i] = {host_data[batch * axis_size + i], i};
      if (largest)
        std::partial_sort(pairs, pairs + k, pairs + axis_size,
                          [](auto &a, auto &b) { return a.first > b.first; });
      else
        std::partial_sort(pairs, pairs + k, pairs + axis_size,
                          [](auto &a, auto &b) { return a.first < b.first; });
      for (int64_t i = 0; i < k; ++i) {
        values_data[batch * k + i] = pairs[i].first;
        indices_data[batch * k + i] = pairs[i].second;
      }
      delete[] pairs;
    }
    cudaMemcpy(out_values->data, values_data, out_n * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(out_indices->data, indices_data, out_n * sizeof(int64_t),
               cudaMemcpyHostToDevice);
    delete[] host_data;
    delete[] values_data;
    delete[] indices_data;
    break;
  }
  case INSIGHT_DTYPE_F64: {
    double *host_data = new double[n];
    cudaMemcpy(host_data, x->data, n * sizeof(double), cudaMemcpyDeviceToHost);
    double *values_data = new double[out_n];
    int64_t *indices_data = new int64_t[out_n];
    for (int64_t batch = 0; batch < outer_size; ++batch) {
      std::pair<double, int64_t> *pairs =
          new std::pair<double, int64_t>[axis_size];
      for (int64_t i = 0; i < axis_size; ++i)
        pairs[i] = {host_data[batch * axis_size + i], i};
      if (largest)
        std::partial_sort(pairs, pairs + k, pairs + axis_size,
                          [](auto &a, auto &b) { return a.first > b.first; });
      else
        std::partial_sort(pairs, pairs + k, pairs + axis_size,
                          [](auto &a, auto &b) { return a.first < b.first; });
      for (int64_t i = 0; i < k; ++i) {
        values_data[batch * k + i] = pairs[i].first;
        indices_data[batch * k + i] = pairs[i].second;
      }
      delete[] pairs;
    }
    cudaMemcpy(out_values->data, values_data, out_n * sizeof(double),
               cudaMemcpyHostToDevice);
    cudaMemcpy(out_indices->data, indices_data, out_n * sizeof(int64_t),
               cudaMemcpyHostToDevice);
    delete[] host_data;
    delete[] values_data;
    delete[] indices_data;
    break;
  }
  default:
    gpu_set_last_error("topk: unsupported dtype on CUDA");
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

REGISTER_ILUVATAR_KERNEL(topk, INSIGHT_DTYPE_F32, topk_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(topk, INSIGHT_DTYPE_F64, topk_kernel_gpu);
