// backends/cuda/kernels/indexing/partition.cu
/**
 * @file partition.cu
 * @brief CUDA kernel for partition operation.
 *
 * Partially sorts array so that the k-th element is in its sorted position.
 * Uses CPU fallback.
 *
 * Input layout (matches CPU kernel):
 *   inputs[0] = InsightArray* output
 *   inputs[1] = InsightArray* input
 *   inputs[2] = int64_t* kth
 *   inputs[3] = int* axis
 */

#include "../../registry/ixuca_registry.h"
#include "insight/c_api/array.h"
#include <algorithm>
#include <cstring>
#include <cuda_runtime.h>
#include <vector>

template <typename T>
static void partition_impl(const T *src, T *dst, int64_t ndim,
                           const int64_t *dims, int axis, int64_t kth) {
  int64_t axis_size = dims[axis];
  int64_t inner_size = 1;
  for (int d = axis + 1; d < ndim; ++d) {
    inner_size *= dims[d];
  }
  int64_t outer_size = 1;
  for (int d = 0; d < axis; ++d) {
    outer_size *= dims[d];
  }

  for (int64_t outer = 0; outer < outer_size; ++outer) {
    for (int64_t inner = 0; inner < inner_size; ++inner) {
      int64_t base = outer * axis_size * inner_size + inner;

      // Copy slice data
      std::vector<T> slice(axis_size);
      for (int64_t i = 0; i < axis_size; ++i) {
        slice[i] = src[base + i * inner_size];
      }

      // Partition
      std::nth_element(slice.begin(), slice.begin() + kth, slice.end());

      // Write back
      for (int64_t i = 0; i < axis_size; ++i) {
        dst[base + i * inner_size] = slice[i];
      }
    }
  }
}

extern "C" {

C_Status partition_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = static_cast<InsightArray *>(outputs[0]);
  InsightArray *x = static_cast<InsightArray *>(inputs[1]);

  if (!out || !x) {
    gpu_set_last_error("partition: null array pointer");
    return C_FAILED;
  }

  int64_t kth = *static_cast<int64_t *>(inputs[2]);
  int axis = *static_cast<int *>(inputs[3]);

  int64_t ndim = x->ndim;
  if (axis < 0)
    axis += ndim;

  int64_t n = x->numel;
  int64_t dims[INSIGHT_MAX_NDIM];
  for (int i = 0; i < ndim; ++i) {
    dims[i] = x->dims[i];
  }

  switch (x->dtype) {
  case INSIGHT_DTYPE_F32: {
    float *host_data = new float[n];
    cudaMemcpy(host_data, x->data, n * sizeof(float), cudaMemcpyDeviceToHost);
    partition_impl(host_data, host_data, ndim, dims, axis, kth);
    cudaMemcpy(out->data, host_data, n * sizeof(float), cudaMemcpyHostToDevice);
    delete[] host_data;
    break;
  }
  case INSIGHT_DTYPE_F64: {
    double *host_data = new double[n];
    cudaMemcpy(host_data, x->data, n * sizeof(double), cudaMemcpyDeviceToHost);
    partition_impl(host_data, host_data, ndim, dims, axis, kth);
    cudaMemcpy(out->data, host_data, n * sizeof(double),
               cudaMemcpyHostToDevice);
    delete[] host_data;
    break;
  }
  case INSIGHT_DTYPE_I32: {
    int32_t *host_data = new int32_t[n];
    cudaMemcpy(host_data, x->data, n * sizeof(int32_t), cudaMemcpyDeviceToHost);
    partition_impl(host_data, host_data, ndim, dims, axis, kth);
    cudaMemcpy(out->data, host_data, n * sizeof(int32_t),
               cudaMemcpyHostToDevice);
    delete[] host_data;
    break;
  }
  case INSIGHT_DTYPE_I64: {
    int64_t *host_data = new int64_t[n];
    cudaMemcpy(host_data, x->data, n * sizeof(int64_t), cudaMemcpyDeviceToHost);
    partition_impl(host_data, host_data, ndim, dims, axis, kth);
    cudaMemcpy(out->data, host_data, n * sizeof(int64_t),
               cudaMemcpyHostToDevice);
    delete[] host_data;
    break;
  }
  default:
    gpu_set_last_error("partition: unsupported dtype");
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

REGISTER_IXUCA_KERNEL(partition, INSIGHT_DTYPE_F32, partition_kernel_gpu);
REGISTER_IXUCA_KERNEL(partition, INSIGHT_DTYPE_F64, partition_kernel_gpu);
REGISTER_IXUCA_KERNEL(partition, INSIGHT_DTYPE_I32, partition_kernel_gpu);
REGISTER_IXUCA_KERNEL(partition, INSIGHT_DTYPE_I64, partition_kernel_gpu);
