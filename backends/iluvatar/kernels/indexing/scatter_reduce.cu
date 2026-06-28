// backends/cuda/kernels/indexing/scatter_reduce.cu
/**
 * @file scatter_reduce.cu
 * @brief CUDA kernel for scatter, scatter_add, scatter_reduce operations.
 *
 * Input layout (matches CPU kernel):
 *   inputs[0] = InsightArray* output (also used as initial values)
 *   inputs[1] = InsightArray* indices
 *   inputs[2] = InsightArray* source values
 *   inputs[3] = int* axis
 *   inputs[4] = const char* reduce mode ("replace", "add", "mul", "max", "min")
 */

#include "../../registry/iluvatar_registry.h"
#include "../common/atomic_helpers.cuh"
#include "insight/c_api/array.h"
#include <cstring>
#include <cuda_runtime.h>

template <typename T>
__global__ void scatter_replace_kernel(T *out, const int64_t *indices,
                                       const T *src, int64_t n,
                                       int64_t index_size, int64_t axis_size) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    int64_t batch = idx / index_size;
    int64_t i = idx % index_size;
    int64_t dst_idx = batch * axis_size + indices[i];
    out[dst_idx] = src[idx];
  }
}

template <typename T>
__global__ void scatter_add_kernel(T *out, const int64_t *indices, const T *src,
                                   int64_t n, int64_t index_size,
                                   int64_t axis_size) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    int64_t batch = idx / index_size;
    int64_t i = idx % index_size;
    int64_t dst_idx = batch * axis_size + indices[i];
    if constexpr (std::is_same_v<T, double>) {
      atomicAddDouble(&out[dst_idx], src[idx]);
    } else {
      atomicAdd(&out[dst_idx], src[idx]);
    }
  }
}

extern "C" {

C_Status scatter_reduce_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = static_cast<InsightArray *>(outputs[0]);
  InsightArray *idx = static_cast<InsightArray *>(inputs[1]);
  InsightArray *src = static_cast<InsightArray *>(inputs[2]);
  int axis = *static_cast<int *>(inputs[3]);
  const char *reduce = static_cast<const char *>(inputs[4]);

  if (!out || !idx || !src) {
    gpu_set_last_error("scatter_reduce: null array pointer");
    return C_FAILED;
  }

  bool is_replace = (strcmp(reduce, "replace") == 0);
  bool is_add = (strcmp(reduce, "add") == 0);

  if (!is_replace && !is_add) {
    gpu_set_last_error(
        "scatter_reduce: only 'replace' and 'add' modes supported on CUDA");
    return C_FAILED;
  }

  int64_t n = src->numel;
  if (n == 0)
    return C_SUCCESS;

  int64_t index_size = idx->numel / (n / src->dims[axis]);
  int64_t axis_size = out->dims[axis];

  int threads = 256;
  int blocks = static_cast<int>((n + threads - 1) / threads);

  switch (out->dtype) {
  case INSIGHT_DTYPE_F32:
    if (is_replace) {
      scatter_replace_kernel<float><<<blocks, threads>>>(
          static_cast<float *>(out->data),
          static_cast<const int64_t *>(idx->data),
          static_cast<const float *>(src->data), n, index_size, axis_size);
    } else {
      scatter_add_kernel<float><<<blocks, threads>>>(
          static_cast<float *>(out->data),
          static_cast<const int64_t *>(idx->data),
          static_cast<const float *>(src->data), n, index_size, axis_size);
    }
    break;
  case INSIGHT_DTYPE_F64:
    if (is_replace) {
      scatter_replace_kernel<double><<<blocks, threads>>>(
          static_cast<double *>(out->data),
          static_cast<const int64_t *>(idx->data),
          static_cast<const double *>(src->data), n, index_size, axis_size);
    } else {
      scatter_add_kernel<double><<<blocks, threads>>>(
          static_cast<double *>(out->data),
          static_cast<const int64_t *>(idx->data),
          static_cast<const double *>(src->data), n, index_size, axis_size);
    }
    break;
  default:
    gpu_set_last_error("scatter_reduce: unsupported dtype");
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

REGISTER_ILUVATAR_KERNEL(scatter_reduce, INSIGHT_DTYPE_F32,
                         scatter_reduce_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(scatter_reduce, INSIGHT_DTYPE_F64,
                         scatter_reduce_kernel_gpu);
