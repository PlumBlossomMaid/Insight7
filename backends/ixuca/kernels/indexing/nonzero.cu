// backends/cuda/kernels/indexing/nonzero.cu
/**
 * @file nonzero.cu
 * @brief CUDA kernel for nonzero operation.
 *
 * Returns indices of non-zero elements.
 * Uses CPU fallback with proper output allocation.
 */

#include "../../registry/ixuca_registry.h"
#include "insight/c_api/array.h"
#include "insight/c_api/place.h"
#include <cstring>
#include <cuda_runtime.h>
#include <vector>

static void allocate_gpu_output(InsightArray *out, int64_t ndim,
                                const int64_t *dims, int64_t numel) {
  size_t bytes = numel * sizeof(int64_t);
  void *data = nullptr;
  if (bytes > 0) {
    cudaMalloc(&data, bytes);
  }
  out->data = data;
  out->storage_nbytes = bytes;
  out->ndim = ndim;
  for (int i = 0; i < ndim; ++i) {
    out->dims[i] = dims[i];
  }
  out->numel = numel;
  out->dtype = INSIGHT_DTYPE_I64;
  if (ndim > 0) {
    out->strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i) {
      out->strides[i] = out->strides[i + 1] * out->dims[i + 1];
    }
  }
  out->offset = 0;
  out->is_view = 0;
  out->device_type = INSIGHT_DEVICE_GPU;
  out->device_id = 0;
  if (!out->ref_count) {
    out->ref_count = new int32_t;
    if (out->ref_count)
      *out->ref_count = 1;
  }
}

template <typename T>
static void nonzero_impl(InsightArray *x, InsightArray *out) {
  int64_t total = x->numel;
  int64_t ndim = x->ndim;

  // Copy input to CPU
  T *host_data = new T[total];
  cudaMemcpy(host_data, x->data, total * sizeof(T), cudaMemcpyDeviceToHost);

  // Copy dims and strides to CPU
  int64_t *dims = new int64_t[ndim];
  int64_t *strides = new int64_t[ndim];
  memcpy(dims, x->dims, ndim * sizeof(int64_t));
  memcpy(strides, x->strides, ndim * sizeof(int64_t));

  // Count non-zero elements
  int64_t nz_count = 0;
  for (int64_t linear = 0; linear < total; ++linear) {
    int64_t indices[INSIGHT_MAX_NDIM];
    int64_t tmp = linear;
    for (int d = ndim - 1; d >= 0; --d) {
      indices[d] = tmp % dims[d];
      tmp /= dims[d];
    }
    int64_t offset = 0;
    for (int d = 0; d < ndim; ++d) {
      offset += indices[d] * strides[d];
    }
    if (host_data[offset] != T(0)) {
      ++nz_count;
    }
  }

  // Allocate output
  int64_t out_dims[2] = {ndim, nz_count};
  allocate_gpu_output(out, 2, out_dims, ndim * nz_count);

  if (nz_count == 0) {
    delete[] host_data;
    delete[] dims;
    delete[] strides;
    return;
  }

  // Fill indices on CPU
  int64_t *host_indices = new int64_t[ndim * nz_count];
  int64_t out_idx = 0;
  for (int64_t linear = 0; linear < total && out_idx < nz_count; ++linear) {
    int64_t indices[INSIGHT_MAX_NDIM];
    int64_t tmp = linear;
    for (int d = ndim - 1; d >= 0; --d) {
      indices[d] = tmp % dims[d];
      tmp /= dims[d];
    }
    int64_t offset = 0;
    for (int d = 0; d < ndim; ++d) {
      offset += indices[d] * strides[d];
    }
    if (host_data[offset] != T(0)) {
      for (int d = 0; d < ndim; ++d) {
        host_indices[d * nz_count + out_idx] = indices[d];
      }
      ++out_idx;
    }
  }

  // Copy indices to GPU
  cudaMemcpy(out->data, host_indices, ndim * nz_count * sizeof(int64_t),
             cudaMemcpyHostToDevice);

  delete[] host_data;
  delete[] dims;
  delete[] strides;
  delete[] host_indices;
}

extern "C" {

C_Status nonzero_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *x = static_cast<InsightArray *>(inputs[0]);
  InsightArray *out = static_cast<InsightArray *>(outputs[0]);

  if (!x || !out) {
    gpu_set_last_error("nonzero: null array pointer");
    return C_FAILED;
  }

  switch (x->dtype) {
  case INSIGHT_DTYPE_BOOL:
    nonzero_impl<bool>(x, out);
    break;
  case INSIGHT_DTYPE_U8:
    nonzero_impl<uint8_t>(x, out);
    break;
  case INSIGHT_DTYPE_I8:
    nonzero_impl<int8_t>(x, out);
    break;
  case INSIGHT_DTYPE_I16:
    nonzero_impl<int16_t>(x, out);
    break;
  case INSIGHT_DTYPE_I32:
    nonzero_impl<int32_t>(x, out);
    break;
  case INSIGHT_DTYPE_I64:
    nonzero_impl<int64_t>(x, out);
    break;
  case INSIGHT_DTYPE_U16:
    nonzero_impl<uint16_t>(x, out);
    break;
  case INSIGHT_DTYPE_U32:
    nonzero_impl<uint32_t>(x, out);
    break;
  case INSIGHT_DTYPE_U64:
    nonzero_impl<uint64_t>(x, out);
    break;
  case INSIGHT_DTYPE_F32:
    nonzero_impl<float>(x, out);
    break;
  case INSIGHT_DTYPE_F64:
    nonzero_impl<double>(x, out);
    break;
  default:
    gpu_set_last_error("nonzero: unsupported dtype");
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_IXUCA_KERNEL(nonzero, INSIGHT_DTYPE_BOOL, nonzero_kernel_gpu);
REGISTER_IXUCA_KERNEL(nonzero, INSIGHT_DTYPE_U8, nonzero_kernel_gpu);
REGISTER_IXUCA_KERNEL(nonzero, INSIGHT_DTYPE_I8, nonzero_kernel_gpu);
REGISTER_IXUCA_KERNEL(nonzero, INSIGHT_DTYPE_I16, nonzero_kernel_gpu);
REGISTER_IXUCA_KERNEL(nonzero, INSIGHT_DTYPE_I32, nonzero_kernel_gpu);
REGISTER_IXUCA_KERNEL(nonzero, INSIGHT_DTYPE_I64, nonzero_kernel_gpu);
REGISTER_IXUCA_KERNEL(nonzero, INSIGHT_DTYPE_U16, nonzero_kernel_gpu);
REGISTER_IXUCA_KERNEL(nonzero, INSIGHT_DTYPE_U32, nonzero_kernel_gpu);
REGISTER_IXUCA_KERNEL(nonzero, INSIGHT_DTYPE_U64, nonzero_kernel_gpu);
REGISTER_IXUCA_KERNEL(nonzero, INSIGHT_DTYPE_F32, nonzero_kernel_gpu);
REGISTER_IXUCA_KERNEL(nonzero, INSIGHT_DTYPE_F64, nonzero_kernel_gpu);
