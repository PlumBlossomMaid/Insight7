// backends/cuda/kernels/manipulation/pad.cu
/**
 * @file pad.cu
 * @brief CUDA kernel for pad operation.
 */

#include "../../registry/iluvatar_registry.h"
#include "insight/c_api/array.h"
#include <cstring>
#include <cuda_runtime.h>

extern "C" {

C_Status pad_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = static_cast<InsightArray *>(outputs[0]);
  InsightArray *x = static_cast<InsightArray *>(inputs[1]);

  if (!out || !x) {
    gpu_set_last_error("pad: null array pointer");
    return C_FAILED;
  }

  int64_t *pad_width = static_cast<int64_t *>(inputs[2]);
  double constant_value = *static_cast<double *>(inputs[3]);

  int64_t ndim = x->ndim;
  int64_t element_size = 1;
  switch (out->dtype) {
  case INSIGHT_DTYPE_BOOL:
  case INSIGHT_DTYPE_U8:
  case INSIGHT_DTYPE_I8:
    element_size = 1;
    break;
  case INSIGHT_DTYPE_I16:
  case INSIGHT_DTYPE_U16:
    element_size = 2;
    break;
  case INSIGHT_DTYPE_I32:
  case INSIGHT_DTYPE_U32:
  case INSIGHT_DTYPE_F32:
    element_size = 4;
    break;
  case INSIGHT_DTYPE_I64:
  case INSIGHT_DTYPE_U64:
  case INSIGHT_DTYPE_F64:
  case INSIGHT_DTYPE_C32:
    element_size = 8;
    break;
  case INSIGHT_DTYPE_C64:
    element_size = 16;
    break;
  default:
    gpu_set_last_error("pad: unsupported dtype");
    return C_FAILED;
  }

  // Fill output with constant value (simplified - assumes contiguous)
  cudaMemset(out->data, 0, out->numel * element_size);

  // Compute strides
  int64_t out_strides[10];
  out_strides[ndim - 1] = 1;
  for (int d = ndim - 2; d >= 0; --d) {
    out_strides[d] = out_strides[d + 1] * out->dims[d + 1];
  }

  // Copy input data to appropriate location in output
  // This is a simplified implementation
  int64_t in_strides[10];
  in_strides[ndim - 1] = 1;
  for (int d = ndim - 2; d >= 0; --d) {
    in_strides[d] = in_strides[d + 1] * x->dims[d + 1];
  }

  // For each element in input, compute output position
  for (int64_t linear = 0; linear < x->numel; ++linear) {
    int64_t indices[10];
    int64_t remaining = linear;
    for (int d = ndim - 1; d >= 0; --d) {
      indices[d] = remaining % x->dims[d];
      remaining /= x->dims[d];
    }

    // Add padding offset
    int64_t out_linear = 0;
    for (int d = 0; d < ndim; ++d) {
      out_linear += (indices[d] + pad_width[2 * d]) * out_strides[d];
    }

    cudaMemcpy(static_cast<char *>(out->data) + out_linear * element_size,
               static_cast<const char *>(x->data) + linear * element_size,
               element_size, cudaMemcpyDeviceToDevice);
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_ILUVATAR_KERNEL(pad, INSIGHT_DTYPE_BOOL, pad_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(pad, INSIGHT_DTYPE_U8, pad_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(pad, INSIGHT_DTYPE_I8, pad_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(pad, INSIGHT_DTYPE_I16, pad_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(pad, INSIGHT_DTYPE_I32, pad_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(pad, INSIGHT_DTYPE_I64, pad_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(pad, INSIGHT_DTYPE_U16, pad_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(pad, INSIGHT_DTYPE_U32, pad_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(pad, INSIGHT_DTYPE_U64, pad_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(pad, INSIGHT_DTYPE_F32, pad_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(pad, INSIGHT_DTYPE_F64, pad_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(pad, INSIGHT_DTYPE_C32, pad_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(pad, INSIGHT_DTYPE_C64, pad_kernel_gpu);
