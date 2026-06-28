// backends/cuda/kernels/indexing/masked_select.cu
/**
 * @file masked_select.cu
 * @brief CUDA kernel for masked_select operation.
 *
 * Returns elements selected by a boolean mask.
 * Uses CPU fallback with dtype-aware byte copy.
 */

#include "../../registry/iluvatar_registry.h"
#include "insight/c_api/array.h"
#include <cstring>
#include <cuda_runtime.h>

extern "C" {

C_Status masked_select_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = static_cast<InsightArray *>(outputs[0]);
  InsightArray *x = static_cast<InsightArray *>(inputs[1]);
  InsightArray *mask = static_cast<InsightArray *>(inputs[2]);

  if (!out || !x || !mask) {
    gpu_set_last_error("masked_select: null array pointer");
    return C_FAILED;
  }

  int64_t n = x->numel;
  size_t elem_size = insight_dtype_size(x->dtype);
  if (elem_size == 0) {
    gpu_set_last_error("masked_select: unknown dtype");
    return C_FAILED;
  }

  // Copy inputs to CPU
  char *x_data = new char[n * elem_size];
  bool *mask_data = new bool[n];
  cudaMemcpy(x_data, x->data, n * elem_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(mask_data, mask->data, n * sizeof(bool), cudaMemcpyDeviceToHost);

  // Select elements where mask is true
  int64_t count = 0;
  for (int64_t i = 0; i < n; ++i) {
    if (mask_data[i]) {
      ++count;
    }
  }

  char *result = new char[count * elem_size];
  int64_t idx = 0;
  for (int64_t i = 0; i < n; ++i) {
    if (mask_data[i]) {
      std::memcpy(result + idx * elem_size, x_data + i * elem_size, elem_size);
      ++idx;
    }
  }

  // Copy result back to GPU
  if (count > 0) {
    cudaMemcpy(out->data, result, count * elem_size, cudaMemcpyHostToDevice);
  }

  delete[] result;
  delete[] x_data;
  delete[] mask_data;

  return C_SUCCESS;
}

} // extern "C"

REGISTER_ILUVATAR_KERNEL(masked_select, INSIGHT_DTYPE_BOOL,
                    masked_select_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(masked_select, INSIGHT_DTYPE_U8, masked_select_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(masked_select, INSIGHT_DTYPE_I8, masked_select_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(masked_select, INSIGHT_DTYPE_I16, masked_select_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(masked_select, INSIGHT_DTYPE_I32, masked_select_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(masked_select, INSIGHT_DTYPE_I64, masked_select_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(masked_select, INSIGHT_DTYPE_U16, masked_select_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(masked_select, INSIGHT_DTYPE_U32, masked_select_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(masked_select, INSIGHT_DTYPE_U64, masked_select_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(masked_select, INSIGHT_DTYPE_F16, masked_select_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(masked_select, INSIGHT_DTYPE_BF16,
                    masked_select_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(masked_select, INSIGHT_DTYPE_F32, masked_select_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(masked_select, INSIGHT_DTYPE_F64, masked_select_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(masked_select, INSIGHT_DTYPE_C32, masked_select_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(masked_select, INSIGHT_DTYPE_C64, masked_select_kernel_gpu);
