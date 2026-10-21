// backends/cuda/kernels/indexing/compress.cu
/**
 * @file compress.cu
 * @brief CUDA kernel for compress operation.
 *
 * Returns selected slices of an array along an axis.
 * Uses CPU fallback.
 */

#include "../../registry/ixuca_registry.h"
#include "insight/c_api/array.h"
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <vector>

extern "C" {

C_Status compress_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = static_cast<InsightArray *>(outputs[0]);
  InsightArray *x = static_cast<InsightArray *>(inputs[1]);
  InsightArray *condition = static_cast<InsightArray *>(inputs[2]);

  if (!out || !x || !condition) {
    gpu_set_last_error("compress: null array pointer");
    return C_FAILED;
  }

  int axis = *static_cast<int *>(inputs[3]);

  int64_t ndim = x->ndim;
  if (axis < 0)
    axis += ndim;

  int64_t n = x->numel;
  int64_t axis_size = x->dims[axis];
  int64_t outer_size = 1;
  for (int d = 0; d < axis; ++d) {
    outer_size *= x->dims[d];
  }
  int64_t inner_size = 1;
  for (int d = axis + 1; d < ndim; ++d) {
    inner_size *= x->dims[d];
  }

  // Copy inputs to CPU
  float *x_data = new float[n];
  bool *cond_data = new bool[axis_size];
  cudaMemcpy(x_data, x->data, n * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(cond_data, condition->data, axis_size * sizeof(bool),
             cudaMemcpyDeviceToHost);

  // Count true conditions
  int64_t count = 0;
  for (int64_t i = 0; i < axis_size; ++i) {
    if (cond_data[i])
      ++count;
  }

  // Compress on CPU
  float *result = new float[outer_size * count * inner_size];
  for (int64_t outer = 0; outer < outer_size; ++outer) {
    int64_t out_idx = 0;
    for (int64_t i = 0; i < axis_size; ++i) {
      if (cond_data[i]) {
        for (int64_t inner = 0; inner < inner_size; ++inner) {
          int64_t src_idx =
              outer * axis_size * inner_size + i * inner_size + inner;
          int64_t dst_idx =
              outer * count * inner_size + out_idx * inner_size + inner;
          result[dst_idx] = x_data[src_idx];
        }
        ++out_idx;
      }
    }
  }

  // Copy result back to GPU
  if (outer_size * count * inner_size > 0) {
    cudaMemcpy(out->data, result,
               outer_size * count * inner_size * sizeof(float),
               cudaMemcpyHostToDevice);
  }

  delete[] x_data;
  delete[] cond_data;
  delete[] result;

  return C_SUCCESS;
}

} // extern "C"

REGISTER_IXUCA_KERNEL(compress, INSIGHT_DTYPE_F32, compress_kernel_gpu);
