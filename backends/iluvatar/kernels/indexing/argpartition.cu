// backends/cuda/kernels/indexing/argpartition.cu
/**
 * @file argpartition.cu
 * @brief CUDA kernel for argpartition operation.
 *
 * Returns indices that partially sort the array.
 * Uses CPU fallback.
 */

#include "../../registry/iluvatar_registry.h"
#include "insight/c_api/array.h"
#include <algorithm>
#include <cstring>
#include <cuda_runtime.h>
#include <vector>

extern "C" {

C_Status argpartition_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = static_cast<InsightArray *>(outputs[0]);
  InsightArray *x = static_cast<InsightArray *>(inputs[1]);

  if (!out || !x) {
    gpu_set_last_error("argpartition: null array pointer");
    return C_FAILED;
  }

  int64_t kth = *static_cast<int64_t *>(inputs[2]);
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

  // Copy input to CPU
  float *x_data = new float[n];
  cudaMemcpy(x_data, x->data, n * sizeof(float), cudaMemcpyDeviceToHost);

  // Compute argpartition on CPU
  int64_t *result = new int64_t[n];
  for (int64_t outer = 0; outer < outer_size; ++outer) {
    for (int64_t inner = 0; inner < inner_size; ++inner) {
      int64_t base = outer * axis_size * inner_size + inner;

      // Create index array
      std::vector<int64_t> indices(axis_size);
      for (int64_t i = 0; i < axis_size; ++i) {
        indices[i] = i;
      }

      // Partial sort (nth element)
      std::nth_element(indices.begin(), indices.begin() + kth, indices.end(),
                       [&x_data, base, inner_size](int64_t a, int64_t b) {
                         return x_data[base + a * inner_size] <
                                x_data[base + b * inner_size];
                       });

      // Write indices
      for (int64_t i = 0; i < axis_size; ++i) {
        result[base + i * inner_size] = indices[i];
      }
    }
  }

  // Copy result back to GPU
  cudaMemcpy(out->data, result, n * sizeof(int64_t), cudaMemcpyHostToDevice);

  delete[] x_data;
  delete[] result;

  return C_SUCCESS;
}

} // extern "C"

REGISTER_ILUVATAR_KERNEL(argpartition, INSIGHT_DTYPE_F32,
                         argpartition_kernel_gpu);
