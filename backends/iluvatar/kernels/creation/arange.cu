// backends/cuda/kernels/creation/arange.cu
/**
 * @file arange.cu
 * @brief CUDA kernel for the arange operation.
 *
 * Generates a 1D array with evenly spaced values within a given interval.
 * Values are computed as: start + i * step for i = 0 to numel-1.
 * Supports integer and floating-point types.
 */
#include "../../registry/iluvatar_registry.h"
#include "common.cuh"
#include "insight/c_api/array.h"
#include <cuda_runtime.h>
#include <string>

/**
 * @brief CUDA kernel to generate arange values.
 *
 * Each thread computes one element: out[i] = static_cast<T>(start + i * step)
 *
 * @tparam T Output element type
 * @param out Output array
 * @param n Number of elements
 * @param start Start value
 * @param step Step size between elements
 */
template <typename T>
__global__ void arange_kernel(T *out, int64_t n, double start, double step) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = static_cast<T>(start + idx * step);
  }
}

extern "C" {

/**
 * @brief GPU entry point for the arange kernel.
 *
 * Generates a 1D array with evenly spaced values.
 *
 * @param inputs  [0] = InsightArray* output, [1] = double* start, [2] = double*
 *                step
 * @param outputs [0] = InsightArray* result (same as inputs[0])
 * @return C_SUCCESS on success, C_FAILED on error
 */
C_Status arange_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = static_cast<InsightArray *>(outputs[0]);

  if (!out) {
    gpu_set_last_error("arange_kernel_gpu: output array is null");
    return C_FAILED;
  }
  if (!inputs[1] || !inputs[2]) {
    gpu_set_last_error("arange_kernel_gpu: start or step is null");
    return C_FAILED;
  }

  double start = *static_cast<double *>(inputs[1]);
  double step = *static_cast<double *>(inputs[2]);
  int64_t n = out->numel;
  if (n == 0)
    return C_SUCCESS;
  int32_t dtype = out->dtype;

  int threads = creation_threads();
  int blocks = creation_blocks(n);

  switch (dtype) {
  case INSIGHT_DTYPE_I32: {
    arange_kernel<<<blocks, threads>>>(static_cast<int32_t *>(out->data), n,
                                       start, step);
    break;
  }
  case INSIGHT_DTYPE_I64: {
    arange_kernel<<<blocks, threads>>>(static_cast<int64_t *>(out->data), n,
                                       start, step);
    break;
  }
  case INSIGHT_DTYPE_F32: {
    arange_kernel<<<blocks, threads>>>(static_cast<float *>(out->data), n,
                                       start, step);
    break;
  }
  case INSIGHT_DTYPE_F64: {
    arange_kernel<<<blocks, threads>>>(static_cast<double *>(out->data), n,
                                       start, step);
    break;
  }
  default:
    gpu_set_last_error(
        ("arange_kernel_gpu: unsupported dtype " + std::to_string(dtype))
            .c_str());
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

// Register for supported types
REGISTER_ILUVATAR_KERNEL(arange, INSIGHT_DTYPE_I32, arange_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(arange, INSIGHT_DTYPE_I64, arange_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(arange, INSIGHT_DTYPE_F32, arange_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(arange, INSIGHT_DTYPE_F64, arange_kernel_gpu);
