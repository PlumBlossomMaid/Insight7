// backends/cuda/kernels/creation/linspace.cu
/**
 * @file linspace.cu
 * @brief CUDA kernel for the linspace operation.
 *
 * Generates a 1D array with linearly spaced values between start and stop.
 * The step is computed on the host as (stop - start) / (n - 1) and passed
 * to the kernel. For n=1, step=0 and the single element equals start.
 * Supports only floating-point types (F32, F64).
 */
#include "../../registry/iluvatar_registry.h"
#include "common.cuh"
#include "insight/c_api/array.h"
#include <cuda_runtime.h>
#include <string>

/**
 * @brief CUDA kernel to generate linspace values.
 *
 * Each thread computes one element: out[i] = static_cast<T>(start + i * step)
 *
 * @tparam T Output element type (float or double)
 * @param out Output array
 * @param n Number of elements
 * @param start Start value
 * @param step Step size between elements
 */
template <typename T>
__global__ void linspace_kernel(T *out, int64_t n, double start, double step) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = static_cast<T>(start + idx * step);
  }
}

extern "C" {

/**
 * @brief GPU entry point for the linspace kernel.
 *
 * Generates a 1D array with linearly spaced values between start and stop.
 * The step size is computed on the host before launching the kernel.
 *
 * @param inputs  [0] = InsightArray* output, [1] = double* start, [2] = double*
 *                stop
 * @param outputs [0] = InsightArray* result (same as inputs[0])
 * @return C_SUCCESS on success, C_FAILED on error
 */
C_Status linspace_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = static_cast<InsightArray *>(outputs[0]);

  if (!out) {
    gpu_set_last_error("linspace_kernel_gpu: output array is null");
    return C_FAILED;
  }
  if (!inputs[1] || !inputs[2]) {
    gpu_set_last_error("linspace_kernel_gpu: start or stop is null");
    return C_FAILED;
  }

  double start = *static_cast<double *>(inputs[1]);
  double stop = *static_cast<double *>(inputs[2]);
  int64_t n = out->numel;
  if (n == 0)
    return C_SUCCESS;
  int32_t dtype = out->dtype;

  // Compute step on host
  double step = (n == 1) ? 0.0 : (stop - start) / (n - 1);

  int threads = creation_threads();
  int blocks = creation_blocks(n);

  switch (dtype) {
  case INSIGHT_DTYPE_F32: {
    linspace_kernel<<<blocks, threads>>>(static_cast<float *>(out->data), n,
                                         start, step);
    break;
  }
  case INSIGHT_DTYPE_F64: {
    linspace_kernel<<<blocks, threads>>>(static_cast<double *>(out->data), n,
                                         start, step);
    break;
  }
  default:
    gpu_set_last_error(
        ("linspace_kernel_gpu: unsupported dtype " + std::to_string(dtype))
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
REGISTER_ILUVATAR_KERNEL(linspace, INSIGHT_DTYPE_F32, linspace_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(linspace, INSIGHT_DTYPE_F64, linspace_kernel_gpu);
