// backends/cuda/kernels/creation/logspace.cu
/**
 * @file logspace.cu
 * @brief CUDA kernel for the logspace operation.
 *
 * Generates a 1D array with logarithmically spaced values.
 * Computes pow(base, start + i * step) for i = 0 to numel-1.
 * The step is computed on the host as (stop - start) / (n - 1).
 * For n=1, step=0 and the single element equals pow(base, start).
 * Supports only floating-point types (F32, F64).
 */
#include "../../registry/ixuca_registry.h"
#include "common.cuh"
#include "insight/c_api/array.h"
#include <cmath>
#include <cuda_runtime.h>
#include <string>

/**
 * @brief CUDA kernel to generate logspace values.
 *
 * Each thread computes one element: out[i] = static_cast<T>(pow(base, start +
 * i * step))
 *
 * @tparam T Output element type (float or double)
 * @param out Output array
 * @param n Number of elements
 * @param start Start exponent
 * @param step Step size between exponents
 * @param base Base of the logarithm
 */
template <typename T>
__global__ void logspace_kernel(T *out, int64_t n, double start, double step,
                                double base) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = static_cast<T>(pow(base, start + idx * step));
  }
}

extern "C" {

/**
 * @brief GPU entry point for the logspace kernel.
 *
 * Generates a 1D array with logarithmically spaced values.
 * The step size is computed on the host before launching the kernel.
 *
 * @param inputs  [0] = InsightArray* output, [1] = double* start, [2] = double*
 *                stop, [3] = double* base
 * @param outputs [0] = InsightArray* result (same as inputs[0])
 * @return C_SUCCESS on success, C_FAILED on error
 */
C_Status logspace_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = static_cast<InsightArray *>(outputs[0]);

  if (!out) {
    gpu_set_last_error("logspace_kernel_gpu: output array is null");
    return C_FAILED;
  }
  if (!inputs[1] || !inputs[2] || !inputs[3]) {
    gpu_set_last_error("logspace_kernel_gpu: start, stop, or base is null");
    return C_FAILED;
  }

  double start = *static_cast<double *>(inputs[1]);
  double stop = *static_cast<double *>(inputs[2]);
  double base = *static_cast<double *>(inputs[3]);
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
    logspace_kernel<<<blocks, threads>>>(static_cast<float *>(out->data), n,
                                         start, step, base);
    break;
  }
  case INSIGHT_DTYPE_F64: {
    logspace_kernel<<<blocks, threads>>>(static_cast<double *>(out->data), n,
                                         start, step, base);
    break;
  }
  default:
    gpu_set_last_error(
        ("logspace_kernel_gpu: unsupported dtype " + std::to_string(dtype))
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

/**
 * @brief Disabled entry point: defers logspace to the CPU implementation.
 *
 * DISABLED PENDING A COREX NUMERICAL FIX. logspace_kernel_gpu above produces
 * wrong results on CoreX (see tests/cuda/cpp/test_creation.cpp,
 * LogspaceSingle). `start`, `step` and `base` cross the void** ABI as `double`,
 * but every .cu in this backend is compiled with `-mllvm --skip-double`
 * (backends/ixuca/ CMakeLists.txt) while the host code that writes them is not.
 *
 * Returning C_FALLBACK makes the framework transfer the output to the host and
 * run the CPU kernel, which is correct but slower. Silently returning wrong
 * numbers is not an option, and an unregistered op would hard-fail
 * (src/core/kernel.cpp: insight_kernel_launch returns C_FAILED when no kernel
 * is found 鈥?the fallback path only runs for a *registered* kernel that
 * returns C_FALLBACK).
 *
 * Re-register logspace_kernel_gpu once the CoreX double-precision issue is
 * fixed.
 */
C_Status logspace_kernel_gpu_fallback(void **inputs, void **outputs) {
  (void)inputs;
  (void)outputs;
  return C_FALLBACK;
}

} // extern "C"

// Register for supported types
REGISTER_IXUCA_KERNEL(logspace, INSIGHT_DTYPE_F32,
                      logspace_kernel_gpu_fallback);
REGISTER_IXUCA_KERNEL(logspace, INSIGHT_DTYPE_F64,
                      logspace_kernel_gpu_fallback);
