// backends/cuda/kernels/creation/full.cu
/**
 * @file full.cu
 * @brief CUDA kernel for the full operation.
 *
 * Fills an output array with a constant scalar value.
 * Supports all numeric dtypes including complex types.
 */
#include "../../registry/iluvatar_registry.h"
#include "common.cuh"
#include "insight/c_api/array.h"
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <string>

/**
 * @brief CUDA kernel to fill an array with a constant value.
 *
 * Each thread writes one element: dst[i] = val
 *
 * @tparam T Element type
 * @param dst Output array
 * @param val Value to fill
 * @param n Number of elements
 */
template <typename T> __global__ void full_kernel(T *dst, T val, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    dst[i] = val;
  }
}

extern "C" {

/**
 * @brief CPU entry point for the full kernel.
 *
 * Fills the output array with a constant value.
 * The fill value is passed as double* in inputs[1] and cast to the output
 * dtype.
 *
 * @param inputs  [0] = unused, [1] = double* fill_value
 * @param outputs [0] = InsightArray* result
 * @return C_SUCCESS on success, C_FAILED on error
 */
C_Status full_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *out = static_cast<InsightArray *>(outputs[0]);

  if (!out) {
    gpu_set_last_error("full_kernel_gpu: output array is null");
    return C_FAILED;
  }
  if (!inputs[1]) {
    gpu_set_last_error("full_kernel_gpu: fill_value is null");
    return C_FAILED;
  }

  double fill_val = *static_cast<double *>(inputs[1]);
  int64_t n = out->numel;
  if (n == 0)
    return C_SUCCESS;
  int32_t dtype = out->dtype;
  size_t elem_size = insight_dtype_size(dtype);
  char *data_with_offset =
      static_cast<char *>(out->data) + out->offset * elem_size;

  int threads = creation_threads();
  int blocks = creation_blocks(n);

  switch (dtype) {
  case INSIGHT_DTYPE_BOOL: {
    bool val = (fill_val != 0.0);
    full_kernel<<<blocks, threads>>>(reinterpret_cast<bool *>(data_with_offset),
                                     val, n);
    break;
  }
  case INSIGHT_DTYPE_U8: {
    uint8_t val = static_cast<uint8_t>(fill_val);
    full_kernel<<<blocks, threads>>>(
        reinterpret_cast<uint8_t *>(data_with_offset), val, n);
    break;
  }
  case INSIGHT_DTYPE_I8: {
    int8_t val = static_cast<int8_t>(fill_val);
    full_kernel<<<blocks, threads>>>(
        reinterpret_cast<int8_t *>(data_with_offset), val, n);
    break;
  }
  case INSIGHT_DTYPE_I16: {
    int16_t val = static_cast<int16_t>(fill_val);
    full_kernel<<<blocks, threads>>>(
        reinterpret_cast<int16_t *>(data_with_offset), val, n);
    break;
  }
  case INSIGHT_DTYPE_I32: {
    int32_t val = static_cast<int32_t>(fill_val);
    full_kernel<<<blocks, threads>>>(
        reinterpret_cast<int32_t *>(data_with_offset), val, n);
    break;
  }
  case INSIGHT_DTYPE_I64: {
    int64_t val = static_cast<int64_t>(fill_val);
    full_kernel<<<blocks, threads>>>(
        reinterpret_cast<int64_t *>(data_with_offset), val, n);
    break;
  }
  case INSIGHT_DTYPE_U16: {
    uint16_t val = static_cast<uint16_t>(fill_val);
    full_kernel<<<blocks, threads>>>(
        reinterpret_cast<uint16_t *>(data_with_offset), val, n);
    break;
  }
  case INSIGHT_DTYPE_U32: {
    uint32_t val = static_cast<uint32_t>(fill_val);
    full_kernel<<<blocks, threads>>>(
        reinterpret_cast<uint32_t *>(data_with_offset), val, n);
    break;
  }
  case INSIGHT_DTYPE_U64: {
    uint64_t val = static_cast<uint64_t>(fill_val);
    full_kernel<<<blocks, threads>>>(
        reinterpret_cast<uint64_t *>(data_with_offset), val, n);
    break;
  }
  case INSIGHT_DTYPE_F32: {
    float val = static_cast<float>(fill_val);
    full_kernel<<<blocks, threads>>>(
        reinterpret_cast<float *>(data_with_offset), val, n);
    break;
  }
  case INSIGHT_DTYPE_F64: {
    full_kernel<<<blocks, threads>>>(
        reinterpret_cast<double *>(data_with_offset), fill_val, n);
    break;
  }
  case INSIGHT_DTYPE_C32: {
    cuFloatComplex val =
        make_cuFloatComplex(static_cast<float>(fill_val), 0.0f);
    full_kernel<<<blocks, threads>>>(
        reinterpret_cast<cuFloatComplex *>(data_with_offset), val, n);
    break;
  }
  case INSIGHT_DTYPE_C64: {
    cuDoubleComplex val = make_cuDoubleComplex(fill_val, 0.0);
    full_kernel<<<blocks, threads>>>(
        reinterpret_cast<cuDoubleComplex *>(data_with_offset), val, n);
    break;
  }
  default:
    gpu_set_last_error(
        ("full_kernel_gpu: unsupported dtype " + std::to_string(dtype))
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

// Register for all supported types
REGISTER_ILUVATAR_KERNEL(full, INSIGHT_DTYPE_BOOL, full_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(full, INSIGHT_DTYPE_U8, full_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(full, INSIGHT_DTYPE_I8, full_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(full, INSIGHT_DTYPE_I16, full_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(full, INSIGHT_DTYPE_I32, full_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(full, INSIGHT_DTYPE_I64, full_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(full, INSIGHT_DTYPE_U16, full_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(full, INSIGHT_DTYPE_U32, full_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(full, INSIGHT_DTYPE_U64, full_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(full, INSIGHT_DTYPE_F32, full_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(full, INSIGHT_DTYPE_F64, full_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(full, INSIGHT_DTYPE_C32, full_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(full, INSIGHT_DTYPE_C64, full_kernel_gpu);
