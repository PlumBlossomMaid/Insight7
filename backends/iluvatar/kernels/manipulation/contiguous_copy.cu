// backends/cuda/kernels/manipulation/contiguous_copy.cu
#include "../../registry/iluvatar_registry.h"
#include "insight/c_api/array.h"
#include <cuda_runtime.h>

/**
 * @brief GPU kernel for copying between arrays with arbitrary strides.
 *
 * Each thread copies one element, computing both source and destination
 * offsets from their respective strides.
 */
__global__ void contiguous_copy_kernel(
    char *__restrict__ dst, const char *__restrict__ src,
    const int64_t *__restrict__ in_dims, const int64_t *__restrict__ in_strides,
    int64_t in_offset, const int64_t *__restrict__ out_dims,
    const int64_t *__restrict__ out_strides, int64_t out_offset, int64_t total,
    int in_ndim, int out_ndim, size_t elem_size) {
  int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
  if (linear >= total)
    return;

  // Compute source offset
  int64_t src_off = in_offset;
  int64_t tmp = linear;
  for (int d = in_ndim - 1; d >= 0; --d) {
    int64_t idx = tmp % in_dims[d];
    src_off += idx * in_strides[d];
    tmp /= in_dims[d];
  }

  // Compute destination offset
  int64_t dst_off = out_offset;
  tmp = linear;
  for (int d = out_ndim - 1; d >= 0; --d) {
    int64_t idx = tmp % out_dims[d];
    dst_off += idx * out_strides[d];
    tmp /= out_dims[d];
  }

  const char *src_ptr = src + src_off * elem_size;
  char *dst_ptr = dst + dst_off * elem_size;

  for (size_t b = 0; b < elem_size; ++b) {
    dst_ptr[b] = src_ptr[b];
  }
}

extern "C" {

C_Status contiguous_copy_gpu(void **inputs, void **outputs) {
  InsightArray *in = static_cast<InsightArray *>(inputs[0]);
  InsightArray *out = static_cast<InsightArray *>(outputs[0]);

  if (!in) {
    gpu_set_last_error("contiguous_copy: input array is null");
    return C_FAILED;
  }
  if (!out) {
    gpu_set_last_error("contiguous_copy: output array is null");
    return C_FAILED;
  }

  size_t elem_size = insight_dtype_size(in->dtype);
  if (elem_size == 0) {
    gpu_set_last_error("contiguous_copy: unknown data type");
    return C_FAILED;
  }

  int64_t total = in->numel;
  if (total == 0)
    return C_SUCCESS;

  // Fast path: both contiguous AND no offset
  if (in->offset == 0 && out->offset == 0 && insight_array_is_contiguous(in) &&
      insight_array_is_contiguous(out)) {
    size_t bytes = total * elem_size;
    cudaError_t err =
        cudaMemcpy(out->data, in->data, bytes, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
      gpu_set_last_error(cudaGetErrorString(err));
      return C_FAILED;
    }
    return C_SUCCESS;
  }

  // Prepare dims and strides on device for both input and output
  int in_ndim = in->ndim;
  int out_ndim = out->ndim;
  int64_t *d_in_dims, *d_in_strides, *d_out_dims, *d_out_strides;
  cudaMalloc(&d_in_dims, in_ndim * sizeof(int64_t));
  cudaMalloc(&d_in_strides, in_ndim * sizeof(int64_t));
  cudaMalloc(&d_out_dims, out_ndim * sizeof(int64_t));
  cudaMalloc(&d_out_strides, out_ndim * sizeof(int64_t));
  cudaMemcpy(d_in_dims, in->dims, in_ndim * sizeof(int64_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_in_strides, in->strides, in_ndim * sizeof(int64_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_out_dims, out->dims, out_ndim * sizeof(int64_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_out_strides, out->strides, out_ndim * sizeof(int64_t),
             cudaMemcpyHostToDevice);

  int threads = 256;
  int blocks = (total + threads - 1) / threads;

  contiguous_copy_kernel<<<blocks, threads>>>(
      static_cast<char *>(out->data), static_cast<const char *>(in->data),
      d_in_dims, d_in_strides, in->offset, d_out_dims, d_out_strides,
      out->offset, total, in_ndim, out_ndim, elem_size);

  cudaFree(d_in_dims);
  cudaFree(d_in_strides);
  cudaFree(d_out_dims);
  cudaFree(d_out_strides);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

REGISTER_ILUVATAR_KERNEL(contiguous_copy, INSIGHT_DTYPE_BOOL,
                         contiguous_copy_gpu);
REGISTER_ILUVATAR_KERNEL(contiguous_copy, INSIGHT_DTYPE_U8,
                         contiguous_copy_gpu);
REGISTER_ILUVATAR_KERNEL(contiguous_copy, INSIGHT_DTYPE_I8,
                         contiguous_copy_gpu);
REGISTER_ILUVATAR_KERNEL(contiguous_copy, INSIGHT_DTYPE_I16,
                         contiguous_copy_gpu);
REGISTER_ILUVATAR_KERNEL(contiguous_copy, INSIGHT_DTYPE_I32,
                         contiguous_copy_gpu);
REGISTER_ILUVATAR_KERNEL(contiguous_copy, INSIGHT_DTYPE_I64,
                         contiguous_copy_gpu);
REGISTER_ILUVATAR_KERNEL(contiguous_copy, INSIGHT_DTYPE_U16,
                         contiguous_copy_gpu);
REGISTER_ILUVATAR_KERNEL(contiguous_copy, INSIGHT_DTYPE_U32,
                         contiguous_copy_gpu);
REGISTER_ILUVATAR_KERNEL(contiguous_copy, INSIGHT_DTYPE_U64,
                         contiguous_copy_gpu);
REGISTER_ILUVATAR_KERNEL(contiguous_copy, INSIGHT_DTYPE_F16,
                         contiguous_copy_gpu);
REGISTER_ILUVATAR_KERNEL(contiguous_copy, INSIGHT_DTYPE_BF16,
                         contiguous_copy_gpu);
REGISTER_ILUVATAR_KERNEL(contiguous_copy, INSIGHT_DTYPE_F32,
                         contiguous_copy_gpu);
REGISTER_ILUVATAR_KERNEL(contiguous_copy, INSIGHT_DTYPE_F64,
                         contiguous_copy_gpu);
REGISTER_ILUVATAR_KERNEL(contiguous_copy, INSIGHT_DTYPE_C32,
                         contiguous_copy_gpu);
REGISTER_ILUVATAR_KERNEL(contiguous_copy, INSIGHT_DTYPE_C64,
                         contiguous_copy_gpu);
REGISTER_ILUVATAR_KERNEL(contiguous_copy, INSIGHT_DTYPE_F8_E4M3,
                         contiguous_copy_gpu);
REGISTER_ILUVATAR_KERNEL(contiguous_copy, INSIGHT_DTYPE_F8_E5M2,
                         contiguous_copy_gpu);