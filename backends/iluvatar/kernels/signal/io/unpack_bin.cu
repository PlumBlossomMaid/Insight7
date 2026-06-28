// backends/cuda/kernels/signal/io/unpack_bin.cu
// CUDA kernel for unpacking bytes to typed array.
// Reinterprets uint8 bytes as target dtype via cudaMemcpy.
// inputs[0]: byte array (U8)
// outputs[0]: typed array (any dtype, pre-allocated with correct size)
#include "../../../registry/iluvatar_registry.h"
#include "insight/c_api/array.h"
#include <cuda_runtime.h>

extern "C" {

C_Status signal_unpack_bin_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *in_arr = (InsightArray *)inputs[0];
  InsightArray *out_arr = (InsightArray *)outputs[0];

  if (!in_arr || !out_arr) {
    gpu_set_last_error("signal_unpack_bin: null array pointer");
    return C_FAILED;
  }

  if (!in_arr->data || !out_arr->data) {
    gpu_set_last_error("signal_unpack_bin: null data pointer");
    return C_FAILED;
  }

  size_t nbytes = (size_t)(out_arr->numel * insight_dtype_size(out_arr->dtype));
  // Clamp to available input bytes to avoid overrun
  size_t in_nbytes = (size_t)in_arr->numel;
  if (nbytes > in_nbytes)
    nbytes = in_nbytes;

  cudaMemcpy(out_arr->data, in_arr->data, nbytes, cudaMemcpyDeviceToDevice);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }
  return C_SUCCESS;
}

} // extern "C"

REGISTER_ILUVATAR_KERNEL(signal_unpack_bin, INSIGHT_DTYPE_U8,
                    signal_unpack_bin_kernel_gpu);
