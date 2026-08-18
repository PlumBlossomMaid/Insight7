// backends/cuda/kernels/signal/io/pack_bin.cu
// CUDA kernel for packing array data to bytes.
// Reinterprets array data as uint8 bytes via cudaMemcpy.
// inputs[0]: data array (any dtype)
// outputs[0]: byte array (U8)
#include "../../../registry/ixuca_registry.h"
#include "insight/c_api/array.h"
#include <cuda_runtime.h>

extern "C" {

C_Status signal_pack_bin_kernel_gpu(void **inputs, void **outputs) {
  InsightArray *in_arr = (InsightArray *)inputs[0];
  InsightArray *out_arr = (InsightArray *)outputs[0];

  if (!in_arr || !out_arr) {
    gpu_set_last_error("signal_pack_bin: null array pointer");
    return C_FAILED;
  }

  if (!in_arr->data || !out_arr->data) {
    gpu_set_last_error("signal_pack_bin: null data pointer");
    return C_FAILED;
  }

  size_t nbytes = (size_t)(in_arr->numel * insight_dtype_size(in_arr->dtype));
  cudaMemcpy(out_arr->data, in_arr->data, nbytes, cudaMemcpyDeviceToDevice);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }
  return C_SUCCESS;
}

} // extern "C"

REGISTER_IXUCA_KERNEL(signal_pack_bin, INSIGHT_DTYPE_U8,
                         signal_pack_bin_kernel_gpu);
