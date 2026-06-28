// backends/cuda/kernels/signal/filtering/lfilter_zi.cu
// lfilter_zi: C_FALLBACK — Gauss-Jordan elimination is not GPU-friendly
// for small matrices. The framework automatically transfers to CPU.
#include "../../../registry/iluvatar_registry.h"
#include "insight/c_api/array.h"

extern "C" {

static C_Status lfilter_zi_kernel_gpu(void **inputs, void **outputs) {
  return C_FALLBACK;
}

} // extern "C"

REGISTER_ILUVATAR_KERNEL(lfilter_zi, INSIGHT_DTYPE_F32, lfilter_zi_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(lfilter_zi, INSIGHT_DTYPE_F64, lfilter_zi_kernel_gpu);
