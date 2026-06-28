// NOTE: This kernel is currently not called from the frontend.
// The frontend uses composite ops for this operation.
// This kernel is preserved for future backend dispatch optimization.
//
// backends/cuda/kernels/signal/filtering/firfilter_zi_state.cu
// FIR filter with initial state: C_FALLBACK for now.
#include "../../../registry/iluvatar_registry.h"
#include "insight/c_api/array.h"

extern "C" {

static C_Status firfilter_zi_state_kernel_gpu(void **inputs, void **outputs) {
  return C_FALLBACK;
}

} // extern "C"

REGISTER_ILUVATAR_KERNEL(firfilter_zi_state, INSIGHT_DTYPE_F32,
                         firfilter_zi_state_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(firfilter_zi_state, INSIGHT_DTYPE_F64,
                         firfilter_zi_state_kernel_gpu);
