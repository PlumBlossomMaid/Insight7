// backends/cuda/kernels/signal/estimation/simple_inv.cu
// simple_inv: C_FALLBACK — Gauss-Jordan elimination on small matrices
// is more efficient on CPU with automatic data transfer.
#include "../../../registry/iluvatar_registry.h"
#include "insight/c_api/array.h"

extern "C" {

static C_Status simple_inv_kernel_gpu(void **inputs, void **outputs) {
  return C_FALLBACK;
}

} // extern "C"

REGISTER_ILUVATAR_KERNEL(simple_inv, INSIGHT_DTYPE_F32, simple_inv_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(simple_inv, INSIGHT_DTYPE_F64, simple_inv_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(simple_inv, INSIGHT_DTYPE_F16, simple_inv_kernel_gpu);
REGISTER_ILUVATAR_KERNEL(simple_inv, INSIGHT_DTYPE_BF16, simple_inv_kernel_gpu);
