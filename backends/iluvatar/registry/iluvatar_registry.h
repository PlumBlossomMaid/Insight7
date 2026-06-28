#pragma once
#include "insight/c_api/kernel.h"

#ifdef __cplusplus
extern "C" {
#endif

void iluvatar_register_kernel(const char *op_name, int32_t dtype,
                              InsightKernel func);
void iluvatar_sync_kernels(C_Status (*register_fn)(const char *, int32_t, int32_t,
                                                   InsightKernel));
void iluvatar_set_last_error(const char *msg);
const char *iluvatar_get_last_error(void);

#ifdef __cplusplus
}
#endif

// Alias for CUDA-compatible kernel code that calls gpu_set_last_error()
// Kernel source files from the CUDA backend use this name; map it to
// the iluvatar backend's error handler.
#ifndef gpu_set_last_error
#define gpu_set_last_error iluvatar_set_last_error
#endif
#ifndef gpu_get_last_error
#define gpu_get_last_error iluvatar_get_last_error
#endif

#define REGISTER_ILUVATAR_KERNEL(op_name, dtype, func)                         \
  static bool _iluvatar_##op_name##_##dtype = []() {                           \
    ::iluvatar_register_kernel(#op_name, static_cast<int32_t>(dtype), func);   \
    return true;                                                               \
  }();
