#pragma once
#include "insight/c_api/kernel.h"

#ifdef __cplusplus
extern "C" {
#endif

void sdaa_register_kernel(const char *op_name, int32_t dtype,
                          InsightKernel func);
void sdaa_sync_kernels(C_Status (*register_fn)(const char *, int32_t, int32_t,
                                               InsightKernel));
void sdaa_set_last_error(const char *msg);
const char *sdaa_get_last_error(void);

#ifdef __cplusplus
}
#endif

#define REGISTER_SDAA_KERNEL(op_name, dtype, func)                             \
  static bool _sdaa_##op_name##_##dtype = []() {                               \
    ::sdaa_register_kernel(#op_name, static_cast<int32_t>(dtype), func);       \
    return true;                                                               \
  }();
