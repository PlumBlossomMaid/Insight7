// backends/cuda/registry/cuda_registry.h
#pragma once
#include "insight/c_api/kernel.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Register a kernel in the GPU backend's local registry.
 *
 * Called by the REGISTER_GPU_KERNEL macro during static initialization.
 *
 * @param op_name Operator name
 * @param dtype   Data type (InsightDType enum value)
 * @param func    Kernel function pointer
 */
void gpu_register_kernel(const char *op_name, int32_t dtype,
                         InsightKernel func);

/**
 * @brief Sync all locally registered kernels to the core framework.
 *
 * Iterates over the local registry and calls the provided registration
 * function for each kernel. Called by InitPlugin during backend loading.
 *
 * @param register_fn Core framework's registration callback
 */
void gpu_sync_kernels(C_Status (*register_fn)(const char *, int32_t, int32_t,
                                              InsightKernel));

/**
 * @brief Set the last error message for the GPU backend.
 *
 * Stores a human-readable error message that can be retrieved by
 * get_last_error. The string is owned by the backend and remains valid
 * until the next operation.
 */
void gpu_set_last_error(const char *msg);

/**
 * @brief Get the last error message for the GPU backend.
 *
 * Returns the last error message set by gpu_set_last_error. The string
 * is owned by the backend and remains valid until the next operation.
 */
const char *gpu_get_last_error(void);

#ifdef __cplusplus
}
#endif

// ========================================================================
// Registration macro for GPU backend kernel files
// ========================================================================
#define REGISTER_GPU_KERNEL(op_name, dtype, func)                              \
  static bool _gpu_##op_name##_##dtype = []() {                                \
    ::gpu_register_kernel(#op_name, static_cast<int32_t>(dtype), func);        \
    return true;                                                               \
  }();
