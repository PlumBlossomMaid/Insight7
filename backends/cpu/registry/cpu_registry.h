// backends/cpu/registry/cpu_registry.h
#pragma once
#include "insight/c_api/kernel.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Register a kernel in the CPU backend's local registry.
 *
 * Called by the REGISTER_CPU_KERNEL macro during static initialization.
 *
 * @param op_name Operator name
 * @param dtype   Data type (InsightDType enum value)
 * @param func    Kernel function pointer
 */
void cpu_register_kernel(const char *op_name, int32_t dtype,
                         InsightKernel func);

/**
 * @brief Sync all locally registered kernels to the core framework.
 *
 * Iterates over the local registry and calls the provided registration
 * function for each kernel. Called by InitPlugin during backend loading.
 *
 * @param register_fn Core framework's registration callback
 */
void cpu_sync_kernels(C_Status (*register_fn)(const char *, int32_t, int32_t,
                                              InsightKernel));

/**
 * @brief Set the last error message for the CPU backend.
 *
 * Stores a thread-local error message that can be retrieved later
 * via cpu_get_last_error(). This function is used by CPU kernel
 * implementations to report detailed error information before
 * returning C_FAILED.
 *
 * @param msg Null-terminated error message string. If NULL, the
 *            error string is cleared to empty.
 */
void cpu_set_last_error(const char *msg);

/**
 * @brief Get the last error message for the CPU backend.
 *
 * Returns the thread-local error message set by the most recent
 * call to cpu_set_last_error(). If no error has been set, returns
 * a default message indicating no error occurred.
 *
 * @return Null-terminated error message string. Never returns NULL.
 */
const char *cpu_get_last_error(void);

#ifdef __cplusplus
}
#endif

// ========================================================================
// Registration macro for CPU backend kernel files
// ========================================================================
#define REGISTER_CPU_KERNEL(op_name, dtype, func)                              \
  static bool _cpu_##op_name##_##dtype = []() {                                \
    ::cpu_register_kernel(#op_name, static_cast<int32_t>(dtype), func);        \
    return true;                                                               \
  }();
