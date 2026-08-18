// include/insight/c_api/kernel.h
#pragma once
#include "dtype.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Standard kernel function signature.
 *
 * All backend kernels must conform to this signature.
 * Input and output arrays are passed as arrays of void pointers,
 * terminated by a NULL sentinel.
 *
 * @param inputs  NULL-terminated array of input array pointers
 * @param outputs NULL-terminated array of output array pointers
 * @return C_SUCCESS on success, C_FALLBACK to request CPU fallback, or error
 * code
 */
typedef C_Status (*InsightKernel)(void **inputs, void **outputs);

/**
 * @brief Kernel argument kind used by schema-aware launches.
 *
 * ARRAY arguments are InsightArray pointers and participate in device transfer
 * during CPU fallback. HOST_SCALAR arguments are host-side scalar/control
 * values and are passed through unchanged.
 */
typedef enum {
  INSIGHT_KERNEL_ARG_ARRAY = 1,
  INSIGHT_KERNEL_ARG_HOST_SCALAR = 2
} InsightKernelArgKind;

/**
 * @brief Register a kernel into the global registry.
 *
 * Called by backend DLLs during module initialization.
 * Each kernel is identified by the tuple (op_name, device_type, dtype).
 *
 * @param op_name     Operator name (e.g., "sum", "add", "matmul")
 * @param device_type Device type (INSIGHT_DEVICE_CPU or INSIGHT_DEVICE_GPU)
 * @param dtype       Data type (INSIGHT_DTYPE_F32, etc.)
 * @param kernel      Kernel function pointer
 * @return C_SUCCESS if registered successfully
 */
C_Status insight_register_kernel(const char *op_name, int32_t device_type,
                                 int32_t dtype, InsightKernel kernel);

/**
 * @brief Look up a kernel from the global registry.
 *
 * @param op_name     Operator name
 * @param device_type Device type
 * @param dtype       Data type
 * @return Kernel function pointer, or NULL if not found
 */
InsightKernel insight_find_kernel(const char *op_name, int32_t device_type,
                                  int32_t dtype);

/**
 * @brief Launch a kernel with automatic CPU fallback.
 *
 * First attempts the kernel on the requested device. This legacy raw-pointer
 * launch path cannot perform CPU fallback because scalar attributes and arrays
 * are not distinguishable. Use insight_kernel_launch_schema() for GPU kernels
 * that may return C_FALLBACK.
 *
 * @param op_name     Operator name
 * @param device_type Preferred device type
 * @param dtype       Data type
 * @param inputs      NULL-terminated array of input pointers
 * @param outputs     NULL-terminated array of output pointers
 * @return C_SUCCESS or error code
 */
C_Status insight_kernel_launch(const char *op_name, int32_t device_type,
                               int32_t dtype, void **inputs, void **outputs);

/**
 * @brief Launch a kernel with explicit input/output argument schemas.
 *
 * Schema entries must match the non-NULL entries in the NULL-terminated input
 * and output pointer arrays. During GPU CPU-fallback, only ARRAY entries are
 * interpreted as InsightArray pointers and transferred between devices;
 * HOST_SCALAR entries are passed through unchanged.
 *
 * @param op_name      Operator name
 * @param device_type  Preferred device type
 * @param dtype        Data type
 * @param inputs       NULL-terminated array of input pointers
 * @param input_kinds  Array of input argument kinds
 * @param input_count  Number of non-NULL input entries
 * @param outputs      NULL-terminated array of output pointers
 * @param output_kinds Array of output argument kinds
 * @param output_count Number of non-NULL output entries
 * @return C_SUCCESS or error code
 */
C_Status insight_kernel_launch_schema(const char *op_name, int32_t device_type,
                                      int32_t dtype, void **inputs,
                                      const InsightKernelArgKind *input_kinds,
                                      int input_count, void **outputs,
                                      const InsightKernelArgKind *output_kinds,
                                      int output_count);

/**
 * @brief Check whether a kernel is registered.
 *
 * @param op_name     Operator name
 * @param device_type Device type
 * @param dtype       Data type
 * @return 1 if a kernel is registered, 0 otherwise
 */
int insight_has_kernel(const char *op_name, int32_t device_type, int32_t dtype);

/**
 * @brief Get the number of registered operator names.
 *
 * @return Number of unique operator names registered
 */
int insight_get_operator_count(void);

/**
 * @brief Get a registered operator name by index.
 *
 * The order of names is the registration order (no specific sorting).
 * Call insight_get_operator_count() first to know the valid index range.
 *
 * @param index   Index in range [0, count-1]
 * @param buffer  Output buffer to receive the name (caller allocated)
 * @param size    Size of the output buffer
 * @return C_SUCCESS if index is valid, C_FAILED otherwise
 */
C_Status insight_get_operator_name(int index, char *buffer, int size);

#ifdef __cplusplus
}
#endif