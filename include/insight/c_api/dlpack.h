// include/insight/c_api/dlpack.h
//
// DLPack C ABI bridge for Insight arrays.
//
// DLPack (https://github.com/dmlc/dlpack) is the language-neutral, zero-copy
// tensor description protocol shared by PyTorch / TensorFlow / JAX / CuPy /
// NumPy. Insight exports its own storage through a DLManagedTensorVersioned
// (to_dlpack) and imports foreign storage through the same structure
// (from_dlpack), so other frameworks can exchange arrays with Insight without
// copying through a Python layer.
//
// Protocol version pinned to v1.3 (see
// third_party/dlpack/include/dlpack/dlpack.h).
//
// Lifecycle rules (mirror the DLPack spec):
//   - to_dlpack: producer creates a DLManagedTensorVersioned whose dl_tensor
//     points at Insight storage. The consumer owns the struct and must call
//     its deleter when done. The deleter releases the manager_ctx, which holds
//     an ins::Array copy keeping the storage alive.
//   - from_dlpack: consumer reads the foreign DLTensor and copies the bytes
//     into Insight-owned storage (the resulting InsightArray is fully owned).
//     The foreign DLManagedTensorVersioned is NOT consumed; the caller keeps
//     ownership and is responsible for its deleter.
#pragma once
#include "insight/c_api/array.h"
#include "insight/c_api/dtype.h"
#include <dlpack/dlpack.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Export an InsightArray as a DLPack managed tensor.
 *
 * The returned DLManagedTensorVersioned borrows the array's storage. The
 * consumer must call `deleter(managed)` when finished. Shape/strides arrays
 * are heap-allocated and freed by the deleter.
 *
 * @param array   Source InsightArray (logical view; offset is honored)
 * @param out     Receives a heap-allocated DLManagedTensorVersioned*
 * @return C_SUCCESS on success, C_FAILED on error (see insight_get_last_error)
 */
C_Status insight_array_to_dlpack(const InsightArray *array,
                                 DLManagedTensorVersioned **out);

/**
 * @brief Import a DLPack managed tensor into a new InsightArray.
 *
 * The data is copied into Insight-owned storage (CPU place). Non-contiguous
 * strides and byte_offset are honored element-wise. The foreign managed
 * tensor remains owned by the caller.
 *
 * @param src  Source DLManagedTensorVersioned (dl_tensor must be valid)
 * @param out  Receives an initialized InsightArray (owned, must be destroyed)
 * @return C_SUCCESS on success, C_FAILED on error (see insight_get_last_error)
 */
C_Status insight_array_from_dlpack(const DLManagedTensorVersioned *src,
                                   InsightArray *out);

/**
 * @brief Map an InsightDType to a DLPack DLDataType.
 * @param dtype InsightDType enum value
 * @param out   Receives the corresponding DLDataType
 * @return C_SUCCESS if mappable, C_FAILED otherwise
 */
C_Status insight_dtype_to_dlpack(int32_t dtype, DLDataType *out);

/**
 * @brief Map a DLPack DLDataType to an InsightDType.
 * @param src  DLPack data type
 * @param out  Receives the corresponding InsightDType
 * @return C_SUCCESS if mappable, C_FAILED otherwise
 */
C_Status insight_dtype_from_dlpack(const DLDataType *src, int32_t *out);

/**
 * @brief Map an Insight device to a DLPack DLDevice.
 * @param device_type InsightDeviceType (INSIGHT_DEVICE_CPU/GPU)
 * @param device_id   Physical device id
 * @param out         Receives the corresponding DLDevice
 * @return C_SUCCESS if mappable, C_FAILED otherwise
 */
C_Status insight_device_to_dlpack(int32_t device_type, int32_t device_id,
                                  DLDevice *out);

#ifdef __cplusplus
}
#endif
