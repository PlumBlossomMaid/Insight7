// include/insight/plugin/op_registry.h
#pragma once
#include "insight/c_api/exception.h"
#include "insight/c_api/kernel.h"
#include "insight/c_api/place.h"
#include "insight/core/dtype.h"
#include "insight/core/exception.h"
#include "insight/core/place.h"
#include <algorithm>
#include <string>
#include <unordered_set>
#include <vector>

namespace ins {

/**
 * @brief Operator registry for kernel dispatch.
 *
 * This class provides a unified interface for launching kernels
 * and querying kernel availability. All methods are static and
 * delegate to the underlying C API for ABI stability.
 */
class OpRegistry {
public:
  enum class ArgKind { Array, HostScalar };

  struct Arg {
    void *ptr;
    ArgKind kind;
  };

  static Arg array_arg(void *ptr) { return {ptr, ArgKind::Array}; }
  static Arg array_arg(const void *ptr) {
    return {const_cast<void *>(ptr), ArgKind::Array};
  }

  template <typename T> static Arg scalar_arg(T *ptr) {
    return {static_cast<void *>(ptr), ArgKind::HostScalar};
  }

  static void handle_status(const std::string &op_name, const Place &place,
                            C_Status status) {
    if (status != C_SUCCESS) {
      const char *error_msg = insight_get_last_error();
      std::string msg = "Kernel '" + op_name + "' failed on " +
                        place.to_string() + " with status " +
                        std::to_string(status);
      if (error_msg && error_msg[0] != '\0') {
        msg += ": " + std::string(error_msg);
      }
      INS_THROW(msg);
    }
  }

  /**
   * @brief Launch a kernel.
   *
   * All inputs and outputs must already reside on the target device
   * and have the correct data type. Device resolution, type promotion,
   * and broadcasting are handled externally before calling this method.
   *
   * The underlying C API automatically falls back to CPU if the GPU
   * kernel returns C_FALLBACK. The caller does not need to handle
   * this case.
   *
   * @param op_name Operator name (e.g., "sum", "add", "matmul")
   * @param place   Target device where the kernel should execute
   * @param dtype   Data type for kernel dispatch
   * @param inputs  Array of input pointers
   * @param outputs Array of output pointers
   * @throws Exception if kernel fails on all available devices
   */
  static void launch(const std::string &op_name, const Place &place,
                     DType dtype, const std::vector<void *> &inputs,
                     const std::vector<void *> &outputs) {

    // Construct NULL-terminated void* arrays for C API
    std::vector<void *> in_ptrs(inputs.size() + 1, nullptr);
    std::vector<void *> out_ptrs(outputs.size() + 1, nullptr);
    std::copy(inputs.begin(), inputs.end(), in_ptrs.begin());
    std::copy(outputs.begin(), outputs.end(), out_ptrs.begin());

    C_Status status = insight_kernel_launch(
        op_name.c_str(), static_cast<int32_t>(place.kind()),
        static_cast<int32_t>(dtype), in_ptrs.data(), out_ptrs.data());

    handle_status(op_name, place, status);
  }

  static void launch_schema(const std::string &op_name, const Place &place,
                            DType dtype, const std::vector<Arg> &inputs,
                            const std::vector<Arg> &outputs) {
    std::vector<void *> in_ptrs(inputs.size() + 1, nullptr);
    std::vector<void *> out_ptrs(outputs.size() + 1, nullptr);
    std::vector<InsightKernelArgKind> input_kinds(inputs.size());
    std::vector<InsightKernelArgKind> output_kinds(outputs.size());

    for (size_t i = 0; i < inputs.size(); ++i) {
      in_ptrs[i] = inputs[i].ptr;
      input_kinds[i] = inputs[i].kind == ArgKind::Array
                           ? INSIGHT_KERNEL_ARG_ARRAY
                           : INSIGHT_KERNEL_ARG_HOST_SCALAR;
    }
    for (size_t i = 0; i < outputs.size(); ++i) {
      out_ptrs[i] = outputs[i].ptr;
      output_kinds[i] = outputs[i].kind == ArgKind::Array
                            ? INSIGHT_KERNEL_ARG_ARRAY
                            : INSIGHT_KERNEL_ARG_HOST_SCALAR;
    }

    C_Status status = insight_kernel_launch_schema(
        op_name.c_str(), static_cast<int32_t>(place.kind()),
        static_cast<int32_t>(dtype), in_ptrs.data(), input_kinds.data(),
        static_cast<int>(input_kinds.size()), out_ptrs.data(),
        output_kinds.data(), static_cast<int>(output_kinds.size()));

    handle_status(op_name, place, status);
  }

  /**
   * @brief Check if a kernel is registered for the given combination.
   *
   * @param op_name Operator name
   * @param place   Target device
   * @param dtype   Data type
   * @return true if a kernel is registered
   */
  static bool has(const std::string &op_name, const Place &place, DType dtype) {
    return insight_has_kernel(op_name.c_str(),
                              static_cast<int32_t>(place.kind()),
                              static_cast<int32_t>(dtype)) != 0;
  }

  /**
   * @brief Check if a kernel is registered for the given combination.
   *
   * Convenience overload using DeviceKind instead of Place.
   *
   * @param op_name Operator name
   * @param device  Device kind (CPU or GPU)
   * @param dtype   Data type
   * @return true if a kernel is registered
   */
  static bool has_kernel(const std::string &op_name, DeviceKind device,
                         DType dtype) {
    int32_t device_type =
        (device == DeviceKind::CPU) ? INSIGHT_DEVICE_CPU : INSIGHT_DEVICE_GPU;
    return insight_has_kernel(op_name.c_str(), device_type,
                              static_cast<int32_t>(dtype)) != 0;
  }

  /**
   * @brief Get all registered operator names.
   *
   * @return std::vector<std::string> List of unique operator names
   */
  static std::vector<std::string> get_operator_names() {
    int count = insight_get_operator_count();
    std::vector<std::string> names;
    names.reserve(count);

    for (int i = 0; i < count; ++i) {
      char buffer[256];
      if (insight_get_operator_name(i, buffer, sizeof(buffer)) == C_SUCCESS) {
        names.emplace_back(buffer);
      }
    }

    return names;
  }

  /**
   * @brief List all registered data types for a given operator and device.
   *
   * @param op_name Operator name
   * @param device  Device kind
   * @return std::vector<DType> List of supported data types
   */
  static std::vector<DType> list_kernels(const std::string &op_name,
                                         DeviceKind device) {
    std::vector<DType> types;

    // Iterate over all possible DType values and check registration
    for (int i = 0; i < static_cast<int>(DType::DTYPE_COUNT); ++i) {
      DType dtype = static_cast<DType>(i);
      if (has_kernel(op_name, device, dtype)) {
        types.push_back(dtype);
      }
    }

    return types;
  }
};

/**
 * @brief Convenience function to access the operator registry.
 *
 * @return OpRegistry& Reference to the registry (stateless, all methods static)
 */
inline OpRegistry &ops() {
  static OpRegistry registry;
  return registry;
}

} // namespace ins