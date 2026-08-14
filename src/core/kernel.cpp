// src/core/kernel.cpp
#include "insight/c_api/kernel.h"
#include "insight/c_api/array.h"
#include "insight/c_api/exception.h"
#include "insight/c_api/place.h"
#include "insight/core/place.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

static std::unordered_map<std::string, InsightKernel> g_kernel_registry;
static thread_local std::string kernel_error_message = "";

static std::vector<std::string> &registered_op_names() {
  static std::vector<std::string> names;
  return names;
}

// Count non-NULL entries in a NULL-terminated void** array.
static int count_ptrs(void **ptrs) {
  if (!ptrs)
    return 0;
  int n = 0;
  while (ptrs[n])
    ++n;
  return n;
}

// ============================================================================
// CPU fallback with GPU→CPU data transfer
// ============================================================================

static bool is_array_kind(const InsightKernelArgKind *kinds, int count,
                          int index) {
  if (!kinds)
    return true;
  return index >= 0 && index < count &&
         kinds[index] == INSIGHT_KERNEL_ARG_ARRAY;
}

static bool validate_arg_schema(const char *op_name, const char *side,
                                void **args, const InsightKernelArgKind *kinds,
                                int count) {
  if (count < 0) {
    kernel_error_message = std::string("insight_kernel_launch: negative ") +
                           side + " count for '" + op_name + "'";
    insight_set_last_error(kernel_error_message.c_str());
    return false;
  }
  if (count > 0 && !kinds) {
    kernel_error_message = std::string("insight_kernel_launch: missing ") +
                           side + " schema for '" + op_name + "'";
    insight_set_last_error(kernel_error_message.c_str());
    return false;
  }
  int actual = count_ptrs(args);
  if (actual != count) {
    kernel_error_message = std::string("insight_kernel_launch: ") + side +
                           " schema count mismatch for '" + op_name +
                           "': expected " + std::to_string(count) + ", got " +
                           std::to_string(actual);
    insight_set_last_error(kernel_error_message.c_str());
    return false;
  }
  for (int i = 0; i < count; ++i) {
    if (kinds[i] != INSIGHT_KERNEL_ARG_ARRAY &&
        kinds[i] != INSIGHT_KERNEL_ARG_HOST_SCALAR) {
      kernel_error_message = std::string("insight_kernel_launch: invalid ") +
                             side + " schema kind for '" + op_name + "'";
      insight_set_last_error(kernel_error_message.c_str());
      return false;
    }
  }
  return true;
}

static C_Status do_cpu_fallback(const char *op_name, int32_t dtype,
                                void **inputs,
                                const InsightKernelArgKind *input_kinds,
                                int input_count, void **outputs,
                                const InsightKernelArgKind *output_kinds,
                                int output_count) {
  InsightKernel cpu_kernel =
      insight_find_kernel(op_name, INSIGHT_DEVICE_CPU, dtype);
  if (!cpu_kernel) {
    kernel_error_message =
        std::string("insight_kernel_launch: CPU fallback kernel not found "
                    "for operator '") +
        op_name + "'";
    insight_set_last_error(kernel_error_message.c_str());
    return C_FAILED;
  }

  int num_inputs = input_kinds ? input_count : count_ptrs(inputs);
  int num_outputs = output_kinds ? output_count : count_ptrs(outputs);

  ins::Place cpu_place = ins::CPUPlace(0);

  struct Transfer {
    InsightArray *arr;
    bool is_output;
    void *orig_data;
    void *cpu_data;
    int32_t orig_device_type;
    int32_t orig_device_id;
    size_t bytes;
  };
  std::vector<Transfer> transfers;
  std::vector<InsightArray *> seen;

  auto transfer_nbytes = [](const InsightArray *arr, size_t *bytes) -> bool {
    size_t logical =
        static_cast<size_t>(arr->numel) * insight_dtype_size(arr->dtype);
    if (arr->storage_nbytes) {
      *bytes = arr->storage_nbytes;
      return true;
    }
    if (arr->is_view || arr->offset != 0 ||
        (arr->numel > 1 && !insight_array_is_contiguous(arr))) {
      return false;
    }
    *bytes = logical;
    return true;
  };

  auto consider = [&](void *ptr, bool is_output) {
    if (!ptr)
      return;
    InsightArray *arr = static_cast<InsightArray *>(ptr);
    if (arr->dtype <= INSIGHT_DTYPE_UNKNOWN ||
        arr->dtype >= INSIGHT_DTYPE_COUNT)
      return;
    if (arr->device_type != INSIGHT_DEVICE_GPU)
      return;
    if (arr->ndim < 0 || arr->ndim > INSIGHT_MAX_NDIM)
      return;
    if (arr->numel < 0)
      return;
    for (size_t i = 0; i < seen.size(); ++i) {
      if (seen[i] == arr) {
        if (is_output)
          transfers[i].is_output = true;
        return;
      }
    }
    transfers.push_back({arr, is_output, nullptr, nullptr, arr->device_type,
                         arr->device_id, 0});
    seen.push_back(arr);
  };

  for (int i = 0; i < num_inputs; ++i) {
    if (is_array_kind(input_kinds, input_count, i))
      consider(inputs[i], false);
  }
  for (int i = 0; i < num_outputs; ++i) {
    if (is_array_kind(output_kinds, output_count, i))
      consider(outputs[i], true);
  }

  for (auto &transfer : transfers) {
    InsightArray *a = transfer.arr;
    if (!transfer_nbytes(a, &transfer.bytes)) {
      kernel_error_message =
          std::string("insight_kernel_launch: CPU fallback cannot safely copy "
                      "view storage for operator '") +
          op_name + "'";
      insight_set_last_error(kernel_error_message.c_str());
      return C_FAILED;
    }
    if (transfer.is_output && !a->data && transfer.bytes > 0) {
      kernel_error_message =
          std::string("insight_kernel_launch: CPU fallback requires "
                      "preallocated GPU outputs for operator '") +
          op_name + "'";
      insight_set_last_error(kernel_error_message.c_str());
      return C_FAILED;
    }
  }

  for (size_t t = 0; t < transfers.size(); ++t) {
    InsightArray *a = transfers[t].arr;
    void *cpu = transfers[t].bytes ? std::malloc(transfers[t].bytes) : nullptr;
    if (transfers[t].bytes && !cpu) {
      for (size_t j = 0; j < t; ++j) {
        std::free(transfers[j].cpu_data);
        transfers[j].arr->data = transfers[j].orig_data;
        transfers[j].arr->device_type = transfers[j].orig_device_type;
        transfers[j].arr->device_id = transfers[j].orig_device_id;
      }
      kernel_error_message = "insight_kernel_launch: CPU malloc failed";
      insight_set_last_error(kernel_error_message.c_str());
      return C_FAILED;
    }
    ins::Place gpu_place(ins::DeviceKind::GPU, a->device_id);
    if (transfers[t].bytes)
      gpu_place.copy_to_host(cpu, a->data, transfers[t].bytes);
    transfers[t].orig_data = a->data;
    transfers[t].cpu_data = cpu;
    a->data = cpu;
    a->device_type = INSIGHT_DEVICE_CPU;
    a->device_id = 0;
  }

  C_Status status = cpu_kernel(inputs, outputs);

  if (status != C_SUCCESS) {
    const char *err = cpu_place.get_last_error();
    if (err && err[0] != '\0')
      insight_set_last_error(err);
  }

  for (size_t t = 0; t < transfers.size(); ++t) {
    InsightArray *a = transfers[t].arr;
    ins::Place gpu_place(ins::DeviceKind::GPU, transfers[t].orig_device_id);
    if (transfers[t].is_output && status == C_SUCCESS && transfers[t].bytes)
      gpu_place.copy_from_host(transfers[t].orig_data, a->data,
                               transfers[t].bytes);
    if (a->data && a->data != transfers[t].cpu_data)
      std::free(a->data);
    std::free(transfers[t].cpu_data);
    a->data = transfers[t].orig_data;
    a->device_type = transfers[t].orig_device_type;
    a->device_id = transfers[t].orig_device_id;
    gpu_place.synchronize();
  }

  return status;
}

// ============================================================================
// insight_kernel_launch — unified entry point
// ============================================================================

extern "C" {

C_Status insight_register_kernel(const char *op_name, int32_t device_type,
                                 int32_t dtype, InsightKernel kernel) {
  if (!op_name) {
    kernel_error_message = "insight_register_kernel: op_name is null";
    insight_set_last_error(kernel_error_message.c_str());
    return C_FAILED;
  }
  if (!kernel) {
    kernel_error_message =
        std::string("insight_register_kernel: kernel is null for '") + op_name +
        "'";
    insight_set_last_error(kernel_error_message.c_str());
    return C_FAILED;
  }
  if (device_type != INSIGHT_DEVICE_CPU && device_type != INSIGHT_DEVICE_GPU) {
    kernel_error_message =
        std::string("insight_register_kernel: invalid device_type ") +
        std::to_string(device_type) + " for '" + op_name + "'";
    insight_set_last_error(kernel_error_message.c_str());
    return C_FAILED;
  }
  if (dtype <= INSIGHT_DTYPE_UNKNOWN || dtype >= INSIGHT_DTYPE_COUNT) {
    kernel_error_message =
        std::string("insight_register_kernel: invalid dtype ") +
        std::to_string(dtype) + " for '" + op_name + "'";
    insight_set_last_error(kernel_error_message.c_str());
    return C_FAILED;
  }

  char key[256];
  snprintf(key, sizeof(key), "%s|%d|%d", op_name, device_type, dtype);
  g_kernel_registry[key] = kernel;

  auto &names = registered_op_names();
  if (std::find(names.begin(), names.end(), op_name) == names.end())
    names.push_back(op_name);
  return C_SUCCESS;
}

InsightKernel insight_find_kernel(const char *op_name, int32_t device_type,
                                  int32_t dtype) {
  if (!op_name)
    return nullptr;
  char key[256];
  snprintf(key, sizeof(key), "%s|%d|%d", op_name, device_type, dtype);
  auto it = g_kernel_registry.find(key);
  return (it != g_kernel_registry.end()) ? it->second : nullptr;
}

static C_Status launch_kernel_common(const char *op_name, int32_t device_type,
                                     int32_t dtype, void **inputs,
                                     const InsightKernelArgKind *input_kinds,
                                     int input_count, void **outputs,
                                     const InsightKernelArgKind *output_kinds,
                                     int output_count, bool has_schema) {
  if (!op_name) {
    kernel_error_message = "insight_kernel_launch: op_name is null";
    insight_set_last_error(kernel_error_message.c_str());
    return C_FAILED;
  }
  if (!inputs || !outputs) {
    kernel_error_message =
        std::string("insight_kernel_launch: null array for '") + op_name + "'";
    insight_set_last_error(kernel_error_message.c_str());
    return C_FAILED;
  }
  if (has_schema && (!validate_arg_schema(op_name, "input", inputs, input_kinds,
                                          input_count) ||
                     !validate_arg_schema(op_name, "output", outputs,
                                          output_kinds, output_count))) {
    return C_FAILED;
  }

  InsightKernel kernel = insight_find_kernel(op_name, device_type, dtype);
  if (!kernel) {
    kernel_error_message =
        std::string("insight_kernel_launch: kernel not found for '") + op_name +
        "', device=" + std::to_string(device_type) +
        ", dtype=" + std::to_string(dtype);
    insight_set_last_error(kernel_error_message.c_str());
    return C_FAILED;
  }

  C_Status status = kernel(inputs, outputs);

  if (status == C_FALLBACK && device_type == INSIGHT_DEVICE_GPU) {
    return do_cpu_fallback(
        op_name, dtype, inputs, has_schema ? input_kinds : nullptr, input_count,
        outputs, has_schema ? output_kinds : nullptr, output_count);
  }

  if (status != C_SUCCESS && status != C_FALLBACK) {
    kernel_error_message =
        std::string("Kernel '") + op_name +
        "' failed (device=" + std::to_string(device_type) +
        ", dtype=" + std::to_string(dtype) +
        ", status=" + std::to_string(status) + ")\nBackend: " +
        ins::Place(static_cast<ins::DeviceKind>(device_type), 0)
            .get_last_error();
    insight_set_last_error(kernel_error_message.c_str());
  }
  return status;
}

C_Status insight_kernel_launch(const char *op_name, int32_t device_type,
                               int32_t dtype, void **inputs, void **outputs) {
  return launch_kernel_common(op_name, device_type, dtype, inputs, nullptr, 0,
                              outputs, nullptr, 0, false);
}

C_Status insight_kernel_launch_schema(const char *op_name, int32_t device_type,
                                      int32_t dtype, void **inputs,
                                      const InsightKernelArgKind *input_kinds,
                                      int input_count, void **outputs,
                                      const InsightKernelArgKind *output_kinds,
                                      int output_count) {
  return launch_kernel_common(op_name, device_type, dtype, inputs, input_kinds,
                              input_count, outputs, output_kinds, output_count,
                              true);
}

int insight_has_kernel(const char *op_name, int32_t device_type,
                       int32_t dtype) {
  return insight_find_kernel(op_name, device_type, dtype) != nullptr ? 1 : 0;
}

int insight_get_operator_count(void) {
  return static_cast<int>(registered_op_names().size());
}

C_Status insight_get_operator_name(int index, char *buffer, int size) {
  auto &names = registered_op_names();
  if (index < 0 || index >= static_cast<int>(names.size()))
    return C_FAILED;
  strncpy(buffer, names[index].c_str(), size - 1);
  buffer[size - 1] = '\0';
  return C_SUCCESS;
}

} // extern "C"
