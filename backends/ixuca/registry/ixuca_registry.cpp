// backends/ixuca/registry/ixuca_registry.cpp
#include "ixuca_registry.h"
#include "insight/c_api/place.h"
#include <cstring>
#include <string>
#include <unordered_map>

struct KernelEntry {
  std::string op_name;
  int32_t dtype;
  InsightKernel func;
};

static std::unordered_map<std::string, KernelEntry> &local_registry() {
  static std::unordered_map<std::string, KernelEntry> reg;
  return reg;
}

void ixuca_register_kernel(const char *op_name, int32_t dtype,
                              InsightKernel func) {
  if (!op_name || !func)
    return;
  std::string key =
      std::string(op_name) + "_" + std::to_string(static_cast<int>(dtype));
  local_registry()[key] = {op_name, dtype, func};
}

void ixuca_sync_kernels(C_Status (*register_fn)(const char *, int32_t,
                                                   int32_t, InsightKernel)) {
  if (!register_fn)
    return;
  for (auto &kv : local_registry()) {
    auto &entry = kv.second;
    register_fn(entry.op_name.c_str(), INSIGHT_DEVICE_GPU, entry.dtype,
                entry.func);
  }
}

static thread_local std::string last_error;

void ixuca_set_last_error(const char *msg) { last_error = msg ? msg : ""; }

const char *ixuca_get_last_error(void) { return last_error.c_str(); }
