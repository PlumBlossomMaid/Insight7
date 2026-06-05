// src/core/place.cpp
#include "insight/core/place.h"
#include "insight/c_api/device_ext.h"
#include "insight/c_api/dtype.h"
#include "insight/c_api/place.h"
#include "insight/core/exception.h"
#include "insight/init.h"
#include <atomic>
#include <cstring>
#include <iostream>
#include <sstream>

extern "C" {

C_Status insight_get_available_devices(char ***devices_out) {
  if (!devices_out) {
    return C_FAILED;
  }

  std::vector<std::string> device_strings;

  // Check CPU
  if (ins::has_device(ins::DeviceKind::CPU)) {
    device_strings.push_back("cpu");
  }

  // Check GPUs
  if (ins::has_device(ins::DeviceKind::GPU)) {
    const C_DeviceInterface *iface =
        ins::get_device_interface(ins::DeviceKind::GPU);
    if (iface && iface->get_device_count) {
      size_t count = 0;
      if (iface->get_device_count(&count) == C_SUCCESS) {
        for (size_t i = 0; i < count; ++i) {
          device_strings.push_back("gpu:" + std::to_string(i));
        }
      }
    }
  }

  // TODO: Add custom backends (ZQ500, ROCm, etc.) when registered

  // Allocate C array
  size_t count = device_strings.size();
  char **devices = (char **)std::malloc(
      (count + 1) * sizeof(char *)); // +1 for NULL terminator
  if (!devices) {
    return C_FAILED;
  }

  for (size_t i = 0; i < count; ++i) {
    devices[i] = (char *)std::malloc(device_strings[i].size() + 1);
    if (!devices[i]) {
      // Cleanup on failure
      for (size_t j = 0; j < i; ++j) {
        std::free(devices[j]);
      }
      std::free(devices);
      return C_FAILED;
    }
    std::strcpy(devices[i], device_strings[i].c_str());
  }
  devices[count] = nullptr; // NULL terminator

  *devices_out = devices;
  return C_SUCCESS;
}

// insight_free_device_list — moved to place_capi.cpp (LOW-level C ABI)

} // extern "C"

namespace ins {

// ========================================================================
// Global device interface registry (indexed by DeviceKind enum value)
// ========================================================================

struct DeviceInfo {
  const C_DeviceInterface *iface = nullptr;
  std::string device_type_name = ""; // "cuda", "rocm", "cpu", etc.
};

static DeviceInfo g_device_infos[2];

const C_DeviceInterface *get_device_interface(DeviceKind kind) {
  size_t idx = static_cast<size_t>(kind);

  INS_CHECK(idx < 2 && idx >= 0,
            "Invalid DeviceKind: ", static_cast<int>(kind));
  return g_device_infos[idx].iface;
}

void set_device_interface(DeviceKind kind, const C_DeviceInterface *iface,
                          const char *device_type_name) {
  size_t idx = static_cast<size_t>(kind);
  if (idx < 2) {
    g_device_infos[idx].iface = iface;
    g_device_infos[idx].device_type_name =
        device_type_name ? device_type_name
                         : (kind == DeviceKind::CPU ? "cpu" : "gpu");
  }
}

bool is_device_available(DeviceKind kind) {
  const C_DeviceInterface *iface = get_device_interface(kind);
  if (!iface)
    return false;

  if (kind == DeviceKind::CPU) {
    return true; // CPU is always available if registered
  }

  // For GPU, check if at least one physical device exists
  if (iface->get_device_count) {
    size_t count = 0;
    C_Status status = iface->get_device_count(&count);
    return (status == C_SUCCESS && count > 0);
  }
  return false;
}

size_t device_count(DeviceKind kind) {
  const C_DeviceInterface *iface = get_device_interface(kind);
  if (!iface || !iface->get_device_count)
    return 0;

  size_t count = 0;
  C_Status status = iface->get_device_count(&count);
  return (status == C_SUCCESS) ? count : 0;
}

// ========================================================================
// Device information queries
// ========================================================================

std::string device_name(DeviceKind kind, int device_id) {
  if (kind == DeviceKind::CPU)
    return "CPU";

  const C_DeviceInterface *iface = get_device_interface(kind);
  if (!iface || !iface->get_device_name)
    return "";

  C_Device_st dev;
  dev.id = device_id;
  char buf[256] = {};
  C_Status status = iface->get_device_name(&dev, buf, sizeof(buf));
  return (status == C_SUCCESS) ? std::string(buf) : "";
}

int cuda_version() {
  const C_DeviceInterface *iface = get_device_interface(DeviceKind::GPU);
  if (!iface || !iface->get_runtime_version)
    return 0;

  C_Device_st dev;
  dev.id = 0;
  size_t ver = 0;
  C_Status status = iface->get_runtime_version(&dev, &ver);
  return (status == C_SUCCESS) ? static_cast<int>(ver) : 0;
}

int driver_version() {
  const C_DeviceInterface *iface = get_device_interface(DeviceKind::GPU);
  if (!iface || !iface->get_driver_version)
    return 0;

  C_Device_st dev;
  dev.id = 0;
  size_t ver = 0;
  C_Status status = iface->get_driver_version(&dev, &ver);
  return (status == C_SUCCESS) ? static_cast<int>(ver) : 0;
}

int compute_capability(int device_id) {
  const C_DeviceInterface *iface = get_device_interface(DeviceKind::GPU);
  if (!iface || !iface->get_compute_capability)
    return 0;

  C_Device_st dev;
  dev.id = device_id;
  size_t cap = 0;
  C_Status status = iface->get_compute_capability(&dev, &cap);
  return (status == C_SUCCESS) ? static_cast<int>(cap) : 0;
}

DeviceMemoryInfo device_memory(int device_id) {
  DeviceMemoryInfo info{0, 0};
  const C_DeviceInterface *iface = get_device_interface(DeviceKind::GPU);
  if (!iface || !iface->device_memory_stats)
    return info;

  C_Device_st dev;
  dev.id = device_id;
  iface->device_memory_stats(&dev, &info.total, &info.free);
  return info;
}

// ========================================================================
// Factory functions
// ========================================================================

Place CPUPlace(int device_id) { return Place(DeviceKind::CPU, device_id); }

Place GPUPlace(int device_id) {
  INS_CHECK(is_device_available(DeviceKind::GPU),
            "GPUPlace: GPU backend is not available");
  return Place(DeviceKind::GPU, device_id);
}

// ========================================================================
// Global default device
// ========================================================================

namespace {
thread_local Place g_default_device = CPUPlace();
thread_local bool g_device_explicitly_set = false;
} // namespace

// Lazy GPU default: if user hasn't explicitly set device and GPU is available,
// default to GPUPlace(0) — matches PaddlePaddle behavior.
Place get_device() {
  if (!g_device_explicitly_set && g_default_device.kind() == DeviceKind::CPU) {
    if (is_device_available(DeviceKind::GPU)) {
      g_default_device = GPUPlace(0);
    }
  }
  return g_default_device;
}

void set_device(const Place &place) {
  g_default_device = place;
  g_device_explicitly_set = true;
}

// ========================================================================
// Place class implementation
// ========================================================================

Place::Place(DeviceKind kind, int device_id)
    : kind_(kind), device_id_(device_id) {
  INS_CHECK(device_id >= 0, "Place: device_id must be non-negative, got ",
            device_id);
}

bool Place::operator==(const Place &other) const {
  return kind_ == other.kind_ && device_id_ == other.device_id_;
}

bool Place::operator!=(const Place &other) const { return !(*this == other); }

const C_DeviceInterface *Place::device_interface() const {
  return get_device_interface(kind_);
}

void Place::set_current() const {
  const C_DeviceInterface *iface = device_interface();
  if (iface && iface->set_device) {
    C_Device_st dev;
    dev.id = device_id_;
    iface->set_device(&dev);
  }
}

void Place::synchronize() const {
  const C_DeviceInterface *iface = device_interface();
  if (iface && iface->synchronize_device) {
    C_Device_st dev;
    dev.id = device_id_;
    iface->synchronize_device(&dev);
  }
}

void *Place::allocate(size_t size) const {
  const C_DeviceInterface *iface = device_interface();
  INS_CHECK(iface != nullptr, "Device interface not available");
  INS_CHECK(iface->device_memory_allocate != nullptr,
            "Device memory_allocate not implemented");

  C_Device_st dev;
  dev.id = device_id_;
  void *ptr = nullptr;
  C_Status status = iface->device_memory_allocate(&dev, &ptr, size);
  INS_CHECK(status == C_SUCCESS && ptr != nullptr, "GPU allocation failed for ",
            size, " bytes on device ", device_id_);
  return ptr;
}

void Place::deallocate(void *ptr, size_t size) const {
  if (!ptr)
    return;
  const C_DeviceInterface *iface = device_interface();
  if (iface && iface->device_memory_deallocate) {
    C_Device_st dev;
    dev.id = device_id_;
    iface->device_memory_deallocate(&dev, ptr, size);
  }
}

void Place::copy_from_host(void *dst, const void *src, size_t size) const {
  if (size == 0)
    return;
  const C_DeviceInterface *iface = device_interface();
  INS_CHECK(iface != nullptr, "Device interface not available");
  INS_CHECK(iface->memory_copy_h2d != nullptr,
            "Device memory_copy_h2d not implemented");

  C_Device_st dev;
  dev.id = device_id_;
  C_Status status = iface->memory_copy_h2d(&dev, dst, src, size);
  INS_CHECK(status == C_SUCCESS, "Device host-to-device copy failed on device ",
            device_id_);
}

void Place::copy_to_host(void *dst, const void *src, size_t size) const {
  if (size == 0)
    return;
  const C_DeviceInterface *iface = device_interface();
  INS_CHECK(iface != nullptr, "Device interface not available");
  INS_CHECK(iface->memory_copy_d2h != nullptr,
            "Device memory_copy_d2h not implemented");

  C_Device_st dev;
  dev.id = device_id_;
  C_Status status = iface->memory_copy_d2h(&dev, dst, src, size);
  INS_CHECK(status == C_SUCCESS, "Device-to-host copy failed on device ",
            device_id_);
}

void Place::copy_on_device(void *dst, const void *src, size_t size) const {
  if (size == 0)
    return;

  const C_DeviceInterface *iface = device_interface();
  INS_CHECK(iface != nullptr, "Device interface not available");
  INS_CHECK(iface->memory_copy_d2d != nullptr,
            "Device memory_copy_d2d not implemented");

  C_Device_st dev;
  dev.id = device_id_;
  C_Status status = iface->memory_copy_d2d(&dev, dst, src, size);
  INS_CHECK(status == C_SUCCESS, "Device-to-device copy failed on device ",
            device_id_);
}

const char *Place::get_last_error() const {
  const C_DeviceInterface *iface = device_interface();
  if (iface && iface->get_last_error) {
    return iface->get_last_error();
  }
  return "No error information available";
}

std::string Place::to_string() const {
  std::ostringstream os;
  os << *this;
  return os.str();
}

const char *get_device_type_name(DeviceKind kind) {
  size_t idx = static_cast<size_t>(kind);
  return (idx < 2) ? g_device_infos[idx].device_type_name.c_str() : "Unknown";
}

std::vector<Place> get_available_devices() {
  std::vector<Place> places;

  if (has_device(DeviceKind::CPU)) {
    places.emplace_back(CPUPlace());
  }

  if (has_device(DeviceKind::GPU)) {
    const C_DeviceInterface *iface = get_device_interface(DeviceKind::GPU);
    if (iface && iface->get_device_count) {
      size_t count = 0;
      if (iface->get_device_count(&count) == C_SUCCESS) {
        for (size_t i = 0; i < count; ++i) {
          places.emplace_back(GPUPlace(static_cast<int>(i)));
        }
      }
    }
  }

  return places;
}

std::vector<std::string> get_available_device_strings() {
  char **devices = nullptr;
  std::vector<std::string> result;

  if (insight_get_available_devices(&devices) == C_SUCCESS) {
    for (size_t i = 0; devices[i] != nullptr; ++i) {
      result.emplace_back(devices[i]);
    }
    insight_free_device_list(devices);
  }

  return result;
}

std::ostream &operator<<(std::ostream &os, const Place &place) {
  const char *name = get_device_type_name(place.kind());
  os << name << ":" << place.device_id();
  return os;
}

} // namespace ins