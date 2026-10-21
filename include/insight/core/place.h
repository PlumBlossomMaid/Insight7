// include/insight/core/place.h
#pragma once
#include "insight/c_api/device_ext.h"
#include <cstdint>
#include <ostream>
#include <vector>

namespace ins {

/**
 * @brief Device kind enumeration.
 *
 * Identifies the type of hardware backend.
 */
enum class DeviceKind {
  CPU,         ///< Host CPU
  GPU,         ///< Accelerator device (CUDA, ROCm, Ascend, etc.)
  DEVICE_COUNT ///< Sentinel for array sizing
};

// ========================================================================
// Device interface access (global read/write)
// ========================================================================

/**
 * @brief Get the C ABI device interface for a given device kind.
 *
 * This is the central access point for all device operations.
 * Returns the function pointer table that was registered by the
 * backend plugin during initialization.
 *
 * @param kind Device kind (CPU or GPU)
 * @return Pointer to C_DeviceInterface, or nullptr if the backend
 *         has not been loaded for this device kind
 */
const C_DeviceInterface *get_device_interface(DeviceKind kind);

/**
 * @brief Set the C ABI device interface for a given device kind.
 *
 * Called by the plugin loading mechanism (InitPlugin) to register
 * a backend. Once set, the interface remains valid for the lifetime
 * of the process. This function is not thread-safe and should only
 * be called during initialization.
 *
 * @param kind  Device kind to associate the interface with
 * @param iface Pointer to a fully populated C_DeviceInterface table
 * @param device_type_name Optional name for the device type (e.g., "CUDA",
 * "ROCM")
 */
void set_device_interface(DeviceKind kind, const C_DeviceInterface *iface,
                          const char *device_type_name,
                          const char *sub_device_type_name = nullptr);

// ========================================================================
// Device availability queries
// ========================================================================

/**
 * @brief Check if a device type is available.
 *
 * A device type is available if its backend plugin has been loaded
 * and at least one physical device is present.
 *
 * @param kind Device kind to check
 * @return true if the device backend is loaded and available
 */
bool is_device_available(DeviceKind kind);

/**
 * @brief Get the number of physical devices of a given kind.
 *
 * Requires the backend plugin to be loaded. If the plugin is not
 * loaded, returns 0.
 *
 * @param kind Device kind
 * @return Number of available devices, or 0 if backend not loaded
 */
size_t device_count(DeviceKind kind);

class Place;

/**
 * @brief Get all available devices as a vector of Place objects.
 *
 * This function queries all initialized backends and returns
 * a Place object for each usable device.
 *
 * @return std::vector<Place> List of available devices
 */
std::vector<Place> get_available_devices();

/**
 * @brief Get all available devices as a vector of strings.
 *
 * Format: "cpu", "gpu:0", "gpu:1", etc.
 *
 * @return std::vector<std::string> List of device strings
 */
std::vector<std::string> get_available_device_strings();

// ========================================================================
// Device information queries
// ========================================================================

/**
 * @brief Get the name of a device (e.g., "NVIDIA A800-SXM4-80GB").
 *
 * Returns "CPU" for CPU devices. Requires the backend to be loaded.
 *
 * @param kind Device kind
 * @param device_id Device index
 * @return Device name string, or empty string on error
 */
std::string device_name(DeviceKind kind, int device_id = 0);

/**
 * @brief Get the active GPU backend runtime version.
 *
 * @return GPU runtime version, or 0 if not available
 */
int gpu_runtime_version();

/**
 * @brief Get the active GPU backend driver version.
 *
 * @return GPU driver version, or 0 if not available
 */
int driver_version();

/**
 * @brief Get the compute capability of a GPU device (e.g., 80 for SM 8.0).
 *
 * @param device_id GPU device index
 * @return Compute capability, or 0 if not available
 */
int compute_capability(int device_id = 0);

/**
 * @brief Device memory information.
 */
struct DeviceMemoryInfo {
  size_t total; ///< Total memory in bytes
  size_t free;  ///< Free memory in bytes
};

/**
 * @brief Get memory information for a device.
 *
 * @param kind Device kind (CPU or GPU)
 * @param device_id Device index
 * @return DeviceMemoryInfo with total and free memory
 */
DeviceMemoryInfo device_memory_info(DeviceKind kind, int device_id = 0);

/**
 * @brief Get the active concrete GPU backend name.
 *
 * The public device kind remains GPU, while the registered backend name
 * identifies the concrete implementation selected at runtime (for example
 * "cuda", "rocm", "ixuca", or "sdaa").
 *
 * @return Active GPU backend name, or empty string if no GPU backend is loaded.
 */
std::string active_gpu_backend_name();

/**
 * @brief Get the active concrete GPU backend version/subtype string.
 *
 * @return Backend version/subtype string, or empty string if unavailable.
 */
std::string active_gpu_backend_version();

/**
 * @brief Get memory information for a GPU device (convenience wrapper).
 *
 * @param device_id GPU device index
 * @return DeviceMemoryInfo with total and free memory
 */
inline DeviceMemoryInfo device_memory(int device_id = 0) {
  return device_memory_info(DeviceKind::GPU, device_id);
}

// ========================================================================
// Factory functions for creating Place objects
// ========================================================================

/**
 * @brief Get a CPU device place.
 *
 * @param device_id CPU device ID (usually 0)
 * @return Place descriptor for the CPU device
 */
Place CPUPlace(int device_id = 0);

/**
 * @brief Get a GPU device place.
 *
 * @param device_id GPU device ID (0, 1, 2, ...)
 * @return Place descriptor for the GPU device
 * @throws Exception if GPU backend is not available
 */
Place GPUPlace(int device_id = 0);

/**
 * @brief Get the current default device.
 *
 * This is the device that will be used for new arrays created
 * without an explicit place.
 *
 * @return Current default place
 */
Place get_device();

/**
 * @brief Set the global default device.
 *
 * Subsequent array creations without an explicit place will use
 * this device. Does not affect already-created arrays.
 *
 * @param place Default device to use
 */
void set_device(const Place &place);

// ========================================================================
// Place class
// ========================================================================

/**
 * @brief Device placement descriptor.
 *
 * A lightweight wrapper around a device kind and device ID.
 * Provides convenience methods that delegate to the C ABI device
 * interface stored in the global registry.
 *
 * Place objects are value types and can be freely copied.
 *
 * Examples:
 * @code
 *   Place cpu = CPUPlace();         // CPU:0
 *   Place gpu0 = GPUPlace(0);       // GPU:0
 *   Place gpu1 = GPUPlace(1);       // GPU:1
 *
 *   gpu0.set_current();             // make GPU:0 active
 *   void* ptr = gpu0.allocate(1024); // allocate 1KB on GPU:0
 *   gpu0.synchronize();             // wait for all work to finish
 * @endcode
 */
class Place {
public:
  /**
   * @brief Default constructor. Creates a CPU:0 place.
   */
  Place() = default;

  /**
   * @brief Construct a place with the given device kind and ID.
   *
   * @param kind      Device kind (CPU or GPU)
   * @param device_id Physical device ID
   */
  Place(DeviceKind kind, int device_id);

  /** @brief Get the device kind. */
  DeviceKind kind() const { return kind_; }

  /** @brief Get the physical device ID. */
  int device_id() const { return device_id_; }

  /** @brief Check if this place represents a CPU device. */
  bool is_cpu() const { return kind_ == DeviceKind::CPU; }

  /** @brief Check if this place represents a GPU device. */
  bool is_gpu() const { return kind_ == DeviceKind::GPU; }

  /** @brief Equality comparison. */
  bool operator==(const Place &other) const;

  /** @brief Inequality comparison. */
  bool operator!=(const Place &other) const;

  // ================================================================
  // Device interface access
  // ================================================================

  /**
   * @brief Get the raw C ABI device interface for this place.
   *
   * Returns the function pointer table that was registered by the
   * backend plugin. For CPU devices, returns nullptr (no backend
   * needed—memory operations use standard malloc/free/memcpy).
   *
   * @return Pointer to C_DeviceInterface, or nullptr for CPU
   */
  const C_DeviceInterface *device_interface() const;

  // ================================================================
  // Convenience methods
  // ================================================================

  /**
   * @brief Set this device as current for the calling thread.
   *
   * Subsequent operations will target this device by default.
   */
  void set_current() const;

  /**
   * @brief Synchronize this device.
   *
   * Blocks until all previously enqueued operations on this device
   * have completed.
   */
  void synchronize() const;

  /**
   * @brief Allocate memory on this device.
   *
   * @param size Size in bytes to allocate
   * @return Pointer to allocated device memory
   * @throws Exception on allocation failure
   */
  void *allocate(size_t size) const;

  /**
   * @brief Deallocate memory on this device.
   *
   * @param ptr  Pointer to free (must have been allocated on this device)
   * @param size Size of the allocation (for bookkeeping)
   */
  void deallocate(void *ptr, size_t size) const;

  /**
   * @brief Copy memory from host to this device.
   *
   * @param dst  Destination pointer (on this device)
   * @param src  Source pointer (on host)
   * @param size Number of bytes to copy
   */
  void copy_from_host(void *dst, const void *src, size_t size) const;

  /**
   * @brief Copy memory from this device to host.
   *
   * @param dst  Destination pointer (on host)
   * @param src  Source pointer (on this device)
   * @param size Number of bytes to copy
   */
  void copy_to_host(void *dst, const void *src, size_t size) const;

  /**
   * @brief Copy memory within this device.
   *
   * @param dst  Destination pointer (on this device)
   * @param src  Source pointer (on this device)
   * @param size Number of bytes to copy
   */
  void copy_on_device(void *dst, const void *src, size_t size) const;

  /**
   * @brief Get the last error message from the device backend.
   *
   * Returns a human-readable description of the last error that
   * occurred on this device. The string is owned by the backend
   * and remains valid until the next operation.
   *
   * @return Error message string (never nullptr)
   */
  const char *get_last_error() const;

  /**
   * @brief Convert to human-readable string.
   *
   * @return String like "cpu:0" or "gpu:1"
   */
  std::string to_string() const;

private:
  DeviceKind kind_ = DeviceKind::CPU;
  int device_id_ = 0;
};

/**
 * @brief Get the human-readable name of a device type.
 *
 * This function returns the name that was registered with set_device_interface.
 * If no interface has been registered for the given device kind, returns
 * "unknown".
 *
 * @param kind Device kind to query
 * @return Human-readable name of the device type
 */
const char *get_device_type_name(DeviceKind kind);

/**
 * @brief Output device place to stream.
 *
 * @param os    Output stream
 * @param place Place to output
 * @return Reference to the output stream
 */
std::ostream &operator<<(std::ostream &os, const Place &place);

} // namespace ins