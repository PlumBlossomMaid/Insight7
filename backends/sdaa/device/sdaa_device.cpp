// backends/sdaa/device/sdaa_device.cpp
#include "insight/c_api/device_ext.h"
#include "insight/c_api/kernel.h"
#include "insight/c_api/place.h"
#include "../registry/sdaa_registry.h"
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef INSIGHT_SDAA_RUNTIME
#include <sdaa_runtime.h>
#endif

namespace {

struct SdaaProfilerEvent {
  std::string name;
  size_t calls = 0;
  double total_ms = 0.0;
  double min_ms = 1e18;
  double max_ms = 0.0;
};

struct SdaaProfiler {
  std::string name;
  int device_id = 0;
  std::chrono::high_resolution_clock::time_point event_start;
  std::string current_event_name;
  bool in_event = false;
  bool running = false;
  std::unordered_map<std::string, SdaaProfilerEvent> events;
  std::vector<C_ProfilerEvent> report_cache;
};

struct SdaaHostCallback {
  C_Device_st device{};
  C_Stream stream = nullptr;
  void (*callback)(C_Device, C_Stream, void *, C_Status *) = nullptr;
  void *user_data = nullptr;
};

static C_Status sdaa_fallback_kernel(void **, void **) { return C_FALLBACK; }

bool has_stub_device() {
#ifdef INSIGHT_SDAA_STUB_DEVICE
  return true;
#else
  return false;
#endif
}

void set_error(const std::string &msg) { sdaa_set_last_error(msg.c_str()); }

#ifdef INSIGHT_SDAA_RUNTIME
C_Status from_sdaa_status(const char *op, sdaaError_t err) {
  if (err == sdaaSuccess) {
    sdaa_set_last_error("");
    return C_SUCCESS;
  }
  const char *name = sdaaGetErrorName(err);
  const char *desc = sdaaGetErrorString(err);
  std::string msg = std::string("sdaa: ") + op + " failed";
  if (name && name[0] != '\0') {
    msg += " [";
    msg += name;
    msg += "]";
  }
  if (desc && desc[0] != '\0') {
    msg += ": ";
    msg += desc;
  }
  set_error(msg);
  return C_FAILED;
}

C_Status from_sdaa_query_status(const char *op, sdaaError_t err) {
  if (err == sdaaSuccess) {
    sdaa_set_last_error("");
    return C_SUCCESS;
  }
  if (err == sdaaErrorNotReady) {
    sdaa_set_last_error("");
    return C_WARNING;
  }
  return from_sdaa_status(op, err);
}

C_Status set_current_device(C_Device device, const char *op) {
  if (!device) {
    set_error(std::string("sdaa: ") + op + " received null device");
    return C_FAILED;
  }
  return from_sdaa_status(op, sdaaSetDevice(device->id));
}

sdaaStream_t to_sdaa_stream(C_Stream stream) {
  return reinterpret_cast<sdaaStream_t>(stream);
}

sdaaEvent_t to_sdaa_event(C_Event event) {
  return reinterpret_cast<sdaaEvent_t>(event);
}

C_Status runtime_device_count(int *count) {
  if (!count)
    return C_FAILED;
  *count = 0;
  return from_sdaa_status("get_device_count", sdaaGetDeviceCount(count));
}
#endif

C_Status require_stub_device(const char *op) {
  if (has_stub_device())
    return C_SUCCESS;
  set_error(std::string("sdaa: ") + op + " requires SDAA runtime");
  return C_FAILED;
}

C_Status sdaa_initialize() {
#ifdef INSIGHT_SDAA_RUNTIME
  int count = 0;
  return runtime_device_count(&count);
#else
  return has_stub_device() ? C_SUCCESS : C_FAILED;
#endif
}

C_Status sdaa_finalize() { return C_SUCCESS; }

C_Status sdaa_init_device(C_Device device) {
  if (!device)
    return C_FAILED;
#ifdef INSIGHT_SDAA_RUNTIME
  return set_current_device(device, "init_device");
#else
  device->id = 0;
  return has_stub_device() ? C_SUCCESS : C_FAILED;
#endif
}

C_Status sdaa_deinit_device(C_Device) { return C_SUCCESS; }

C_Status sdaa_set_device(C_Device device) {
#ifdef INSIGHT_SDAA_RUNTIME
  return set_current_device(device, "set_device");
#else
  if (!device)
    return C_FAILED;
  return require_stub_device("set_device");
#endif
}

C_Status sdaa_get_device(C_Device device) {
  if (!device)
    return C_FAILED;
#ifdef INSIGHT_SDAA_RUNTIME
  int id = 0;
  C_Status status = from_sdaa_status("get_device", sdaaGetDevice(&id));
  if (status == C_SUCCESS)
    device->id = id;
  return status;
#else
  device->id = 0;
  return has_stub_device() ? C_SUCCESS : C_FAILED;
#endif
}

C_Status sdaa_synchronize_device(C_Device device) {
#ifdef INSIGHT_SDAA_RUNTIME
  C_Status status = set_current_device(device, "synchronize_device");
  if (status != C_SUCCESS)
    return status;
  return from_sdaa_status("synchronize_device", sdaaDeviceSynchronize());
#else
  if (!device)
    return C_FAILED;
  return require_stub_device("synchronize_device");
#endif
}

C_Status sdaa_get_device_count(size_t *count) {
  if (!count)
    return C_FAILED;
#ifdef INSIGHT_SDAA_RUNTIME
  int c = 0;
  C_Status status = runtime_device_count(&c);
  *count = c > 0 ? static_cast<size_t>(c) : 0;
  return status;
#else
  *count = has_stub_device() ? 1 : 0;
  return C_SUCCESS;
#endif
}

C_Status sdaa_get_device_list(size_t *devices) {
  if (!devices)
    return C_FAILED;
  size_t count = 0;
  C_Status status = sdaa_get_device_count(&count);
  if (status != C_SUCCESS)
    return status;
  for (size_t i = 0; i < count; ++i)
    devices[i] = i;
  return C_SUCCESS;
}

C_Status sdaa_device_memory_allocate(C_Device device, void **ptr, size_t size) {
  if (!ptr || !device)
    return C_FAILED;
  *ptr = nullptr;
  if (size == 0)
    return C_SUCCESS;
#ifdef INSIGHT_SDAA_RUNTIME
  C_Status status = set_current_device(device, "device_memory_allocate");
  if (status != C_SUCCESS)
    return status;
  return from_sdaa_status("device_memory_allocate", sdaaMalloc(ptr, size));
#else
  if (require_stub_device("device_memory_allocate") != C_SUCCESS)
    return C_FAILED;
  *ptr = std::malloc(size);
  if (!*ptr) {
    set_error("sdaa: allocation failed");
    return C_FAILED;
  }
  return C_SUCCESS;
#endif
}

C_Status sdaa_device_memory_deallocate(C_Device device, void *ptr, size_t) {
  if (!ptr)
    return C_SUCCESS;
#ifdef INSIGHT_SDAA_RUNTIME
  if (device)
    set_current_device(device, "device_memory_deallocate");
  return from_sdaa_status("device_memory_deallocate", sdaaFree(ptr));
#else
  std::free(ptr);
  return C_SUCCESS;
#endif
}

C_Status sdaa_host_memory_allocate(C_Device device, void **ptr, size_t size) {
  if (!ptr || !device)
    return C_FAILED;
  *ptr = nullptr;
  if (size == 0)
    return C_SUCCESS;
#ifdef INSIGHT_SDAA_RUNTIME
  return from_sdaa_status("host_memory_allocate", sdaaMallocHost(ptr, size));
#else
  return sdaa_device_memory_allocate(device, ptr, size);
#endif
}

C_Status sdaa_host_memory_deallocate(C_Device device, void *ptr, size_t size) {
#ifdef INSIGHT_SDAA_RUNTIME
  (void)device;
  (void)size;
  if (!ptr)
    return C_SUCCESS;
  return from_sdaa_status("host_memory_deallocate", sdaaFreeHost(ptr));
#else
  return sdaa_device_memory_deallocate(device, ptr, size);
#endif
}

#ifdef INSIGHT_SDAA_RUNTIME
C_Status sdaa_memory_copy_kind(C_Device device, void *dst, const void *src,
                               size_t size, sdaaMemcpyKind kind,
                               const char *op) {
  if (!device || (!dst && size > 0) || (!src && size > 0))
    return C_FAILED;
  if (size == 0)
    return C_SUCCESS;
  C_Status status = set_current_device(device, op);
  if (status != C_SUCCESS)
    return status;
  return from_sdaa_status(op, sdaaMemcpy(dst, src, size, kind));
}

C_Status sdaa_async_memory_copy_kind(C_Device device, C_Stream stream, void *dst,
                                     const void *src, size_t size,
                                     sdaaMemcpyKind kind, const char *op) {
  if (!device || (!dst && size > 0) || (!src && size > 0))
    return C_FAILED;
  if (size == 0)
    return C_SUCCESS;
  C_Status status = set_current_device(device, op);
  if (status != C_SUCCESS)
    return status;
  return from_sdaa_status(
      op, sdaaMemcpyAsync(dst, src, size, kind, to_sdaa_stream(stream)));
}
#endif

C_Status sdaa_memory_copy_h2d(C_Device device, void *dst, const void *src,
                              size_t size) {
#ifdef INSIGHT_SDAA_RUNTIME
  return sdaa_memory_copy_kind(device, dst, src, size, sdaaMemcpyHostToDevice,
                               "memory_copy_h2d");
#else
  if (!device || (!dst && size > 0) || (!src && size > 0))
    return C_FAILED;
  if (require_stub_device("memory_copy_h2d") != C_SUCCESS)
    return C_FAILED;
  if (size > 0)
    std::memcpy(dst, src, size);
  return C_SUCCESS;
#endif
}

C_Status sdaa_memory_copy_d2h(C_Device device, void *dst, const void *src,
                              size_t size) {
#ifdef INSIGHT_SDAA_RUNTIME
  return sdaa_memory_copy_kind(device, dst, src, size, sdaaMemcpyDeviceToHost,
                               "memory_copy_d2h");
#else
  return sdaa_memory_copy_h2d(device, dst, src, size);
#endif
}

C_Status sdaa_memory_copy_d2d(C_Device device, void *dst, const void *src,
                              size_t size) {
#ifdef INSIGHT_SDAA_RUNTIME
  return sdaa_memory_copy_kind(device, dst, src, size, sdaaMemcpyDeviceToDevice,
                               "memory_copy_d2d");
#else
  return sdaa_memory_copy_h2d(device, dst, src, size);
#endif
}

C_Status sdaa_memory_copy_p2p(C_Device dst_device, C_Device src_device,
                              void *dst, const void *src, size_t size) {
  if (!dst_device || !src_device)
    return C_FAILED;
#ifdef INSIGHT_SDAA_RUNTIME
  if (dst_device->id == src_device->id)
    return sdaa_memory_copy_d2d(dst_device, dst, src, size);
  void *host = size ? std::malloc(size) : nullptr;
  if (size && !host) {
    set_error("sdaa: memory_copy_p2p host staging allocation failed");
    return C_FAILED;
  }
  C_Status status = sdaa_memory_copy_d2h(src_device, host, src, size);
  if (status == C_SUCCESS)
    status = sdaa_memory_copy_h2d(dst_device, dst, host, size);
  std::free(host);
  return status;
#else
  return sdaa_memory_copy_h2d(dst_device, dst, src, size);
#endif
}

C_Status sdaa_async_memory_copy_h2d(C_Device device, C_Stream stream, void *dst,
                                    const void *src, size_t size) {
#ifdef INSIGHT_SDAA_RUNTIME
  return sdaa_async_memory_copy_kind(device, stream, dst, src, size,
                                     sdaaMemcpyHostToDevice,
                                     "async_memory_copy_h2d");
#else
  return sdaa_memory_copy_h2d(device, dst, src, size);
#endif
}

C_Status sdaa_async_memory_copy_d2h(C_Device device, C_Stream stream, void *dst,
                                    const void *src, size_t size) {
#ifdef INSIGHT_SDAA_RUNTIME
  return sdaa_async_memory_copy_kind(device, stream, dst, src, size,
                                     sdaaMemcpyDeviceToHost,
                                     "async_memory_copy_d2h");
#else
  return sdaa_memory_copy_d2h(device, dst, src, size);
#endif
}

C_Status sdaa_async_memory_copy_d2d(C_Device device, C_Stream stream, void *dst,
                                    const void *src, size_t size) {
#ifdef INSIGHT_SDAA_RUNTIME
  return sdaa_async_memory_copy_kind(device, stream, dst, src, size,
                                     sdaaMemcpyDeviceToDevice,
                                     "async_memory_copy_d2d");
#else
  return sdaa_memory_copy_d2d(device, dst, src, size);
#endif
}

C_Status sdaa_async_memory_copy_p2p(C_Device dst_device, C_Device src_device,
                                    C_Stream, void *dst, const void *src,
                                    size_t size) {
  return sdaa_memory_copy_p2p(dst_device, src_device, dst, src, size);
}

C_Status sdaa_device_memory_set(C_Device device, void *ptr, unsigned char value,
                                size_t size) {
  if (!device || (!ptr && size > 0))
    return C_FAILED;
  if (size == 0)
    return C_SUCCESS;
#ifdef INSIGHT_SDAA_RUNTIME
  C_Status status = set_current_device(device, "device_memory_set");
  if (status != C_SUCCESS)
    return status;
  return from_sdaa_status("device_memory_set",
                          sdaaMemset(ptr, static_cast<int>(value), size));
#else
  if (require_stub_device("device_memory_set") != C_SUCCESS)
    return C_FAILED;
  std::memset(ptr, value, size);
  return C_SUCCESS;
#endif
}

C_Status sdaa_device_memory_stats(C_Device device, size_t *total_memory,
                                  size_t *free_memory) {
  if (!device || !total_memory || !free_memory)
    return C_FAILED;
#ifdef INSIGHT_SDAA_RUNTIME
  C_Status status = set_current_device(device, "device_memory_stats");
  if (status != C_SUCCESS)
    return status;
  return from_sdaa_status("device_memory_stats",
                          sdaaMemGetInfo(free_memory, total_memory));
#else
  if (require_stub_device("device_memory_stats") != C_SUCCESS)
    return C_FAILED;
  *total_memory = 0;
  *free_memory = 0;
  return C_SUCCESS;
#endif
}

C_Status sdaa_create_stream(C_Device device, C_Stream *stream) {
  if (!stream || !device)
    return C_FAILED;
  *stream = nullptr;
#ifdef INSIGHT_SDAA_RUNTIME
  C_Status status = set_current_device(device, "create_stream");
  if (status != C_SUCCESS)
    return status;
  sdaaStream_t s = nullptr;
  status = from_sdaa_status("create_stream", sdaaStreamCreate(&s));
  if (status == C_SUCCESS)
    *stream = reinterpret_cast<C_Stream>(s);
  return status;
#else
  return require_stub_device("create_stream");
#endif
}

C_Status sdaa_destroy_stream(C_Device, C_Stream stream) {
#ifdef INSIGHT_SDAA_RUNTIME
  if (!stream)
    return C_SUCCESS;
  return from_sdaa_status("destroy_stream",
                          sdaaStreamDestroy(to_sdaa_stream(stream)));
#else
  (void)stream;
  return C_SUCCESS;
#endif
}

C_Status sdaa_query_stream(C_Device, C_Stream stream) {
#ifdef INSIGHT_SDAA_RUNTIME
  return from_sdaa_query_status("query_stream",
                                sdaaStreamQuery(to_sdaa_stream(stream)));
#else
  (void)stream;
  return C_SUCCESS;
#endif
}

C_Status sdaa_synchronize_stream(C_Device, C_Stream stream) {
#ifdef INSIGHT_SDAA_RUNTIME
  return from_sdaa_status("synchronize_stream",
                          sdaaStreamSynchronize(to_sdaa_stream(stream)));
#else
  (void)stream;
  return C_SUCCESS;
#endif
}

#ifdef INSIGHT_SDAA_RUNTIME
void sdaa_host_callback_trampoline(void *user_data) {
  auto *ctx = reinterpret_cast<SdaaHostCallback *>(user_data);
  C_Status status = C_SUCCESS;
  if (ctx->callback)
    ctx->callback(&ctx->device, ctx->stream, ctx->user_data, &status);
  delete ctx;
}
#endif

C_Status sdaa_stream_add_callback(C_Device device, C_Stream stream,
                                  void (*callback)(C_Device, C_Stream, void *,
                                                   C_Status *),
                                  void *user_data) {
  if (!callback)
    return C_SUCCESS;
#ifdef INSIGHT_SDAA_RUNTIME
  auto *ctx = new (std::nothrow) SdaaHostCallback();
  if (!ctx)
    return C_FAILED;
  if (device)
    ctx->device = *device;
  ctx->stream = stream;
  ctx->callback = callback;
  ctx->user_data = user_data;
  C_Status status = from_sdaa_status(
      "stream_add_callback",
      sdaaLaunchHostFunc(to_sdaa_stream(stream), sdaa_host_callback_trampoline,
                         ctx));
  if (status != C_SUCCESS)
    delete ctx;
  return status;
#else
  C_Status status = C_SUCCESS;
  callback(device, stream, user_data, &status);
  return status;
#endif
}

C_Status sdaa_stream_wait_event(C_Device, C_Stream stream, C_Event event) {
#ifdef INSIGHT_SDAA_RUNTIME
  return from_sdaa_status(
      "stream_wait_event",
      sdaaStreamWaitEvent(to_sdaa_stream(stream), to_sdaa_event(event), 0));
#else
  (void)stream;
  (void)event;
  return C_SUCCESS;
#endif
}

C_Status sdaa_create_event(C_Device device, C_Event *event) {
  if (!event || !device)
    return C_FAILED;
  *event = nullptr;
#ifdef INSIGHT_SDAA_RUNTIME
  C_Status status = set_current_device(device, "create_event");
  if (status != C_SUCCESS)
    return status;
  sdaaEvent_t e = nullptr;
  status = from_sdaa_status("create_event", sdaaEventCreate(&e));
  if (status == C_SUCCESS)
    *event = reinterpret_cast<C_Event>(e);
  return status;
#else
  auto *e = new (std::nothrow) std::chrono::high_resolution_clock::time_point();
  if (!e)
    return C_FAILED;
  *event = reinterpret_cast<C_Event>(e);
  return C_SUCCESS;
#endif
}

C_Status sdaa_destroy_event(C_Device, C_Event event) {
#ifdef INSIGHT_SDAA_RUNTIME
  if (!event)
    return C_SUCCESS;
  return from_sdaa_status("destroy_event", sdaaEventDestroy(to_sdaa_event(event)));
#else
  delete reinterpret_cast<std::chrono::high_resolution_clock::time_point *>(event);
  return C_SUCCESS;
#endif
}

C_Status sdaa_record_event(C_Device, C_Stream stream, C_Event event) {
  if (!event)
    return C_FAILED;
#ifdef INSIGHT_SDAA_RUNTIME
  return from_sdaa_status("record_event",
                          sdaaEventRecord(to_sdaa_event(event),
                                          to_sdaa_stream(stream)));
#else
  *reinterpret_cast<std::chrono::high_resolution_clock::time_point *>(event) =
      std::chrono::high_resolution_clock::now();
  return C_SUCCESS;
#endif
}

C_Status sdaa_query_event(C_Device, C_Event event) {
#ifdef INSIGHT_SDAA_RUNTIME
  return from_sdaa_query_status("query_event",
                                sdaaEventQuery(to_sdaa_event(event)));
#else
  (void)event;
  return C_SUCCESS;
#endif
}

C_Status sdaa_synchronize_event(C_Device, C_Event event) {
#ifdef INSIGHT_SDAA_RUNTIME
  return from_sdaa_status("synchronize_event",
                          sdaaEventSynchronize(to_sdaa_event(event)));
#else
  (void)event;
  return C_SUCCESS;
#endif
}

C_Status sdaa_elapsed_time(C_Event start, C_Event end, float *ms) {
  if (!start || !end || !ms)
    return C_FAILED;
#ifdef INSIGHT_SDAA_RUNTIME
  return from_sdaa_status("elapsed_time",
                          sdaaEventElapsedTime(ms, to_sdaa_event(start),
                                               to_sdaa_event(end)));
#else
  auto *s = reinterpret_cast<std::chrono::high_resolution_clock::time_point *>(start);
  auto *e = reinterpret_cast<std::chrono::high_resolution_clock::time_point *>(end);
  *ms = std::chrono::duration<float, std::milli>(*e - *s).count();
  return C_SUCCESS;
#endif
}

C_Status sdaa_get_compute_capability(C_Device device, size_t *capability) {
  if (!device || !capability)
    return C_FAILED;
  *capability = 0;
#ifdef INSIGHT_SDAA_RUNTIME
  int arch = 0;
  if (sdaaDeviceGetAttribute(&arch, sdaaDevAttrArch, device->id) == sdaaSuccess)
    *capability = static_cast<size_t>(arch);
  return C_SUCCESS;
#else
  return require_stub_device("get_compute_capability");
#endif
}

C_Status sdaa_get_runtime_version(C_Device, size_t *version) {
  if (!version)
    return C_FAILED;
  *version = 0;
#ifdef INSIGHT_SDAA_RUNTIME
  int v = 0;
  C_Status status = from_sdaa_status("get_runtime_version",
                                     sdaaRuntimeGetVersion(&v));
  if (status == C_SUCCESS)
    *version = static_cast<size_t>(v);
  return status;
#else
  return require_stub_device("get_runtime_version");
#endif
}

C_Status sdaa_get_driver_version(C_Device, size_t *version) {
  if (!version)
    return C_FAILED;
  *version = 0;
#ifdef INSIGHT_SDAA_RUNTIME
  int v = 0;
  C_Status status = from_sdaa_status("get_driver_version",
                                     sdaaDriverGetVersion(&v));
  if (status == C_SUCCESS)
    *version = static_cast<size_t>(v);
  return status;
#else
  return require_stub_device("get_driver_version");
#endif
}

C_Status sdaa_get_multi_process(C_Device, size_t *multi_process) {
  if (!multi_process)
    return C_FAILED;
  *multi_process = 0;
  return C_SUCCESS;
}

C_Status sdaa_get_max_threads_per_mp(C_Device, size_t *threads) {
  if (!threads)
    return C_FAILED;
  *threads = 0;
  return C_SUCCESS;
}

C_Status sdaa_get_max_threads_per_block(C_Device device, size_t *threads) {
  if (!device || !threads)
    return C_FAILED;
  *threads = 0;
#ifdef INSIGHT_SDAA_RUNTIME
  int value = 0;
  if (sdaaDeviceGetAttribute(&value, sdaaDevAttrMaxThreadsPerBlock,
                             device->id) == sdaaSuccess)
    *threads = static_cast<size_t>(value);
  return C_SUCCESS;
#else
  return require_stub_device("get_max_threads_per_block");
#endif
}

C_Status sdaa_get_max_grid_dim_size(C_Device, size_t dims[3]) {
  if (!dims)
    return C_FAILED;
  dims[0] = dims[1] = dims[2] = 0;
  return C_SUCCESS;
}

C_Status sdaa_get_device_name(C_Device device, char *buf, size_t buf_size) {
  if (!device || !buf || buf_size == 0)
    return C_FAILED;
#ifdef INSIGHT_SDAA_RUNTIME
  sdaaDeviceProp_t props{};
  C_Status status = from_sdaa_status(
      "get_device_name", sdaaGetDeviceProperties(&props, device->id));
  if (status != C_SUCCESS)
    return status;
  std::strncpy(buf, props.name, buf_size - 1);
#else
  const char *name = has_stub_device() ? "SDAA Stub" : "SDAA";
  std::strncpy(buf, name, buf_size - 1);
#endif
  buf[buf_size - 1] = '\0';
  return C_SUCCESS;
}

C_Status sdaa_profiler_create(C_Profiler *prof, const char *name,
                              int device_id) {
  if (!prof)
    return C_FAILED;
  auto *p = new (std::nothrow) SdaaProfiler();
  if (!p)
    return C_FAILED;
  if (name)
    p->name = name;
  p->device_id = device_id;
  *prof = reinterpret_cast<C_Profiler>(p);
  return C_SUCCESS;
}

C_Status sdaa_profiler_destroy(C_Profiler prof) {
  delete reinterpret_cast<SdaaProfiler *>(prof);
  return C_SUCCESS;
}

C_Status sdaa_profiler_start(C_Profiler prof) {
  if (!prof)
    return C_FAILED;
  reinterpret_cast<SdaaProfiler *>(prof)->running = true;
  return C_SUCCESS;
}

C_Status sdaa_profiler_stop(C_Profiler prof) {
  if (!prof)
    return C_FAILED;
  reinterpret_cast<SdaaProfiler *>(prof)->running = false;
  return C_SUCCESS;
}

C_Status sdaa_profiler_reset(C_Profiler prof) {
  if (!prof)
    return C_FAILED;
  auto *p = reinterpret_cast<SdaaProfiler *>(prof);
  p->events.clear();
  p->report_cache.clear();
  p->in_event = false;
  return C_SUCCESS;
}

C_Status sdaa_profiler_begin_event(C_Profiler prof, const char *name) {
  if (!prof || !name)
    return C_FAILED;
  auto *p = reinterpret_cast<SdaaProfiler *>(prof);
  if (!p->running)
    return C_SUCCESS;
  p->current_event_name = name;
  p->event_start = std::chrono::high_resolution_clock::now();
  p->in_event = true;
  return C_SUCCESS;
}

C_Status sdaa_profiler_end_event(C_Profiler prof) {
  if (!prof)
    return C_FAILED;
  auto *p = reinterpret_cast<SdaaProfiler *>(prof);
  if (!p->running || !p->in_event)
    return C_SUCCESS;
  auto end = std::chrono::high_resolution_clock::now();
  double ms =
      std::chrono::duration<double, std::milli>(end - p->event_start).count();
  p->in_event = false;
  auto &event = p->events[p->current_event_name];
  event.name = p->current_event_name;
  event.calls++;
  event.total_ms += ms;
  if (ms < event.min_ms)
    event.min_ms = ms;
  if (ms > event.max_ms)
    event.max_ms = ms;
  return C_SUCCESS;
}

C_Status sdaa_profiler_get_events(C_Profiler prof, C_ProfilerEvent **events,
                                  size_t *count) {
  if (!prof || !events || !count)
    return C_FAILED;
  auto *p = reinterpret_cast<SdaaProfiler *>(prof);
  p->report_cache.clear();
  for (auto &kv : p->events) {
    auto &src = kv.second;
    C_ProfilerEvent event;
    event.name = src.name.c_str();
    event.calls = src.calls;
    event.total_ms = static_cast<float>(src.total_ms);
    event.min_ms = static_cast<float>(src.min_ms < 1e17 ? src.min_ms : 0.0);
    event.max_ms = static_cast<float>(src.max_ms);
    p->report_cache.push_back(event);
  }
  *events = p->report_cache.data();
  *count = p->report_cache.size();
  return C_SUCCESS;
}

void register_cpu_fallback_kernels(C_Status (*register_fn)(const char *, int32_t,
                                                            int32_t,
                                                            InsightKernel)) {
  if (!register_fn)
    return;
  int op_count = insight_get_operator_count();
  char op_name[256];
  for (int i = 0; i < op_count; ++i) {
    if (insight_get_operator_name(i, op_name, sizeof(op_name)) != C_SUCCESS)
      continue;
    for (int dtype = INSIGHT_DTYPE_UNKNOWN + 1; dtype < INSIGHT_DTYPE_COUNT;
         ++dtype) {
      if (insight_has_kernel(op_name, INSIGHT_DEVICE_CPU, dtype))
        register_fn(op_name, INSIGHT_DEVICE_GPU, dtype, sdaa_fallback_kernel);
    }
  }
}

} // namespace

extern "C" {

INSIGHT_GPU_API C_Status InitPluginGPU(CustomRuntimeParams *params) {
  if (!params || !params->interface)
    return C_FAILED;

  INSIGHT_CHECK_CUSTOM_DEVICE_VERSION(params);

  C_DeviceInterface *iface = params->interface;
  iface->size = sizeof(C_DeviceInterface);

  iface->initialize = sdaa_initialize;
  iface->finalize = sdaa_finalize;
  iface->init_device = sdaa_init_device;
  iface->deinit_device = sdaa_deinit_device;

  iface->set_device = sdaa_set_device;
  iface->get_device = sdaa_get_device;
  iface->synchronize_device = sdaa_synchronize_device;
  iface->get_device_count = sdaa_get_device_count;
  iface->get_device_list = sdaa_get_device_list;

  iface->device_memory_allocate = sdaa_device_memory_allocate;
  iface->device_memory_deallocate = sdaa_device_memory_deallocate;
  iface->host_memory_allocate = sdaa_host_memory_allocate;
  iface->host_memory_deallocate = sdaa_host_memory_deallocate;
  iface->memory_copy_h2d = sdaa_memory_copy_h2d;
  iface->memory_copy_d2h = sdaa_memory_copy_d2h;
  iface->memory_copy_d2d = sdaa_memory_copy_d2d;
  iface->memory_copy_p2p = sdaa_memory_copy_p2p;
  iface->device_memory_set = sdaa_device_memory_set;
  iface->device_memory_stats = sdaa_device_memory_stats;

  iface->async_memory_copy_h2d = sdaa_async_memory_copy_h2d;
  iface->async_memory_copy_d2h = sdaa_async_memory_copy_d2h;
  iface->async_memory_copy_d2d = sdaa_async_memory_copy_d2d;
  iface->async_memory_copy_p2p = sdaa_async_memory_copy_p2p;

  iface->create_stream = sdaa_create_stream;
  iface->destroy_stream = sdaa_destroy_stream;
  iface->query_stream = sdaa_query_stream;
  iface->synchronize_stream = sdaa_synchronize_stream;
  iface->stream_add_callback = sdaa_stream_add_callback;
  iface->stream_wait_event = sdaa_stream_wait_event;

  iface->create_event = sdaa_create_event;
  iface->destroy_event = sdaa_destroy_event;
  iface->record_event = sdaa_record_event;
  iface->query_event = sdaa_query_event;
  iface->synchronize_event = sdaa_synchronize_event;
  iface->elapsed_time = sdaa_elapsed_time;

  iface->get_compute_capability = sdaa_get_compute_capability;
  iface->get_runtime_version = sdaa_get_runtime_version;
  iface->get_driver_version = sdaa_get_driver_version;
  iface->get_multi_process = sdaa_get_multi_process;
  iface->get_max_threads_per_mp = sdaa_get_max_threads_per_mp;
  iface->get_max_threads_per_block = sdaa_get_max_threads_per_block;
  iface->get_max_grid_dim_size = sdaa_get_max_grid_dim_size;
  iface->get_device_name = sdaa_get_device_name;

  iface->profiler_create = sdaa_profiler_create;
  iface->profiler_destroy = sdaa_profiler_destroy;
  iface->profiler_start = sdaa_profiler_start;
  iface->profiler_stop = sdaa_profiler_stop;
  iface->profiler_reset = sdaa_profiler_reset;
  iface->profiler_begin_event = sdaa_profiler_begin_event;
  iface->profiler_end_event = sdaa_profiler_end_event;
  iface->profiler_get_events = sdaa_profiler_get_events;

  iface->get_last_error = sdaa_get_last_error;

  for (int i = 0; i < 8; ++i)
    iface->reserved[i] = nullptr;

  params->device_type = const_cast<char *>("sdaa");
#ifdef INSIGHT_SDAA_RUNTIME
  params->sub_device_type = const_cast<char *>("runtime");
#else
  params->sub_device_type = const_cast<char *>("stub");
#endif

  if (params->register_kernel) {
    register_cpu_fallback_kernels(params->register_kernel);
    sdaa_sync_kernels(params->register_kernel);
  }

  return C_SUCCESS;
}

} // extern "C"
