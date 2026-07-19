#pragma once

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <new>
#include <sstream>
#include <string>

namespace ait {

inline thread_local bool target_has_graph_mode = false;

using DeviceError = int;
using Handle = void*;

struct CPUDeviceProperties {
  std::string name{"CPU"};

  // AITemplate's shared profiling runtime expects this field.
  // CPU cache detection will be implemented later.
  size_t l2CacheSize{0};
};

struct CPUStream {};

struct CPUEvent {
  std::chrono::steady_clock::time_point timestamp{};
  bool recorded{false};
};

struct CPUGraph {};
struct CPUGraphExec {};

using DevicePropertyType = CPUDeviceProperties;
using StreamType = CPUStream*;
using EventType = CPUEvent*;
using GraphType = CPUGraph*;
using GraphExecType = CPUGraphExec*;

constexpr DeviceError kCPUSuccess = 0;
constexpr DeviceError kCPUNotReady = 1;
constexpr DeviceError kCPUInvalidValue = 2;
constexpr DeviceError kCPUUnsupported = 3;

inline thread_local DeviceError cpu_last_error = kCPUSuccess;

inline DeviceError SetCPUError(DeviceError error) {
  cpu_last_error = error;
  return error;
}

inline DeviceError GetDevice(int* device_idx) {
  if (device_idx == nullptr) {
    return SetCPUError(kCPUInvalidValue);
  }

  *device_idx = 0;
  return SetCPUError(kCPUSuccess);
}

inline DeviceError GetDeviceProperties(
    DevicePropertyType* prop,
    int device_idx) {
  (void)device_idx;

  if (prop == nullptr) {
    return SetCPUError(kCPUInvalidValue);
  }

  *prop = CPUDeviceProperties{};
  return SetCPUError(kCPUSuccess);
}

inline std::string PrintInfoDeviceProperties(
    const DevicePropertyType& prop) {
  std::ostringstream oss;
  oss << "CPU device properties:"
      << "\n  Device: " << prop.name;
  return oss.str();
}

inline std::string PrintDebugDeviceProperties(
    const DevicePropertyType& prop) {
  return PrintInfoDeviceProperties(prop);
}

inline DeviceError StreamCreate(
    StreamType* stream,
    bool non_blocking = false) {
  (void)non_blocking;

  if (stream == nullptr) {
    return SetCPUError(kCPUInvalidValue);
  }

  *stream = new (std::nothrow) CPUStream();
  if (*stream == nullptr) {
    return SetCPUError(kCPUInvalidValue);
  }

  return SetCPUError(kCPUSuccess);
}

inline DeviceError StreamDestroy(StreamType stream) {
  delete stream;
  return SetCPUError(kCPUSuccess);
}

inline DeviceError StreamSynchronize(StreamType stream) {
  (void)stream;
  return SetCPUError(kCPUSuccess);
}

inline DeviceError StreamWaitEvent(
    StreamType stream,
    EventType event) {
  (void)stream;
  (void)event;
  return SetCPUError(kCPUSuccess);
}

inline DeviceError StreamBeginCapture(
    StreamType stream,
    bool global = true) {
  (void)stream;
  (void)global;
  return SetCPUError(kCPUUnsupported);
}

inline DeviceError StreamEndCapture(
    StreamType stream,
    GraphType* graph) {
  (void)stream;

  if (graph != nullptr) {
    *graph = nullptr;
  }

  return SetCPUError(kCPUUnsupported);
}

inline DeviceError GraphInstantiate(
    GraphExecType* graph_exec,
    GraphType graph) {
  (void)graph;

  if (graph_exec != nullptr) {
    *graph_exec = nullptr;
  }

  return SetCPUError(kCPUUnsupported);
}

inline DeviceError GraphDestroy(GraphType graph) {
  delete graph;
  return SetCPUError(kCPUSuccess);
}

inline DeviceError GraphExecUpdate(
    GraphExecType graph_exec,
    GraphType graph) {
  (void)graph_exec;
  (void)graph;
  return SetCPUError(kCPUUnsupported);
}

inline DeviceError GraphExecDestroy(GraphExecType graph_exec) {
  delete graph_exec;
  return SetCPUError(kCPUSuccess);
}

inline DeviceError GraphExecLaunch(
    GraphExecType graph_exec,
    StreamType stream) {
  (void)graph_exec;
  (void)stream;
  return SetCPUError(kCPUUnsupported);
}

inline DeviceError CPUCopy(
    void* dst,
    const void* src,
    size_t size) {
  if (size == 0) {
    return SetCPUError(kCPUSuccess);
  }

  if (dst == nullptr || src == nullptr) {
    return SetCPUError(kCPUInvalidValue);
  }

  std::memcpy(dst, src, size);
  return SetCPUError(kCPUSuccess);
}

inline DeviceError CopyToDevice(
    Handle dst,
    const void* src,
    size_t size,
    StreamType stream = nullptr) {
  (void)stream;
  return CPUCopy(dst, src, size);
}

inline DeviceError CopyToHost(
    Handle dst,
    const void* src,
    size_t size,
    StreamType stream = nullptr) {
  (void)stream;
  return CPUCopy(dst, src, size);
}

inline DeviceError DeviceToDeviceCopy(
    Handle dst,
    const void* src,
    size_t size,
    StreamType stream = nullptr) {
  (void)stream;
  return CPUCopy(dst, src, size);
}

inline DeviceError DeviceMalloc(
    Handle* dst,
    size_t size) {
  if (dst == nullptr) {
    return SetCPUError(kCPUInvalidValue);
  }

  // Keep zero-byte allocations non-null because parts of the existing
  // runtime perform pointer arithmetic on workspace pointers.
  const size_t allocation_size = size == 0 ? 1 : size;

  *dst = std::malloc(allocation_size);
  if (*dst == nullptr) {
    return SetCPUError(kCPUInvalidValue);
  }

  return SetCPUError(kCPUSuccess);
}

inline DeviceError DeviceMallocHost(
    Handle* dst,
    size_t size) {
  return DeviceMalloc(dst, size);
}

inline DeviceError DeviceMallocAsync(
    Handle* dst,
    size_t size,
    StreamType stream = nullptr) {
  (void)stream;
  return DeviceMalloc(dst, size);
}

inline DeviceError FreeDeviceMemory(Handle src) {
  std::free(src);
  return SetCPUError(kCPUSuccess);
}

inline DeviceError FreeDeviceHostMemory(Handle src) {
  std::free(src);
  return SetCPUError(kCPUSuccess);
}

inline DeviceError FreeDeviceMemoryAsync(
    Handle src,
    StreamType stream = nullptr) {
  (void)stream;
  std::free(src);
  return SetCPUError(kCPUSuccess);
}

inline DeviceError DeviceMemset(
    Handle src,
    int value,
    size_t size) {
  if (size == 0) {
    return SetCPUError(kCPUSuccess);
  }

  if (src == nullptr) {
    return SetCPUError(kCPUInvalidValue);
  }

  std::memset(src, value, size);
  return SetCPUError(kCPUSuccess);
}

inline DeviceError GetDeviceSuccess() {
  return kCPUSuccess;
}

inline DeviceError GetLastError() {
  return cpu_last_error;
}

inline const char* GetErrorString(DeviceError error) {
  switch (error) {
    case kCPUSuccess:
      return "success";
    case kCPUNotReady:
      return "not ready";
    case kCPUInvalidValue:
      return "invalid value";
    case kCPUUnsupported:
      return "unsupported operation";
    default:
      return "unknown CPU runtime error";
  }
}

inline std::string GetLastErrorString() {
  return GetErrorString(cpu_last_error);
}

inline DeviceError CreateEvent(
    EventType* event,
    bool measure_time = true) {
  (void)measure_time;

  if (event == nullptr) {
    return SetCPUError(kCPUInvalidValue);
  }

  *event = new (std::nothrow) CPUEvent();
  if (*event == nullptr) {
    return SetCPUError(kCPUInvalidValue);
  }

  return SetCPUError(kCPUSuccess);
}

inline DeviceError DestroyEvent(EventType event) {
  delete event;
  return SetCPUError(kCPUSuccess);
}

inline DeviceError EventRecord(
    EventType event,
    StreamType stream = nullptr) {
  (void)stream;

  if (event == nullptr) {
    return SetCPUError(kCPUInvalidValue);
  }

  event->timestamp = std::chrono::steady_clock::now();
  event->recorded = true;

  return SetCPUError(kCPUSuccess);
}

inline DeviceError EventSynchronize(EventType event) {
  if (event == nullptr) {
    return SetCPUError(kCPUInvalidValue);
  }

  return SetCPUError(kCPUSuccess);
}

inline DeviceError EventElapsedTime(
    float* ms,
    EventType start,
    EventType end) {
  if (ms == nullptr ||
      start == nullptr ||
      end == nullptr ||
      !start->recorded ||
      !end->recorded) {
    return SetCPUError(kCPUInvalidValue);
  }

  *ms = std::chrono::duration<float, std::milli>(
            end->timestamp - start->timestamp)
            .count();

  return SetCPUError(kCPUSuccess);
}

inline DeviceError QueryEvent(EventType event) {
  if (event == nullptr) {
    return SetCPUError(kCPUInvalidValue);
  }

  // CPU execution is synchronous in the initial backend.
  return SetCPUError(kCPUSuccess);
}

inline DeviceError GetDeviceNotReady() {
  return kCPUNotReady;
}

inline DeviceError GetDriverVersion(int* driver_version) {
  if (driver_version == nullptr) {
    return SetCPUError(kCPUInvalidValue);
  }

  *driver_version = 0;
  return SetCPUError(kCPUSuccess);
}

inline DeviceError GetRuntimeVersion(int* runtime_version) {
  if (runtime_version == nullptr) {
    return SetCPUError(kCPUInvalidValue);
  }

  *runtime_version = 0;
  return SetCPUError(kCPUSuccess);
}

inline void ProfilerRangePush(const char* msg) {
  (void)msg;
}

inline void ProfilerRangePop() {}

} // namespace ait
