#pragma once

#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>

#include <pthreadpool.h>


namespace ait {


inline size_t cpu_num_threads() {
  static const size_t num_threads =
      []() -> size_t {
        const char* value =
            std::getenv(
                "AIT_CPU_NUM_THREADS");

        if (
            value == nullptr ||
            value[0] == '\0') {
          return 1;
        }

        if (
            value[0] < '0' ||
            value[0] > '9') {
          throw std::runtime_error(
              "AIT_CPU_NUM_THREADS must be "
              "a positive integer");
        }

        errno = 0;

        char* end = nullptr;

        const unsigned long long parsed =
            std::strtoull(
                value,
                &end,
                10);

        if (
            errno != 0 ||
            end == value ||
            *end != '\0' ||
            parsed == 0 ||
            parsed >
                static_cast<unsigned long long>(
                    std::numeric_limits<
                        size_t>::max())) {
          throw std::runtime_error(
              "AIT_CPU_NUM_THREADS must be "
              "a positive integer");
        }

        return static_cast<size_t>(
            parsed);
      }();

  return num_threads;
}


class CpuThreadPoolHolder {
 public:
  CpuThreadPoolHolder()
      : requested_threads_(
            cpu_num_threads()) {

    if (requested_threads_ > 1) {
      pool_ =
          pthreadpool_create(
              requested_threads_);

      if (pool_ == nullptr) {
        throw std::runtime_error(
            "pthreadpool_create failed");
      }
    }

    const char* debug =
        std::getenv(
            "AIT_CPU_THREADPOOL_DEBUG");

    const bool debug_enabled =
        debug != nullptr &&
        debug[0] != '\0' &&
        !(
            debug[0] == '0' &&
            debug[1] == '\0'
        );

    if (debug_enabled) {
      std::fprintf(
          stderr,
          "[AIT CPU] requested_threads=%zu "
          "actual_threads=%zu\n",
          requested_threads_,
          threads());
    }
  }

  ~CpuThreadPoolHolder() {
    if (pool_ != nullptr) {
      pthreadpool_destroy(
          pool_);
    }
  }

  CpuThreadPoolHolder(
      const CpuThreadPoolHolder&) =
      delete;

  CpuThreadPoolHolder& operator=(
      const CpuThreadPoolHolder&) =
      delete;

  pthreadpool_t get() const {
    return pool_;
  }

  size_t threads() const {
    if (pool_ == nullptr) {
      return 1;
    }

    return
        pthreadpool_get_threads_count(
            pool_);
  }

 private:
  pthreadpool_t pool_ = nullptr;
  size_t requested_threads_ = 1;
};


inline CpuThreadPoolHolder&
cpu_threadpool_holder() {
  static CpuThreadPoolHolder holder;
  return holder;
}


inline pthreadpool_t cpu_threadpool() {
  return
      cpu_threadpool_holder().get();
}


inline size_t cpu_threadpool_num_threads() {
  return
      cpu_threadpool_holder().threads();
}


inline void parallelize_1d(
    pthreadpool_task_1d_t task,
    void* context,
    size_t range) {

  if (range == 0) {
    return;
  }

  pthreadpool_t pool =
      cpu_threadpool();

  if (
      pool == nullptr ||
      range == 1) {

    for (
        size_t i = 0;
        i < range;
        ++i) {
      task(
          context,
          i);
    }

    return;
  }

  pthreadpool_parallelize_1d(
      pool,
      task,
      context,
      range,
      0);
}


}  // namespace ait
