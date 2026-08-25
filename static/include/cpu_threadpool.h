#pragma once

#include <cstddef>
#include <cstdlib>

#include <pthreadpool.h>

namespace ait {

class CPUThreadPoolHolder {
 public:
  CPUThreadPoolHolder() {
    const char* env = std::getenv("AIT_CPU_THREADS");
    long threads = 1;
    if (env != nullptr) {
      char* end = nullptr;
      const long parsed = std::strtol(env, &end, 10);
      if (end != env && parsed > 0) {
        threads = parsed;
      }
    }
    if (threads > 1) {
      pool_ = pthreadpool_create(static_cast<size_t>(threads));
    }
  }

  ~CPUThreadPoolHolder() {
    if (pool_ != nullptr) {
      pthreadpool_destroy(pool_);
    }
  }

  CPUThreadPoolHolder(const CPUThreadPoolHolder&) = delete;
  CPUThreadPoolHolder& operator=(const CPUThreadPoolHolder&) = delete;

  pthreadpool_t get() const {
    return pool_;
  }

 private:
  pthreadpool_t pool_ = nullptr;
};

inline pthreadpool_t cpu_threadpool() {
  static CPUThreadPoolHolder holder;
  return holder.get();
}

using CPUParallelTask1D = void (*)(void*, size_t);

inline void parallelize_1d(
    CPUParallelTask1D task,
    void* context,
    size_t range) {
  if (range == 0) {
    return;
  }

  pthreadpool_t pool = cpu_threadpool();

  if (pool == nullptr || range == 1) {
    for (size_t i = 0; i < range; ++i) {
      task(context, i);
    }
    return;
  }

  pthreadpool_parallelize_1d(
      pool,
      task,
      context,
      range,
      0 /* flags */);
}

}  // namespace ait
