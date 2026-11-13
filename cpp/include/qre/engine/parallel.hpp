#pragma once
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>
//Phase 14: a minimal parallel loop for independent tasks
//Workers take the next task index from a shared atomic counter (dynamic scheduling), so one slow task (a GARCH-t fit)
//does not hold up a whole block. Each task writes only to its own output slot, so the results are identical for any
//number of threads and any scheduling order. The first exception thrown by a task is rethrown on the calling thread.
namespace qre::engine{
    inline std::size_t resolve_threads(std::size_t requested){
        if(requested > 0) return requested;
        const unsigned hw = std::thread::hardware_concurrency();
        return hw > 0 ? hw : 1;
    }

    template<class F>
    void parallel_for(std::size_t n_tasks, std::size_t n_threads, F&& task){
        n_threads = std::min(resolve_threads(n_threads), std::max<std::size_t>(n_tasks, 1));
        if(n_threads <= 1){
            for(std::size_t i = 0; i < n_tasks; ++i) task(i);
            return;
        }
        std::atomic<std::size_t> next{0};
        std::atomic<bool> failed{false};
        std::exception_ptr error;
        std::mutex error_mutex;
        auto worker = [&]{
            for(;;){
                const std::size_t i = next.fetch_add(1, std::memory_order_relaxed);
                if(i >= n_tasks || failed.load(std::memory_order_relaxed)) return;
                try{ task(i); }
                catch(...){
                    std::lock_guard<std::mutex> lock(error_mutex);
                    if(!error) error = std::current_exception();
                    failed.store(true, std::memory_order_relaxed);
                }
            }
        };
        {
            std::vector<std::jthread> pool;
            pool.reserve(n_threads - 1);
            for(std::size_t k = 1; k < n_threads; ++k) pool.emplace_back(worker);
            worker();
        }   //jthreads join here
        if(error) std::rethrow_exception(error);
    }
}
