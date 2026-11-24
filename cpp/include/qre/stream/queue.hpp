#pragma once
#include <atomic>
#include <cstddef>
#include <deque>
#include <mutex>
#include <new>
#include <optional>
#include <vector>
#include "qre/core/exceptions.hpp"

//Phase 22: queues between the feed thread and the risk thread
//  SpscQueue   lock-free ring buffer for exactly one producer thread and one consumer thread
//  MutexQueue  std::mutex + std::deque, the baseline it is measured against (Experiment 27)
//
//SpscQueue design: capacity is a power of two, so an index wraps with a mask. head_ (next slot to read) is written
//only by the consumer and tail_ (next slot to write) only by the producer, so no locks or compare-and-swap loops
//are needed: each side publishes its index with a release store and reads the other's with an acquire load, which
//makes the slot written before the store visible to the other thread after its load. The two indices sit on
//separate cache lines (no false sharing), and each side keeps a private copy of the other's index and re-reads the
//shared one only when the copy says the queue is full (producer) or empty (consumer), which keeps cache traffic low.
namespace qre::stream{
    inline constexpr std::size_t kCacheLine = 64;

    template<typename T>
    class SpscQueue{
    public:
        explicit SpscQueue(std::size_t capacity){
            if(capacity < 2 || (capacity & (capacity - 1)) != 0)
                throw InvalidInputError("SpscQueue capacity must be a power of two, at least 2");
            buf_.resize(capacity);
            mask_ = capacity - 1;
        }
        SpscQueue(const SpscQueue&) = delete;
        SpscQueue& operator=(const SpscQueue&) = delete;

        //producer only; false when full
        bool try_push(const T& v){
            const std::size_t t = tail_.load(std::memory_order_relaxed);
            if(t - head_cache_ == buf_.size()){
                head_cache_ = head_.load(std::memory_order_acquire);
                if(t - head_cache_ == buf_.size()) return false;
            }
            buf_[t & mask_] = v;
            tail_.store(t + 1, std::memory_order_release);
            return true;
        }

        //consumer only; false when empty
        bool try_pop(T& out){
            const std::size_t h = head_.load(std::memory_order_relaxed);
            if(h == tail_cache_){
                tail_cache_ = tail_.load(std::memory_order_acquire);
                if(h == tail_cache_) return false;
            }
            out = buf_[h & mask_];
            head_.store(h + 1, std::memory_order_release);
            return true;
        }

        std::size_t capacity() const{ return buf_.size(); }
        //approximate when called while both threads run
        std::size_t size() const{ return tail_.load(std::memory_order_acquire) - head_.load(std::memory_order_acquire); }

    private:
        std::vector<T> buf_;
        std::size_t mask_ = 0;
        alignas(kCacheLine) std::atomic<std::size_t> head_{0};
        alignas(kCacheLine) std::size_t tail_cache_ = 0;            //consumer's copy of tail_
        alignas(kCacheLine) std::atomic<std::size_t> tail_{0};
        alignas(kCacheLine) std::size_t head_cache_ = 0;            //producer's copy of head_
    };

    template<typename T>
    class MutexQueue{
    public:
        explicit MutexQueue(std::size_t capacity) : cap_(capacity){
            if(capacity < 1) throw InvalidInputError("MutexQueue capacity must be at least 1");
        }
        bool try_push(const T& v){
            std::lock_guard<std::mutex> g(m_);
            if(q_.size() == cap_) return false;
            q_.push_back(v);
            return true;
        }
        bool try_pop(T& out){
            std::lock_guard<std::mutex> g(m_);
            if(q_.empty()) return false;
            out = q_.front();
            q_.pop_front();
            return true;
        }
        std::size_t capacity() const{ return cap_; }
    private:
        std::size_t cap_;
        std::mutex m_;
        std::deque<T> q_;
    };
}
