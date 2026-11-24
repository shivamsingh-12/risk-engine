#pragma once
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <utility>
#include <cstdint>
#include <limits>

//Phase 22: latency histogram. Records nanosecond durations in O(1) with fixed memory, so it can sit in a hot loop.
//Buckets are log-linear (as in HdrHistogram): each power of two is split into 32 equal sub-buckets, so a reported
//percentile is within 1/32 (about 3%) of the true value at any scale from 1 ns to about 2^40 ns (18 minutes).
namespace qre::stream{
    inline std::int64_t now_ns(){
        return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    class LatencyHistogram{
    public:
        static constexpr int kSubBits = 5;                       //32 sub-buckets per power of two
        static constexpr int kSub = 1 << kSubBits;
        static constexpr int kTop = 41;                          //powers of two covered
        static constexpr int kBuckets = kTop * kSub;

        void record(std::int64_t ns){
            const auto v = static_cast<std::uint64_t>(std::max<std::int64_t>(ns, 0));
            ++counts_[static_cast<std::size_t>(index(v))];
            ++n_;
            sum_ += static_cast<double>(v);
            max_ = std::max(max_, v);
            min_ = std::min(min_, v);
        }

        void merge(const LatencyHistogram& o){
            for(int i = 0; i < kBuckets; ++i) counts_[static_cast<std::size_t>(i)] += o.counts_[static_cast<std::size_t>(i)];
            n_ += o.n_; sum_ += o.sum_; max_ = std::max(max_, o.max_); min_ = std::min(min_, o.min_);
        }

        void reset(){ *this = LatencyHistogram{}; }

        std::uint64_t count() const{ return n_; }
        double mean() const{ return n_ ? sum_ / static_cast<double>(n_) : 0.0; }
        std::uint64_t max() const{ return n_ ? max_ : 0; }
        std::uint64_t min() const{ return n_ ? min_ : 0; }

        //value at percentile p in [0, 100]: the midpoint of the bucket that holds it (exact for values below 32)
        double percentile(double p) const{
            if(n_ == 0) return 0.0;
            const auto rank = static_cast<std::uint64_t>(std::max(1.0, std::ceil(p / 100.0 * static_cast<double>(n_))));
            std::uint64_t seen = 0;
            for(int i = 0; i < kBuckets; ++i){
                seen += counts_[static_cast<std::size_t>(i)];
                if(seen >= rank){
                    const auto [lo, hi] = bounds(i);
                    return std::min(static_cast<double>(max_), 0.5 * (static_cast<double>(lo) + static_cast<double>(hi)));
                }
            }
            return static_cast<double>(max_);
        }

        //bucket of a value: values below 32 get their own bucket; above, the power of two and the top 5 bits below it
        static int index(std::uint64_t v){
            if(v < static_cast<std::uint64_t>(kSub)) return static_cast<int>(v);
            const int e = std::bit_width(v) - 1;                 //v in [2^e, 2^(e+1))
            const int shift = e - kSubBits;
            const int sub = static_cast<int>((v >> shift) & (kSub - 1));
            const int i = (shift + 1) * kSub + sub;
            return std::min(i, kBuckets - 1);
        }

        //inclusive value range of bucket i
        static std::pair<std::uint64_t, std::uint64_t> bounds(int i){
            if(i < kSub) return {static_cast<std::uint64_t>(i), static_cast<std::uint64_t>(i)};
            const int shift = i / kSub - 1;
            const std::uint64_t lo = (static_cast<std::uint64_t>(kSub + i % kSub)) << shift;
            return {lo, lo + (std::uint64_t{1} << shift) - 1};
        }

    private:
        std::array<std::uint64_t, kBuckets> counts_{};
        std::uint64_t n_ = 0;
        double sum_ = 0.0;
        std::uint64_t max_ = 0;
        std::uint64_t min_ = std::numeric_limits<std::uint64_t>::max();
    };
}
