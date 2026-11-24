#pragma once
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <istream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <Eigen/Dense>
#include "qre/core/exceptions.hpp"
#include "qre/data/csv_loader.hpp"
#include "qre/linalg/covariance.hpp"
#include "qre/stream/feed.hpp"
#include "qre/stream/incremental.hpp"
#include "qre/stream/latency.hpp"
#include "qre/stream/queue.hpp"

//Phase 22: the streaming risk pipeline behind qre_stream
//
//    feed thread --Tick--> SpscQueue --> risk thread --snapshot (mutex, a few times a second)--> server thread
//
//The feed thread stamps each tick with the time it is created and pushes it into the lock-free queue. The risk thread
//pops ticks, applies each to IncrementalRisk in O(N + T), and records tick-to-update latency in a histogram. Every
//publish interval it builds a snapshot (VaR by selection, latency percentiles) and swaps it in under a mutex; the
//server thread only ever touches that snapshot, so slow clients can never stall the risk thread.
namespace qre::stream{
    struct Holding{ std::string ticker; double amount; };     //shares, or a weight when building from weights

    //"AAPL=100,JPM=50" -> holdings
    inline std::vector<Holding> parse_holdings(const std::string& s){
        std::vector<Holding> out;
        std::stringstream ss(s);
        std::string item;
        while(std::getline(ss, item, ',')){
            const auto eq = item.find('=');
            if(eq == std::string::npos || eq == 0) throw InvalidInputError("holdings look like AAPL=100,JPM=50 (got '" + item + "')");
            double v;
            try{ v = std::stod(item.substr(eq + 1)); } catch(const std::exception&){ throw InvalidInputError("bad amount in '" + item + "'"); }
            if(!std::isfinite(v)) throw InvalidInputError("bad amount in '" + item + "'");
            out.push_back({item.substr(0, eq), v});
        }
        if(out.empty()) throw InvalidInputError("no holdings given");
        return out;
    }

    struct PortfolioSetup{
        std::vector<std::string> tickers;
        Eigen::VectorXd shares, prices;
        Eigen::MatrixXd cov, scenarios;
        std::string first_date, last_date;
    };

    //From a price table: the last `window` daily returns on which every holding trades, their covariance, and the
    //last prices. weights=true reads the amounts as weights of `value` (negative = short) and converts to shares.
    inline PortfolioSetup make_setup(const data::PriceTable& t, const std::vector<Holding>& h, int window, bool weights, double value){
        PortfolioSetup s;
        std::vector<Eigen::Index> cols;
        for(const auto& x : h){
            const auto it = std::find(t.tickers.begin(), t.tickers.end(), x.ticker);
            if(it == t.tickers.end()) throw InvalidInputError(x.ticker + " is not in the price file");
            if(std::find(s.tickers.begin(), s.tickers.end(), x.ticker) != s.tickers.end()) throw InvalidInputError(x.ticker + " appears twice");
            cols.push_back(static_cast<Eigen::Index>(it - t.tickers.begin()));
            s.tickers.push_back(x.ticker);
        }
        std::vector<Eigen::Index> rows;
        for(Eigen::Index r = 0; r < t.prices.rows(); ++r){
            bool ok = true;
            for(auto c : cols) ok = ok && std::isfinite(t.prices(r, c)) && t.prices(r, c) > 0.0;
            if(ok) rows.push_back(r);
        }
        if(static_cast<int>(rows.size()) < window + 1) throw InvalidInputError("need " + std::to_string(window + 1) + " days on which every holding has a price");
        rows.erase(rows.begin(), rows.end() - (window + 1));
        const auto n = static_cast<Eigen::Index>(cols.size());
        s.scenarios.resize(window, n);
        for(int k = 0; k < window; ++k)
            for(Eigen::Index j = 0; j < n; ++j)
                s.scenarios(k, j) = t.prices(rows[static_cast<std::size_t>(k) + 1], cols[static_cast<std::size_t>(j)]) /
                                    t.prices(rows[static_cast<std::size_t>(k)], cols[static_cast<std::size_t>(j)]) - 1.0;
        s.cov = linalg::sample_cov(s.scenarios);
        s.prices.resize(n);
        s.shares.resize(n);
        double wsum = 0.0;
        for(const auto& x : h) wsum += x.amount;
        if(weights && !(wsum > 0.0)) throw InvalidInputError("weights must have a positive sum");
        for(Eigen::Index j = 0; j < n; ++j){
            s.prices(j) = t.prices(rows.back(), cols[static_cast<std::size_t>(j)]);
            const double a = h[static_cast<std::size_t>(j)].amount;
            s.shares(j) = weights ? a / wsum * value / s.prices(j) : a;
        }
        s.first_date = t.dates[static_cast<std::size_t>(rows.front())];
        s.last_date = t.dates[static_cast<std::size_t>(rows.back())];
        return s;
    }

    struct PipelineOptions{
        std::string source = "synthetic";       //synthetic | lines (an input stream of TICKER,PRICE lines)
        double rate = 1000.0;                   //synthetic ticks per second; 0 = as fast as possible
        double speed = 0.02;                    //synthetic: simulated trading days per second of wall time
        double confidence = 0.99;
        double publish_hz = 10.0;
        long long resync_every = 100000;        //ticks between full recomputations
        std::size_t queue_capacity = 1 << 16;
        std::uint64_t seed = 1;
        bool busy_poll = false;                 //risk thread spins on an empty queue: lowest latency, one core at 100%
    };

    class Pipeline{
    public:
        Pipeline(PortfolioSetup setup, PipelineOptions opt, std::istream* lines = nullptr)
            : s_(std::move(setup)), o_(opt), lines_(lines), q_(opt.queue_capacity),
              risk_(s_.shares, s_.prices, s_.cov, s_.scenarios, opt.confidence){
            if(o_.source != "synthetic" && o_.source != "lines") throw InvalidInputError("source must be synthetic or lines");
            if(o_.source == "lines" && !lines_) throw InvalidInputError("the lines source needs an input stream");
            if(!(o_.publish_hz > 0.0 && o_.publish_hz <= 1000.0)) throw InvalidInputError("publish rate must be in (0, 1000] per second");
            for(std::size_t i = 0; i < s_.tickers.size(); ++i) index_[s_.tickers[i]] = static_cast<std::uint32_t>(i);
            publish(risk_.snapshot(), 0.0);
        }
        ~Pipeline(){ stop(); }

        void start(){
            stop_ = false;
            started_ns_ = now_ns();
            risk_thread_ = std::thread([this]{ risk_loop(); });
            feed_thread_ = std::thread([this]{ feed_loop(); });
        }

        void stop(){
            stop_ = true;
            if(feed_thread_.joinable()) feed_thread_.join();
            if(risk_thread_.joinable()) risk_thread_.join();
        }

        //true once a finite feed (the lines source at end of input) is exhausted and every tick processed
        bool finished() const{ return finished_; }

        std::string json() const{
            std::lock_guard<std::mutex> g(m_);
            return json_;
        }

        const PortfolioSetup& setup() const{ return s_; }

    private:
        void feed_loop(){
            std::uint64_t seq = 0;
            auto push = [&](Tick t){
                t.seq = seq++;
                t.t_ns = now_ns();
                while(!q_.try_push(t)){
                    ++full_;
                    if(stop_) return false;
                    std::this_thread::yield();
                }
                return true;
            };
            if(o_.source == "synthetic"){
                const double ticks_per_day = o_.rate > 0.0 ? o_.rate / o_.speed : 1e6;
                SyntheticFeed feed(s_.prices, s_.cov, static_cast<double>(s_.tickers.size()) / ticks_per_day, o_.seed);
                const std::int64_t t0 = now_ns();
                while(!stop_){
                    if(o_.rate > 0.0){
                        //pace in small batches: emit what the schedule says is due, then sleep briefly
                        const auto due = static_cast<std::uint64_t>(static_cast<double>(now_ns() - t0) * 1e-9 * o_.rate);
                        if(seq >= due){ std::this_thread::sleep_for(std::chrono::microseconds(200)); continue; }
                    }
                    if(!push(feed.next())) return;
                }
            } else {
                std::string line;
                while(!stop_ && std::getline(*lines_, line)){
                    const auto c = line.find(',');
                    if(c == std::string::npos) continue;
                    const auto it = index_.find(line.substr(0, c));
                    if(it == index_.end()) continue;
                    double p;
                    try{ p = std::stod(line.substr(c + 1)); } catch(const std::exception&){ continue; }
                    if(!(p > 0.0) || !std::isfinite(p)) continue;
                    if(!push(Tick{it->second, p, 0, 0})) return;
                }
                feed_done_ = true;
            }
        }

        void risk_loop(){
            LatencyHistogram recent;
            const auto every = static_cast<std::int64_t>(1e9 / o_.publish_hz);
            std::int64_t next_pub = now_ns() + every, last_pub = now_ns();
            long long last_ticks = 0;
            int idle = 0;
            Tick t;
            for(;;){
                if(q_.try_pop(t)){
                    idle = 0;
                    risk_.on_tick(t.asset, t.price);
                    const std::int64_t lat = now_ns() - t.t_ns;
                    hist_.record(lat);
                    recent.record(lat);
                    if(o_.resync_every > 0 && risk_.ticks() % o_.resync_every == 0) risk_.resync();
                } else {
                    if(stop_ || (feed_done_ && q_.size() == 0)) break;
                    //briefly spin, then sleep: low latency when busy, little CPU when idle (or always spin)
                    if(o_.busy_poll || ++idle < 2000) continue;
                    std::this_thread::sleep_for(std::chrono::microseconds(50));
                }
                const std::int64_t now = now_ns();
                if(now >= next_pub){
                    const double secs = static_cast<double>(now - last_pub) * 1e-9;
                    const double tps = static_cast<double>(risk_.ticks() - last_ticks) / std::max(secs, 1e-9);
                    publish(risk_.snapshot(), tps, &recent);
                    recent.reset();
                    last_ticks = risk_.ticks();
                    last_pub = now;
                    next_pub = now + every;
                }
            }
            publish(risk_.snapshot(), 0.0, &recent);
            finished_ = true;
        }

        void publish(const RiskSnapshot& r, double tps, const LatencyHistogram* recent = nullptr){
            std::string j;
            j.reserve(512 + 160 * s_.tickers.size());
            char b[256];
            std::snprintf(b, sizeof b, "{\"type\":\"snapshot\",\"source\":\"%s\",\"ticks\":%lld,\"ticks_per_sec\":%.1f,\"confidence\":%.4f,",
                          o_.source.c_str(), r.ticks, tps, o_.confidence);
            j += b;
            std::snprintf(b, sizeof b, "\"value\":%.6f,\"pnl\":%.6f,\"sigma\":%.6f,\"var_parametric\":%.6f,\"es_parametric\":%.6f,",
                          r.value, r.pnl, r.sigma, r.var_parametric, r.es_parametric);
            j += b;
            std::snprintf(b, sizeof b, "\"var_historical\":%.6f,\"es_historical\":%.6f,\"uptime_s\":%.3f,\"queue_full_events\":%llu,",
                          r.var_historical, r.es_historical, started_ns_ ? static_cast<double>(now_ns() - started_ns_) * 1e-9 : 0.0,
                          static_cast<unsigned long long>(full_.load()));
            j += b;
            auto lat = [&](const char* name, const LatencyHistogram& h){
                std::snprintf(b, sizeof b, "\"%s\":{\"count\":%llu,\"p50\":%.0f,\"p99\":%.0f,\"p999\":%.0f,\"max\":%llu,\"mean\":%.1f}",
                              name, static_cast<unsigned long long>(h.count()), h.percentile(50), h.percentile(99), h.percentile(99.9),
                              static_cast<unsigned long long>(h.max()), h.mean());
                j += b;
            };
            j += "\"latency_ns\":{";
            lat("all", hist_);
            j += ",";
            lat("recent", recent ? *recent : LatencyHistogram{});
            j += "},\"window\":{\"days\":" + std::to_string(s_.scenarios.rows()) + ",\"from\":\"" + s_.first_date + "\",\"to\":\"" + s_.last_date + "\"},";
            j += "\"assets\":[";
            for(std::size_t i = 0; i < s_.tickers.size(); ++i){
                const auto k = static_cast<Eigen::Index>(i);
                std::snprintf(b, sizeof b, "%s{\"ticker\":\"%s\",\"price\":%.6f,\"shares\":%.6f,\"exposure\":%.6f,\"component_var\":%.6f}",
                              i ? "," : "", s_.tickers[i].c_str(), risk_.price(k), s_.shares(k), s_.shares(k) * risk_.price(k), r.component_var(k));
                j += b;
            }
            j += "]}";
            std::lock_guard<std::mutex> g(m_);
            json_ = std::move(j);
        }

        PortfolioSetup s_;
        PipelineOptions o_;
        std::istream* lines_;
        SpscQueue<Tick> q_;
        IncrementalRisk risk_;
        std::unordered_map<std::string, std::uint32_t> index_;
        LatencyHistogram hist_;
        std::atomic<bool> stop_{false}, feed_done_{false}, finished_{false};
        std::atomic<std::uint64_t> full_{0};
        std::int64_t started_ns_ = 0;
        std::thread feed_thread_, risk_thread_;
        mutable std::mutex m_;
        std::string json_;
    };
}
