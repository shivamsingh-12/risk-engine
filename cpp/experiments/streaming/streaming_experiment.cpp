//Experiment 27 (Phase 22): streaming risk. Run from the repo root:
//    build/experiments/streaming_experiment
//Writes cpp/experiments/streaming/results/
//  Part 1  cost of one tick: incremental update vs full recomputation, as the portfolio grows
//  Part 2  rounding drift of the incremental updates over 10 million ticks
//  Part 3  lock-free SPSC queue vs mutex queue: throughput, and latency under load with real work per tick
//  Part 4  the whole qre_stream pipeline: latency and CPU at different tick rates, busy-poll vs sleep-when-idle
#include <sys/resource.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include <Eigen/Dense>
#include "qre/data/csv_loader.hpp"
#include "qre/stream/incremental.hpp"
#include "qre/stream/latency.hpp"
#include "qre/stream/pipeline.hpp"
#include "qre/stream/queue.hpp"

using namespace qre;
using namespace qre::stream;

namespace{
const std::string kOut = "cpp/experiments/streaming/results/";

struct Book{ Eigen::VectorXd q, p; Eigen::MatrixXd cov, r; };

//N assets, T scenarios: returns from a 3-factor model, so the covariance looks like a real equity book
Book make_book(int n, int t, std::uint64_t seed){
    std::mt19937_64 g(seed);
    std::normal_distribution<double> nd;
    Eigen::MatrixXd f(t, 3), load(3, n);
    for(int i = 0; i < t; ++i) for(int k = 0; k < 3; ++k) f(i, k) = 0.01 * nd(g) / (k + 1);
    for(int k = 0; k < 3; ++k) for(int j = 0; j < n; ++j) load(k, j) = (k == 0 ? 1.0 : 0.0) + 0.4 * nd(g);
    Book b;
    b.r = f * load;
    for(int i = 0; i < t; ++i) for(int j = 0; j < n; ++j) b.r(i, j) += 0.012 * nd(g);
    b.cov = linalg::sample_cov(b.r);
    b.p.resize(n); b.q.resize(n);
    std::uniform_real_distribution<double> u(20.0, 400.0);
    for(int j = 0; j < n; ++j){ b.p(j) = u(g); b.q(j) = std::round(10000.0 / b.p(j)) * (j % 7 == 6 ? -1.0 : 1.0); }
    return b;
}

double cpu_seconds(){
    rusage u{};
    getrusage(RUSAGE_SELF, &u);
    return static_cast<double>(u.ru_utime.tv_sec + u.ru_stime.tv_sec) + 1e-6 * static_cast<double>(u.ru_utime.tv_usec + u.ru_stime.tv_usec);
}

// ------------------------------------------------------------------------------------------------ Part 1
void part1(){
    std::ofstream f(kOut + "tick_cost.csv");
    f << "assets,scenarios,update_ns,update_plus_snapshot_ns,full_recompute_ns,speedup_update,speedup_with_snapshot\n";
    std::printf("Part 1: nanoseconds per tick (mean over many ticks)\n%7s %9s %10s %13s %14s %10s %10s\n", "assets", "scenarios",
                "update", "+ snapshot", "full recompute", "speed-up", "w/ snap");
    for(int t : {250, 1000}){
        for(int n : {10, 50, 100, 250, 500, 1000}){
            const Book b = make_book(n, t, 11);
            IncrementalRisk risk(b.q, b.p, b.cov, b.r, 0.99);
            std::mt19937_64 g(3);
            std::uniform_int_distribution<int> pick(0, n - 1);
            std::normal_distribution<double> nd;
            std::vector<std::pair<int, double>> ticks(20000);
            Eigen::VectorXd p = b.p;
            for(auto& tk : ticks){ tk.first = pick(g); p(tk.first) *= std::exp(0.001 * nd(g)); tk.second = p(tk.first); }
            auto time_it = [&](auto&& fn, std::size_t reps){
                const auto t0 = std::chrono::steady_clock::now();
                double sink = 0.0;
                for(std::size_t k = 0; k < reps; ++k) sink += fn(ticks[k % ticks.size()]);
                const double ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count();
                if(sink == 42.0) std::printf(" ");
                return ns / static_cast<double>(reps);
            };
            const std::size_t reps = n <= 100 ? 400000 : 100000;
            const double upd = time_it([&](auto tk){ risk.on_tick(tk.first, tk.second); return 0.0; }, reps);
            const double snap = time_it([&](auto tk){ risk.on_tick(tk.first, tk.second); return risk.snapshot().var_historical; }, reps / 4);
            Eigen::VectorXd pp = b.p;
            std::vector<double> scratch;
            const double v0 = b.q.dot(b.p);
            const std::size_t full_reps = std::max<std::size_t>(200, static_cast<std::size_t>(4e8 / (static_cast<double>(n) * n + static_cast<double>(n) * t)));
            const double full = time_it([&](auto tk){ pp(tk.first) = tk.second; return full_recompute(b.q, pp, b.cov, b.r, 0.99, v0, scratch).var_historical; }, full_reps);
            f << n << "," << t << "," << upd << "," << snap << "," << full << "," << full / upd << "," << full / snap << "\n";
            std::printf("%7d %9d %9.0f %12.0f %14.0f %9.0fx %9.1fx\n", n, t, upd, snap, full, full / upd, full / snap);
        }
    }
}

// ------------------------------------------------------------------------------------------------ Part 2
void part2(){
    std::ofstream f(kOut + "drift.csv");
    f << "ticks,variance_rel_error,exposure_rel_error,scenario_loss_rel_error,var_historical_rel_error\n";
    std::printf("\nPart 2: drift of the running values from a full recomputation, 100 assets, no resync\n%12s %14s %14s %14s %14s\n",
                "ticks", "variance", "exposure", "scenario loss", "hist. VaR");
    const Book b = make_book(100, 250, 5);
    IncrementalRisk risk(b.q, b.p, b.cov, b.r, 0.99);
    std::mt19937_64 g(9);
    std::uniform_int_distribution<int> pick(0, 99);
    std::normal_distribution<double> nd;
    Eigen::VectorXd p = b.p;
    std::vector<double> scratch;
    long long done = 0;
    for(long long checkpoint : {1000LL, 10000LL, 100000LL, 1000000LL, 3000000LL, 10000000LL}){
        for(; done < checkpoint; ++done){
            const int i = pick(g);
            p(i) *= std::exp(0.002 * nd(g) - 0.000002);     //prices wander; exposures change sign never
            risk.on_tick(i, p(i));
        }
        const auto d = risk.drift();
        const double vh = risk.snapshot().var_historical;
        const double vf = full_recompute(b.q, p, b.cov, b.r, 0.99, 0.0, scratch).var_historical;
        f << checkpoint << "," << d.variance << "," << d.exposure << "," << d.scenario_loss << "," << std::abs(vh - vf) / vf << "\n";
        std::printf("%12lld %14.2e %14.2e %14.2e %14.2e\n", checkpoint, d.variance, d.exposure, d.scenario_loss, std::abs(vh - vf) / vf);
    }
    std::printf("  (prices moved by a factor of up to %.1f from their start)\n", (p.array() / b.p.array()).maxCoeff());
}

// ------------------------------------------------------------------------------------------------ Part 3
struct Msg{ std::uint32_t asset; double price; std::int64_t t_ns; };

template<typename Q>
double max_throughput(std::size_t n){
    Q q(1 << 14);
    std::atomic<bool> go{false};
    std::thread c([&]{
        Msg m{};
        std::size_t got = 0;
        while(!go) {}
        while(got < n) if(q.try_pop(m)) ++got;
    });
    const auto t0 = std::chrono::steady_clock::now();
    go = true;
    for(std::size_t i = 0; i < n; ++i) while(!q.try_push(Msg{0, 1.0, 0})) {}
    c.join();
    return static_cast<double>(n) / std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

//producer paced at `rate` messages per second (busy-wait pacing), consumer applies each tick to a 100-asset book
template<typename Q>
LatencyHistogram loaded_latency(double rate, double seconds, const Book& b){
    Q q(1 << 16);
    LatencyHistogram h;
    const auto n = static_cast<std::size_t>(rate * seconds);
    std::atomic<bool> done{false};
    std::thread c([&]{
        IncrementalRisk risk(b.q, b.p, b.cov, b.r, 0.99);
        Msg m{};
        std::size_t got = 0;
        while(got < n){
            if(q.try_pop(m)){
                risk.on_tick(m.asset, m.price);
                h.record(now_ns() - m.t_ns);
                ++got;
            }
        }
        done = true;
    });
    const std::int64_t t0 = now_ns();
    const double gap = 1e9 / rate;
    std::mt19937_64 g(1);
    const int na = static_cast<int>(b.p.size());
    for(std::size_t i = 0; i < n; ++i){
        const auto due = t0 + static_cast<std::int64_t>(static_cast<double>(i) * gap);
        while(now_ns() < due) {}
        const auto a = static_cast<std::uint32_t>(g() % static_cast<std::uint64_t>(na));
        Msg m{a, b.p(a) * (1.0 + 1e-4 * static_cast<double>(i % 7)), now_ns()};
        while(!q.try_push(m)) {}
    }
    c.join();
    return h;
}

void part3(){
    std::ofstream f(kOut + "queues.csv");
    f << "queue,test,rate,messages_per_sec,p50_ns,p99_ns,p999_ns,max_ns,mean_ns\n";
    std::printf("\nPart 3: queues between two threads\n");
    for(int rep = 0; rep < 3; ++rep){
        const double s = max_throughput<SpscQueue<Msg>>(20'000'000), m = max_throughput<MutexQueue<Msg>>(5'000'000);
        f << "spsc,max_throughput,," << s << ",,,,,\n" << "mutex,max_throughput,," << m << ",,,,,\n";
        std::printf("  run %d, maximum throughput: SPSC %.1f million/s, mutex %.1f million/s (%.0fx)\n", rep + 1, s / 1e6, m / 1e6, s / m);
    }
    const Book b = make_book(100, 250, 2);
    std::printf("  latency, producer paced, consumer updates a 100-asset book per tick (ns; median of 3 runs by p99):\n  %-6s %10s %8s %8s %9s %9s %9s\n",
                "queue", "rate/s", "p50", "p99", "p99.9", "max", "mean");
    for(double rate : {10000.0, 100000.0, 1000000.0}){
        const double secs = rate >= 1e6 ? 2.0 : 3.0;
        for(int which = 0; which < 2; ++which){
            //three runs; report the one with the median p99 (tails on a shared machine vary from run to run)
            std::vector<LatencyHistogram> runs;
            for(int k = 0; k < 3; ++k)
                runs.push_back(which == 0 ? loaded_latency<SpscQueue<Msg>>(rate, secs, b) : loaded_latency<MutexQueue<Msg>>(rate, secs, b));
            std::sort(runs.begin(), runs.end(), [](const auto& x, const auto& y){ return x.percentile(99) < y.percentile(99); });
            const LatencyHistogram& h = runs[1];
            const char* name = which == 0 ? "spsc" : "mutex";
            f << name << ",latency," << rate << ",," << h.percentile(50) << "," << h.percentile(99) << "," << h.percentile(99.9) << "," << h.max() << "," << h.mean() << "\n";
            std::printf("  %-6s %10.0f %8.0f %8.0f %9.0f %9llu %9.0f\n", name, rate, h.percentile(50), h.percentile(99), h.percentile(99.9),
                        static_cast<unsigned long long>(h.max()), h.mean());
        }
    }
}

// ------------------------------------------------------------------------------------------------ Part 4
void part4(){
    std::ofstream f(kOut + "pipeline.csv");
    f << "mode,target_rate,achieved_rate,cpu_cores,p50_ns,p99_ns,p999_ns,max_ns\n";
    const auto table = data::load_price_csv("data/market_data/adjusted_close.csv");
    std::vector<Holding> h;
    for(const auto& t : table.tickers) h.push_back({t, 1.0});
    const PortfolioSetup setup = make_setup(table, h, 250, true, 1'000'000.0);
    std::printf("\nPart 4: qre_stream pipeline (6 holdings, synthetic feed), 4 seconds per setting\n  %-10s %10s %12s %6s %8s %8s %9s %9s\n",
                "mode", "target/s", "achieved/s", "cores", "p50", "p99", "p99.9", "max");
    for(bool busy : {false, true}){
        for(double rate : {1000.0, 10000.0, 100000.0, 0.0}){
            PipelineOptions o;
            o.rate = rate;
            o.busy_poll = busy;
            o.publish_hz = 10.0;
            Pipeline p(setup, o);
            const double c0 = cpu_seconds();
            const auto t0 = std::chrono::steady_clock::now();
            p.start();
            std::this_thread::sleep_for(std::chrono::seconds(4));
            p.stop();
            const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            const double cores = (cpu_seconds() - c0) / wall;
            const std::string j = p.json();
            auto num = [&](const std::string& key, std::size_t from = 0){
                const auto k = j.find("\"" + key + "\":", from);
                return std::stod(j.substr(k + key.size() + 3));
            };
            const auto all = j.find("\"all\":");
            const double ticks = num("ticks");
            const double p50 = num("p50", all), p99 = num("p99", all), p999 = num("p999", all), mx = num("max", all);
            const char* mode = busy ? "busy-poll" : "sleep-idle";
            f << mode << "," << rate << "," << ticks / wall << "," << cores << "," << p50 << "," << p99 << "," << p999 << "," << mx << "\n";
            std::printf("  %-10s %10s %12.0f %6.2f %8.0f %8.0f %9.0f %9.0f\n", mode, rate > 0 ? std::to_string(static_cast<int>(rate)).c_str() : "max",
                        ticks / wall, cores, p50, p99, p999, mx);
        }
    }
}
}

int main(){
    std::filesystem::create_directories(kOut);
    std::printf("Hardware threads: %u\n\n", std::thread::hardware_concurrency());
    const auto t0 = std::chrono::steady_clock::now();
    part1();
    part2();
    part3();
    try{ part4(); } catch(const std::exception& e){ std::printf("Part 4 skipped: %s (run from the repo root)\n", e.what()); }
    std::printf("\n%.1f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    return 0;
}
