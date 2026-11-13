//Phase 17 experiment: can a neural network learn to hedge better than Black-Scholes when Black-Scholes' assumptions
//fail? Short one 30-day at-the-money call (S = K = 100, zero rates), 30 daily hedging dates, objective CVaR 95%
//Case A: GBM (vol 20%), no trading costs -- Black-Scholes' only flaw is discrete hedging
//Case B, C: GBM with proportional costs of 0.2% and 1% of the traded value
//Case D: Heston (v0 = theta = 0.04, kappa = 2, vol of vol 0.6, rho = -0.7), no costs; the network also sees the
//        current variance, and is compared with Black-Scholes deltas at the current volatility sqrt(v_k) and at 20%
//Every network: 2 x 16 tanh units, 3000 Adam steps on fresh batches of 2,048 paths, learning rate 0.005 decaying
//linearly to 0.00025; evaluated on 100,000 independent test paths. Also: the network's no-trade band with costs
//Run from the repo root
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include "qre/hedging/deep_hedging.hpp"
using namespace qre::hedging;

namespace {
const double K = 100.0, T = 30.0 / 252.0, ALPHA = 0.95;
const int N = 30;

struct Stats{ double mean, sd, var95, cvar95, turnover, costs; };
Stats stats(const Matrix& S, const Matrix& d, double cost){
    const Vector L = hedge_loss(S, d, K, cost);
    const double m = L.mean();
    double turn = 0.0, paid = 0.0;
    for(Eigen::Index p = 0; p < S.rows(); ++p){
        double prev = 0.0;
        for(Eigen::Index k = 0; k <= d.cols(); ++k){
            const double now = k < d.cols() ? d(p, k) : 0.0;
            turn += std::abs(now - prev);
            paid += cost * S(p, k) * std::abs(now - prev);
            prev = now;
        }
    }
    const double n = static_cast<double>(S.rows());
    return {m, std::sqrt((L.array() - m).square().sum() / (n - 1.0)), quantile(L, ALPHA), cvar(L, ALPHA), turn / n, paid / n};
}
void row(std::ofstream& f, const char* cs, const char* strategy, const Stats& s){
    f << cs << "," << strategy << "," << s.mean << "," << s.sd << "," << s.var95 << "," << s.cvar95 << "," << s.turnover << "," << s.costs << "\n";
}
}

int main(){
    const std::string out_dir = "cpp/experiments/deep-hedging/results/";
    std::ofstream c1(out_dir + "hedging_performance.csv"), c2(out_dir + "training_curves.csv"), c3(out_dir + "no_trade_band.csv"),
                  c4(out_dir + "delta_comparison.csv");
    if(!c1.is_open() || !c2.is_open() || !c3.is_open() || !c4.is_open()){
        std::cerr << "Failed to open output files in " << out_dir << "\nRun from repo root \n";
        return 1;
    }
    for(auto* f : {&c1, &c2, &c3, &c4}) f->precision(10);
    c1 << "case,strategy,mean_loss,sd_loss,var95,cvar95,turnover,costs_paid\n";
    c2 << "case,iteration,batch_objective\n";
    c3 << "case,moneyness,bs_delta,band_low,band_high\n";
    c4 << "case,date,mean_abs_diff_nn_vs_bs,max_abs_diff\n";
    TrainConfig cfg;
    cfg.iterations = 3000; cfg.batch = 2048; cfg.lr = 5e-3; cfg.lr_final_fraction = 0.05; cfg.log_every = 100;
    std::cout << "Black-Scholes price of the call: " << bs_call(100.0, K, 0.2, T) << "\n";

    struct GbmCase{ const char* name; double cost; uint64_t seed; };
    for(const GbmCase& gc : {GbmCase{"A_gbm_no_cost", 0.0, 1701}, GbmCase{"B_gbm_cost_0.2pct", 0.002, 1702}, GbmCase{"C_gbm_cost_1pct", 0.01, 1703}}){
        Sampler gbm = [](std::size_t n, std::mt19937_64& e, Matrix& S, Matrix&){ S = simulate_gbm(n, N, 100.0, 0.0, 0.2, T, e); };
        cfg.seed = gc.seed;
        TrainLog log;
        const auto t0 = std::chrono::steady_clock::now();
        const Policy p = train(Policy::init(3, 16, gc.seed), gbm, K, T, gc.cost, ALPHA, cfg, &log);
        std::cout << gc.name << " trained in " << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << " s\n";
        for(auto [it, j] : log.objective) c2 << gc.name << "," << it << "," << j << "\n";
        std::mt19937_64 te(gc.seed + 100);
        const Matrix S = simulate_gbm(100000, N, 100.0, 0.0, 0.2, T, te);
        const Matrix dn = policy_deltas(p, S, K, T), db = bs_hedge_deltas(S, K, T, 0.2);
        row(c1, gc.name, "deep_hedge", stats(S, dn, gc.cost));
        row(c1, gc.name, "black_scholes_delta", stats(S, db, gc.cost));
        row(c1, gc.name, "no_hedge", stats(S, Matrix::Zero(S.rows(), N), gc.cost));
        for(int k : {0, 10, 20, 29}){
            const Vector diff = (dn.col(k) - db.col(k)).cwiseAbs();
            c4 << gc.name << "," << k << "," << diff.mean() << "," << diff.maxCoeff() << "\n";
        }
        //no-trade band at the middle date: previous positions for which the network trades less than 0.02
        const int k = 15;
        for(double m = 0.94; m <= 1.0601; m += 0.02){
            const double tau = T * (N - k) / N;
            const double d1 = (std::log(m) + 0.5 * 0.04 * tau) / (0.2 * std::sqrt(tau));
            double lo = 2.0, hi = -1.0;
            for(int j = 0; j <= 100; ++j){
                const double prev = j / 100.0;
                Matrix x(1, 3);
                x << static_cast<double>(k) / N, std::log(m) / std::sqrt(T), prev;
                const Matrix h1 = ((x * p.W1).rowwise() + p.b1).array().tanh();
                const Matrix h2 = ((h1 * p.W2).rowwise() + p.b2).array().tanh();
                const double out = (h2 * p.W3)(0, 0) + p.b3;
                if(std::abs(out - prev) < 0.02){ lo = std::min(lo, prev); hi = std::max(hi, prev); }
            }
            c3 << gc.name << "," << m << "," << 0.5 * std::erfc(-d1 / std::sqrt(2.0)) << ",";
            if(hi < 0.0) c3 << ",\n"; else c3 << lo << "," << hi << "\n";
        }
    }

    //Case D: Heston
    {
        const char* name = "D_heston_no_cost";
        Sampler hes = [](std::size_t n, std::mt19937_64& e, Matrix& S, Matrix& V){
            simulate_heston(n, N, 100.0, 0.0, 0.04, 2.0, 0.04, 0.6, -0.7, T, e, S, V);
        };
        cfg.seed = 1704;
        TrainLog log;
        const auto t0 = std::chrono::steady_clock::now();
        const Policy p = train(Policy::init(4, 16, 1704), hes, K, T, 0.0, ALPHA, cfg, &log);
        std::cout << name << " trained in " << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << " s\n";
        for(auto [it, j] : log.objective) c2 << name << "," << it << "," << j << "\n";
        std::mt19937_64 te(1804);
        Matrix S, V;
        simulate_heston(100000, N, 100.0, 0.0, 0.04, 2.0, 0.04, 0.6, -0.7, T, te, S, V);
        const Matrix dn = policy_deltas(p, S, K, T, &V);
        row(c1, name, "deep_hedge_with_variance", stats(S, dn, 0.0));
        row(c1, name, "black_scholes_delta_current_vol", stats(S, bs_hedge_deltas(S, K, T, 0.2, &V), 0.0));
        row(c1, name, "black_scholes_delta_20pct", stats(S, bs_hedge_deltas(S, K, T, 0.2), 0.0));
        row(c1, name, "no_hedge", stats(S, Matrix::Zero(S.rows(), N), 0.0));
        const Matrix db = bs_hedge_deltas(S, K, T, 0.2, &V);
        for(int k : {0, 10, 20, 29}){
            const Vector diff = (dn.col(k) - db.col(k)).cwiseAbs();
            c4 << name << "," << k << "," << diff.mean() << "," << diff.maxCoeff() << "\n";
        }
    }
    std::cout << "Results in " << out_dir << "\n";
    return 0;
}
