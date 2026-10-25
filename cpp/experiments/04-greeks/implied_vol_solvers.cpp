//Phase 5 experiment: Implied volatility recovery -> Newton-Raphson vs bisection vs Brent
//Each grid point: pick a true vol, price the option with Black-Scholes, then ask each solver
//to recover the vol from that price alone. Error = |recovered - true|
//C++ only: this tests how the Phase 4 solvers behave on the Phase 5 pricer (iteration counts,
//failure modes), not a mathematical fact, so there is no independent library to validate against
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include "qre/core/exceptions.hpp"
#include "qre/optimization/calibration.hpp"
#include "qre/pricing/black_scholes.hpp"
using qre::pricing::BSParams;
using qre::pricing::OptionType;
using qre::optimization::FailureReason;
using qre::optimization::RootResult;

std::string reason_to_string(FailureReason reason){
    switch (reason){
        case FailureReason::None: return "None";
        case FailureReason::MaxIterations: return "MaxIterations";
        case FailureReason::DerivativeNearZero: return "DerivativeNearZero";
        case FailureReason::Diverged: return "Diverged";
    }
    return "Unknown";
}

//One solver run -> one CSV row (and one console line)
void write_row(std::ofstream& out, double moneyness, double true_vol, double time,
               const std::string& solver, bool converged, int iterations,
               double error, const std::string& reason){
    out << moneyness << "," << true_vol << "," << time << "," << solver << ","
        << (converged ? "true" : "false") << "," << iterations << "," << error << "," << reason << "\n";
}

int main(){
    const std::string output_path = "cpp/experiments/04-greeks/results/implied_vol_results.csv";
    std::ofstream out(output_path);
    if(!out.is_open()){
        std::cerr << "Failed to open output file " << output_path << "\n";
        std::cerr << "Run from repo root \n";
        return 1;
    }
    out.precision(10);
    out << "moneyness,true_vol,time,solver,converged,iterations,abs_error,failure_reason\n";

    const double spot = 100.0;
    const double rate = 0.05;
    const double init_guess = 0.2;         //Newton's starting point
    const double lower = 1e-6;             //bracket for bisection and Brent
    const double upper = 5.0;
    const double tol = 1e-10;
    const double moneyness_grid[] = {0.7, 0.85, 1.0, 1.15, 1.3};   //K/S
    const double vol_grid[] = {0.05, 0.2, 0.5, 1.0, 1.5};
    const double time_grid[] = {0.1, 1.0};

    int total = 0, newton_fail = 0, bisect_fail = 0, brent_fail = 0, skipped = 0;
    long newton_it = 0, bisect_it = 0, brent_it = 0;
    int newton_ok = 0, bisect_ok = 0, brent_ok = 0;

    for(double m : moneyness_grid){
        for(double true_vol : vol_grid){
            for(double t : time_grid){
                BSParams p{spot, spot * m, rate, true_vol, t, OptionType::Call};
                const double mkt = qre::pricing::price(p);
                //Deep OTM short-dated options can price to ~0 -> no-arbitrage check would reject, nothing to invert
                const double lo_bound = std::max(spot - p.strike * std::exp(-rate * t), 0.0);
                if(!(mkt > lo_bound + 1e-12)){
                    ++skipped;
                    continue;
                }
                ++total;
                auto model = [p](double sigma){
                    BSParams q = p;
                    q.vol = sigma;
                    return qre::pricing::price(q);
                };
                //Lambda so one try/catch pattern handles all three solvers
                auto run = [&](const std::string& name, auto solve, int& fails, long& its, int& oks){
                    try{
                        auto res = solve();
                        if(res.has_value()){
                            const RootResult& r = res.value();
                            const double err = std::abs(r.root - true_vol);
                            write_row(out, m, true_vol, t, name, r.converged, r.iterations, err, reason_to_string(r.reason));
                            if(r.converged){ ++oks; its += r.iterations; } else ++fails;
                        } else {
                            write_row(out, m, true_vol, t, name, false, 0, NAN, "SolverError");
                            ++fails;
                        }
                    } catch(const qre::NumericalError& e){
                        write_row(out, m, true_vol, t, name, false, 0, NAN, "Threw");
                        ++fails;
                    }
                };
                run("newton", [&]{ return qre::optimization::calibrate_newton(model, mkt, init_guess, tol); }, newton_fail, newton_it, newton_ok);
                run("bisection", [&]{ return qre::optimization::calibrate_bisection(model, mkt, lower, upper, tol); }, bisect_fail, bisect_it, bisect_ok);
                run("brent", [&]{ return qre::optimization::calibrate_brent(model, mkt, lower, upper, tol); }, brent_fail, brent_it, brent_ok);
            }
        }
    }
    out.close();
    std::cout << "grid points inverted: " << total << " (skipped " << skipped << " with no extractable vol)\n";
    std::cout << "newton:    failures=" << newton_fail << " mean iters (converged)=" << (newton_ok ? (double)newton_it / newton_ok : 0) << "\n";
    std::cout << "bisection: failures=" << bisect_fail << " mean iters (converged)=" << (bisect_ok ? (double)bisect_it / bisect_ok : 0) << "\n";
    std::cout << "brent:     failures=" << brent_fail << " mean iters (converged)=" << (brent_ok ? (double)brent_it / brent_ok : 0) << "\n";
    std::cout << "Results written to " << output_path << "\n";
    return 0;
}
