//Phase 4 experiment: Characterize Newton-Raphson's distinct failure modes using 3 specifically
//chosen functions, one per failure -> run same problems through bisection and Brent to demonstrate
//C++ only: This experiment tests the specific behavior of the solvers in root_finding/calibration
//Not a mathematical fact, using another library may make different choices about tolerances/divergence detection -> impossible to help with validation
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include "qre/optimization/root_finding.hpp"
using qre::optimization::RootResult;
using qre::optimization::FailureReason;
using qre::optimization::bisection;
using qre::optimization::newton_raphson;
using qre::optimization::brent;

std::string reason_to_string(FailureReason reason){
    switch (reason){
        case FailureReason::None: return "None";
        case FailureReason::MaxIterations: return "MaxIterations";
        case FailureReason::DerivativeNearZero: return "DerivativeNearZero";
        case FailureReason::Diverged: return "Diverged";
    }
    return "Unknown";
}

void write_row(
    std::ofstream& out,
    const std::string& func_name,
    const std::string& solver,
    bool converged,
    int iterations,
    const std::string& reason 
) {
    out << func_name << "," << solver << "," << (converged ? "true" : "false") << "," << iterations << "," << reason << "\n";
    std::cout << func_name << " / " << solver << " converged=" << (converged ? "true" : "false") << " iterations=" << iterations << " reason=" << reason << "\n";
}

int main(){
    const std::string output_path = "cpp/experiments/04-numerical-methods/results/newton_failure_results.csv";
    std::ofstream out(output_path);
    if(!out.is_open()) {
        std::cerr << "Failed to open output file" << output_path << "\n";
        std::cerr << "Run from repo root \n";
        return 1;
    }
    out << "function_name,solver,converged,iterations,failure_reason\n";

    //Case 1: DerivativeNearZero
    //f(x) = x^2-4, derivative of 2x
    {
    auto f = [](double x) { return x*x-4.0;};
    auto f_prime = [](double x) { return 2.0*x;};
    auto newton_result = newton_raphson(f, f_prime, 0.0);
    if(newton_result.has_value()){
        const auto& r = newton_result.value();
        write_row(out, "x^2-4", "newton", r.converged, r.iterations, reason_to_string(r.reason));
    }
    auto bisection_result = bisection(f, 0.0, 5.0);
    if(bisection_result.has_value()){
        const auto& r = bisection_result.value();
        write_row(out, "x^2-4", "bisection", r.converged, r.iterations, reason_to_string(r.reason));
    }
    auto brent_result = brent(f, 0.0, 5.0);
    if(brent_result.has_value()){
        const auto& r = brent_result.value();
        write_row(out, "x^2-4", "brent", r.converged, r.iterations, reason_to_string(r.reason));
    }
}

    //Case 2: Diverged
    //f(x) = cube root of x, root at x=0, classic textbook case where Newton diverges from almost any nonzero starting guess
    {
    auto f = [](double x){return std::cbrt(x);};
    auto f_prime = [](double x){ return (1.0/3.0) * std::pow(std::abs(x), -2.0/3.0);};
    auto newton_result = newton_raphson(f, f_prime, 1.0);
    if(newton_result.has_value()){
        const auto& r = newton_result.value();
        write_row(out, "cbrt(x)", "newton", r.converged, r.iterations, reason_to_string(r.reason));
    }
    auto bisection_result = bisection(f, -2.0, 2.0);
    if(bisection_result.has_value()){
        const auto& r = bisection_result.value();
        write_row(out, "cbrt(x)", "bisection", r.converged, r.iterations, reason_to_string(r.reason));
    }
    auto brent_result = brent(f, -2.0, 2.0);
    if(brent_result.has_value()){
        const auto& r = brent_result.value();
        write_row(out, "cbrt(x)", "brent", r.converged, r.iterations, reason_to_string(r.reason));
    }
}
    //Case 3: MaxIterations
    //f(x) = cos(x) - x, real root near x - 0.739. Well behaved function but max_iterations is set too low
    {
        auto f = [](double x){return std::cos(x) - x; };
        auto f_prime = [](double x){return -std::sin(x) - 1.0;};
        auto newton_result = newton_raphson(f, f_prime, 0.0, qre::config::kDefTolerance, 2);
        if(newton_result.has_value()){
            const auto& r = newton_result.value();
            write_row(out, "cos(x)-x", "newton", r.converged, r.iterations, reason_to_string(r.reason));
        }
        auto bisection_result = bisection(f, 0.0, 1.0);
        if(bisection_result.has_value()){
            const auto& r = bisection_result.value();
            write_row(out, "cos(x)-x", "bisection", r.converged, r.iterations, reason_to_string(r.reason));
        }
        auto brent_result = brent(f, 0.0, 1.0);
        if(brent_result.has_value()){
            const auto& r = brent_result.value();
            write_row(out, "cos(x)-x", "brent", r.converged, r.iterations, reason_to_string(r.reason));
        }
    }
    out.close();
    std::cout << "\nResults written to " << output_path << "\n";
    return 0;
}
