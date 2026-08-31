#pragma once
#include <functional>
#include "qre/core/config.hpp"
#include "qre/optimization/root_finding.hpp"
namespace qre::optimization{
    //Builds model param objective every calibration function needs -> root finding with specific framing
    inline std::function<double(double)> make_cal_obj(
        std::function<double(double)> model,
        double target
    ) {
        return [model, target](double param) {return model(param) - target;};
    }
    //Calibrate via Newton-Raphson -> call root finding
    inline Result<RootResult> calibrate_newton(
        std::function<double(double)> model,
        double target,
        double init_guess,
        double tolerance = config::kDefTolerance,
        int max_iterations = config::kDefMaxIterations
    ) {
        auto objective = make_cal_obj(model, target);
        auto objective_derivative = [objective](double param){
            constexpr double h = 1e-6;
            return(objective(param + h) - objective(param - h)) / (2.0 * h);
        };
        return newton_raphson(objective, objective_derivative, init_guess, tolerance, max_iterations);
    }
    //Calibrate via bisection -> slow but guaranteed to converge given a valid bracket
    inline Result<RootResult> calibrate_bisection(
        std::function<double(double)> model,
        double target,
        double lower_bound,
        double upper_bound,
        double tolerance = config::kDefTolerance,
        int max_iterations = config::kDefMaxIterations
    ) {
        auto objective = make_cal_obj(model, target);
        return bisection(objective, lower_bound, upper_bound, tolerance,max_iterations);
    }
    //Calibrate via Brent's method -> bisection with faster interpolation steps when they're trustworthy
    inline Result<RootResult> calibrate_brent(
        std::function<double(double)> model,
        double target,
        double lower_bound,
        double upper_bound,
        double tolerance = config::kDefTolerance,
        int max_iterations = config::kDefMaxIterations
    ) {
        auto objective = make_cal_obj(model, target);
        return brent(objective, lower_bound, upper_bound, tolerance, max_iterations);
    }
}