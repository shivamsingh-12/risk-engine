#pragma once
#include <cmath>
#include <functional>
#include <algorithm>
#include <utility>
#include "qre/core/config.hpp"
#include "qre/core/exceptions.hpp"
#include "qre/core/result.hpp"
namespace qre::optimization{
    //Why a failure occurred, circled back and added thsi to differentiate between the errors that could occur in Phase 4's experiments
    //Names should be self explanatory
    enum class FailureReason {
        None,
        MaxIterations,
        DerivativeNearZero,
        Diverged
    };
    //Create a shared result type across all root finding methods
    struct RootResult{
        double root;
        int iterations;
        bool converged;
        FailureReason reason;
    };
    //Bisection method implementation (refer to 05-numerical-methods notes)
    inline Result<RootResult> bisection(
        std::function<double(double)>f,
        double a,
        double b,
        double tolerance = config::kDefTolerance,
        int max_iterations = config::kDefMaxIterations
    ) {
        double fa = f(a);
        double fb = f(b);
        //check to ensure they have opposite signs -> precondition of root existing
        if (fa * fb > 0.0){
            throw InvalidInputError("bisection: requires opposite signs");
        }
        double midpoint = a;
        for(int iteration = 0; iteration < max_iterations; ++iteration){
            midpoint = (a+b)/2.0;
            const double f_mid = f(midpoint);
            if(std::abs(f_mid) < tolerance){
                return RootResult{midpoint, iteration + 1, true, FailureReason::None};
            }
            //Root between a and midpoint -> keep that half
            if(fa * f_mid < 0.0){
                b = midpoint;
                fb = f_mid;
            //Root between midpoint and b -> keep that half
            } else {
                a = midpoint;
                fa = f_mid;
            }
        }
        //Ran out of iterations without reaching tolerance
        return RootResult{midpoint, max_iterations, false, FailureReason::MaxIterations};
    }

    //Newton-Raphson - tangent line at current guess to jump towards where it crosses zero
    //Converges quadratically, faster than bisection, can fail in certain ways
    inline Result<RootResult> newton_raphson(
        std::function<double(double)> f,
        std::function<double(double)> f_prime,
        double x0,
        double tolerance = config::kDefTolerance,
        int max_iterations = config::kDefMaxIterations,
        //this is kept distinct form tolerance to measure different things
        //tolerance answers how close to the true root is close enough to stop
        //derivative threshold answers how small a slope is too dangerous to divide by
        double derivative_threshold = 1e-10
    ) {
        double x = x0;
        for(int iteration = 0; iteration < max_iterations; ++iteration){
            const double fx = f(x);
            if(std::abs(fx) < tolerance){
                return RootResult{x, iteration, true, FailureReason::None};
            }
            const double fpx = f_prime(x);
            if(std::abs(fpx) < derivative_threshold){
                return RootResult{x, iteration, false, FailureReason::DerivativeNearZero};
            }
            const double x_next = x - fx / fpx;
            //Guess blowing up/becoming non finite -> simple divergence signal/slow drift
            if(std::isnan(x_next) || std::isinf(x_next) || std::abs(x_next) > 1e10){
                return RootResult{x_next, iteration, false, FailureReason::Diverged};
            }
            x = x_next;
        }
        return RootResult{x, max_iterations, false, FailureReason::MaxIterations};
    }
    //Brent's method -> combine bisection's guaranteed convergence with speed of secant/inverse-quadratic interpolation
    //Maintains a bracket around root but faster interpolation
    inline Result<RootResult> brent(
        std::function<double(double)> f,
        double a,
        double b,
        double tolerance = config::kDefTolerance,
        int max_iterations = config::kDefMaxIterations
    ) {
        double fa = f(a);
        double fb = f(b);
        if(fa * fb > 0.0){
            throw InvalidInputError("brent: f(a) and f(b) must be opposite signs");
        }
        if(std::abs(fa) < std::abs(fb)){
            std::swap(a,b);
            std::swap(fa, fb);
        }
        double c = a;
        double fc = fa;
        bool mflag = true;
        double d = 0.0;

        for(int iteration = 0; iteration < max_iterations; ++iteration){
            if(std::abs(fb) < tolerance){
                return RootResult{b, iteration, true, FailureReason::None};
            }
            if(std::abs(b-a) < tolerance){
                return RootResult{b, iteration, true, FailureReason::None};
            }
            double s;
            if(fa != fc && fb != fc){
                //inverse quad interpolation
                s = a * fb * fc / ((fa - fb) * (fa - fc)) + b * fa * fc / ((fb - fa) * (fb - fc)) + c * fa * fb / ((fc - fa) * (fc - fb));
            } else {
                //other is secant method
                s = b - fb * (b-a) / (fb - fa);
            }
            //extra conditions where interpolated step is rejected for bisection
            const double lower_bound = (3.0 * a + b) / 4.0;
            const bool s_oor = (s < std::min(lower_bound, b) || s > std::max(lower_bound, b));
            const bool step_shrink_inc = (mflag && std::abs(s-b) >= std::abs(b-c)/2.0) || (!mflag && std::abs(s-b) >= std::abs(c-d)/2.0) || (mflag && std::abs(b-c) < tolerance) || (!mflag && std::abs(c-d) < tolerance);
            if(s_oor || step_shrink_inc){
                s = (a+b)/2.0; //bisection fallback
                mflag = true;
            } else {
                mflag = false;
            }
            const double fs = f(s);
            d = c;
            c = b;
            fc = fb;
            if(fa * fs < 0.0){
                b = s;
                fb = fs;
            } else {
                a = s;
                fa = fs;
            }
        //if b remains best estimate -> maintain 
        if(std::abs(fa) < std::abs(fb)){
            std::swap(a,b);
            std::swap(fa,fb);
        }
    }
    return RootResult{b, max_iterations, false, FailureReason::MaxIterations};
    
    }
}