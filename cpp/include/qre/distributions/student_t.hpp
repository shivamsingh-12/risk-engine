#pragma once
#include <cmath>
#include <numbers>
#include "qre/core/exceptions.hpp"
#include "qre/optimization/root_finding.hpp"
//Student-t distribution with nu > 0 degrees of freedom (standard form: location 0, scale 1)
//pdf  f(x) = Gamma((nu+1)/2) / (sqrt(nu pi) Gamma(nu/2)) (1 + x^2/nu)^(-(nu+1)/2)
//cdf  via the regularized incomplete beta function: for x >= 0, F(x) = 1 - I_{nu/(nu+x^2)}(nu/2, 1/2) / 2
//quantile: Brent on cdf(x) - p (Phase 4), then Newton polish with the pdf for far-tail accuracy
//The standard library has no Student-t, so the incomplete beta is implemented here with the
//continued-fraction expansion evaluated by the modified Lentz method
namespace qre::distributions::student_t{
    namespace detail{
        inline void validate_nu(double nu){
            if(!(nu > 0.0) || !std::isfinite(nu)) throw InvalidInputError("nu must be positive and finite");
        }
        //Continued fraction for the incomplete beta (modified Lentz)
        inline double beta_continued_fraction(double a, double b, double x){
            constexpr int kMaxIter = 500;
            constexpr double kEps = 1e-16, kTiny = 1e-300;
            const double qab = a + b, qap = a + 1.0, qam = a - 1.0;
            double c = 1.0, d = 1.0 - qab * x / qap;
            if(std::abs(d) < kTiny) d = kTiny;
            d = 1.0 / d;
            double h = d;
            for(int m = 1; m <= kMaxIter; ++m){
                const double dm = static_cast<double>(m), m2 = 2.0 * dm;
                double aa = dm * (b - dm) * x / ((qam + m2) * (a + m2));
                d = 1.0 + aa * d; if(std::abs(d) < kTiny) d = kTiny;
                c = 1.0 + aa / c; if(std::abs(c) < kTiny) c = kTiny;
                d = 1.0 / d; h *= d * c;
                aa = -(a + dm) * (qab + dm) * x / ((a + m2) * (qap + m2));
                d = 1.0 + aa * d; if(std::abs(d) < kTiny) d = kTiny;
                c = 1.0 + aa / c; if(std::abs(c) < kTiny) c = kTiny;
                d = 1.0 / d;
                const double del = d * c;
                h *= del;
                if(std::abs(del - 1.0) < kEps) return h;
            }
            throw ConvergenceError("incomplete beta continued fraction did not converge");
        }
        //Regularized incomplete beta I_x(a, b)
        inline double incomplete_beta(double a, double b, double x){
            if(a <= 0.0 || b <= 0.0) throw InvalidInputError("incomplete beta needs a, b > 0");
            if(x <= 0.0) return 0.0;
            if(x >= 1.0) return 1.0;
            const double front = std::exp(std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b)
                                          + a * std::log(x) + b * std::log1p(-x));
            //Use the symmetry I_x(a,b) = 1 - I_{1-x}(b,a) where the fraction converges fastest
            if(x < (a + 1.0) / (a + b + 2.0)) return front * beta_continued_fraction(a, b, x) / a;
            return 1.0 - front * beta_continued_fraction(b, a, 1.0 - x) / b;
        }
    }

    inline double pdf(double x, double nu){
        detail::validate_nu(nu);
        const double log_norm = std::lgamma((nu + 1.0) / 2.0) - std::lgamma(nu / 2.0) - 0.5 * std::log(nu * std::numbers::pi);
        return std::exp(log_norm - 0.5 * (nu + 1.0) * std::log1p(x * x / nu));
    }

    inline double cdf(double x, double nu){
        detail::validate_nu(nu);
        const double tail = 0.5 * detail::incomplete_beta(nu / 2.0, 0.5, nu / (nu + x * x));
        return x >= 0.0 ? 1.0 - tail : tail;
    }

    inline double quantile(double p, double nu){
        detail::validate_nu(nu);
        if(!(p > 0.0 && p < 1.0)) throw InvalidInputError("p must be in (0, 1)");
        if(p == 0.5) return 0.0;
        if(p < 0.5) return -quantile(1.0 - p, nu);          //symmetry
        double hi = 1.0;
        while(cdf(hi, nu) < p){
            hi *= 2.0;
            if(hi > 1e12) throw InvalidInputError("quantile is too far in the tail to bracket");
        }
        auto f = [p, nu](double x){ return cdf(x, nu) - p; };
        auto res = optimization::brent(f, 0.0, hi, 1e-15, 300);
        if(!res.has_value()) throw ConvergenceError("student_t quantile: Brent failed");
        double x = res->root;
        for(int i = 0; i < 3; ++i){                         //Newton polish: cdf is very flat in the far tail
            const double step = (cdf(x, nu) - p) / pdf(x, nu);
            if(!std::isfinite(step)) break;
            x -= step;
        }
        return x;
    }
}
