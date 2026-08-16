#pragma once
#include <cmath>
#include <random>
#include <vector>
#include "qre/core/rng.hpp"
#include "qre/core/types.hpp"
#include "qre/core/exceptions.hpp"
    //Precalculate some values to potentially save time when called inside loops
namespace qre::distributions::normal{
    //sub name space -> internal helper specfically to check sigma
     namespace detail{
        inline void validate_sigma(double sigma){
            if(sigma <= 0.0){
                throw InvalidInputError("sigma must be positive");
            }
        }
    }
    //Function calculates exact density of x occuring under a normal distribution with PDF
    inline double pdf(double x, double mu, double sigma){
        detail::validate_sigma(sigma);
        static constexpr double kInvSqrt2Pi = 0.3989422804014327; //Calculated 1/sqrt(2*pi)
        const double z = (x - mu)/sigma; //z-score
        return (kInvSqrt2Pi/sigma) * std::exp(-0.5*z*z); //full formula
    }
    //Calculates total probability random value witll be less than/equal to x
    inline double cdf(double x, double mu, double sigma){
        detail::validate_sigma(sigma);
        static constexpr double kInvSqrt2 = 0.7071067811865476; // 1/sqrt(2)
        return 0.5 * std::erfc(-(x-mu) * kInvSqrt2/sigma); //erfc not erf avoids catastrophic cancellation
    }
    //Next 2 functions matter more for lognormal and others are added
    //sigma and then mu -> silence compiler warning & consistency
    inline double mean_analytical(double mu, double /*sigma*/){
        return mu;
    }
    inline double var_analytical(double /*mu*/, double sigma){
        detail::validate_sigma(sigma);
        return sigma * sigma;
    }
    //Future TODO: nothing stops a non-positive sigma -> fix when possible -> done
    inline std::vector<double> sample(std::size_t n, double mu, double sigma, Rng& rng){
        detail::validate_sigma(sigma);
        //Distribution object init with mean and SD
        //note: use std::normal_distribution so it can handle the actual sampling algorithm
        std::normal_distribution<double> dist(mu, sigma);
        std::vector<double> result;
        result.reserve(n); //allocalates memory for n floating point nums in vector
        for(std::size_t i = 0; i < n; ++i){
            result.push_back(dist(rng.engine())); //dist obj called as function -> rng engine -> random to normally distributed val
        }
        //appends generated nums to result and returns filled vector
        return result;
    }
}