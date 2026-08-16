#pragma once
#include <cmath>
#include <numeric>
#include <vector>
#include "qre/core/exceptions.hpp"
namespace qre::estimators::empirical{
    //mean
    //Pass data by const ref -> save time, use std lib for accumulate instead of for loop
    inline double sample_mean(const std::vector<double>& data){
        //base
        if (data.empty()) throw InvalidInputError("data must not be empty");
        const double sum = std::accumulate(data.begin(), data.end(), 0.0);
        //explicit cast for precaution
        return sum/static_cast<double>(data.size());
    }
    //Divide by n-1 not n because n would give you a biased estimate (Bessel's correction)
    inline double sample_var(const std::vector<double>& data){
        if (data.size() < 2) throw InvalidInputError("need min of 2 data points");
        const double mean = sample_mean(data);
        double sum_sq_diff = 0.0;
        //Formula in for loop
        for(double x : data){
            const double diff = x - mean;
            sum_sq_diff += diff * diff;
        }
        return sum_sq_diff/static_cast<double>(data.size()-1);
    }
    //Nothing but square root of variance -> usually more useful
    inline double sample_stddev(const std::vector<double>& data){
        return std::sqrt(sample_var(data));
    }
    //Standard error of the mean, just implemented formula
    inline double std_error(const std::vector<double>& data){
        return sample_stddev(data)/std::sqrt(static_cast<double>(data.size()));
    }
}