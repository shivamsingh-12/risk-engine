#pragma once
#include <Eigen/Dense>
#include <vector>
#include "qre/core/exceptions.hpp"
#include "qre/core/types.hpp"
#include "qre/estimators/empirical.hpp"
namespace qre::data{
    //Converts a column's window slice into a vector double
    inline std::vector<double> window_slice(const ReturnMatrix& returns, int col, int start, int window){
        std::vector<double> slice;
        slice.reserve(static_cast<std::size_t>(window));
        for(int i = 0; i < window; ++i){
            slice.push_back(returns(start + i, col));
        }
        return slice;
    }
    //Rolling mean per asset - for input and length n and window w, output has length n - w + 1
    //Also reuses empirical's sample mean rather than recomputing
    inline Eigen::MatrixXd rolling_mean(const ReturnMatrix& returns, int window){
        if(window <= 0 || window > returns.rows()) {
            throw InvalidInputError("rolling mean: window must be positive and less than num of observations");
        }
        const int n_win = static_cast<int>(returns.rows()) - window + 1;
        Eigen::MatrixXd result(n_win, returns.cols());
        for(int c = 0; c < returns.cols(); ++c){
            for(int s = 0; s < n_win; ++s){
                std::vector<double> slice = window_slice(returns, c, s, window);
                result(s, c) = qre::estimators::empirical::sample_mean(slice);
            }
        }
        return result;
    }
    //Rolling volatility or std dev per asset - also reuses empirical's stddev
    inline Eigen::MatrixXd rolling_vol(const ReturnMatrix& returns, int window){
        if(window <= 0 || window > returns.rows()) {
            throw InvalidInputError("rolling volatility: window must be positive and less than num of observations");
        }
        const int n_win = static_cast<int>(returns.rows()) - window + 1;
        Eigen::MatrixXd result(n_win, returns.cols());
        for(int c = 0; c < returns.cols(); ++c){
            for(int s = 0; s < n_win; ++s){
                std::vector<double> slice = window_slice(returns, c, s, window);
                result(s, c) = qre::estimators::empirical::sample_stddev(slice);
            }
        }
        return result;
    }
}