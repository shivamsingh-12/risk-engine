#pragma once
#include <Eigen/Dense>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>
#include "qre/core/exceptions.hpp"
#include "qre/core/types.hpp"
namespace qre::data{
    //count missing entries per column
    inline Eigen::VectorXi missing_per_column(const Eigen::MatrixXd& prices){
        const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> is_nan = prices.array().isNaN();
        Eigen::VectorXi counts(prices.cols());
        for(int col = 0; col < prices.cols(); ++col){
            counts(col) = is_nan.col(col).cast<int>().sum();
        }
        return counts;
    }
    //Tags coordinates where prices lower than 0, which is irrelevant but diff from function above
    inline std::vector<std::pair<int,int>> flag_invalid_prices(const Eigen::MatrixXd& prices){
        std::vector<std::pair<int,int>> invalid;
        for(int c = 0; c < prices.cols(); ++c){
            for(int r = 0; r < prices.rows(); ++r){
                if(prices(r, c) <= 0.0){
                    invalid.emplace_back(r, c);
                }
            }
        }
        return invalid;
    }
    //forward fills a specific amount of days -> gap longer is left unfilled to stop fabrication, next func drops if goes over
    inline Eigen::MatrixXd forward_fill(const Eigen::MatrixXd& prices, int max_consec_days){
        Eigen::MatrixXd result = prices;
        for(int c = 0; c < result.cols(); ++c){
            double last = std::numeric_limits<double>::quiet_NaN();
            int gap = 0;
            for(int r = 0; r < result.rows(); ++r){
                if(std::isnan(result(r, c))){
                    if(std::isnan(last)){
                        continue;
                    }
                    if(gap < max_consec_days){
                        result(r,c) = last;
                        ++gap;
                    }
                } else {
                    last = result(r, c);
                    gap = 0;
                }
            }
        }
        return result;
    }
    //Result of next func dropping missing vals, and which cols were dropped -- don't silently vanish
    struct CleanResult{
        Eigen::MatrixXd clean_prices;
        std::vector<int> drop_col_indices;
    };
    //
    inline CleanResult drop_missing_assets(const Eigen::MatrixXd& prices, double max_miss_pct){
        const Eigen::VectorXi missing_counts = missing_per_column(prices);
        const double totrows = static_cast<double>(prices.rows());
        std::vector<int> keep;
        std::vector<int> dropped;
        for(int c = 0; c < prices.cols(); ++c){
            const double missing_pct = static_cast<double>(missing_counts(c))/totrows;
            if(missing_pct > max_miss_pct){
                dropped.push_back(c);
            } else {
                keep.push_back(c);
            }
        }
        Eigen::MatrixXd clean_prices(prices.rows(), static_cast<int>(keep.size()));
        for(std::size_t i = 0; i < keep.size(); ++i){
            clean_prices.col(static_cast<int>(i)) = prices.col(keep[i]);
        }
        return CleanResult{clean_prices, dropped};
    }
}