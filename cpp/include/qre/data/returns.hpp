#pragma once
#include <Eigen/Dense>
#include "qre/core/exceptions.hpp"
#include "qre/core/types.hpp"

namespace qre::data{
    // Simple returns: (P_t / P_t-1) - 1 -> take current / prev then subtract
    //When aggregating returns across multiple assets at a single point in time
    inline ReturnMatrix simple_return(const PriceMatrix& prices){
        if(prices.rows() < 2){
            throw InvalidInputError("simple: need at least 2 price observations");
        }
        const Eigen::MatrixXd current = prices.bottomRows(prices.rows()-1);
        const Eigen::MatrixXd prev = prices.topRows(prices.rows()-1);
        return current.array()/prev.array() - 1.0;
    }
    //Log returns: ln(P_t / P_t-1) -> Appropriate 
    //When aggregating returns across time for a single asset
    inline ReturnMatrix log_return(const PriceMatrix& prices){
        if(prices.rows() < 2){
            throw InvalidInputError("log: need at least 2 price observations");
        }
        const Eigen::MatrixXd current = prices.bottomRows(prices.rows()-1);
        const Eigen::MatrixXd prev = prices.topRows(prices.rows()-1);
        return (current.array()/prev.array()).log();
    }
}