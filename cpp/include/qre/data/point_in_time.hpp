#pragma once
#include <cmath>
#include <vector>
#include <Eigen/Dense>
#include "qre/core/exceptions.hpp"
#include "qre/core/types.hpp"
//Data hardening: point-in-time windows -> a forecast must only use data that existed when it was made
//The forecast for the return of row t (P_t / P_{t-1} - 1) may use rows <= t-1 only
//Note: fully back-adjusted LEVELS before t embed future corporate actions, but RETURNS before t are
//unaffected (see adjust_as_of in corporate_actions.hpp and the tests) -> estimate from returns
namespace qre::data{
    //Rows [t-1-window, t-1]: window + 1 prices -> window returns. t may equal rows() (forecast the next day)
    inline PriceMatrix pit_window(const PriceMatrix& prices, Eigen::Index t, Eigen::Index window){
        if(window < 1 || t - 1 - window < 0 || t > prices.rows()){
            throw InvalidInputError("window does not fit before row t");
        }
        return prices.middleRows(t - 1 - window, window + 1);
    }

    //Assets with a complete history in the point-in-time window -> universe membership known at t-1
    inline std::vector<Eigen::Index> universe_at(const PriceMatrix& prices, Eigen::Index t, Eigen::Index window){
        const PriceMatrix w = pit_window(prices, t, window);
        std::vector<Eigen::Index> out;
        for(Eigen::Index c = 0; c < w.cols(); ++c){
            if(w.col(c).allFinite()) out.push_back(c);
        }
        return out;
    }
}
