#pragma once
#include <cmath>
#include <vector>
#include <Eigen/Dense>
#include "qre/core/exceptions.hpp"
#include "qre/core/types.hpp"
//Data hardening: splits and cash dividends
//An action's row is its ex-date: the first row trading on the post-action basis
//Back-adjustment multiplies every row BEFORE the ex-date so the latest prices stay as traded
//  split (value = shares out per share in, 2.0 for 2-for-1, 0.5 for 1-for-2): factor 1 / ratio
//  cash dividend (value = cash per share): factor 1 - D / P(ex-1), raw close before the ex-date
//Returns from the adjusted series are total returns: split-free and dividend-inclusive
namespace qre::data{
    enum class ActionType {Split, CashDividend};

    struct CorporateAction{
        Eigen::Index asset;
        Eigen::Index row;
        ActionType type;
        double value;
    };

    namespace detail{
        inline double action_factor(const PriceMatrix& prices, const CorporateAction& a){
            if(a.asset < 0 || a.asset >= prices.cols() || a.row < 1 || a.row >= prices.rows()){
                throw InvalidInputError("corporate action asset/row out of range");
            }
            if(a.type == ActionType::Split){
                if(a.value <= 0.0) throw InvalidInputError("split ratio must be positive");
                return 1.0 / a.value;
            }
            const double prev = prices(a.row - 1, a.asset);
            if(!std::isfinite(prev) || a.value <= 0.0 || a.value >= prev){
                throw InvalidInputError("dividend must be positive and below the prior close");
            }
            return 1.0 - a.value / prev;
        }
    }

    inline PriceMatrix back_adjust(const PriceMatrix& prices, const std::vector<CorporateAction>& actions){
        Eigen::MatrixXd factor = Eigen::MatrixXd::Ones(prices.rows(), prices.cols());
        for(const auto& a : actions){
            const double f = detail::action_factor(prices, a);
            factor.block(0, a.asset, a.row, 1) *= f;
        }
        return prices.cwiseProduct(factor);
    }

    //Point-in-time adjustment: only actions with ex-date at or before row t-1 are known at decision time
    inline PriceMatrix adjust_as_of(const PriceMatrix& prices, const std::vector<CorporateAction>& actions, Eigen::Index t){
        std::vector<CorporateAction> known;
        for(const auto& a : actions) if(a.row <= t - 1) known.push_back(a);
        return back_adjust(prices, known);
    }

    struct SplitCandidate{
        Eigen::Index asset;
        Eigen::Index row;
        double ratio;
    };

    //Days whose ratio P(t-1)/P(t) is within tol of a common split ratio AND moved at least min_move
    //Detection only: a genuine -50% crash is indistinguishable from an unadjusted 2-for-1 split,
    //so candidates are flagged for confirmation, never applied automatically
    inline std::vector<SplitCandidate> detect_split_candidates(const PriceMatrix& prices, double min_move = 0.25, double tol = 0.03){
        static constexpr double kRatios[] = {10.0, 5.0, 4.0, 3.0, 2.0, 1.5, 1.0 / 1.5, 0.5, 1.0 / 3.0, 0.25, 0.2, 0.1};
        std::vector<SplitCandidate> out;
        for(Eigen::Index c = 0; c < prices.cols(); ++c){
            for(Eigen::Index r = 1; r < prices.rows(); ++r){
                const double p0 = prices(r - 1, c), p1 = prices(r, c);
                if(!std::isfinite(p0) || !std::isfinite(p1) || p0 <= 0.0 || p1 <= 0.0) continue;
                if(std::abs(p1 / p0 - 1.0) < min_move) continue;
                const double ratio = p0 / p1;
                for(double s : kRatios){
                    if(std::abs(ratio / s - 1.0) <= tol){
                        out.push_back(SplitCandidate{c, r, s});
                        break;
                    }
                }
            }
        }
        return out;
    }
}
