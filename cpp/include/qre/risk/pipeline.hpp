#pragma once
#include <Eigen/Dense>
#include "qre/core/exceptions.hpp"
#include "qre/core/rng.hpp"
#include "qre/core/types.hpp"
#include "qre/data/cleaning.hpp"
#include "qre/data/returns.hpp"
#include "qre/linalg/covariance.hpp"
#include "qre/risk/var_es.hpp"
//Phase 7 glue: prices -> returns -> covariance -> portfolio -> simulation -> VaR/ES in one call
//Every stage is the module built in its own phase, so this file only wires them together
namespace qre::risk{
    struct RiskConfig{
        double confidence = 0.99;
        int horizon_days = 1;
        std::size_t n_paths = 100000;
        int max_fill_days = 5;          //longest NaN gap forward filled (data layer)
        int trading_days = 252;
        bool antithetic = true;
    };

    struct RiskReport{
        VarEs historical;
        VarEs parametric;               //zero mean, sigma from the sample covariance
        VarEs monte_carlo;
        double portfolio_vol;           //1-day
        Eigen::VectorXd risk_contributions;   //sums to portfolio_vol
        std::size_t n_returns;
    };

    inline RiskReport run_risk_pipeline(const PriceMatrix& prices, const WeightVector& weights, Rng& rng,
                                        const RiskConfig& cfg = {}){
        if(prices.rows() < 3) throw InvalidInputError("need at least 3 price rows");
        if(prices.cols() != weights.size()) throw InvalidInputError("prices columns must match number of weights");
        const PriceMatrix filled = data::forward_fill(prices, cfg.max_fill_days);
        if(filled.hasNaN()) throw InvalidInputError("prices still contain NaN after forward fill; drop or repair those assets");
        if((filled.array() <= 0.0).any()) throw InvalidInputError("prices must be positive");

        const ReturnMatrix returns = data::simple_return(filled);     //simple returns aggregate across assets
        const CovarianceMatrix cov = linalg::sample_cov(returns);
        const double sigma_p = portfolio_vol(weights, cov);
        return RiskReport{
            historical_var_es(portfolio_returns(returns, weights), cfg.confidence, cfg.horizon_days),
            parametric_var_es(0.0, sigma_p, cfg.confidence, cfg.horizon_days),
            monte_carlo_var_es(weights, cov, cfg.confidence, cfg.horizon_days, cfg.n_paths, rng,
                               cfg.trading_days, cfg.antithetic),
            sigma_p,
            risk_contributions(weights, cov),
            static_cast<std::size_t>(returns.rows())
        };
    }
}
