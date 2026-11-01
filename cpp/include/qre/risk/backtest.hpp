#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include "qre/core/exceptions.hpp"
#include "qre/core/rng.hpp"
#include "qre/core/types.hpp"
#include "qre/data/point_in_time.hpp"
#include "qre/data/returns.hpp"
#include "qre/linalg/covariance.hpp"
#include "qre/risk/ewma.hpp"
#include "qre/risk/garch.hpp"
#include "qre/risk/var_es.hpp"
//Backtesting: rolling out-of-sample VaR/ES forecasts and the statistical tests that judge them
//Conventions: losses positive; breach on day t means loss_t > VaR_t (strictly); alpha = 1 - confidence
//The forecast for day t uses only price rows <= t-1 (data::pit_window) -> no look-ahead
//Weights are held constant every day (daily rebalanced to fixed weights)
//Methods: Historical, Parametric, MonteCarlo (equal-weight window) and the adaptive EwmaParametric and
//FilteredHistorical (both use EWMA decay lam, see ewma.hpp), and GarchParametric / GarchFiltered
//(GARCH(1,1) refitted every day on the portfolio return window, see garch.hpp), and GarchStudentT
//(GARCH(1,1) with standardized Student-t shocks, nu estimated every day)
//  Kupiec POF        H0: breach rate == alpha                  LR ~ chi2(1)
//  Christoffersen    H0: breaches independent (no clustering)  LR ~ chi2(1)
//  Conditional cov.  both of the above                         LR ~ chi2(2)
//  Traffic light     Basel zones from the binomial cumulative probability of the breach count
//  Z2 (Acerbi-Szekely, loss convention) = sum(L_t 1[L_t > VaR_t] / ES_t) / (T alpha) - 1
//     correct model: E[L 1(L > VaR)] = alpha ES -> E[Z2] = 0; Z2 > 0 means ES was too small
//     descriptive only: its null distribution depends on the model, so no p-value here
namespace qre::risk{
    struct TestResult{
        double lr;          //likelihood-ratio statistic
        double p_value;
    };

    enum class TrafficLight {Green, Yellow, Red};

    inline std::string to_string(TrafficLight z){
        switch(z){
            case TrafficLight::Green: return "green";
            case TrafficLight::Yellow: return "yellow";
            case TrafficLight::Red: return "red";
        }
        return "unknown";
    }

    namespace detail{
        inline double xlogy(double x, double y){ return x == 0.0 ? 0.0 : x * std::log(y); }    //0 ln 0 = 0
        inline double chi2_sf_df1(double x){ return std::erfc(std::sqrt(std::max(x, 0.0) / 2.0)); }
        inline double chi2_sf_df2(double x){ return std::exp(-std::max(x, 0.0) / 2.0); }
        inline void check_alpha(double alpha){
            if(!(alpha > 0.0 && alpha < 1.0)) throw InvalidInputError("alpha must be in (0, 1)");
        }
        //P(X <= x), X ~ Binomial(n, p), summed in log space for stability
        inline double binomial_cdf(std::size_t x, std::size_t n, double p){
            const double dn = static_cast<double>(n);
            double total = 0.0;
            for(std::size_t k = 0; k <= x; ++k){
                const double dk = static_cast<double>(k);
                const double log_pmf = std::lgamma(dn + 1.0) - std::lgamma(dk + 1.0) - std::lgamma(dn - dk + 1.0)
                                       + dk * std::log(p) + (dn - dk) * std::log1p(-p);
                total += std::exp(log_pmf);
            }
            return std::min(total, 1.0);
        }
    }

    //Proportion of failures: is the breach rate consistent with alpha?
    inline TestResult kupiec_pof(std::size_t n_obs, std::size_t n_breaches, double alpha){
        detail::check_alpha(alpha);
        if(n_obs < 1 || n_breaches > n_obs) throw InvalidInputError("need n_obs >= 1 and n_breaches <= n_obs");
        const double n = static_cast<double>(n_obs), x = static_cast<double>(n_breaches);
        const double phat = x / n;
        const double ll_null = detail::xlogy(n - x, 1.0 - alpha) + detail::xlogy(x, alpha);
        const double ll_alt = detail::xlogy(n - x, 1.0 - phat) + detail::xlogy(x, phat);
        const double lr = std::max(-2.0 * (ll_null - ll_alt), 0.0);
        return TestResult{lr, detail::chi2_sf_df1(lr)};
    }

    //First-order Markov test: does a breach today change the chance of a breach tomorrow?
    inline TestResult christoffersen_independence(const std::vector<bool>& breaches){
        if(breaches.size() < 2) throw InvalidInputError("need at least 2 observations");
        double n00 = 0, n01 = 0, n10 = 0, n11 = 0;
        for(std::size_t i = 1; i < breaches.size(); ++i){
            const bool a = breaches[i - 1], b = breaches[i];
            if(!a && !b) ++n00; else if(!a && b) ++n01; else if(a && !b) ++n10; else ++n11;
        }
        const double pi01 = (n00 + n01) > 0 ? n01 / (n00 + n01) : 0.0;
        const double pi11 = (n10 + n11) > 0 ? n11 / (n10 + n11) : 0.0;
        const double pi = (n01 + n11) / (n00 + n01 + n10 + n11);
        const double ll_null = detail::xlogy(n00 + n10, 1.0 - pi) + detail::xlogy(n01 + n11, pi);
        const double ll_alt = detail::xlogy(n00, 1.0 - pi01) + detail::xlogy(n01, pi01)
                            + detail::xlogy(n10, 1.0 - pi11) + detail::xlogy(n11, pi11);
        const double lr = std::max(-2.0 * (ll_null - ll_alt), 0.0);
        return TestResult{lr, detail::chi2_sf_df1(lr)};
    }

    inline TestResult conditional_coverage(const std::vector<bool>& breaches, double alpha){
        const auto x = static_cast<std::size_t>(std::count(breaches.begin(), breaches.end(), true));
        const double lr = kupiec_pof(breaches.size(), x, alpha).lr + christoffersen_independence(breaches).lr;
        return TestResult{lr, detail::chi2_sf_df2(lr)};
    }

    //Basel: green if P(X <= x) < 95%, yellow if < 99.99%, otherwise red
    inline TrafficLight traffic_light(std::size_t n_breaches, std::size_t n_obs = 250, double alpha = 0.01){
        detail::check_alpha(alpha);
        if(n_obs < 1 || n_breaches > n_obs) throw InvalidInputError("need n_obs >= 1 and n_breaches <= n_obs");
        const double cum = detail::binomial_cdf(n_breaches, n_obs, alpha);
        if(cum < 0.95) return TrafficLight::Green;
        if(cum < 0.9999) return TrafficLight::Yellow;
        return TrafficLight::Red;
    }

    inline double es_z2(const std::vector<double>& losses, const std::vector<double>& var,
                        const std::vector<double>& es, double alpha){
        detail::check_alpha(alpha);
        if(losses.empty() || losses.size() != var.size() || losses.size() != es.size()){
            throw InvalidInputError("losses, var and es must be non-empty and the same length");
        }
        double sum = 0.0;
        for(std::size_t i = 0; i < losses.size(); ++i){
            if(es[i] <= 0.0) throw InvalidInputError("ES forecasts must be positive");
            if(losses[i] > var[i]) sum += losses[i] / es[i];
        }
        return sum / (static_cast<double>(losses.size()) * alpha) - 1.0;
    }

    enum class BacktestMethod {Historical, Parametric, MonteCarlo, EwmaParametric, FilteredHistorical,
                               GarchParametric, GarchFiltered, GarchStudentT};

    struct BacktestSeries{
        std::vector<Eigen::Index> rows;     //forecast day t (row index into prices)
        std::vector<double> var;
        std::vector<double> es;
        std::vector<double> loss;           //realized portfolio loss on day t
        std::vector<bool> breaches() const {
            std::vector<bool> b(loss.size());
            for(std::size_t i = 0; i < loss.size(); ++i) b[i] = loss[i] > var[i];
            return b;
        }
    };

    //One forecast per day t = window+1 .. T-1, each from the `window` returns ending at row t-1
    inline BacktestSeries rolling_backtest(const PriceMatrix& prices, const WeightVector& weights,
                                           Eigen::Index window = 250, double confidence = 0.99,
                                           BacktestMethod method = BacktestMethod::Historical,
                                           std::size_t n_paths = 20000, uint64_t seed = 42, int trading_days = 252,
                                           double lam = 0.94){
        if(prices.cols() != weights.size()) throw InvalidInputError("prices columns must match number of weights");
        if(window < 2 || prices.rows() < window + 2) throw InvalidInputError("need window >= 2 and at least window + 2 price rows");
        if(!prices.allFinite() || (prices.array() <= 0.0).any()){
            throw InvalidInputError("prices must be finite and positive; run data hardening first");
        }
        BacktestSeries s;
        for(Eigen::Index t = window + 1; t < prices.rows(); ++t){
            const ReturnMatrix rets = data::simple_return(data::pit_window(prices, t, window));
            VarEs f{};
            if(method == BacktestMethod::Historical){
                f = historical_var_es(portfolio_returns(rets, weights), confidence);
            } else if(method == BacktestMethod::FilteredHistorical){
                f = fhs_var_es(portfolio_returns(rets, weights), confidence, lam);
            } else if(method == BacktestMethod::GarchStudentT){
                f = garch_var_es(portfolio_returns(rets, weights), confidence, false, GarchDist::StudentT);
            } else if(method == BacktestMethod::GarchParametric || method == BacktestMethod::GarchFiltered){
                f = garch_var_es(portfolio_returns(rets, weights), confidence, method == BacktestMethod::GarchFiltered);
            } else if(method == BacktestMethod::EwmaParametric){
                f = ewma_parametric_var_es(rets, weights, confidence, lam);
            } else {
                const CovarianceMatrix cov = linalg::sample_cov(rets);
                if(method == BacktestMethod::Parametric){
                    f = parametric_var_es(0.0, portfolio_vol(weights, cov), confidence);
                } else {
                    Rng rng(seed + static_cast<uint64_t>(t));
                    f = monte_carlo_var_es(weights, cov, confidence, 1, n_paths, rng, trading_days);
                }
            }
            const Eigen::VectorXd day = (prices.row(t).array() / prices.row(t - 1).array() - 1.0).matrix().transpose();
            s.rows.push_back(t);
            s.var.push_back(f.var);
            s.es.push_back(f.es);
            s.loss.push_back(-weights.dot(day));
        }
        return s;
    }

    struct BacktestSummary{
        std::size_t n_obs;
        std::size_t n_breaches;
        double breach_rate;
        TestResult kupiec;
        TestResult christoffersen;
        TestResult conditional;
        TrafficLight traffic_light;     //on the last 250 forecasts (or all, if fewer)
        double z2;
    };

    inline BacktestSummary summarize(const BacktestSeries& s, double confidence){
        const double alpha = 1.0 - confidence;
        const std::vector<bool> b = s.breaches();
        const auto x = static_cast<std::size_t>(std::count(b.begin(), b.end(), true));
        const std::size_t last_n = std::min<std::size_t>(250, b.size());
        const auto last_x = static_cast<std::size_t>(std::count(b.end() - static_cast<std::ptrdiff_t>(last_n), b.end(), true));
        return BacktestSummary{
            b.size(), x, static_cast<double>(x) / static_cast<double>(b.size()),
            kupiec_pof(b.size(), x, alpha), christoffersen_independence(b), conditional_coverage(b, alpha),
            traffic_light(last_x, last_n, alpha), es_z2(s.loss, s.var, s.es, alpha)
        };
    }
}
