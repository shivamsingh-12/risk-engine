#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>
#include <Eigen/Dense>
#include "qre/core/exceptions.hpp"
#include "qre/core/types.hpp"
#include "qre/risk/var_es.hpp"

//Phase 20: stress testing. Historical replay of past crises, hypothetical shocks by Gaussian conditioning, and
//reverse stress tests. Python reference: python_reference/qre/risk/stress.py (same names, same numbers)
//
//Historical replay: each asset's actual daily returns over a crisis window, applied to today's portfolio held
//buy-and-hold. An asset with no prices then is proxied day by day: r_i,t = beta_i * r_proxy,t.
//Hypothetical shock: E[r_R | r_S = s] = Sigma_RS Sigma_SS^-1 s, with Sigma_RR - Sigma_RS Sigma_SS^-1 Sigma_SR left
//over (scaled to the shock's horizon). Reverse stress: the market return s = -L / beta_p that loses L on average
namespace qre::risk{
    struct Scenario{
        std::string_view key, name, start, end;
    };

    //Market peak to trough (S&P 500) unless noted
    inline constexpr Scenario kScenarios[] = {
        {"gfc", "Global financial crisis", "2007-10-09", "2009-03-09"},
        {"lehman", "Lehman collapse", "2008-09-12", "2008-11-20"},
        {"downgrade_2011", "US downgrade and euro crisis", "2011-07-22", "2011-10-03"},
        {"china_oil", "China devaluation and oil crash", "2015-08-17", "2016-02-11"},
        {"volmageddon", "February 2018 volatility spike", "2018-01-26", "2018-02-08"},
        {"q4_2018", "Q4 2018 sell-off", "2018-09-20", "2018-12-24"},
        {"covid", "COVID crash", "2020-02-19", "2020-03-23"},
        {"rates_2022", "2022 inflation and rate shock", "2022-01-03", "2022-10-12"},
        {"banks_2023", "2023 regional bank failures", "2023-03-08", "2023-03-17"},     //bank stocks, not the index
        {"tariffs_2025", "April 2025 tariff shock", "2025-04-02", "2025-04-08"},
    };

    inline const Scenario& find_scenario(std::string_view key){
        for(const auto& s : kScenarios) if(s.key == key) return s;
        throw InvalidInputError("unknown scenario '" + std::string(key) + "'");
    }

    //Rows of the last dates on or before start and end (ISO dates, increasing)
    inline std::pair<Eigen::Index, Eigen::Index> window_indices(const std::vector<std::string>& dates,
                                                                std::string_view start, std::string_view end){
        if(start >= end) throw InvalidInputError("scenario start must be before its end");
        auto last_on_or_before = [&](std::string_view d){
            const auto it = std::upper_bound(dates.begin(), dates.end(), d,
                                             [](std::string_view a, const std::string& b){ return a < b; });
            return static_cast<Eigen::Index>(it - dates.begin()) - 1;
        };
        const Eigen::Index i0 = last_on_or_before(start), i1 = last_on_or_before(end);
        if(i0 < 0) throw InvalidInputError("data starts " + (dates.empty() ? std::string("never") : dates.front()) +
                                           ", after the scenario start " + std::string(start));
        if(i1 <= i0) throw InvalidInputError("no trading days between " + std::string(start) + " and " + std::string(end));
        return {i0, i1};
    }

    //OLS slope of asset on proxy over days where both are finite (at least 20)
    inline double beta(const std::vector<double>& asset, const std::vector<double>& proxy){
        if(asset.size() != proxy.size()) throw InvalidInputError("asset and proxy returns must have the same length");
        double sa = 0.0, sp = 0.0;
        std::size_t n = 0;
        for(std::size_t i = 0; i < asset.size(); ++i){
            if(std::isfinite(asset[i]) && std::isfinite(proxy[i])){ sa += asset[i]; sp += proxy[i]; ++n; }
        }
        if(n < 20) throw InvalidInputError("need at least 20 days on which both the asset and the proxy trade to estimate a beta");
        const double ma = sa / static_cast<double>(n), mp = sp / static_cast<double>(n);
        double sxy = 0.0, sxx = 0.0;
        for(std::size_t i = 0; i < asset.size(); ++i){
            if(std::isfinite(asset[i]) && std::isfinite(proxy[i])){
                sxy += (proxy[i] - mp) * (asset[i] - ma);
                sxx += (proxy[i] - mp) * (proxy[i] - mp);
            }
        }
        if(sxx <= 0.0) throw InvalidInputError("proxy returns have no variance");
        return sxy / sxx;
    }

    struct ScenarioReturns{
        Eigen::MatrixXd returns;                //n x N daily simple returns
        std::vector<bool> proxied;              //asset needed the proxy on at least one day
        std::vector<int> proxied_days;
    };

    //prices (n+1) x N over the window, NaN where an asset did not trade; proxy n+1 finite closes; one beta per asset
    inline ScenarioReturns scenario_returns(const Eigen::MatrixXd& prices, const Eigen::VectorXd& proxy,
                                            const Eigen::VectorXd& betas){
        const Eigen::Index n1 = prices.rows(), na = prices.cols();
        if(n1 < 2 || proxy.size() != n1 || betas.size() != na)
            throw InvalidInputError("need (n+1) x N prices, n+1 proxy prices and N betas, n >= 1");
        if(!proxy.allFinite() || (proxy.array() <= 0.0).any())
            throw InvalidInputError("proxy prices must be finite and positive over the whole window");
        for(Eigen::Index i = 0; i < n1; ++i)
            for(Eigen::Index j = 0; j < na; ++j)
                if(std::isfinite(prices(i, j)) && prices(i, j) <= 0.0) throw InvalidInputError("prices must be positive");
        ScenarioReturns out{Eigen::MatrixXd(n1 - 1, na), std::vector<bool>(static_cast<std::size_t>(na), false),
                            std::vector<int>(static_cast<std::size_t>(na), 0)};
        for(Eigen::Index t = 1; t < n1; ++t){
            const double rp = proxy(t) / proxy(t - 1) - 1.0;
            for(Eigen::Index j = 0; j < na; ++j){
                const double own = prices(t, j) / prices(t - 1, j) - 1.0;
                if(std::isfinite(own)){
                    out.returns(t - 1, j) = own;
                } else {
                    if(!std::isfinite(betas(j))) throw InvalidInputError("an asset with missing prices needs a finite beta");
                    out.returns(t - 1, j) = betas(j) * rp;
                    out.proxied[static_cast<std::size_t>(j)] = true;
                    ++out.proxied_days[static_cast<std::size_t>(j)];
                }
            }
        }
        return out;
    }

    struct Replay{
        double total_return;
        Eigen::VectorXd contributions;      //w_i (growth_i - 1); sums to total_return
        Eigen::VectorXd asset_returns;
        Eigen::VectorXd value;              //n + 1 points from 1
        double worst_day;
        Eigen::Index worst_day_index;       //1..n
        double worst_10d;
        double max_drawdown;
        Eigen::Index days;
    };

    //Buy-and-hold replay. Weights are fractions of starting capital: negative is short, the rest is cash
    inline Replay replay(const Eigen::MatrixXd& returns, const Eigen::VectorXd& weights){
        const Eigen::Index n = returns.rows(), na = returns.cols();
        if(n < 1 || weights.size() != na) throw InvalidInputError("returns must be n x N with n >= 1, and weights N");
        if(!returns.allFinite() || (returns.array() <= -1.0).any())
            throw InvalidInputError("returns must be finite and above -100%");
        Eigen::VectorXd growth = Eigen::VectorXd::Ones(na);
        Eigen::VectorXd value(n + 1);
        const double cash = 1.0 - weights.sum();
        value(0) = 1.0;
        for(Eigen::Index t = 0; t < n; ++t){
            growth.array() *= (1.0 + returns.row(t).transpose().array());
            value(t + 1) = cash + growth.dot(weights);
        }
        Replay r;
        r.asset_returns = growth.array() - 1.0;
        r.contributions = weights.cwiseProduct(r.asset_returns);
        r.total_return = value(n) - 1.0;
        r.worst_day = std::numeric_limits<double>::infinity();
        r.worst_day_index = 0;
        for(Eigen::Index t = 1; t <= n; ++t){
            const double d = value(t) / value(t - 1) - 1.0;
            if(d < r.worst_day){ r.worst_day = d; r.worst_day_index = t; }
        }
        const Eigen::Index k = std::min<Eigen::Index>(10, n);
        r.worst_10d = std::numeric_limits<double>::infinity();
        for(Eigen::Index t = k; t <= n; ++t) r.worst_10d = std::min(r.worst_10d, value(t) / value(t - k) - 1.0);
        double peak = value(0);
        r.max_drawdown = 0.0;
        for(Eigen::Index t = 0; t <= n; ++t){
            peak = std::max(peak, value(t));
            r.max_drawdown = std::max(r.max_drawdown, 1.0 - value(t) / peak);
        }
        r.value = std::move(value);
        r.days = n;
        return r;
    }

    struct Shock{
        Eigen::VectorXd mean;               //expected return given the shock (shocked assets: the shock)
        Eigen::MatrixXd cond_cov;           //covariance left given the shock, at the shock's horizon
        std::vector<Eigen::Index> shocked;
    };

    //Gaussian conditioning on r_S = s; cov is the covariance of daily simple returns
    inline Shock conditional_shock(const CovarianceMatrix& cov, const std::vector<Eigen::Index>& shocked,
                                   const std::vector<double>& shocks, double horizon_days = 1.0){
        const Eigen::Index n = cov.rows();
        if(n == 0 || cov.cols() != n || (cov - cov.transpose()).cwiseAbs().maxCoeff() > 1e-14)
            throw InvalidInputError("cov must be a symmetric square matrix");
        if(shocked.empty() || shocked.size() != shocks.size())
            throw InvalidInputError("give one shock per shocked asset, at least one, no repeats");
        std::unordered_set<Eigen::Index> seen;
        for(const auto i : shocked){
            if(!seen.insert(i).second) throw InvalidInputError("give one shock per shocked asset, at least one, no repeats");
            if(i < 0 || i >= n) throw InvalidInputError("shocked index out of range");
        }
        for(const double s : shocks) if(!std::isfinite(s) || s <= -1.0) throw InvalidInputError("shocks must be finite returns above -100%");
        if(!(horizon_days > 0.0)) throw InvalidInputError("horizon must be positive");
        const auto k = static_cast<Eigen::Index>(shocked.size());
        Eigen::MatrixXd sss(k, k), a(n, k);
        Eigen::VectorXd s(k);
        for(Eigen::Index p = 0; p < k; ++p){
            const auto ip = shocked[static_cast<std::size_t>(p)];
            s(p) = shocks[static_cast<std::size_t>(p)];
            a.col(p) = cov.col(ip);
            for(Eigen::Index q = 0; q < k; ++q) sss(p, q) = cov(ip, shocked[static_cast<std::size_t>(q)]);
        }
        const Eigen::LLT<Eigen::MatrixXd> llt(sss);
        if(llt.info() != Eigen::Success)
            throw InvalidInputError("the shocked assets' covariance is singular (two shocked assets move identically?)");
        Shock out;
        out.shocked = shocked;
        out.mean = a * llt.solve(s);
        for(Eigen::Index p = 0; p < k; ++p) out.mean(shocked[static_cast<std::size_t>(p)]) = s(p);
        out.mean = out.mean.cwiseMax(-1.0);
        Eigen::MatrixXd cond = (cov - a * llt.solve(a.transpose())) * horizon_days;
        cond = 0.5 * (cond + cond.transpose());
        for(const auto i : shocked){ cond.row(i).setZero(); cond.col(i).setZero(); }
        out.cond_cov = std::move(cond);
        return out;
    }

    struct ShockPnl{
        double expected_return;
        double stdev;
        double loss_at_confidence;          //-expected + z stdev
        Eigen::VectorXd contributions;      //w_i mean_i
    };

    inline ShockPnl shock_pnl(const WeightVector& weights, const Shock& shock, double confidence = 0.99){
        if(weights.size() != shock.mean.size()) throw InvalidInputError("one weight per asset in the covariance");
        if(!(confidence >= 0.5 && confidence < 1.0)) throw InvalidInputError("confidence must be in [0.5, 1)");
        ShockPnl p;
        p.expected_return = weights.dot(shock.mean);
        p.stdev = std::sqrt(std::max(weights.dot(shock.cond_cov * weights), 0.0));
        p.loss_at_confidence = -p.expected_return + detail::normal_quantile(confidence) * p.stdev;
        p.contributions = weights.cwiseProduct(shock.mean);
        return p;
    }

    inline double portfolio_beta(const WeightVector& weights, const CovarianceMatrix& cov, Eigen::Index market){
        if(market < 0 || market >= cov.rows()) throw InvalidInputError("market index out of range");
        if(cov(market, market) <= 0.0) throw InvalidInputError("market variance must be positive");
        return weights.dot(cov.col(market)) / cov(market, market);
    }

    //Market return with an expected portfolio loss of target_loss; nullopt if the portfolio's market beta is <= 0
    inline std::optional<double> reverse_stress(const WeightVector& weights, const CovarianceMatrix& cov,
                                                Eigen::Index market, double target_loss){
        if(!(target_loss > 0.0)) throw InvalidInputError("target loss must be positive");
        const double b = portfolio_beta(weights, cov, market);
        if(b <= 0.0) return std::nullopt;
        return -target_loss / b;
    }
}
