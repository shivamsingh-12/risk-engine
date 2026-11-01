#pragma once
#include <cmath>
#include <limits>
#include <optional>
#include <utility>
#include <vector>
#include <Eigen/Dense>
#include "qre/core/types.hpp"
#include "qre/data/corporate_actions.hpp"
//Data hardening: detect everything, repair only what is unambiguous
//  repaired: non-positive prices and spike-and-revert bad ticks -> missing, then short gaps filled
//            within each asset's listed life
//  flagged only: split candidates (could be a real crash) and stale runs (nothing to recover them from)
//Unlike cleaning.hpp's forward_fill, nothing is filled before listing or after delisting, and no asset is
//dropped for having a partial history (dropping names that later delisted is survivorship bias)
namespace qre::data{
    struct StaleRun{
        Eigen::Index asset;
        Eigen::Index start;     //first row that repeats the previous price
        Eigen::Index length;    //number of consecutive repeated rows
    };

    struct BadTick{
        Eigen::Index asset;
        Eigen::Index row;
    };

    struct Lifetime{
        Eigen::Index first;     //first valid row (listing)
        Eigen::Index last;      //last valid row (delisting or end of data)
    };

    inline std::vector<StaleRun> stale_runs(const PriceMatrix& prices, Eigen::Index min_run = 5){
        std::vector<StaleRun> out;
        for(Eigen::Index c = 0; c < prices.cols(); ++c){
            Eigen::Index start = -1, length = 0;
            for(Eigen::Index r = 1; r <= prices.rows(); ++r){
                const bool same = r < prices.rows() && std::isfinite(prices(r, c)) && std::isfinite(prices(r - 1, c))
                                  && prices(r, c) == prices(r - 1, c);
                if(same){
                    if(length == 0) start = r;
                    ++length;
                } else {
                    if(length >= min_run) out.push_back(StaleRun{c, start, length});
                    length = 0;
                }
            }
        }
        return out;
    }

    //Spike-and-revert: a big log move that reverses the next day back to near the prior level
    inline std::vector<BadTick> bad_ticks(const PriceMatrix& prices, double threshold = 0.15){
        std::vector<BadTick> out;
        for(Eigen::Index c = 0; c < prices.cols(); ++c){
            for(Eigen::Index r = 1; r + 1 < prices.rows(); ++r){
                const double p0 = prices(r - 1, c), p1 = prices(r, c), p2 = prices(r + 1, c);
                if(!std::isfinite(p0) || !std::isfinite(p1) || !std::isfinite(p2)) continue;
                if(p0 <= 0.0 || p1 <= 0.0 || p2 <= 0.0) continue;
                const double a = std::log(p1 / p0), b = std::log(p2 / p1), back = std::log(p2 / p0);
                if(std::abs(a) > threshold && std::abs(b) > threshold && a * b < 0.0 && std::abs(back) < threshold / 2.0){
                    out.push_back(BadTick{c, r});
                }
            }
        }
        return out;
    }

    inline std::vector<std::optional<Lifetime>> lifetimes(const PriceMatrix& prices){
        std::vector<std::optional<Lifetime>> out;
        for(Eigen::Index c = 0; c < prices.cols(); ++c){
            Eigen::Index first = -1, last = -1;
            for(Eigen::Index r = 0; r < prices.rows(); ++r){
                if(std::isfinite(prices(r, c))){
                    if(first < 0) first = r;
                    last = r;
                }
            }
            if(first < 0) out.emplace_back(std::nullopt);
            else out.emplace_back(Lifetime{first, last});
        }
        return out;
    }

    //Fill gaps of at most max_consec_days, only strictly inside an asset's listed life
    inline PriceMatrix forward_fill_within_life(const PriceMatrix& prices, Eigen::Index max_consec_days){
        PriceMatrix out = prices;
        const auto lives = lifetimes(prices);
        for(Eigen::Index c = 0; c < out.cols(); ++c){
            const auto& life = lives[static_cast<std::size_t>(c)];
            if(!life) continue;
            Eigen::Index r = life->first + 1;
            while(r <= life->last){
                if(std::isnan(out(r, c))){
                    Eigen::Index end = r;
                    while(std::isnan(out(end, c))) ++end;          //terminates: row life->last is valid
                    if(end - r <= max_consec_days) out.block(r, c, end - r, 1).setConstant(out(r - 1, c));
                    r = end;
                } else {
                    ++r;
                }
            }
        }
        return out;
    }

    struct QualityReport{
        std::vector<std::pair<Eigen::Index, Eigen::Index>> non_positive;   //(asset, row)
        std::vector<Eigen::Index> leading_missing;        //per asset: rows before listing
        std::vector<Eigen::Index> trailing_missing;       //per asset: rows after delisting
        std::vector<Eigen::Index> internal_missing;       //per asset: gaps inside the listed life (before filling)
        std::vector<Eigen::Index> remaining_internal_missing;   //after filling
        std::vector<StaleRun> stale;
        std::vector<BadTick> bad_ticks;
        std::vector<SplitCandidate> split_candidates;
        bool clean() const {
            for(auto v : internal_missing) if(v) return false;
            return non_positive.empty() && stale.empty() && bad_ticks.empty() && split_candidates.empty();
        }
    };

    struct HardenConfig{
        Eigen::Index max_fill_days = 5;
        Eigen::Index min_stale_run = 5;
        double bad_tick_threshold = 0.15;   //log move
        double split_min_move = 0.25;
        double split_tol = 0.03;
    };

    struct HardenResult{
        PriceMatrix prices;
        QualityReport report;
    };

    namespace detail{
        inline void missing_counts(const PriceMatrix& prices, std::vector<Eigen::Index>& lead,
                                   std::vector<Eigen::Index>& trail, std::vector<Eigen::Index>& internal){
            lead.clear(); trail.clear(); internal.clear();
            const auto lives = lifetimes(prices);
            for(Eigen::Index c = 0; c < prices.cols(); ++c){
                const auto& life = lives[static_cast<std::size_t>(c)];
                if(!life){
                    lead.push_back(prices.rows()); trail.push_back(0); internal.push_back(0);
                    continue;
                }
                lead.push_back(life->first);
                trail.push_back(prices.rows() - 1 - life->last);
                Eigen::Index n = 0;
                for(Eigen::Index r = life->first; r <= life->last; ++r) if(std::isnan(prices(r, c))) ++n;
                internal.push_back(n);
            }
        }
    }

    inline HardenResult harden(const PriceMatrix& raw, const HardenConfig& cfg = {}){
        HardenResult res{raw, {}};
        QualityReport& rep = res.report;
        const double nan = std::numeric_limits<double>::quiet_NaN();
        for(Eigen::Index c = 0; c < raw.cols(); ++c){
            for(Eigen::Index r = 0; r < raw.rows(); ++r){
                if(raw(r, c) <= 0.0){
                    rep.non_positive.emplace_back(c, r);
                    res.prices(r, c) = nan;
                }
            }
        }
        rep.split_candidates = detect_split_candidates(res.prices, cfg.split_min_move, cfg.split_tol);
        rep.stale = stale_runs(res.prices, cfg.min_stale_run);
        rep.bad_ticks = bad_ticks(res.prices, cfg.bad_tick_threshold);
        for(const auto& b : rep.bad_ticks) res.prices(b.row, b.asset) = nan;
        std::vector<Eigen::Index> dummy_lead, dummy_trail;
        detail::missing_counts(res.prices, rep.leading_missing, rep.trailing_missing, rep.internal_missing);
        res.prices = forward_fill_within_life(res.prices, cfg.max_fill_days);
        detail::missing_counts(res.prices, dummy_lead, dummy_trail, rep.remaining_internal_missing);
        return res;
    }
}
