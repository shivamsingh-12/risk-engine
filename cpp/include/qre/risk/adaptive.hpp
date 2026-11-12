#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include "qre/core/exceptions.hpp"
#include "qre/core/types.hpp"
#include "qre/risk/backtest.hpp"
#include "qre/risk/scoring.hpp"
//Phase 13: an adaptive risk engine -- run every model side by side, score them daily, combine them
//Each day the weights depend only on the models' past FZ0 scores (no look-ahead):
//  Equal      : average VaR and ES of all models (no learning; the classic hard-to-beat combination)
//  SelectBest : the single model with the lowest mean FZ0 over the last `lookback` days
//  ExpWeights : w_k proportional to exp(-eta S_k), S_k = summed FZ0 over the last `lookback` days ("Hedge");
//               eta -> 0 gives Equal, eta -> infinity gives SelectBest
//Combined forecasts are convex combinations of (VaR, ES) pairs, so ES >= VaR is preserved.
//The first `burn` days only build up scores; evaluation starts after them
namespace qre::risk{
    enum class Combination {Equal, SelectBest, ExpWeights};

    struct ModelForecasts{
        std::vector<BacktestMethod> methods;
        Eigen::MatrixXd var;                    //T x K
        Eigen::MatrixXd es;                     //T x K
        std::vector<double> loss;               //T realized losses
        std::vector<Eigen::Index> rows;         //forecast day (row index into prices)
    };

    struct CombinedForecast{
        std::vector<double> var;                //T - burn
        std::vector<double> es;
        Eigen::MatrixXd weights;                //(T - burn) x K
    };

    inline ModelForecasts run_models(const PriceMatrix& prices, const WeightVector& weights,
                                     const std::vector<BacktestMethod>& methods, Eigen::Index window = 250,
                                     double confidence = 0.99){
        if(methods.empty()) throw InvalidInputError("need at least one model");
        ModelForecasts f;
        f.methods = methods;
        for(std::size_t k = 0; k < methods.size(); ++k){
            const BacktestSeries s = rolling_backtest(prices, weights, window, confidence, methods[k]);
            if(k == 0){
                f.rows = s.rows;
                f.loss = s.loss;
                f.var.resize(static_cast<Eigen::Index>(s.var.size()), static_cast<Eigen::Index>(methods.size()));
                f.es.resize(f.var.rows(), f.var.cols());
            } else if(s.rows != f.rows){
                throw InvalidInputError("models must forecast the same days");
            }
            for(std::size_t i = 0; i < s.var.size(); ++i){
                f.var(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(k)) = s.var[i];
                f.es(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(k)) = s.es[i];
            }
        }
        return f;
    }

    //FZ0 score of every model on every day, T x K
    inline Eigen::MatrixXd model_scores(const Eigen::MatrixXd& var, const Eigen::MatrixXd& es,
                                        const std::vector<double>& loss, double confidence){
        if(es.rows() != var.rows() || es.cols() != var.cols() || static_cast<Eigen::Index>(loss.size()) != var.rows()){
            throw InvalidInputError("var and es must be T x K and loss of length T");
        }
        Eigen::MatrixXd s(var.rows(), var.cols());
        for(Eigen::Index t = 0; t < var.rows(); ++t){
            for(Eigen::Index k = 0; k < var.cols(); ++k){
                s(t, k) = fz0_loss(loss[static_cast<std::size_t>(t)], var(t, k), es(t, k), confidence);
            }
        }
        return s;
    }

    //Weights for day t from the scores of days max(0, t - lookback) .. t-1 (t = scores.rows() -> the next day)
    inline Eigen::VectorXd combination_weights(const Eigen::MatrixXd& scores, Eigen::Index t, Combination scheme,
                                               Eigen::Index lookback = 100, double eta = 1.0){
        const Eigen::Index K = scores.cols();
        if(t < 1 || t > scores.rows() || lookback < 1 || eta < 0.0) throw InvalidInputError("need 1 <= t <= T, lookback >= 1 and eta >= 0");
        const Eigen::Index start = std::max<Eigen::Index>(0, t - lookback);
        const Eigen::VectorXd sum = scores.middleRows(start, t - start).colwise().sum().transpose();
        Eigen::VectorXd w(K);
        if(scheme == Combination::Equal){
            w.setConstant(1.0 / static_cast<double>(K));
        } else if(scheme == Combination::SelectBest){
            Eigen::Index best = 0;
            sum.minCoeff(&best);            //first minimum, like numpy argmin
            w.setZero();
            w(best) = 1.0;
        } else {
            w = (-eta * (sum.array() - sum.minCoeff())).exp().matrix();
            w /= w.sum();
        }
        return w;
    }

    inline CombinedForecast combine(const Eigen::MatrixXd& var, const Eigen::MatrixXd& es, const std::vector<double>& loss,
                                    double confidence, Combination scheme = Combination::Equal, Eigen::Index lookback = 100,
                                    double eta = 1.0, Eigen::Index burn = 100){
        const Eigen::MatrixXd scores = model_scores(var, es, loss, confidence);
        const Eigen::Index T = var.rows(), K = var.cols();
        if(burn < 1 || burn >= T || lookback < 1 || eta < 0.0) throw InvalidInputError("need 1 <= burn < T, lookback >= 1 and eta >= 0");
        CombinedForecast c{std::vector<double>(static_cast<std::size_t>(T - burn)), std::vector<double>(static_cast<std::size_t>(T - burn)),
                           Eigen::MatrixXd(T - burn, K)};
        for(Eigen::Index t = burn; t < T; ++t){
            const Eigen::VectorXd w = combination_weights(scores, t, scheme, lookback, eta);
            const auto i = static_cast<std::size_t>(t - burn);
            c.weights.row(t - burn) = w.transpose();
            c.var[i] = (w.array() * var.row(t).transpose().array()).sum();
            c.es[i] = (w.array() * es.row(t).transpose().array()).sum();
        }
        return c;
    }

    //Package forecasts from day `burn` on as a BacktestSeries, so summarize() and the coverage tests apply
    inline BacktestSeries as_series(const ModelForecasts& f, const std::vector<double>& var, const std::vector<double>& es,
                                    Eigen::Index burn){
        const auto b = static_cast<std::size_t>(burn);
        if(b + var.size() != f.loss.size() || es.size() != var.size()) throw InvalidInputError("forecasts do not line up with the evaluation days");
        BacktestSeries s;
        s.rows.assign(f.rows.begin() + static_cast<std::ptrdiff_t>(b), f.rows.end());
        s.loss.assign(f.loss.begin() + static_cast<std::ptrdiff_t>(b), f.loss.end());
        s.var = var;
        s.es = es;
        return s;
    }

    inline BacktestSeries model_series(const ModelForecasts& f, Eigen::Index k, Eigen::Index burn){
        std::vector<double> v, e;
        for(Eigen::Index t = burn; t < f.var.rows(); ++t){ v.push_back(f.var(t, k)); e.push_back(f.es(t, k)); }
        return as_series(f, v, e, burn);
    }
}
