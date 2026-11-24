#pragma once
#include <cmath>
#include <vector>
#include <Eigen/Dense>
#include "qre/core/exceptions.hpp"
#include "qre/core/types.hpp"
#include "qre/distributions/normal.hpp"
#include "qre/risk/var_es.hpp"

//Phase 21: component VaR and ES (Euler allocation). Python reference: python_reference/qre/risk/component.py
//Parametric (normal, zero mean): C_i = k w_i (Sw)_i / sigma_p with k = z for VaR, phi(z)/(1-c) for ES
//Historical: component ES_i = mean of -w_i r_t,i over the tail days (exact); component VaR_i = Gaussian-kernel
//weighted mean around the VaR (Epperlein-Smillie), bandwidth 1.06 sd(L) T^-1/5, rescaled to sum to the VaR
namespace qre::risk{
    struct ComponentRisk{
        double var;
        double es;
        Eigen::VectorXd component_var;
        Eigen::VectorXd component_es;
        int tail_days = 0;
    };

    namespace component_detail{
        inline void check(const Eigen::VectorXd& w, double confidence){
            if(w.size() == 0 || !w.allFinite()) throw InvalidInputError("weights must be a non-empty finite vector");
            if(!(confidence >= 0.5 && confidence < 1.0)) throw InvalidInputError("confidence must be in [0.5, 1)");
        }
    }

    inline ComponentRisk parametric_components(const WeightVector& w, const CovarianceMatrix& cov, double confidence){
        component_detail::check(w, confidence);
        if(cov.rows() != w.size() || cov.cols() != w.size()) throw InvalidInputError("cov must be square and match the number of weights");
        const Eigen::VectorXd m = cov * w;
        const double v = w.dot(m);
        if(!(v > 0.0)) throw InvalidInputError("portfolio variance must be positive");
        const double s = std::sqrt(v);
        const double z = detail::normal_quantile(confidence);
        const double k_es = distributions::normal::pdf(z, 0.0, 1.0) / (1.0 - confidence);
        const Eigen::VectorXd share = w.cwiseProduct(m) / s;
        return ComponentRisk{z * s, k_es * s, z * share, k_es * share, 0};
    }

    //returns: T x N asset return scenarios
    inline ComponentRisk historical_components(const ReturnMatrix& returns, const WeightVector& w, double confidence){
        component_detail::check(w, confidence);
        const Eigen::Index t = returns.rows(), n = returns.cols();
        if(n != w.size() || t < 20 || !returns.allFinite())
            throw InvalidInputError("returns must be finite, T x N with T >= 20 and one column per weight");
        const Eigen::MatrixXd pos = -(returns.array().rowwise() * w.transpose().array()).matrix();
        const Eigen::VectorXd loss = pos.rowwise().sum();
        const VarEs ve = var_es_from_losses(std::vector<double>(loss.data(), loss.data() + t), confidence);
        ComponentRisk out{ve.var, ve.es, Eigen::VectorXd::Zero(n), Eigen::VectorXd::Zero(n), 0};
        for(Eigen::Index i = 0; i < t; ++i){
            if(loss(i) >= ve.var){ out.component_es += pos.row(i).transpose(); ++out.tail_days; }
        }
        out.component_es /= static_cast<double>(out.tail_days);
        const double mean = loss.mean();
        const double sd = std::sqrt((loss.array() - mean).square().sum() / static_cast<double>(t - 1));
        const double h = sd > 0.0 ? 1.06 * sd * std::pow(static_cast<double>(t), -0.2) : 1.0;
        Eigen::VectorXd raw = Eigen::VectorXd::Zero(n);
        double ksum = 0.0;
        for(Eigen::Index i = 0; i < t; ++i){
            const double u = (loss(i) - ve.var) / h;
            const double k = std::exp(-0.5 * u * u);
            raw += k * pos.row(i).transpose();
            ksum += k;
        }
        raw /= ksum;
        const double total = raw.sum();
        out.component_var = total != 0.0 ? (raw * (ve.var / total)).eval() : raw;
        return out;
    }
}
