#pragma once
#include <algorithm>
#include <cmath>
#include <optional>
#include <Eigen/Dense>
#include "qre/core/exceptions.hpp"
#include "qre/core/types.hpp"

//Phase 21: portfolio construction. Fully invested weights (sum 1) within per-asset bounds lo <= w <= hi
//(negative lo allows shorts). Python reference: python_reference/qre/portfolio/construction.py (same algorithms)
//  min_variance   min w'Sw                          mean_variance  max mu'w - (gamma/2) w'Sw
//  risk_parity    w_i (Sw)_i equal (long-only)      inverse_vol    w_i proportional to 1/sigma_i
//QPs min 0.5 w'Qw - c'w are solved by projected gradient with Nesterov momentum and adaptive restart (FISTA);
//the projection onto {sum w = 1, lo <= w <= hi} is clip(v - tau, lo, hi) with tau found by bisection.
//Risk parity: cyclical coordinate descent on Spinu's convex problem min 0.5 x'Sx - sum b_i ln x_i
namespace qre::portfolio{
    struct QpResult{
        Eigen::VectorXd weights;
        int iterations;
        bool converged;
    };

    struct Bounds{
        Eigen::VectorXd lo, hi;
    };

    namespace detail{
        inline void check_cov(const CovarianceMatrix& cov){
            if(cov.rows() == 0 || cov.rows() != cov.cols()) throw InvalidInputError("covariance must be a non-empty square matrix");
            const double scale = std::max(1.0, cov.cwiseAbs().maxCoeff());
            if(!cov.allFinite() || (cov - cov.transpose()).cwiseAbs().maxCoeff() > 1e-12 * scale)
                throw InvalidInputError("covariance must be finite and symmetric");
        }
    }

    inline Bounds make_bounds(Eigen::Index n, double lo = 0.0, double hi = 1.0){
        Bounds b{Eigen::VectorXd::Constant(n, lo), Eigen::VectorXd::Constant(n, hi)};
        if(!std::isfinite(lo) || !std::isfinite(hi) || lo > hi) throw InvalidInputError("bounds must be finite with lo <= hi");
        if(b.lo.sum() > 1.0 + 1e-12 || b.hi.sum() < 1.0 - 1e-12)
            throw InvalidInputError("no fully invested portfolio fits these bounds (need sum(lo) <= 1 <= sum(hi))");
        return b;
    }

    //Euclidean projection onto {sum w = total, lo <= w <= hi}
    inline Eigen::VectorXd project(const Eigen::VectorXd& v, const Bounds& b, double total = 1.0){
        double a = (v - b.hi).minCoeff(), z = (v - b.lo).maxCoeff();
        auto clip = [&](double tau){ return (v.array() - tau).max(b.lo.array()).min(b.hi.array()).matrix().eval(); };
        for(int i = 0; i < 200; ++i){
            const double mid = 0.5 * (a + z);
            if(mid <= a || mid >= z) break;
            if(clip(mid).sum() > total) a = mid; else z = mid;
        }
        Eigen::VectorXd w = clip(0.5 * (a + z));
        int free = 0;
        for(Eigen::Index i = 0; i < w.size(); ++i) if(w(i) > b.lo(i) && w(i) < b.hi(i)) ++free;
        if(free > 0){
            const double adj = (total - w.sum()) / free;
            for(Eigen::Index i = 0; i < w.size(); ++i) if(w(i) > b.lo(i) && w(i) < b.hi(i)) w(i) += adj;
        }
        return w;
    }

    inline QpResult solve_qp(const Eigen::MatrixXd& q, const Eigen::VectorXd& c, const Bounds& b,
                             double tol = 1e-12, int max_iter = 200000){
        detail::check_cov(q);
        const Eigen::Index n = q.rows();
        if(c.size() != n || b.lo.size() != n || b.hi.size() != n) throw InvalidInputError("c and bounds need one entry per asset");
        const double lmax = Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(q, Eigen::EigenvaluesOnly).eigenvalues().maxCoeff();
        if(!(lmax > 0.0)) throw InvalidInputError("Q has no positive eigenvalue");
        const double step = 1.0 / lmax;
        Eigen::VectorXd x = project(Eigen::VectorXd::Constant(n, 1.0 / static_cast<double>(n)), b);
        Eigen::VectorXd y = x;
        double t = 1.0;
        const double scale = std::max(1.0, x.cwiseAbs().maxCoeff());
        for(int k = 1; k <= max_iter; ++k){
            const Eigen::VectorXd xn = project(y - step * (q * y - c), b);
            if((xn - x).cwiseAbs().maxCoeff() <= tol * scale) return {xn, k, true};
            if((y - xn).dot(xn - x) > 0.0){
                t = 1.0;
                y = xn;
            } else {
                const double tn = 0.5 * (1.0 + std::sqrt(1.0 + 4.0 * t * t));
                y = xn + ((t - 1.0) / tn) * (xn - x);
                t = tn;
            }
            x = xn;
        }
        return {x, max_iter, false};
    }

    inline Eigen::VectorXd equal_weight(Eigen::Index n){
        if(n < 1) throw InvalidInputError("need at least one asset");
        return Eigen::VectorXd::Constant(n, 1.0 / static_cast<double>(n));
    }

    inline Eigen::VectorXd inverse_vol(const CovarianceMatrix& cov){
        detail::check_cov(cov);
        const Eigen::VectorXd d = cov.diagonal();
        if((d.array() <= 0.0).any()) throw InvalidInputError("every asset needs a positive variance");
        const Eigen::VectorXd iv = d.cwiseSqrt().cwiseInverse();
        return iv / iv.sum();
    }

    inline QpResult min_variance(const CovarianceMatrix& cov, const Bounds& b){
        detail::check_cov(cov);
        return solve_qp(cov, Eigen::VectorXd::Zero(cov.rows()), b);
    }

    inline QpResult mean_variance(const Eigen::VectorXd& mu, const CovarianceMatrix& cov, double gamma, const Bounds& b){
        detail::check_cov(cov);
        if(mu.size() != cov.rows() || !mu.allFinite()) throw InvalidInputError("one finite expected return per asset");
        if(!(gamma > 0.0)) throw InvalidInputError("risk aversion gamma must be positive");
        return solve_qp(gamma * cov, mu, b);
    }

    //w_i (Sw)_i / w'Sw: each asset's share of portfolio variance
    inline Eigen::VectorXd risk_contribution_shares(const Eigen::VectorXd& w, const CovarianceMatrix& cov){
        detail::check_cov(cov);
        const Eigen::VectorXd m = cov * w;
        const double v = w.dot(m);
        if(!(v > 0.0)) throw InvalidInputError("portfolio variance must be positive");
        return w.cwiseProduct(m) / v;
    }

    //Long-only weights with risk contributions proportional to budgets (equal when budgets is empty)
    inline QpResult risk_parity(const CovarianceMatrix& cov, Eigen::VectorXd budgets = {}, double tol = 1e-12,
                                int max_iter = 10000){
        detail::check_cov(cov);
        const Eigen::Index n = cov.rows();
        const Eigen::VectorXd d = cov.diagonal();
        if((d.array() <= 0.0).any()) throw InvalidInputError("every asset needs a positive variance");
        if(budgets.size() == 0) budgets = Eigen::VectorXd::Constant(n, 1.0 / static_cast<double>(n));
        if(budgets.size() != n || (budgets.array() <= 0.0).any() || !budgets.allFinite())
            throw InvalidInputError("one positive budget per asset");
        budgets /= budgets.sum();
        Eigen::VectorXd x = d.cwiseSqrt().cwiseInverse();
        x /= std::sqrt(x.dot(cov * x));
        for(int k = 1; k <= max_iter; ++k){
            for(Eigen::Index i = 0; i < n; ++i){
                const double r = cov.row(i).dot(x) - d(i) * x(i);
                x(i) = (-r + std::sqrt(r * r + 4.0 * d(i) * budgets(i))) / (2.0 * d(i));
            }
            const Eigen::VectorXd w = x / x.sum();
            if((risk_contribution_shares(w, cov) - budgets).cwiseAbs().maxCoeff() <= tol) return {w, k, true};
        }
        return {x / x.sum(), max_iter, false};
    }
}
