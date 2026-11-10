#pragma once
#include <algorithm>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include "qre/core/exceptions.hpp"
#include "qre/core/types.hpp"
//Phase 10: covariance estimation when the number of assets N is not small relative to T
//The sample covariance has N(N+1)/2 entries estimated from T observations. As N/T grows its eigenvalues spread:
//the largest are biased up, the smallest down (exactly zero once N >= T), so it becomes ill-conditioned or singular
//and anything that inverts it (minimum-variance weights, Cholesky for simulation) amplifies the noise
//
//Ledoit-Wolf (2004) linear shrinkage towards a scaled identity (matches sklearn.covariance.ledoit_wolf):
//  S = X'X / T (X demeaned, 1/T as in the derivation), mu = tr(S) / N, F = mu I, <A,B> = tr(A B') / N
//  d2 = ||S - F||^2, b2_bar = (1/T^2) sum_t ||x_t x_t' - S||^2, b2 = min(b2_bar, d2), delta = b2 / d2
//  Sigma_LW = delta F + (1 - delta) S  -> every eigenvalue moves towards mu, positive definite whenever delta > 0
//
//Random matrix theory (Marchenko-Pastur) clipping on the correlation matrix:
//  pure noise with q = N/T puts sample correlation eigenvalues in [(1 - sqrt q)^2, (1 + sqrt q)^2]
//  eigenvalues above lambda_plus = (1 + sqrt q)^2 are kept as signal, the rest replaced by their average
//  (trace preserved), rebuilt, rescaled to unit diagonal and converted back with the sample volatilities
namespace qre::linalg{
    struct ShrinkageResult{
        CovarianceMatrix cov;
        double intensity;       //delta in [0, 1]
    };

    struct RmtResult{
        CovarianceMatrix cov;
        double lambda_plus;     //Marchenko-Pastur upper edge
        Eigen::Index n_signal;  //eigenvalues kept
    };

    namespace shrinkage_detail{
        inline Eigen::MatrixXd demean(const ReturnMatrix& r){
            if(r.rows() < 2 || r.cols() < 1) throw InvalidInputError("need T >= 2 observations and N >= 1 assets");
            return r.rowwise() - r.colwise().mean();
        }
    }

    inline ShrinkageResult ledoit_wolf(const ReturnMatrix& returns){
        const Eigen::MatrixXd x = shrinkage_detail::demean(returns);
        const double t = static_cast<double>(x.rows()), n = static_cast<double>(x.cols());
        const Eigen::MatrixXd s = x.transpose() * x / t;
        const double mu = s.trace() / n;
        const Eigen::MatrixXd f = mu * Eigen::MatrixXd::Identity(x.cols(), x.cols());
        const double d2 = (s - f).squaredNorm() / n;
        double b2_bar = 0.0;
        for(Eigen::Index i = 0; i < x.rows(); ++i){
            const Eigen::VectorXd xi = x.row(i).transpose();
            b2_bar += (xi * xi.transpose() - s).squaredNorm() / n;
        }
        b2_bar /= t * t;
        const double delta = d2 == 0.0 ? 0.0 : std::min(b2_bar, d2) / d2;
        return ShrinkageResult{delta * f + (1.0 - delta) * s, delta};
    }

    inline double marchenko_pastur_upper(Eigen::Index n, Eigen::Index t){
        const double q = static_cast<double>(n) / static_cast<double>(t);
        return (1.0 + std::sqrt(q)) * (1.0 + std::sqrt(q));
    }

    inline RmtResult rmt_clip(const ReturnMatrix& returns){
        const Eigen::MatrixXd x = shrinkage_detail::demean(returns);
        const Eigen::MatrixXd s = x.transpose() * x / static_cast<double>(x.rows() - 1);
        const Eigen::VectorXd sd = s.diagonal().cwiseSqrt();
        if((sd.array() <= 0.0).any()) throw InvalidInputError("variances must be positive");
        const Eigen::MatrixXd corr = s.array() / (sd * sd.transpose()).array();
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(corr);
        Eigen::VectorXd vals = es.eigenvalues();
        const double lam_plus = marchenko_pastur_upper(x.cols(), x.rows());
        double noise_sum = 0.0;
        Eigen::Index n_noise = 0;
        for(Eigen::Index i = 0; i < vals.size(); ++i) if(vals(i) <= lam_plus){ noise_sum += vals(i); ++n_noise; }
        if(n_noise > 0) for(Eigen::Index i = 0; i < vals.size(); ++i) if(vals(i) <= lam_plus) vals(i) = noise_sum / static_cast<double>(n_noise);
        Eigen::MatrixXd cleaned = es.eigenvectors() * vals.asDiagonal() * es.eigenvectors().transpose();
        const Eigen::VectorXd csd = cleaned.diagonal().cwiseSqrt();
        cleaned = cleaned.array() / (csd * csd.transpose()).array();
        return RmtResult{cleaned.array() * (sd * sd.transpose()).array(), lam_plus, vals.size() - n_noise};
    }

    //Fully invested minimum-variance weights w = C^-1 1 / (1' C^-1 1); throws unless C is positive definite
    inline Eigen::VectorXd min_variance_weights(const CovarianceMatrix& cov){
        Eigen::LLT<Eigen::MatrixXd> llt(cov);
        if(llt.info() != Eigen::Success) throw InvalidInputError("covariance is not positive definite");
        const Eigen::VectorXd y = llt.solve(Eigen::VectorXd::Ones(cov.rows()));
        return y / y.sum();
    }
}
