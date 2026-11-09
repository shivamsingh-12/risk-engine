#pragma once
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include "qre/core/exceptions.hpp"
#include "qre/core/types.hpp"
//Phase 10: principal component analysis of a covariance (or correlation) matrix
//C = V diag(lambda) V', eigenvalues sorted largest first. Column k of V is the k-th principal component (a portfolio
//of the assets), lambda_k its variance, lambda_k / sum(lambda) the share of total variance it explains.
//For equities the first component is usually close to an equal-weighted "market" portfolio.
//Eigenvector signs are arbitrary; fixed here so each component's loadings sum to >= 0
namespace qre::stats{
    struct PcaResult{
        Eigen::VectorXd eigenvalues;    //descending
        Eigen::MatrixXd components;     //columns are eigenvectors, same order
        Eigen::VectorXd explained;      //share of total variance per component
    };

    inline PcaResult pca(const CovarianceMatrix& cov){
        if(cov.rows() < 1 || cov.rows() != cov.cols()) throw InvalidInputError("need a square matrix");
        if(!cov.isApprox(cov.transpose())) throw InvalidInputError("matrix is not symmetric");
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(cov);
        const Eigen::Index n = cov.rows();
        PcaResult r{Eigen::VectorXd(n), Eigen::MatrixXd(n, n), Eigen::VectorXd(n)};
        for(Eigen::Index k = 0; k < n; ++k){                     //eigen returns ascending -> reverse
            r.eigenvalues(k) = es.eigenvalues()(n - 1 - k);
            Eigen::VectorXd v = es.eigenvectors().col(n - 1 - k);
            if(v.sum() < 0.0) v = -v;
            r.components.col(k) = v;
        }
        const double total = r.eigenvalues.sum();
        if(total <= 0.0) throw InvalidInputError("total variance must be positive");
        r.explained = r.eigenvalues / total;
        return r;
    }
}
