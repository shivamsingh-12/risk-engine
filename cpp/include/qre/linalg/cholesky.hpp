#pragma once
#include <Eigen/Dense>
#include "qre/core/types.hpp"
#include "qre/core/exceptions.hpp"
namespace qre::linalg{
    //Decomposes a covariance matrix A into a lower-triangular L where L * L^T = A.
    //Requirement of Cholesky decomp is being positive definite, therefore check and return
    inline Eigen::MatrixXd cholesky_decompose(const CovarianceMatrix& cov){
        //Eigen::LLT requires input to be symmetric - but can still fail, handle manually
        if(!cov.isApprox(cov.transpose())){
            throw InvalidInputError("matrix is not symmetric");
        }
        Eigen::LLT<Eigen::MatrixXd> llt(cov);
        if(llt.info() != Eigen::Success){
            throw InvalidInputError("covariance matrix is not positive-definite");
        }
        return llt.matrixL();
    }
    //If z has covariance I, then L*z has covariance L * I * L^T = L * L^T = A
    inline Eigen::MatrixXd correlated_normals(const Eigen::MatrixXd& L, const Eigen::MatrixXd& ind_draws){
        if(L.cols() != ind_draws.rows()){
            throw InvalidInputError("L and ind_draws have incompatible dimensions");
        }
        return L * ind_draws;
    }
}