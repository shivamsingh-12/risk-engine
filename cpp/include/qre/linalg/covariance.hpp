#pragma once
#include <Eigen/Dense>
#include "qre/core/types.hpp"
#include "qre/core/exceptions.hpp"
//Calculates sample covariance using Eigen library
namespace qre::linalg{
    inline CovarianceMatrix sample_covariance(const ReturnMatrix& returns){
        //Basic validation checks to make sure it fits, 2 rows -> time steps, 1 column -> asset
        if(returns.rows() < 2){
            throw InvalidInputError("need at least 2 observations");
        }
        if(returns.cols() < 1){
            throw InvalidInputError("need at least 1 asset column");
        }
        //avg return for each asset -> row vector of means
        const Eigen::RowVectorXd col_means = returns.colwise().mean();
        //centers data around zero
        const Eigen::MatrixXd demean = returns.rowwise() - col_means;
        //total count
        const double n = static_cast<double>(returns.rows());
        //Sum of cross products then divided for sample covariance formula
        //n-1 for bessel's correction, 1.0 because n is a double
        return (demean.transpose() * demean)/(n-1.0);
    }
}