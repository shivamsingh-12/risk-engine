#pragma once
#include <Eigen/Dense>
namespace qre{
        //Custom names for the Eigen types for more feasible readability
        //Note: these are aliases, not new types
        //Compiler just sees as Eigen vectors or matrices
    using PriceSeries = Eigen::VectorXd;
    using ReturnVector = Eigen::VectorXd;
    using ReturnMatrix = Eigen::MatrixXd;
    using CovarianceMatrix = Eigen::MatrixXd;
    using CorrelationMatrix = Eigen::MatrixXd;
    using WeightVector = Eigen::VectorXd;
}