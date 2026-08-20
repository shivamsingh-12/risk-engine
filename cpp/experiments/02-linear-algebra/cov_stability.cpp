//Phase 2 experiment: number of assets (N) grows relative to a fixed amount of observations T
//Check behavior with covariance and cholesky decomposition in 2 different languages
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <fstream>
#include <iostream>
#include <vector>
#include "qre/core/rng.hpp"
#include "qre/core/exceptions.hpp"
#include "qre/distributions/normal.hpp"
#include "qre/linalg/covariance.hpp"
#include "qre/linalg/cholesky.hpp"

Eigen::MatrixXd true_covariance(int n, double rho){
    //Build NxN matrix with val of rho
    Eigen::MatrixXd cov = Eigen::MatrixXd::Constant(n, n, rho);
    //Overwrites diagonal to 1, sets variance to 1 -> matrix represents normalized, standard variables
    cov.diagonal().setOnes();
    return cov;
}
Eigen::MatrixXd ind_draw(int n, int t, qre::Rng& rng){
    Eigen::MatrixXd draws(n, t);
    //once every column calls sample to get n independent normal values
    for(int col = 0; col < t; ++col){
        std::vector<double> samples = qre::distributions::normal::sample(static_cast<std::size_t>(n), 0.0, 1.0, rng);
        //copies n values row by row and coverts
        for(int row = 0; row < n; ++row){
            draws(row, col) = samples[static_cast<std::size_t>(row)];
        }
    }
    return draws;
}
int main(){
    const double rho = 0.3;
    const int t = 50;
    const std::vector<int> ns = {5,10,25,40,50,60};
    const std::string output_path = "cpp/experiments/02-linear-algebra/results/cov_stability_results.csv";
    std::ofstream out(output_path);
    if(!out.is_open()){
        std::cerr << "Failed to open output file: " << output_path << "\n";
        std::cerr << "Make sure you're running this from the repo root.\n";
        return 1;
    }
    out << "N,T,N/T,condition_num,cholesky_success\n";
    qre::Rng base;
    int iteration = 0;
    for(int n : ns){
        qre::Rng stream = base.spawn(static_cast<uint64_t>(iteration));
        ++iteration;
    
    //Cov with Cholesky factor
    Eigen::MatrixXd true_cov = true_covariance(n, rho);
    Eigen::MatrixXd L_true = qre::linalg::cholesky_decompose(true_cov);
    //Independent draws for N x T
    Eigen::MatrixXd ind_draws = ind_draw(n, t, stream);
    Eigen::MatrixXd correlation = qre::linalg::correlated_normals(L_true, ind_draws);
    //transpose
    Eigen::MatrixXd transpose = correlation.transpose();
    Eigen::MatrixXd est_cov = qre::linalg::sample_cov(transpose);
    //Condition number but since matrix is symmetric for covariance -> SelfAdjointEigenSolver
    //Looking for smaller number
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(est_cov);
    Eigen::VectorXd eigenvalues = solver.eigenvalues();
    const double condition_num = eigenvalues(eigenvalues.size() - 1)/ eigenvalues(0);
    //attempt cholesky on estimated covariance
    bool cholesky_success = true;
    try{
        qre::linalg::cholesky_decompose(est_cov);
    } catch(const qre::InvalidInputError&){
        cholesky_success = false;
    }
    //N/T ratio
    const double novert = static_cast<double>(n)/static_cast<double>(t);
    out << n << "," << t << "," << novert << "," << condition_num << "," << (cholesky_success ? "true" : "false") << "\n";
    std::cout << "N=" << n << "N/T" << novert << " condition number=" << condition_num << " cholesky succes:" << (cholesky_success ? "true" : "false") << "\n";
    }
    out.close();
    std::cout << "Results written to cov_stability_results.csv" << "\n";
    return 0;
}