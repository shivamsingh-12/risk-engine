//Phase 1 experiment: Does mean/var of samples from N actually converge with the growth of N?
//More specifically at the theoretical value of (1/sqrt(N))?
//Results to be seen in csv file 👍

#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include "qre/core/rng.hpp"
#include "qre/distributions/normal.hpp"
#include "qre/estimators/empirical.hpp"

int main(){
    //Fix parameters
    const double mu = 0.0;
    const double sigma = 1.0;
    const double ana_mean = qre::distributions::normal::mean_analytical(mu,sigma);
    const double ana_var = qre::distributions::normal::var_analytical(mu,sigma);
    //Rng spawns independent streams per iteration
    qre::Rng base_rng;
    //Geometric loop for N to see actual convergence -> linear takes too long
    const std::size_t max_n = 1000000; //note: 1e6 would throw compiler warning
    const std::string output_path = "cpp/experiments/01-probability/results/convergence_results.csv";
    std::ofstream out(output_path);
    if(!out.is_open()){
        std::cerr << "Failed to open file";
        return 1;
    }
    //Formatting
    out << "N,emp_mean,ana_mean,mean_abs_error,"
    "empirical_var,ana_var,var_abs_error\n";
    std::size_t iteration = 0;
    for(std::size_t n = 10; n <= max_n; n*= 10){
        //Independent stream per N like rng.hpp
        qre::Rng stream = base_rng.spawn(iteration);
        ++iteration;
        std::vector<double> samples = qre::distributions::normal::sample(n,mu,sigma,stream);
        const double emp_mean = qre::estimators::empirical::sample_mean(samples);
        const double emp_var = qre::estimators::empirical::sample_var(samples);
        const double mean_error = std::abs(emp_mean - ana_mean);
        const double var_error = std::abs(emp_var - ana_var);
        //Formatting again
        out << n << "," << emp_mean << "," << ana_mean << "," << mean_error << "," << emp_var  << "," << ana_var  << "," << var_error  << "\n";
        //Print to terminal for immediate feedback without CSV
        std::cout << "N=" << n << " mean_error" << mean_error << " var_error" << var_error << "\n";
    }
    out.close();
    std::cout << "\nResults written to " << output_path;
    return 0;
}

