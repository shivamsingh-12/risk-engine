//Phase 8 experiment: stochastic calculus, checked numerically
//Part 1: convergence orders of Euler-Maruyama and Milstein on GBM (exact solution known). Strong error uses the
//        same Brownian path at every step size; weak error uses E[X_T^2] and is compared with the exact EM bias
//Part 2: Ito calculus. Quadratic variation of Brownian motion -> T (with variance 2T^2/n), and the Ito and
//        Stratonovich integrals of W dW differ by half the quadratic variation -> T/2
//Part 3: Merton jump-diffusion. Monte Carlo vs closed form across strikes and maturities, and the implied
//        volatility smile the jumps create (Phase 5's implied vol solver on Merton prices)
//Part 4: Ornstein-Uhlenbeck. How well can mean reversion be estimated from 1, 5 and 25 years of daily data?
//C++ only: every function is cross-validated against the Python reference in both test suites
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "qre/models/merton.hpp"
#include "qre/models/ou.hpp"
#include "qre/montecarlo/sde.hpp"
#include "qre/pricing/black_scholes.hpp"
using namespace qre;

namespace {
double slope(const std::vector<double>& h, const std::vector<double>& e){
    double mx = 0, my = 0;
    for(std::size_t i = 0; i < h.size(); ++i){ mx += std::log(h[i]); my += std::log(e[i]); }
    mx /= static_cast<double>(h.size()); my /= static_cast<double>(h.size());
    double sxy = 0, sxx = 0;
    for(std::size_t i = 0; i < h.size(); ++i){ sxy += (std::log(h[i]) - mx) * (std::log(e[i]) - my); sxx += std::pow(std::log(h[i]) - mx, 2); }
    return sxy / sxx;
}
double sd(const Eigen::VectorXd& v){
    const double m = v.mean();
    return std::sqrt((v.array() - m).square().sum() / static_cast<double>(v.size() - 1));
}
double sd(const std::vector<double>& v){
    return estimators::empirical::sample_stddev(v);
}
}

int main(){
    const std::string out_dir = "cpp/experiments/stochastic-processes/results/";
    std::ofstream conv(out_dir + "convergence.csv"), ito(out_dir + "ito_calculus.csv"),
                  smile(out_dir + "merton_smile.csv"), oucsv(out_dir + "ou_estimation.csv");
    if(!conv.is_open() || !ito.is_open() || !smile.is_open() || !oucsv.is_open()){
        std::cerr << "Failed to open output files in " << out_dir << "\nRun from repo root \n";
        return 1;
    }
    conv.precision(10); ito.precision(10); smile.precision(10); oucsv.precision(10);

    //------------------------------------------------------------ Part 1: convergence orders
    constexpr double MU = 0.05, SIG = 0.4, T = 1.0;
    const montecarlo::SdeFn a = [](double x, double){ return MU * x; };
    const montecarlo::SdeFn b = [](double x, double){ return SIG * x; };
    const montecarlo::SdeFn bx = [](double, double){ return SIG; };
    Rng rng(2031);
    {
        const Eigen::MatrixXd fine = montecarlo::brownian_increments(20000, 512, T, rng);
        const Eigen::VectorXd exact = ((MU - 0.5 * SIG * SIG) * T + SIG * fine.rowwise().sum().array()).exp().matrix();
        std::vector<double> h, em, mil;
        conv << "kind,steps,h,euler_maruyama,milstein,em_theory\n";
        for(Eigen::Index steps : {4, 8, 16, 32, 64, 128, 256}){
            const Eigen::MatrixXd dW = montecarlo::coarsen(fine, 512 / steps);
            const double e = (montecarlo::euler_maruyama(1.0, a, b, T, dW).col(steps) - exact).cwiseAbs().mean();
            const double m = (montecarlo::milstein(1.0, a, b, bx, T, dW).col(steps) - exact).cwiseAbs().mean();
            h.push_back(T / static_cast<double>(steps)); em.push_back(e); mil.push_back(m);
            conv << "strong," << steps << "," << T / static_cast<double>(steps) << "," << e << "," << m << ",\n";
        }
        std::cout << "strong order: Euler-Maruyama " << slope(h, em) << ", Milstein " << slope(h, mil) << "\n";
    }
    {
        const Eigen::MatrixXd fine = montecarlo::brownian_increments(200000, 64, T, rng);
        const double exact2 = (((MU - 0.5 * SIG * SIG) * T + SIG * fine.rowwise().sum().array()).exp().square()).mean();
        std::vector<double> h, em, mil;
        for(Eigen::Index steps : {2, 4, 8, 16, 32, 64}){
            const double hh = T / static_cast<double>(steps);
            const Eigen::MatrixXd dW = montecarlo::coarsen(fine, 64 / steps);
            const double e = std::abs(montecarlo::euler_maruyama(1.0, a, b, T, dW).col(steps).array().square().mean() - exact2);
            const double m = std::abs(montecarlo::milstein(1.0, a, b, bx, T, dW).col(steps).array().square().mean() - exact2);
            const double th = std::abs(std::pow(std::pow(1.0 + MU * hh, 2) + SIG * SIG * hh, static_cast<double>(steps)) - std::exp((2.0 * MU + SIG * SIG) * T));
            h.push_back(hh); em.push_back(e); mil.push_back(m);
            conv << "weak_second_moment," << steps << "," << hh << "," << e << "," << m << "," << th << "\n";
        }
        std::cout << "weak order (E[X^2]): Euler-Maruyama " << slope(h, em) << ", Milstein " << slope(h, mil) << "\n";
    }

    //------------------------------------------------------------ Part 2: Ito calculus
    ito << "steps,mean_qv,sd_qv,theory_sd_qv,mean_strat_minus_ito,half_T,mean_ito,var_ito,theory_var_ito\n";
    for(Eigen::Index steps : {10, 100, 1000, 10000}){
        const Eigen::MatrixXd dW = montecarlo::brownian_increments(2000, steps, T, rng);
        const Eigen::VectorXd qv = montecarlo::quadratic_variation(dW);
        const Eigen::VectorXd i = montecarlo::ito_integral_w_dw(dW);
        const Eigen::VectorXd s = montecarlo::stratonovich_integral_w_dw(dW);
        const double vi = sd(i) * sd(i);
        ito << steps << "," << qv.mean() << "," << sd(qv) << "," << T * std::sqrt(2.0 / static_cast<double>(steps)) << ","
            << (s - i).mean() << "," << T / 2.0 << "," << i.mean() << "," << vi << "," << T * T / 2.0 * (1.0 - 1.0 / static_cast<double>(steps)) << "\n";
        std::cout << "steps " << steps << ": QV mean " << qv.mean() << " sd " << sd(qv) << " (theory " << T * std::sqrt(2.0 / static_cast<double>(steps))
                  << "), Strat-Ito " << (s - i).mean() << ", Ito mean " << i.mean() << " var " << vi << "\n";
    }

    //------------------------------------------------------------ Part 3: Merton smile
    smile << "maturity,strike,closed_form,mc_price,mc_se,z_score,implied_vol\n";
    for(double mat : {0.1, 0.5, 2.0}){
        double iv80 = 0, iv100 = 0, iv120 = 0, worst_z = 0;
        for(double k = 70.0; k <= 130.0 + 1e-9; k += 5.0){
            const models::MertonParams p{100.0, k, 0.05, 0.2, mat, 1.0, -0.1, 0.15};
            const double cf = models::merton_price(p);
            const auto mc = models::merton_mc_price(p, 400000, rng);
            const double z = (mc.price - cf) / mc.std_error;
            const auto iv = pricing::implied_vol(cf, {100.0, k, 0.05, 0.2, mat, pricing::OptionType::Call});
            const double v = iv.has_value() ? iv->root : std::nan("");
            worst_z = std::max(worst_z, std::abs(z));
            if(k == 80.0) iv80 = v;
            if(k == 100.0) iv100 = v;
            if(k == 120.0) iv120 = v;
            smile << mat << "," << k << "," << cf << "," << mc.price << "," << mc.std_error << "," << z << "," << v << "\n";
        }
        std::cout << "Merton T=" << mat << ": IV(80)=" << iv80 << " IV(100)=" << iv100 << " IV(120)=" << iv120
                  << "  skew IV(80)-IV(120)=" << iv80 - iv120 << "  max |MC z| " << worst_z << "\n";
    }

    //------------------------------------------------------------ Part 4: OU estimation
    const double kappa = 3.0, theta = 0.04, sigma = 0.05, dt = 1.0 / 252.0;
    oucsv << "years,n_obs,reps,kappa_true,kappa_mean,kappa_sd,kappa_median,theta_mean,theta_sd,sigma_mean,sigma_sd,failed_fits\n";
    for(int years : {1, 5, 25}){
        const Eigen::Index n = 252 * years;
        std::vector<double> ks, ts, ss;
        int failed = 0;
        const int reps = 1000;
        for(int r = 0; r < reps; ++r){
            const Eigen::RowVectorXd path = models::simulate_ou(theta, kappa, theta, sigma, dt, n, 1, rng).row(0);
            try{
                const auto f = models::fit_ou(std::vector<double>(path.data(), path.data() + path.size()), dt);
                ks.push_back(f.kappa); ts.push_back(f.theta); ss.push_back(f.sigma);
            } catch(const InvalidInputError&){ ++failed; }
        }
        std::vector<double> sorted = ks;
        std::sort(sorted.begin(), sorted.end());
        const double med = sorted[sorted.size() / 2];
        oucsv << years << "," << n << "," << reps << "," << kappa << "," << estimators::empirical::sample_mean(ks) << "," << sd(ks) << "," << med << ","
              << estimators::empirical::sample_mean(ts) << "," << sd(ts) << "," << estimators::empirical::sample_mean(ss) << "," << sd(ss) << "," << failed << "\n";
        std::cout << "OU " << years << "y: kappa mean " << estimators::empirical::sample_mean(ks) << " sd " << sd(ks) << " median " << med
                  << " | theta mean " << estimators::empirical::sample_mean(ts) << " sd " << sd(ts)
                  << " | sigma mean " << estimators::empirical::sample_mean(ss) << " sd " << sd(ss) << " | failed " << failed << "\n";
    }
    std::cout << "Results written to " << out_dir << "\n";
    return 0;
}
