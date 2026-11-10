//Phase 10 experiment: fixing the covariance instability found in Phase 2
//Part 1: the exact Phase 2 setup (true correlation 0.3, unit variances, T = 50 observations, N growing), extended past
//        N = T and replicated 200 times per N. Three estimators: the sample covariance (Phase 2), Ledoit-Wolf shrinkage and
//        Marchenko-Pastur eigenvalue clipping. Measured: condition number, Cholesky success, error against the true
//        matrix, and out-of-sample quality of the minimum-variance portfolio each estimate produces (its variance under the
//        TRUE covariance, divided by the best achievable)
//Part 2: PCA, Ledoit-Wolf and the Marchenko-Pastur edge on the real 6-stock history
//C++ only: Ledoit-Wolf matches sklearn to 1e-10 (Python tests) and the C++ matches the Python reference in both suites
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include "qre/core/rng.hpp"
#include "qre/data/csv_loader.hpp"
#include "qre/data/quality.hpp"
#include "qre/data/returns.hpp"
#include "qre/distributions/normal.hpp"
#include "qre/linalg/cholesky.hpp"
#include "qre/linalg/covariance.hpp"
#include "qre/linalg/shrinkage.hpp"
#include "qre/stats/pca.hpp"
using namespace qre;

namespace {
CovarianceMatrix equicorr(Eigen::Index n, double rho){
    CovarianceMatrix c = CovarianceMatrix::Constant(n, n, rho);
    c.diagonal().setOnes();
    return c;
}
//Condition number; infinity when the matrix is (numerically) singular
double condition(const CovarianceMatrix& c){
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(c);
    const double lo = es.eigenvalues().minCoeff(), hi = es.eigenvalues().maxCoeff();
    return lo > 1e-12 * hi ? hi / lo : std::numeric_limits<double>::infinity();
}
double median(std::vector<double> v){
    std::sort(v.begin(), v.end());
    const std::size_t m = v.size() / 2;
    return v.size() % 2 ? v[m] : 0.5 * (v[m - 1] + v[m]);
}
double mean(const std::vector<double>& v){
    double s = 0.0;
    for(double x : v) s += x;
    return v.empty() ? std::nan("") : s / static_cast<double>(v.size());
}
}

int main(){
    const std::string out_dir = "cpp/experiments/08-shrinkage/results/";
    std::ofstream out(out_dir + "estimator_comparison.csv"), real(out_dir + "real_data_pca.csv");
    if(!out.is_open() || !real.is_open()){
        std::cerr << "Failed to open output files in " << out_dir << "\nRun from repo root \n";
        return 1;
    }
    out.precision(10); real.precision(10);

    //------------------------------------------------------------ Part 1
    const double rho = 0.3;
    const Eigen::Index t = 50;
    const int reps = 200;
    out << "N,T,N_over_T,estimator,median_condition,cholesky_success_rate,mean_rel_frobenius_error,"
           "mean_minvar_variance_ratio,minvar_computable_share,mean_shrinkage_intensity,mean_signal_eigenvalues\n";
    Rng master(2034);
    for(Eigen::Index n : {5, 10, 25, 40, 50, 60, 75, 100}){
        const CovarianceMatrix truth = equicorr(n, rho);
        const Eigen::MatrixXd L = linalg::cholesky_decompose(truth);
        const double best = 1.0 / truth.llt().solve(Eigen::VectorXd::Ones(n)).sum();   //optimal min-variance
        struct Acc{ std::vector<double> cond, err, ratio; int chol = 0; double extra = 0.0; };
        Acc acc[3];
        for(int r = 0; r < reps; ++r){
            Rng rng = master.spawn(static_cast<uint64_t>(n * 1000 + r));
            const auto z = distributions::normal::sample(static_cast<std::size_t>(n * t), 0.0, 1.0, rng);
            Eigen::MatrixXd zm(n, t);
            for(Eigen::Index j = 0; j < t; ++j) for(Eigen::Index i = 0; i < n; ++i) zm(i, j) = z[static_cast<std::size_t>(j * n + i)];
            const ReturnMatrix x = (L * zm).transpose();
            const auto lw = linalg::ledoit_wolf(x);
            const auto rmt = linalg::rmt_clip(x);
            const CovarianceMatrix est[3] = {linalg::sample_cov(x), lw.cov, rmt.cov};
            acc[1].extra += lw.intensity;
            acc[2].extra += static_cast<double>(rmt.n_signal);
            for(int k = 0; k < 3; ++k){
                acc[k].cond.push_back(condition(est[k]));
                acc[k].err.push_back((est[k] - truth).norm() / truth.norm());
                try{
                    linalg::cholesky_decompose(est[k]);
                    ++acc[k].chol;
                    const Eigen::VectorXd w = linalg::min_variance_weights(est[k]);
                    acc[k].ratio.push_back(w.dot(truth * w) / best);
                } catch(const InvalidInputError&){}
            }
        }
        const char* names[3] = {"sample", "ledoit_wolf", "rmt_clipping"};
        for(int k = 0; k < 3; ++k){
            const double chol_rate = static_cast<double>(acc[k].chol) / reps;
            out << n << "," << t << "," << static_cast<double>(n) / static_cast<double>(t) << "," << names[k] << "," << median(acc[k].cond) << ","
                << chol_rate << "," << mean(acc[k].err) << "," << mean(acc[k].ratio) << "," << static_cast<double>(acc[k].ratio.size()) / reps << ","
                << (k == 1 ? acc[1].extra / reps : std::nan("")) << "," << (k == 2 ? acc[2].extra / reps : std::nan("")) << "\n";
            std::cout << "N=" << n << " " << names[k] << ": median cond " << median(acc[k].cond) << "  Cholesky " << chol_rate * 100
                      << "%  rel err " << mean(acc[k].err) << "  minvar ratio " << mean(acc[k].ratio);
            if(k == 1) std::cout << "  delta " << acc[1].extra / reps;
            if(k == 2) std::cout << "  signal eigs " << acc[2].extra / reps;
            std::cout << "\n";
        }
    }

    //------------------------------------------------------------ Part 2
    data::PriceTable table;
    try{ table = data::load_price_csv("data/market_data/adjusted_close.csv"); }
    catch(const std::exception& e){
        std::cerr << e.what() << "\nRun from repo root \n";
        return 1;
    }
    const auto quality = data::harden(table.prices);
    if(!quality.report.clean()){
        std::cerr << "price data failed the quality report\n";
        return 1;
    }
    const ReturnMatrix rets = data::simple_return(quality.prices);
    const CovarianceMatrix s = linalg::sample_cov(rets);
    const Eigen::VectorXd sd = s.diagonal().cwiseSqrt();
    const CovarianceMatrix corr = s.array() / (sd * sd.transpose()).array();
    const auto p = stats::pca(corr);
    const double lam_plus = linalg::marchenko_pastur_upper(rets.cols(), rets.rows());
    const auto lw = linalg::ledoit_wolf(rets);
    const auto rmt = linalg::rmt_clip(rets);
    real << "component,eigenvalue,explained,above_marchenko_pastur";
    for(const auto& tk : table.tickers) real << "," << tk;
    real << "\n";
    std::cout << "\nreal data: T=" << rets.rows() << " N=" << rets.cols() << " MP edge " << lam_plus << "  Ledoit-Wolf delta " << lw.intensity
              << "  RMT signal eigenvalues " << rmt.n_signal << "\n";
    for(Eigen::Index k = 0; k < p.eigenvalues.size(); ++k){
        real << k + 1 << "," << p.eigenvalues(k) << "," << p.explained(k) << "," << (p.eigenvalues(k) > lam_plus ? "true" : "false");
        for(Eigen::Index i = 0; i < p.components.rows(); ++i) real << "," << p.components(i, k);
        real << "\n";
        std::cout << "PC" << k + 1 << ": eigenvalue " << p.eigenvalues(k) << " explains " << p.explained(k) * 100 << "%  loadings";
        for(Eigen::Index i = 0; i < p.components.rows(); ++i) std::cout << " " << table.tickers[static_cast<std::size_t>(i)] << "=" << p.components(i, k);
        std::cout << "\n";
    }
    real << "summary,T=" << rets.rows() << ";N=" << rets.cols() << ";marchenko_pastur_edge=" << lam_plus << ";ledoit_wolf_delta=" << lw.intensity
         << ";rmt_signal_eigenvalues=" << rmt.n_signal << ",,";
    for(Eigen::Index i = 0; i < rets.cols(); ++i) real << ",";
    real << "\n";
    std::cout << "Results written to " << out_dir << "\n";
    return 0;
}
