#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "qre/linalg/covariance.hpp"
#include "qre/montecarlo/gbm_path.hpp"
using namespace qre;
using qre::montecarlo::McOptions;
using qre::pricing::BSParams;
using qre::pricing::OptionType;

namespace {
constexpr double S = 100.0, K = 100.0, R = 0.05, SIG = 0.2, T = 1.0;
BSParams base(OptionType t = OptionType::Call){ return BSParams{S, K, R, SIG, T, t}; }

std::vector<double> terminal_log_returns(const Eigen::MatrixXd& paths){
    std::vector<double> out;
    for(Eigen::Index p = 0; p < paths.rows(); ++p){
        out.push_back(std::log(paths(p, paths.cols() - 1) / paths(p, 0)));
    }
    return out;
}
}

TEST(GbmPaths, ShapeStartAndPositive){
    Rng rng(1);
    const auto paths = montecarlo::simulate_gbm_paths(S, R, SIG, T, 50, 1000, rng);
    EXPECT_EQ(paths.rows(), 1000);
    EXPECT_EQ(paths.cols(), 51);
    EXPECT_EQ(paths.col(0).minCoeff(), S);
    EXPECT_EQ(paths.col(0).maxCoeff(), S);
    EXPECT_GT(paths.minCoeff(), 0.0);
}

TEST(GbmPaths, TerminalLogReturnMoments){
    Rng rng(2);
    const auto lr = terminal_log_returns(montecarlo::simulate_gbm_paths(S, R, SIG, T, 10, 200000, rng));
    const double se = SIG * std::sqrt(T) / std::sqrt(static_cast<double>(lr.size()));
    EXPECT_NEAR(estimators::empirical::sample_mean(lr), (R - 0.5 * SIG * SIG) * T, 4 * se);
    EXPECT_NEAR(estimators::empirical::sample_stddev(lr), SIG * std::sqrt(T), 0.01 * SIG * std::sqrt(T));
}

TEST(GbmPaths, StepCountDoesNotChangeTerminalDistribution){
    Rng r1(4), r2(5);
    const auto a = terminal_log_returns(montecarlo::simulate_gbm_paths(S, R, SIG, T, 1, 200000, r1));
    const auto b = terminal_log_returns(montecarlo::simulate_gbm_paths(S, R, SIG, T, 100, 200000, r2));
    EXPECT_NEAR(estimators::empirical::sample_mean(a), estimators::empirical::sample_mean(b), 0.01);
    EXPECT_NEAR(estimators::empirical::sample_stddev(a), estimators::empirical::sample_stddev(b), 0.005);
}

TEST(GbmPaths, AntitheticHalvesAreMirrored){
    Rng rng(5);
    const auto p = montecarlo::simulate_gbm_paths(S, R, SIG, T, 3, 10, rng, true);
    const double drift = (R - 0.5 * SIG * SIG) * T / 3.0;
    for(Eigen::Index t = 0; t < 3; ++t){
        for(Eigen::Index i = 0; i < 5; ++i){
            const double up = std::log(p(i, t + 1) / p(i, t)) - drift;
            const double dn = std::log(p(i + 5, t + 1) / p(i + 5, t)) - drift;
            EXPECT_NEAR(up, -dn, 1e-12);
        }
    }
}

TEST(GbmMonteCarlo, SameSeedReproducibleDifferentSeedDiffers){
    Rng a(7), b(7), c(8);
    const auto ra = montecarlo::mc_european_price(base(), 10000, a);
    const auto rb = montecarlo::mc_european_price(base(), 10000, b);
    const auto rc = montecarlo::mc_european_price(base(), 10000, c);
    EXPECT_EQ(ra.price, rb.price);
    EXPECT_NE(ra.price, rc.price);
}

TEST(GbmMonteCarlo, PriceWithinFourStandardErrorsOfBlackScholes){
    for(OptionType t : {OptionType::Call, OptionType::Put}){
        for(bool anti : {false, true}){
            for(bool cv : {false, true}){
                Rng rng(11);
                const auto r = montecarlo::mc_european_price(base(t), 200000, rng, McOptions{anti, cv});
                EXPECT_LT(std::abs(r.price - pricing::price(base(t))), 4 * r.std_error)
                    << "type=" << (t == OptionType::Call ? "call" : "put") << " anti=" << anti << " cv=" << cv;
            }
        }
    }
}

TEST(GbmMonteCarlo, VarianceReductionLowersStandardError){
    auto se = [](McOptions o){ Rng rng(21); return montecarlo::mc_european_price(base(), 100000, rng, o).std_error; };
    const double plain = se({false, false});
    EXPECT_LT(se({true, false}), plain);
    EXPECT_LT(se({false, true}), plain);
    EXPECT_LT(se({true, true}), plain);
}

TEST(GbmMonteCarlo, ErrorScalesAsInverseSqrtN){
    const double exact = pricing::price(base());
    auto rmse = [&](std::size_t n){
        double sum_sq = 0.0;
        for(int i = 0; i < 200; ++i){
            Rng rng(1000 + static_cast<uint64_t>(i));
            const double e = montecarlo::mc_european_price(base(), n, rng).price - exact;
            sum_sq += e * e;
        }
        return std::sqrt(sum_sq / 200.0);
    };
    const double ratio = rmse(1000) / rmse(10000);          //expect sqrt(10) ~ 3.16
    EXPECT_NEAR(ratio, std::sqrt(10.0), 0.25 * std::sqrt(10.0));
}

TEST(GbmMonteCarlo, ReportedStandardErrorMatchesEmpiricalSpread){
    std::vector<double> prices;
    for(int i = 0; i < 300; ++i){
        Rng rng(2000 + static_cast<uint64_t>(i));
        prices.push_back(montecarlo::mc_european_price(base(), 5000, rng, McOptions{true, false}).price);
    }
    Rng rng(1);
    const double se = montecarlo::mc_european_price(base(), 5000, rng, McOptions{true, false}).std_error;
    EXPECT_NEAR(estimators::empirical::sample_stddev(prices), se, 0.2 * se);
}

TEST(GbmCorrelated, ReproducesTargetCorrelationAndMarginals){
    Eigen::MatrixXd corr(3, 3);
    corr << 1.0, 0.6, -0.3,
            0.6, 1.0, 0.1,
           -0.3, 0.1, 1.0;
    Eigen::VectorXd spots(3), vols(3);
    spots << 100, 50, 80;
    vols << 0.2, 0.3, 0.25;
    Rng rng(31);
    const auto assets = montecarlo::simulate_correlated_gbm(spots, R, vols, corr, T, 4, 200000, rng);
    ASSERT_EQ(assets.size(), 3u);
    Eigen::MatrixXd lr(200000, 3);
    for(Eigen::Index a = 0; a < 3; ++a){
        for(Eigen::Index p = 0; p < 200000; ++p){
            lr(p, a) = std::log(assets[static_cast<std::size_t>(a)](p, 4) / spots(a));
        }
    }
    const Eigen::MatrixXd cov = linalg::sample_cov(lr);
    for(Eigen::Index i = 0; i < 3; ++i){
        EXPECT_NEAR(std::sqrt(cov(i, i)), vols(i) * std::sqrt(T), 0.01 * vols(i));
        for(Eigen::Index j = 0; j < 3; ++j){
            EXPECT_NEAR(cov(i, j) / std::sqrt(cov(i, i) * cov(j, j)), corr(i, j), 0.01);
        }
    }
}

TEST(GbmInvalid, BadInputsThrow){
    Rng rng(1);
    BSParams p = base(); p.spot = -1.0;
    EXPECT_THROW(montecarlo::mc_european_price(p, 100, rng), InvalidInputError);
    p = base(); p.strike = 0.0;
    EXPECT_THROW(montecarlo::mc_european_price(p, 100, rng), InvalidInputError);
    p = base(); p.vol = 0.0;
    EXPECT_THROW(montecarlo::mc_european_price(p, 100, rng), InvalidInputError);
    p = base(); p.time = 0.0;
    EXPECT_THROW(montecarlo::mc_european_price(p, 100, rng), InvalidInputError);
    EXPECT_THROW(montecarlo::mc_european_price(base(), 1, rng), InvalidInputError);
    EXPECT_THROW(montecarlo::mc_european_price(base(), 101, rng, McOptions{true, false}), InvalidInputError);
}

TEST(GbmInvalid, NonPositiveDefiniteCorrelationThrows){
    Eigen::MatrixXd bad(3, 3);
    bad << 1.0, 0.99, -0.99,
           0.99, 1.0, 0.99,
          -0.99, 0.99, 1.0;
    Eigen::VectorXd spots = Eigen::VectorXd::Constant(3, 100.0), vols = Eigen::VectorXd::Constant(3, 0.2);
    Rng rng(1);
    EXPECT_THROW(montecarlo::simulate_correlated_gbm(spots, R, vols, bad, T, 2, 100, rng), InvalidInputError);
}
