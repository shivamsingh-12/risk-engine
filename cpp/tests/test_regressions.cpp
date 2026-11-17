#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include <vector>
#include "qre/risk/evt.hpp"
#include "qre/risk/garch.hpp"
using namespace qre;
using namespace qre::risk;

//Regressions found while building the Phase 15 portfolio app

//A min-variance portfolio's conditional-EVT window: the method-of-moments start (xi ~ -0.4) leaves the largest
//exceedance outside the GPD support, and Nelder-Mead refused the non-finite start, stopping the whole qre_run
TEST(Phase15Regression, GpdFitSkipsInfeasibleMomentStart){
    const std::vector<double> y{0.629533, 0.922216, 0.200356, 0.996887, 0.721952, 0.189688, 1.264614, 0.794235, 1.525641,
                                1.147761, 0.711705, 0.321565, 0.152588, 0.790589, 0.421305, 0.482209, 0.076213, 0.508952,
                                0.51812, 0.48874, 0.652854, 2.431878, 0.680863, 0.053342, 0.581883};
    const GpdFit f = fit_gpd(y);
    EXPECT_NEAR(f.xi, -0.2505940805162809, 1e-6);    //Python reference (scipy Nelder-Mead)
    EXPECT_NEAR(f.beta, 0.8551165870936019, 1e-6);
    EXPECT_FALSE(f.at_bound);
}

//On near-normal returns nu runs to the bound; a +inf wall there made Nelder-Mead crawl for thousands of iterations.
//The likelihood is now flat above kNuMax and the fitted nu is reported at the bound
TEST(Phase15Regression, GarchTNuIsClampedNotWalled){
    std::mt19937_64 eng(1313);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<double> r;
    for(int i = 0; i < 250; ++i) r.push_back(0.008 * nd(eng));
    const GarchFit f = fit_garch(r, GarchDist::StudentT);
    EXPECT_DOUBLE_EQ(f.nu, kNuMax);
    Eigen::VectorXd theta(3);
    theta << 0.0, 0.0, 50.0;                         //nu = 2 + e^50, far above the bound
    double v = 0.0;
    for(double x : r) v += x * x;
    EXPECT_TRUE(std::isfinite(garch_t_neg_loglik(theta, r, v / 250.0)));
}
