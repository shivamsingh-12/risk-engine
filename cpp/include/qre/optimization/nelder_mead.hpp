#pragma once
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <vector>
#include <Eigen/Dense>
#include "qre/core/exceptions.hpp"
//Nelder-Mead: derivative-free minimization in n dimensions (Phase 4 only had 1-D root finding)
//Keeps a simplex of n+1 points and replaces the worst one each step by reflecting it through the centroid
//of the others, expanding if that helps a lot, contracting if it doesn't, shrinking toward the best otherwise
//Good fit for likelihoods (GARCH, later Heston calibration): no gradients needed, robust to noise
//Not guaranteed to find a global minimum -> callers should use several starting points
//Coefficients: reflection 1, expansion 2, contraction 0.5, shrink 0.5 (the standard choice)
//Stops when max |f_i - f_best| <= ftol AND max ||x_i - x_best||_inf <= xtol, or after max_iter iterations
namespace qre::optimization{
    struct MinimizeResult{
        Eigen::VectorXd x;
        double fx;
        int iterations;
        bool converged;
    };

    inline MinimizeResult nelder_mead(const std::function<double(const Eigen::VectorXd&)>& f,
                                      const Eigen::VectorXd& x0, double step = 0.5,
                                      double xtol = 1e-10, double ftol = 1e-12, int max_iter = 4000){
        const Eigen::Index n = x0.size();
        if(n < 1) throw InvalidInputError("nelder_mead: need at least one parameter");
        //Non-finite objective values count as +infinity so the simplex moves away from them
        auto eval = [&f](const Eigen::VectorXd& x){
            const double v = f(x);
            return std::isfinite(v) ? v : std::numeric_limits<double>::infinity();
        };
        std::vector<Eigen::VectorXd> pts(static_cast<std::size_t>(n + 1), x0);
        for(Eigen::Index i = 0; i < n; ++i) pts[static_cast<std::size_t>(i + 1)](i) += step;
        std::vector<double> fv(pts.size());
        for(std::size_t i = 0; i < pts.size(); ++i) fv[i] = eval(pts[i]);
        if(!std::isfinite(fv[0])) throw InvalidInputError("nelder_mead: objective is not finite at the starting point");

        std::vector<std::size_t> order(pts.size());
        int it = 0;
        for(; it < max_iter; ++it){
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(), [&fv](std::size_t a, std::size_t b){ return fv[a] < fv[b]; });
            const std::size_t best = order.front(), worst = order.back(), second = order[order.size() - 2];

            double fspread = 0.0, xspread = 0.0;
            for(std::size_t i = 0; i < pts.size(); ++i){
                fspread = std::max(fspread, std::abs(fv[i] - fv[best]));
                xspread = std::max(xspread, (pts[i] - pts[best]).cwiseAbs().maxCoeff());
            }
            if(fspread <= ftol && xspread <= xtol) break;

            Eigen::VectorXd centroid = Eigen::VectorXd::Zero(n);
            for(std::size_t i = 0; i < pts.size(); ++i) if(i != worst) centroid += pts[i];
            centroid /= static_cast<double>(n);

            const Eigen::VectorXd xr = centroid + (centroid - pts[worst]);
            const double fr = eval(xr);
            if(fr < fv[best]){
                const Eigen::VectorXd xe = centroid + 2.0 * (centroid - pts[worst]);
                const double fe = eval(xe);
                if(fe < fr){ pts[worst] = xe; fv[worst] = fe; }
                else { pts[worst] = xr; fv[worst] = fr; }
                continue;
            }
            if(fr < fv[second]){ pts[worst] = xr; fv[worst] = fr; continue; }
            //Contract: outside if the reflected point beat the worst, inside otherwise
            const bool outside = fr < fv[worst];
            const Eigen::VectorXd xc = outside ? Eigen::VectorXd(centroid + 0.5 * (xr - centroid))
                                               : Eigen::VectorXd(centroid + 0.5 * (pts[worst] - centroid));
            const double fc = eval(xc);
            if(fc < (outside ? fr : fv[worst])){ pts[worst] = xc; fv[worst] = fc; continue; }
            //Shrink everything toward the best point
            for(std::size_t i = 0; i < pts.size(); ++i){
                if(i == best) continue;
                pts[i] = pts[best] + 0.5 * (pts[i] - pts[best]);
                fv[i] = eval(pts[i]);
            }
        }
        const std::size_t b = static_cast<std::size_t>(std::min_element(fv.begin(), fv.end()) - fv.begin());
        return MinimizeResult{pts[b], fv[b], it, it < max_iter};
    }
}
