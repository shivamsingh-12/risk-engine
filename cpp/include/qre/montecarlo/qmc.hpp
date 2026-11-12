#pragma once
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <utility>
#include <vector>
#include <Eigen/Dense>
#include "qre/core/exceptions.hpp"
#include "qre/core/rng.hpp"
//Phase 12: quasi-Monte Carlo with Sobol sequences, and the Brownian bridge
//MC error ~ N^(-1/2) because random points clump by chance; a low-discrepancy sequence fills [0,1)^d evenly,
//so for smooth integrands with low effective dimension QMC error is close to O(1/N)
//Sobol (Joe-Kuo 2008 direction numbers, same table as scipy.stats.qmc.Sobol): per dimension a primitive polynomial
//of degree s with coefficients a_1..a_{s-1} and odd initial m_1..m_s give
//  m_k = 2^s m_{k-s} XOR m_{k-s} XOR (XOR_{i=1}^{s-1} 2^i a_i m_{k-i}),   V_k = m_k 2^(32-k)
//  x_i = XOR of V_{k+1} over the set bits k of the Gray code i ^ (i >> 1),   u_i = x_i / 2^32
//Randomized QMC: XOR every point with one random 32-bit integer per dimension (random digital shift) -> each point
//uniform, estimator unbiased, evenness kept, independent shifts give a standard error. Points at cell midpoints
//(x + 0.5) / 2^32 so no coordinate is exactly 0 before the inverse normal cdf
//Brownian bridge: QMC's early coordinates are its best, so Z_1 sets W(T), Z_2 the midpoint given both ends, ...
//  W(t_l) | W(t_j), W(t_k) ~ N(((t_k - t_l) W(t_j) + (t_l - t_j) W(t_k)) / (t_k - t_j), (t_l - t_j)(t_k - t_l) / (t_k - t_j))
//Same path distribution as the incremental construction, only the coordinate -> path mapping changes (Jaeckel 2002)
namespace qre::montecarlo::qmc{
    inline constexpr int kMaxBits = 32;

    struct DirectionEntry{
        uint32_t poly;                  //primitive polynomial including leading and trailing bits
        std::vector<uint32_t> m;        //initial m_1..m_s
    };

    //Dimensions 2..64 (dimension 1 is van der Corput, all m_k = 1)
    inline const std::vector<DirectionEntry>& direction_table(){
        static const std::vector<DirectionEntry> table{
            {3, {1}},
            {7, {1, 3}},
            {11, {1, 3, 1}},
            {13, {1, 1, 1}},
            {19, {1, 1, 3, 3}},
            {25, {1, 3, 5, 13}},
            {37, {1, 1, 5, 5, 17}},
            {41, {1, 1, 5, 5, 5}},
            {47, {1, 1, 7, 11, 19}},
            {55, {1, 1, 5, 1, 1}},
            {59, {1, 1, 1, 3, 11}},
            {61, {1, 3, 5, 5, 31}},
            {67, {1, 3, 3, 9, 7, 49}},
            {91, {1, 1, 1, 15, 21, 21}},
            {97, {1, 3, 1, 13, 27, 49}},
            {103, {1, 1, 1, 15, 7, 5}},
            {109, {1, 3, 1, 15, 13, 25}},
            {115, {1, 1, 5, 5, 19, 61}},
            {131, {1, 3, 7, 11, 23, 15, 103}},
            {137, {1, 3, 7, 13, 13, 15, 69}},
            {143, {1, 1, 3, 13, 7, 35, 63}},
            {145, {1, 3, 5, 9, 1, 25, 53}},
            {157, {1, 3, 1, 13, 9, 35, 107}},
            {167, {1, 3, 1, 5, 27, 61, 31}},
            {171, {1, 1, 5, 11, 19, 41, 61}},
            {185, {1, 3, 5, 3, 3, 13, 69}},
            {191, {1, 1, 7, 13, 1, 19, 1}},
            {193, {1, 3, 7, 5, 13, 19, 59}},
            {203, {1, 1, 3, 9, 25, 29, 41}},
            {211, {1, 3, 5, 13, 23, 1, 55}},
            {213, {1, 3, 7, 3, 13, 59, 17}},
            {229, {1, 3, 1, 3, 5, 53, 69}},
            {239, {1, 1, 5, 5, 23, 33, 13}},
            {241, {1, 1, 7, 7, 1, 61, 123}},
            {247, {1, 1, 7, 9, 13, 61, 49}},
            {253, {1, 3, 3, 5, 3, 55, 33}},
            {285, {1, 3, 1, 15, 31, 13, 49, 245}},
            {299, {1, 3, 5, 15, 31, 59, 63, 97}},
            {301, {1, 3, 1, 11, 11, 11, 77, 249}},
            {333, {1, 3, 1, 11, 27, 43, 71, 9}},
            {351, {1, 1, 7, 15, 21, 11, 81, 45}},
            {355, {1, 3, 7, 3, 25, 31, 65, 79}},
            {357, {1, 3, 1, 1, 19, 11, 3, 205}},
            {361, {1, 1, 5, 9, 19, 21, 29, 157}},
            {369, {1, 3, 7, 11, 1, 33, 89, 185}},
            {391, {1, 3, 3, 3, 15, 9, 79, 71}},
            {397, {1, 3, 7, 11, 15, 39, 119, 27}},
            {425, {1, 1, 3, 1, 11, 31, 97, 225}},
            {451, {1, 1, 1, 3, 23, 43, 57, 177}},
            {463, {1, 3, 7, 7, 17, 17, 37, 71}},
            {487, {1, 3, 1, 5, 27, 63, 123, 213}},
            {501, {1, 1, 3, 5, 11, 43, 53, 133}},
            {529, {1, 3, 5, 5, 29, 17, 47, 173, 479}},
            {539, {1, 3, 3, 11, 3, 1, 109, 9, 69}},
            {545, {1, 1, 1, 5, 17, 39, 23, 5, 343}},
            {557, {1, 3, 1, 5, 25, 15, 31, 103, 499}},
            {563, {1, 1, 1, 11, 11, 17, 63, 105, 183}},
            {601, {1, 1, 5, 11, 9, 29, 97, 231, 363}},
            {607, {1, 1, 5, 15, 19, 45, 41, 7, 383}},
            {617, {1, 3, 7, 7, 31, 19, 83, 137, 221}},
            {623, {1, 1, 1, 3, 23, 15, 111, 223, 83}},
            {631, {1, 1, 5, 13, 31, 15, 55, 25, 161}},
            {637, {1, 1, 3, 13, 25, 47, 39, 87, 257}}
        };
        return table;
    }

    inline std::size_t max_dim(){ return direction_table().size() + 1; }

    //dim x 32 direction numbers V_1..V_32
    inline std::vector<std::array<uint32_t, kMaxBits>> direction_numbers(std::size_t dim){
        if(dim < 1 || dim > max_dim()) throw InvalidInputError("Sobol dimension out of range");
        std::vector<std::array<uint32_t, kMaxBits>> v(dim);
        for(int k = 0; k < kMaxBits; ++k) v[0][static_cast<std::size_t>(k)] = uint32_t{1} << (kMaxBits - 1 - k);
        for(std::size_t j = 1; j < dim; ++j){
            const DirectionEntry& e = direction_table()[j - 1];
            const int s = static_cast<int>(std::bit_width(e.poly)) - 1;
            const uint32_t a = (e.poly >> 1) & ((uint32_t{1} << (s - 1)) - 1u);
            std::vector<uint64_t> m(e.m.begin(), e.m.end());
            for(int k = s; k < kMaxBits; ++k){
                const auto ks = static_cast<std::size_t>(k - s);
                uint64_t next = m[ks] ^ (m[ks] << s);
                for(int i = 1; i < s; ++i){
                    if((a >> (s - 1 - i)) & 1u) next ^= m[static_cast<std::size_t>(k - i)] << i;
                }
                m.push_back(next);
            }
            for(int k = 0; k < kMaxBits; ++k){
                v[j][static_cast<std::size_t>(k)] = static_cast<uint32_t>(m[static_cast<std::size_t>(k)] << (kMaxBits - 1 - k));
            }
        }
        return v;
    }

    //First n points (origin included) as 32-bit integers, row-major n x dim, generated in Gray-code order
    inline std::vector<uint32_t> sobol_ints(std::size_t n, std::size_t dim){
        if(n < 1) throw InvalidInputError("need at least one Sobol point");
        const auto v = direction_numbers(dim);
        std::vector<uint32_t> out(n * dim, 0u);
        std::vector<uint32_t> x(dim, 0u);
        for(std::size_t i = 1; i < n; ++i){
            const auto c = static_cast<std::size_t>(std::countr_one(i - 1));     //rightmost zero bit of i - 1
            if(c >= static_cast<std::size_t>(kMaxBits)) throw InvalidInputError("at most 2^32 Sobol points");
            for(std::size_t d = 0; d < dim; ++d){
                x[d] ^= v[d][c];
                out[i * dim + d] = x[d];
            }
        }
        return out;
    }

    //Unscrambled points in [0, 1), identical to scipy.stats.qmc.Sobol(dim, scramble=False)
    inline Eigen::MatrixXd sobol_points(std::size_t n, std::size_t dim){
        const std::vector<uint32_t> x = sobol_ints(n, dim);
        Eigen::MatrixXd u(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(dim));
        for(std::size_t i = 0; i < n; ++i){
            for(std::size_t d = 0; d < dim; ++d){
                u(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(d)) = std::ldexp(static_cast<double>(x[i * dim + d]), -kMaxBits);
            }
        }
        return u;
    }

    inline std::vector<uint32_t> random_shift(std::size_t dim, Rng& rng){
        std::vector<uint32_t> s(dim);
        for(auto& v : s) v = static_cast<uint32_t>(rng.engine()() >> 32);
        return s;
    }

    //Standard normal quantile: Acklam's rational approximation (rel. error 1e-9) polished by one Halley step on
    //the exact cdf. Works in the lower tail and uses symmetry above 0.5, where 1 - p is exact
    inline double inv_normal_cdf(double p){
        if(!(p > 0.0 && p < 1.0)) throw InvalidInputError("probability must be in (0, 1)");
        if(p > 0.5) return -inv_normal_cdf(1.0 - p);
        static constexpr std::array<double, 6> a{-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
                                                  1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00};
        static constexpr std::array<double, 5> b{-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
                                                  6.680131188771972e+01, -1.328068155288572e+01};
        static constexpr std::array<double, 6> c{-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
                                                  -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00};
        static constexpr std::array<double, 4> d{7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
                                                  3.754408661907416e+00};
        double x;
        if(p < 0.02425){
            const double q = std::sqrt(-2.0 * std::log(p));
            x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
        } else {
            const double q = p - 0.5, r = q * q;
            x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
                (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
        }
        const double e = 0.5 * std::erfc(-x / std::numbers::sqrt2) - p;
        const double u = e * std::sqrt(2.0 * std::numbers::pi) * std::exp(0.5 * x * x);
        return x - u / (1.0 + 0.5 * x * u);
    }

    //n x dim standard normals from (optionally digitally shifted) Sobol points at cell midpoints
    inline Eigen::MatrixXd sobol_normals(std::size_t n, std::size_t dim, const std::vector<uint32_t>& shift = {}){
        if(!shift.empty() && shift.size() != dim) throw InvalidInputError("shift must have one entry per dimension");
        const std::vector<uint32_t> x = sobol_ints(n, dim);
        Eigen::MatrixXd z(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(dim));
        for(std::size_t i = 0; i < n; ++i){
            for(std::size_t d = 0; d < dim; ++d){
                const uint32_t xi = shift.empty() ? x[i * dim + d] : (x[i * dim + d] ^ shift[d]);
                z(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(d)) =
                    inv_normal_cdf(std::ldexp(static_cast<double>(xi) + 0.5, -kMaxBits));
            }
        }
        return z;
    }

    //W at t_k = k T / m from normals z (n x m), by summing increments
    inline Eigen::MatrixXd incremental_path(const Eigen::MatrixXd& z, double time){
        if(z.cols() < 1 || time <= 0.0) throw InvalidInputError("need at least one step and positive time");
        Eigen::MatrixXd w(z.rows(), z.cols());
        const double sd = std::sqrt(time / static_cast<double>(z.cols()));
        w.col(0) = sd * z.col(0);
        for(Eigen::Index k = 1; k < z.cols(); ++k) w.col(k) = w.col(k - 1) + sd * z.col(k);
        return w;
    }

    //Bridge construction order and weights for m equally spaced dates, built once per m
    class BrownianBridge{
        public:
            BrownianBridge(Eigen::Index steps, double time) : m_(steps){
                if(steps < 1 || time <= 0.0) throw InvalidInputError("need at least one step and positive time");
                const auto m = static_cast<std::size_t>(steps);
                std::vector<double> t(m);
                for(std::size_t k = 0; k < m; ++k) t[k] = time * static_cast<double>(k + 1) / static_cast<double>(m);
                std::vector<bool> populated(m, false);
                populated[m - 1] = true;
                plan_.push_back({m - 1, -1, -1, 0.0, 0.0, std::sqrt(t[m - 1])});
                std::size_t j = 0;
                for(std::size_t i = 1; i < m; ++i){
                    while(populated[j]) ++j;
                    std::size_t k = j;
                    while(!populated[k]) ++k;
                    const std::size_t l = j + ((k - 1 - j) >> 1);
                    populated[l] = true;
                    const double t_left = j > 0 ? t[j - 1] : 0.0;
                    const double span = t[k] - t_left;
                    plan_.push_back({l, static_cast<long>(j) - 1, static_cast<long>(k), (t[k] - t[l]) / span,
                                     (t[l] - t_left) / span, std::sqrt((t[l] - t_left) * (t[k] - t[l]) / span)});
                    j = k + 1;
                    if(j >= m) j = 0;
                }
            }

            Eigen::MatrixXd build(const Eigen::MatrixXd& z) const{
                if(z.cols() != m_) throw InvalidInputError("normals must have one column per step");
                Eigen::MatrixXd w(z.rows(), z.cols());
                for(std::size_t i = 0; i < plan_.size(); ++i){
                    const Step& s = plan_[i];
                    const auto l = static_cast<Eigen::Index>(s.target);
                    w.col(l) = s.sd * z.col(static_cast<Eigen::Index>(i));
                    if(s.right >= 0) w.col(l) += s.wr * w.col(s.right);
                    if(s.left >= 0) w.col(l) += s.wl * w.col(s.left);
                }
                return w;
            }

        private:
            struct Step{
                std::size_t target;
                long left, right;       //-1 -> W(0) = 0 or no right point
                double wl, wr, sd;
            };
            Eigen::Index m_;
            std::vector<Step> plan_;
    };

    inline Eigen::MatrixXd brownian_bridge(const Eigen::MatrixXd& z, double time){
        return BrownianBridge(z.cols(), time).build(z);
    }
}
