#pragma once
#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <vector>
#include "qre/core/exceptions.hpp"
//Phase 12: adjoint algorithmic differentiation (reverse-mode AD) with a tape
//Every elementary operation is recorded as a node with its parents and the local partials d node / d parent.
//Backward sweep: adjoint(output) = 1, then for nodes in reverse order adjoint(parent) += partial * adjoint(node)
//-> d output / d every input in one sweep, at a small constant multiple of the cost of the computation itself,
//   whereas bumping needs one or two extra evaluations per input
//Derivatives are exact to rounding, of the program as written: a jump in the program (indicator, digital payoff)
//has derivative 0 away from the jump, exactly like the pathwise Monte Carlo estimator
//Monte Carlo use: record inputs and anything shared by all paths, take mark = size(), then per path record, sweep
//from the path's output back to the mark (inputs below the mark accumulate), and rewind to the mark so the tape
//never grows. One final sweep from the mark down to 0 pushes the accumulated adjoints into the inputs
namespace qre::autodiff{
    class Tape;

    class Var{
        public:
            Var() = default;
            double value() const { return v_; }
            std::size_t index() const { return i_; }
            Tape* tape() const { return t_; }
        private:
            friend class Tape;
            Var(double v, std::size_t i, Tape* t) : v_(v), i_(i), t_(t) {}
            double v_ = 0.0;
            std::size_t i_ = 0;
            Tape* t_ = nullptr;
    };

    class Tape{
        public:
            Var variable(double value){ return push(value, 0, {0, 0}, {0.0, 0.0}); }
            std::size_t size() const { return nodes_.size(); }

            //drop every node recorded after `mark` (their adjoints too)
            void rewind(std::size_t mark){
                if(mark > nodes_.size()) throw InvalidInputError("cannot rewind past the end of the tape");
                nodes_.resize(mark);
                if(adj_.size() > mark) adj_.resize(mark);
            }

            void clear_adjoints(){ adj_.assign(nodes_.size(), 0.0); }

            //adjoint(y) += seed, then propagate through nodes y.index() .. stop (inclusive) downward.
            //Nodes below `stop` only receive adjoint; nodes at or above `stop` are zeroed first, except y's own
            void accumulate(const Var& y, std::size_t stop = 0, double seed = 1.0){
                check(y);
                if(stop > y.index()) throw InvalidInputError("stop must not be above the output");
                adj_.resize(nodes_.size(), 0.0);
                for(std::size_t i = stop; i < nodes_.size(); ++i) adj_[i] = 0.0;
                adj_[y.index()] += seed;
                sweep(y.index(), stop);
            }

            //propagate already accumulated adjoints from node `from` down to `stop` (inclusive)
            void propagate(std::size_t from, std::size_t stop = 0){
                if(from >= nodes_.size() || stop > from) throw InvalidInputError("bad propagation range");
                adj_.resize(nodes_.size(), 0.0);
                sweep(from, stop);
            }

            //plain one-shot gradient: zero everything, seed y, sweep to the start
            void gradient(const Var& y){
                check(y);
                clear_adjoints();
                adj_[y.index()] = 1.0;
                sweep(y.index(), 0);
            }

            double adjoint(const Var& x) const{
                if(x.tape() != this) throw InvalidInputError("variable is not on this tape");
                return x.index() < adj_.size() ? adj_[x.index()] : 0.0;
            }

            //record a node: n parents (0, 1 or 2)
            Var push(double value, std::size_t n, std::array<std::size_t, 2> parents, std::array<double, 2> partials){
                nodes_.push_back(Node{parents, partials, n});
                return Var(value, nodes_.size() - 1, this);
            }

        private:
            struct Node{
                std::array<std::size_t, 2> parent;
                std::array<double, 2> partial;
                std::size_t n;
            };
            void check(const Var& y) const{
                if(y.tape() != this) throw InvalidInputError("output was not recorded on this tape");
            }
            void sweep(std::size_t from, std::size_t stop){
                for(std::size_t i = from + 1; i-- > stop;){
                    const double a = adj_[i];
                    if(a == 0.0) continue;
                    const Node& nd = nodes_[i];
                    for(std::size_t k = 0; k < nd.n; ++k) adj_[nd.parent[k]] += nd.partial[k] * a;
                }
            }
            std::vector<Node> nodes_;
            std::vector<double> adj_;
    };

    namespace detail{
        inline Var unary(const Var& x, double value, double d){
            return x.tape()->push(value, 1, {x.index(), 0}, {d, 0.0});
        }
        inline Var binary(const Var& x, const Var& y, double value, double dx, double dy){
            if(x.tape() != y.tape()) throw InvalidInputError("operands are on different tapes");
            return x.tape()->push(value, 2, {x.index(), y.index()}, {dx, dy});
        }
    }

    inline Var operator+(const Var& x, const Var& y){ return detail::binary(x, y, x.value() + y.value(), 1.0, 1.0); }
    inline Var operator+(const Var& x, double c){ return detail::unary(x, x.value() + c, 1.0); }
    inline Var operator+(double c, const Var& x){ return x + c; }
    inline Var operator-(const Var& x, const Var& y){ return detail::binary(x, y, x.value() - y.value(), 1.0, -1.0); }
    inline Var operator-(const Var& x, double c){ return detail::unary(x, x.value() - c, 1.0); }
    inline Var operator-(double c, const Var& x){ return detail::unary(x, c - x.value(), -1.0); }
    inline Var operator-(const Var& x){ return detail::unary(x, -x.value(), -1.0); }
    inline Var operator*(const Var& x, const Var& y){ return detail::binary(x, y, x.value() * y.value(), y.value(), x.value()); }
    inline Var operator*(const Var& x, double c){ return detail::unary(x, x.value() * c, c); }
    inline Var operator*(double c, const Var& x){ return x * c; }
    inline Var operator/(const Var& x, const Var& y){
        return detail::binary(x, y, x.value() / y.value(), 1.0 / y.value(), -x.value() / (y.value() * y.value()));
    }
    inline Var operator/(const Var& x, double c){ return detail::unary(x, x.value() / c, 1.0 / c); }
    inline Var operator/(double c, const Var& x){ return detail::unary(x, c / x.value(), -c / (x.value() * x.value())); }

    inline Var exp(const Var& x){ const double e = std::exp(x.value()); return detail::unary(x, e, e); }
    inline Var log(const Var& x){ return detail::unary(x, std::log(x.value()), 1.0 / x.value()); }
    inline Var sqrt(const Var& x){ const double s = std::sqrt(x.value()); return detail::unary(x, s, 0.5 / s); }
    //standard normal cdf; derivative is the pdf
    inline Var ncdf(const Var& x){
        const double v = x.value();
        return detail::unary(x, 0.5 * std::erfc(-v / std::numbers::sqrt2), std::exp(-0.5 * v * v) / std::sqrt(2.0 * std::numbers::pi));
    }
    //max(x, 0) with derivative 1{x > 0}
    inline Var max0(const Var& x){ return detail::unary(x, x.value() > 0.0 ? x.value() : 0.0, x.value() > 0.0 ? 1.0 : 0.0); }
}
