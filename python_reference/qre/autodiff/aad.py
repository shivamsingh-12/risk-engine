"""Phase 12 reference: adjoint algorithmic differentiation (reverse-mode automatic differentiation).

Any program is a chain of elementary operations. Recording each operation on a tape, with the local
partial derivatives of its result with respect to its arguments, lets the chain rule run backwards:
    adjoint(x) = d output / d x,    adjoint(output) = 1,
    for each node v = g(a, b) in reverse order:  adjoint(a) += dg/da * adjoint(v),  adjoint(b) += dg/db * adjoint(v)
One backward sweep gives the derivative of one output with respect to every input. Its cost is a small
constant multiple of the original computation (roughly 2 to 5 times) however many inputs there are.
Bumping costs one or two extra evaluations per input, so AAD wins as soon as there are more than a few.

Derivatives are exact to rounding (no step size h), but only of the program as written: where the program
has a jump (an indicator, a digital payoff), the derivative is 0 away from the jump and the jump itself is
invisible, exactly like the pathwise Monte Carlo estimator.

This is a deliberately small scalar implementation to make the mechanics visible; the C++ version
(qre/autodiff/aad.hpp) is the one used for speed.
"""
import math


class Tape:
    def __init__(self):
        self.parents = []      # per node: tuple of parent indices
        self.partials = []     # per node: tuple of d node / d parent

    def __len__(self):
        return len(self.parents)

    def _push(self, parents, partials):
        self.parents.append(parents)
        self.partials.append(partials)
        return len(self.parents) - 1

    def variable(self, value):
        """An input: a node with no parents."""
        return Var(float(value), self, self._push((), ()))

    def gradient(self, output, inputs):
        """d output / d input for each input, from one backward sweep."""
        if output.tape is not self:
            raise ValueError("output was not recorded on this tape")
        adj = [0.0] * len(self.parents)
        adj[output.idx] = 1.0
        for i in range(output.idx, -1, -1):
            a = adj[i]
            if a == 0.0:
                continue
            for p, d in zip(self.parents[i], self.partials[i]):
                adj[p] += d * a
        return [adj[x.idx] for x in inputs]


class Var:
    __slots__ = ("value", "tape", "idx")

    def __init__(self, value, tape, idx):
        self.value, self.tape, self.idx = value, tape, idx

    def _unary(self, value, d):
        return Var(value, self.tape, self.tape._push((self.idx,), (d,)))

    def _binary(self, other, value, d_self, d_other):
        return Var(value, self.tape, self.tape._push((self.idx, other.idx), (d_self, d_other)))

    def __add__(self, o):
        if isinstance(o, Var):
            return self._binary(o, self.value + o.value, 1.0, 1.0)
        return self._unary(self.value + o, 1.0)

    __radd__ = __add__

    def __sub__(self, o):
        if isinstance(o, Var):
            return self._binary(o, self.value - o.value, 1.0, -1.0)
        return self._unary(self.value - o, 1.0)

    def __rsub__(self, o):
        return self._unary(o - self.value, -1.0)

    def __mul__(self, o):
        if isinstance(o, Var):
            return self._binary(o, self.value * o.value, o.value, self.value)
        return self._unary(self.value * o, float(o))

    __rmul__ = __mul__

    def __truediv__(self, o):
        if isinstance(o, Var):
            return self._binary(o, self.value / o.value, 1.0 / o.value, -self.value / (o.value * o.value))
        return self._unary(self.value / o, 1.0 / o)

    def __rtruediv__(self, o):
        return self._unary(o / self.value, -o / (self.value * self.value))

    def __neg__(self):
        return self._unary(-self.value, -1.0)


def exp(x):
    e = math.exp(x.value)
    return x._unary(e, e)


def log(x):
    return x._unary(math.log(x.value), 1.0 / x.value)


def sqrt(x):
    s = math.sqrt(x.value)
    return x._unary(s, 0.5 / s)


def ncdf(x):
    """Standard normal cdf, derivative is the pdf."""
    return x._unary(0.5 * math.erfc(-x.value / math.sqrt(2.0)), math.exp(-0.5 * x.value ** 2) / math.sqrt(2.0 * math.pi))


def max0(x):
    """max(x, 0) with derivative 1{x > 0} (the kink at 0 has measure zero for continuous inputs)."""
    return x._unary(max(x.value, 0.0), 1.0 if x.value > 0.0 else 0.0)
