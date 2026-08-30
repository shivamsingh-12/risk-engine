Numerical methods:

Floating point representation - Method used in computers to store and process real numbers
                                with fractional parts across a wide range of values.
                                Sign(S) - One bit showing positive/negative
                                Exponent(E) - Scale factor that raises base (usually 2) to a power, shifted by bias to handle poiitive/negative powers
                                Mantissa(M) - The significant digits of the number.

Numerical stability - Property of a computer algorithm describing how much it magnifies
                      rounding or input errors during calculation.
                      Forward error - Difference between computed result and true answer
                      Backward error - How small of a change to the input data makes the computed result exact
                      Condition number - Already implemented in project

Numerical differentiation - Estimates the derivative of a mathematical function or a set of
                            discrete data points using finite difference formulas.
                            Forward Difference - Approximates derivative using current point and a next point ahead (x+h), error of order h
                                Formula: f'(x) ~ (f(x+h)-f(x))/h
                            Backward Difference - Approximates the derivative using the current point and a previous point behind (x-h), error of order h
                                Formula: f'(x) ~ (f(x)-f(x-h))/h
                            Central Difference: Uses points on both sides (x+h and x-h), providing higher accuracy with an error of order h^2.
                                Formula: f'(x) ~ (f(x+h)-f(x-h))/2h

Numerical integration - Set of math rules used to find the approximate value of a definite
                        integral when a function is too hard to solve by hand or comes from data tables.
                        Common methods: midpoint rule, trapezoidal rule (self explanatory)

Root finding - Mathematical and computational process of finding values of x (called roots 
               or zeroes) that make a given function equal 0.
                   Fall in two main types: Bracketing methods (which use a safe interval containing the root) and open methods (which use one or more initial guesses to rapidly home in on the answer).
                   Bisection Method: Slow but safe bracketing method that repeatedly halves an interval [a,b] where f(a) and f(b) have opposite signs.
                   Newton-Raphson Method: An open method that uses the function's derivative and tangent lines to quickly converge on a root.
                   Secant method: Similar to Newton's method, estimates the derivative using a secant line instead of a derivative formula.
                   Brent's method: Combines root bracketing, interval bysection, secant method, and inverse quadratic interpolation to find the root of a function safely and quickly. 

Convergence rate (linear vs quadratic) - Linear convergence reduces the error by a constant
                                         fraction each step, while quadratic convergence squares the error at each step.

Interpolation in secant method - Linear interpolation between the two most recent guesses.

Interpolation in inverse quadratic - Root finding algorithm that fits a quadratic polynomial 
                                     to the inverse of a function (x as a function of y) using three preceding points.

Note: Formula for last 2 would be incorrect if typed directly, best to look on google