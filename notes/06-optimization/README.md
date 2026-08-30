Optimization
Note: Gradient, Hessians, Critical Points, Convexity have all been covered in 02-calculus

Local vs global minimum/maximum - Local minimum or maximum is the lowest or highest point in
                                  a small local area, while a global minimum or maximum is the absolute lowest or highest point across the entire domain of a function.
                                
Gradient descent - An iterative optimization algorithm used to find the minimum of a function
                   Formula: Xn+1 = Xn - ⍺▽f(Xn)
                   Xn: Current position or parameter value
                   Xn+1: The new updated position or parameter value
                   ⍺: Learning rate, which controls size of the step taken
                   ▽f(Xn): Gradient of function f at point Xn
                   Minus sign: Moves position in direction of steepest descent to minimize

Newton's method for optimization - Iterative numerical technique used to find the local
                                   minimum or maximum of a twice-differentiable function by using both first and second order derivatives.
                                   Goal: Instead of finding where the function equals zero, optimization seeks where the derivative equals zero, representing a critical point.
                                   Curvature Tracking: Uses first derivative (gradient) and second derivative (Hessian matrix) to approximate the objective function locally as a quadratic curve.
                                   Update Rule: Xk+1 = Xk - [▽^2f(Xk)]^-1▽f(Xk)

Constrained vs unconstrained optimization - Constrained optimization finds the best solution
                                            within specific limits, while unconstrained optimization searches the entire space without restrictions.
                                            Unconstrained: Uses gradient descent or Newton's method. An example would be training neural network loss functions.
                                            Constrained: Uses linear programming or sequential quadratic programming, An example would be maximizing a company's profit while staying under a budget.

Calibration (to fit model parameters to match observed data) - Iterative process of adjusting
            unknown mathematical or simulation paramters so that a model's outputs closely match real-world observed data.
            Works by defining targets, setting parameter bounds, measuring error, and then optimizing iteratively.
