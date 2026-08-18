Calculus:

Limit: The value that a function or sequence approaches as the input gets closer and closer to a given number.
    Written as lim x->c f(x) = L

Derivative: The instant rate of change of a function. Shows the slope of a line touching a curve at one exact point. 
    Written as f'(x) or dy/dx

Integral: Measures the area under a curve or adds up many small slices to find a whole. Reverse process of a derivative.
    Definite: Has set limits and calculates a specific net numerical area.
    Indefinite: Has no limits and finds a general function (an antiderivative) plus a constant.

FTC (Fundamental Theorem of Calculus):
    Part 1: Derivative of an Integral
        States if f is a continuous function on an interval, then defining a function F(x) as the accumulation of area from a constant a to a variable x:
            F(x) = a integral x f(t)dt

            Gives back: F'(x) = d/dx [a integral x f(t)dt ] = f(x)

    Part 2: Evaluation of Definite Integrals
        Provides primary tool for computing definite integrals without using Riemann sums. If f is continuous on [a,b] and F is any antiderivative of f (so that F' = f) then:
            a integral b f(x)dx = F(b) - F(a)

Partial Derivative - A derivative of a function with multiple variables where you find the rate of change
                     with respect to just one variable while holding the rest constant.
                     Example - f(x,y) = x^2y^3 + 5x
                     To find with respect to x (∂f/∂x), y is the constant: ∂f/∂x = 2xy^3 + 5
                     To find with respect to y (∂f/∂y), x is the constant: ∂f/∂y = 3x^2y^2 + 0 = 3x^2y^2

Gradient - A vector that represents the maximum rate of change of a scalar function at any given point. It
           points directly in the direction of the steepest ascent and its magnitude measures the steepness of that slope.
           Steps: Find the partial derivatives (f(x,y) = 2x^2+3y^2 -> ∂f/∂x = 4x, ∂f/∂y = 6y), form the gradient vector with ▽f(x,y) = (4x,6y), evaluate at a specific point(f(1,2) = (4,12))
              -> Direction of steepest ascent: Standing at 1,2 then moving in 4,12 will take you up the hill the fastest
              Can use Pythagorean Theorem for magnitude: sqrt(4^2 + 12^2) = sqrt(16 + 144) = sqrt(160) = 12.65

Jacobian - The Jacobian or the Jacobian Matrix acts as a multi-dimensional derivative that shows how a   
           system of equations changes locally.
    If a function takes inputs (x,y) and ouputs (u,v), the matrix organizes all partial derivatives into rows(outputs) and columns(inputs)

            J = [ ∂u/∂x    ∂u/∂y ]
                [ ∂v/∂x    ∂v/∂y ]

    Jacobian Determinant is the calculated value of this matrix:
    |J| = (∂u/∂x * ∂v/∂y) - (∂u/∂y * ∂v/∂x)

    Example: Polar to Cartesian Coordinates

    1. Define equations
    x = rcos(θ)
    y = rsin(θ)

    2.Partial Derivatives
    Row 1 for x
    ∂x/∂r = cos(θ)       
    ∂x/∂θ = -rsin(θ)
    Row 2 for y
    ∂y/∂r = sin(θ)       
    ∂y/∂θ = rcos(θ)

    3.Matrix
    J = [cos(θ)  -rsin(θ)]
        [sin(θ)   rcos(θ)]

    4.Determinant
    det(J) = (cos(θ) * rcos(θ)) - (-rsin(θ) * sin(θ))
    det(J) = rcos^2(θ)+rsin^2(θ)
    cos^2(θ) + sin^2(θ) = 1 -> det(J) = r

Hessian - The Hessian Matrix is a square matrix of second order partial derivatives of a scalar valued 
          function. It describes the local curvature of a multivariable function, extending the concept of the single-variable second derivative (f''(x)) to higher dimensions.
    Mathematical function -> For a twice-differentiable function R^n -> R, the Hessian matrix H is an nxn matrix as follows:
        H = [∂^2f/∂x^21   ∂^2f/∂x1∂x2  ...  ∂^2f/∂x1∂xn   ]
            [∂^2f/∂x2∂x1  ∂^2f/∂x^22   ...  ∂^2f/∂x2∂xn   ]
            [      .          .                    .      ]
            [      .          .                    .      ]
            [      .          .                    .      ]
            [∂^2f/∂xn∂x1  ∂^2f/∂xn∂x2  ...  ∂^2f/∂x^2n    ]

    Step by step:
    f(x,y) = x^3-3xy+y^3
    Step 1: First Partial Derivatives
    ∂f/∂x: 3x^2-3y
    ∂f/∂y: -3x+3y^2
    ▿f(x,y) = [3x^2-3y, -3x+3y^2]^T
    Step 2: Find Second Partial Derivatives
    ∂^2f/∂x^2: ∂/∂x(3x^2-3y) = 6x
    ∂^2f/∂y^2: ∂/∂y(-3x+3y^2) = 6y
    ∂^2f/∂x∂y: ∂/∂y(3x^2-3y) = -3
    ∂^2f/∂y∂x: ∂/∂x(-3x+3y^2) = -3
    Step 3: Construct Hessian Matrix
        H(x,y) = [6x -3]
                 [-3 6y]
    Allows us to classify crit points (where gradient f = 0) into local minima/maxima or saddle points.
    Our first derivatives yield (0,0) and (1,1).
    Hessian Determinant: D = det(H) = fxxfyy-(fxy)^2
    Case 1: Test (0,0)
        H(0,0) = [0 -3]
                 [-3 0]
        Determinant: 0*0 - (-3)^2 = -9
        Because determinant is negative, the function curves upward one direction and down in another, making (0,0) a saddle point.
    Case 2: Test (1,1)
        H(1,1) = [6 -3]
                 [-3 6]
        Determinant: 6*6 - (-3)^2 = 36 - 9 = 27
        Because determinant is positve, check top left element. Since 6 > 0, the function curves upwards in all directions -> (1,1) is a local minimum.

Critical point: Value x = c in a function's domain where the first derivative is zero or does not exist.
                These points are vital because local max/min and horizontal inflection points occur here.

Convexity: Curves upward in a sense when two points are chosen and a line is drawn it's above the graph. If
        a function has a second derivative, it is convex when f''(x) >= 0 everywhere in its domain.

Multivariable Chain Rule - Finds the derivative of a composite function where an output depends on 
                           multiple intermediate variables, which in turn depend on one or more independent inputs.
                           Case 1: One indepedent variable
                           For z = f(x,y) where x = x(t) and y = y(t), total derivative is ∂z/∂t = ∂f/∂x * dx/dt + ∂f/∂y * dy/dt.
                           Case 2: Multiple Independent Variables (Partial Derivatives)
                           If z = f(x,y) where x = x(u,v) and y = (u,v) depend on two variables, you look
                           for partial changes with respect to both u and v.
                           To find ∂z/∂u, use: ∂z/∂x * ∂x/∂u + ∂z/∂y * ∂y/∂u
                           To find ∂z/∂v, use: ∂z/∂x * ∂x/∂v + ∂z/∂y * ∂y/∂v
                           General Matrix Form: For vector-valued functions where y=f(u) and u=g(x), the derivative of the composite function is the product of their individual Jacobian matrices.
                           D(f⊙g)(x) = Df(g(x)) * Dg(x)                         
