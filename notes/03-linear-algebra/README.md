Linear Algebra:

Vector Space: A collection of objects called vectors that can be added together and multipled or scaled by
              numbers called scalars.
              Requires four elements:
                1. A set of vectors (V)
                2. A field of scalars (F), typically real numbers (R)
                3. Vector addition, combining two to make a third
                4. Scalar multiplication, resizing a vector by a number
              Must satisfy ten fundamental properties:
                1. Closure under addition: Adding two vectors stays in the space.
                2. Closure under scalar multiplication: Scaling a vector stays in the space.
                3. Commutativity: u + v = v + u
                4. Associativity of addition: (u + v) + w = u + (v + w)
                5. Additive identity: A zero vector 0 exists where v + 0 = w
                6. Additive inverse: For every v, there is -v such that v + -v = 0.
                7. Distributivity of scalar sums: a(u + v) = au + av
                8. Distributivity of vector sums: (a + b)v = av + bv
                9. Associativity of scalar multiplication: a(bv) = (ab)v
                10. Multiplicative identity: Multiplying by 1 leaves vecto unchanged (1v = v).

Linear Combination: A math expression made by multiplying each term in a set by a scalar and adding all
                    the results together. For example, a linear combination of x and y is written as ax + by, where a and b are numbers.

Inner Product: An inner product is a generalization of the dot product. In a vector space, it is a way to
               multiply vectors together, with the result being a scalar.
               Satisfy 4 properties:
                1. <u + v, w> = <u,w> + <v,w>
                2. <⍺ v, w> = ⍺ <v,w>
                3. <v,w> = <w,v>
                4. <v,v> >= 0 and equal if and only v = 0. (positive-definite)

L1 and L2 Norm: Mathematical methods used to measure the size, length, or magnitude of a vector.
                L1 Norm (Manhattan/Taxicab Norm):
                Formula: ||x1|| = sigma(n, i=1) |xi|
                Measures distance as if walking on a grid of city blocks
                L2 Norm (Euclidean Norm):
                Formula: ||x2|| = sqrt(sigma(n, i=1)xi^2)
                Measures shortest straight line distance from the origin

Orthogonality: The geometric idea of perpendicularity to vector spaces. Two vectors u and v are orthogonal
               if u * v = 0.

Linear transformation: A special type of function between vector spaces that preserves the operations of
                       vector addition and scalar multiplication.

Eigenvalue: A special set of scalars associated with a linear system of equations (a matrix equation)
            sometimes also known as characteristic roots, characteristic values, proper values, or latent roots.

Eigenvector: A special non-zero vector that does not change direction when a linear transformation (matrix)
             is applied to it; it only gets stretched, shrunk, or flipped by a scalar value called an eigenvalue.

Eigendecomposition: The breakdown of a square matrix into a set of its eigenvectors and eigenvalues. It
                    writes a matrix A as A = PDP^-1, where P holds the eigenvectors and D is a diagonal matrix with the eigenvalues on its main diagonal.
                    Step by step example for 2x2 matrix:
                    
                    A = [4    1]
                        [6    3]

                    Step 1: Find the eigenvalues(ƛ)
                    Solve characteristic equation det(A - ƛl) = 0

                    det(4-ƛ    1)
                       (6    3-ƛ) = 0
                    
                    (4 - ƛ)(3 - ƛ)-(1)(6) = 0
                    ƛ^2 - 7√ + 12 - 6 = 0
                    ƛ^2 - 7ƛ + 6 = 0
                    (ƛ - 6)(ƛ - 1) = 0
                    Eigenvalues are 6 and 1, giving us D

                    D = (6   0)
                        (0   1)

                    Step 2: Find the eigenvectors
                    For each eigenvalue, solve (A-ƛl)v = 0 to find corresponding vector
                    For ƛ1 = 6:
                    (4 - 6    1) (x) =  (0)   ->  (-2    1) (x) = (0)
                    (6    3 - 6) (y)    (0)   ->  (6    -3) (y)   (0)
                    This reduces -2x+y=0 or y=2x. If x = 1, then y = 2.
                    The first eigenvector is v1 = (1)
                                                  (2)
                    For ƛ2 = 1:
                    (4 - 1    1) (x) =  (0)   ->  (3    1) (x) = (0)
                    (6    3 - 1) (y)    (0)   ->  (6    2) (y)   (0)
                    This reduces 3x+y=0 or y=-3x. If x = 1, then y = -3.
                    The first eigenvector is v1 = (1)
                                                  (-3)   

                    Step 3: Construct Matrix P and Find P^-1
                    Put the eigenvectors together as columns to form P:

                    P = (v1   v2) = (1   1)
                                    (2  -3)
                    Next find the inverse matrix P^-1 using the 2x2 formula:
                     1/(ad-bc)(d   -b)
                              (-c   a)
                    det(P) = (1)(-3) - (1)(2) = -5

                    P^-1 = 1/-5 (-3   -1)   (3/5   1/5)
                                (-2    1) = (2/5  -1/5)                 

                    Step 4: Final Eigendecomposition Formula
                    Can fully express A = PDP^-1
                    [4    1] = (1   1)  (6   0)  (3/5   1/5)
                    [6    3] = (2  -3)  (0   1)  (2/5  -1/5)

Positive definite matrix: A square matrix where the quadratic form x^TAx is strictly greater than zero for
                          every non-zero vector x. It acts like a positive real number in higher dimensions, meaning all its eigenvalues are strictly positive, and it creates an upward-curving bowl shape geometrically.
                          -> Guarantees a strict local minimum when checking the second-deriviative Hessian matrix.

Positive semi-definite matrix: A square, symmetric matrix whose quadratic form x^TAx >= 0 for all vectors 
                               x, and all of its eigenvalues are greater than or equal to zero. It acts like a non-negative number in multi-dimensional space.

Matrix decomposition: Splits a complex matrix into a product of simpler, constituent matrices. It acts like
                      factoring a number into primes, making heavy computations like solving linear equations, finding inverses, and extracting data features faster and easier.

Cholesky decomposition: Factorizes a symmetric, positive-definite matrix into the product of a lower
                        triangular matrix and its transpose (A = LL^T). Twice as fast as LU decomposition.
                        Core requirements and formula:
                        -> Symmetry: matrix A must equal its tranpose (A=A^T)
                        -> Positive-definite: All eigenvalues and leading principal subdeterminants must be strictly positive.
                        -> Factorization form: A = LL^T where L has real, positive numbers on its main diagonal.
                        Example:
                            [4     12      -16]
                        A = [12    37      -43]
                            [-16   -43      98]

                        Because A is symmetric and positive-definite, we can find lower triangular matrix L

                            [L11    0      0]
                        L = [L21   L22     0]
                            [L31   L32   L33]

                        Step by Step Derivation
                        Calculate entries of L column by column:
                        Column 1:
                            L11: sqrt(A11) = sqrt(4) = 2
                            L21: L21 = A21/L11 = 12/2 = 6
                            L31 = A31/L11 = -16/2 = -8
                        Column 2:
                            L22: Subtract the squared prior elements in this row from A22, then take the square root.
                            L22 = sqrt(A22 - (L21)^2) = sqrt(37 - 6^2) = sqrt(1) = 1
                            L32 = (A32 - (L31 * L21))/L22 = (-43-(-8 * 6))/1 = (-43 + 48)/ 1 = 5
                        Column 3:
                            L33: Subtract all prior squared elements in this row from A33, then take the square root.
                            L33 = sqrt(A33 - ((L31)^2 + (L32)^2)) = sqrt(98 - ((-8)^2) + 5^2) = sqrt(98 -89) = 3
                        
                            [2    0    0]
                        L = [6    1    0]
                            [-8   5    3]

                              [2    6  -8]
                        L^T = [0    1   5]
                              [0    0   3]

Condition number - For matrix A, the condition number x(A) equals the norm of A multiplied by the norm of
                   of its inverse matrix (A^-1).
