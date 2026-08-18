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