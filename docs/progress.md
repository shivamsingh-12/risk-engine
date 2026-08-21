August 15 - Define probability and calculus terms, pushed
            Research how to implement the CMake files as necessary
            ci.yml implementation -> youtube and AI for help
            Work on core within cpp folder -> edited types.hpp, exceptions.hpp, result.hpp (refer back when caller inspects object), config.hpp
            Research for rng.hpp:
            Mersenne Twister -> PRNG
                Most popular variant MT19937, period of 2^(19937) - 1
                Maintains state of 624 ints & updates using bit manipulation
            Can use PCG or CSPRNGs as well but MT is fastest because of bitwise manipulation -> most suitable for this project
            Can return with engine^ stream index * golden ratio to ensure bits redistribute evenly (0x9E3779B97F4A7C15)
August 16 - Learned how to implement pdf and cdf
            Silence compiler warnings for future lognormal input vals -> reduced build log noise
            #std::normal_distribution to return in vector -> handles sampling algorithm
            Normal.hpp work
            AI suggested erfc instead of erf for better numerical precision
            Precalculate some values to potentially save time when called inside loops
            Complete internal helper to avoid confusion of negative sigma
            Using accumulate instead of a manual for loop is cleaner and less error-prone
August 17 - Finished intial phase of setting up, need to test
            Writing convergence test in C++ -> geometric loop for N, set values for mu and sigma,
            write to CSV, track larger values = lower error
            Want to write in Python - what changes?
                -> Use numpy RNG
                -> Writing to CSV will be different, see how to implement
                -> Do formulas/math transfer?
                -> Consistent geometric loop for N
August 18 - Calc and Linear Algebra notes -> learned and implemented examples of Hessian, Jacobian,
            Cholesky, etc.
August 19 - Implement covariance matrix in covariance.hpp
            Also implemented cholesky decomp in C++ after ensuring requirements are met and handling with Eigen library
            Cross language reference for implementation - stick with C++ and Python
            C++ -> use real headers and build correlation matrix by hand, fill Eigen::MatrixXd columns with normal::sample(), transpose to match ReturnMatrix convention, return N, T, N/T ratio, write everything to CSV like last experiment for comparision
            Python -> Use numpy's cholesky, transpose, calculate eigenvalues using eigvalsh, then condition number, write everything to CSV
August 20 - Finished actual implementation of both files
            Self review: Hard part was certainly Python, C++ is a language I am a lot more familiar with and the use of the Eigen library makes things so much more convienient and easy to understand and implement. Math logic is by FAR the hardest thing to understand, I needed a lot of Khan academy and youtube, Python initally was hard but is starting to feel more like pseudocode and becomes a lot easier when I can Google and see that the vast majority of math can be left to numpy, solidifying the cross language references. CSV output feels like a reward, debugging hasn't been an issue so far, but do expect it to be a problem in the future as the math gets harder and I delve further into uncharted waters.