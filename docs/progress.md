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
            Want to write in Python - what changes?
                -> Use numpy RNG
                -> Writing to CSV will be different, see how to implement
                -> Do formulas/math transfer?
                -> Consistent geometric loop for N