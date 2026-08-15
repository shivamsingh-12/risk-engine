August 15 - Define probability and calculus terms, pushed
            Research how to implement the CMake files as necessary
            ci.yml implementation -> youtube and Claude for help
            Work on core within cpp folder -> edited types.hpp, exceptions.hpp, result.hpp (refer back when caller inspects object), config.hpp
            Research for rng.hpp:
            Mersenne Twister -> PRNG
                Most popular variant MT19937, period of 2^(19937) - 1
                Maintains state of 624 ints & updates using bit manipulation
            Can use PCG or CSPRNGs as well but MT is fastest because of bitwise manipulation -> most suitable for this project
            Can return with engine^ stream index * golden ratio to ensure bits redistribute evenly (0x9E3779B97F4A7C15)