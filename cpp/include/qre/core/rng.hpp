#pragma once
#include <cstdint>
#include <random>
#include "qre/core/config.hpp"
        //Throughout the file when engine_() is called it advances its internal state and returns a raw
        //random number.
namespace qre{
    class Rng{
        public:
            explicit Rng(uint64_t seed = config::kDefSeed) : engine_(seed) {}
            void seed(uint64_t seed) { engine_.seed(seed); }
            //std::mt19937_64 is the Mersenne Twister engine -> standard general PRNG in C++ lib
            std::mt19937_64& engine() {return engine_;}
            // spawn() specifically uses engine_() + XOR with a large odd constant -> produce new but
            // different seed per stream index
            Rng spawn(uint64_t stream_index){
                return Rng(engine_() ^ (stream_index * 0x9E3779B97F4A7C15ull));
            }
        private:
            std::mt19937_64 engine_;
    };
}