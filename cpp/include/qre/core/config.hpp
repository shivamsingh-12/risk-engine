#pragma once
#include <cstdint>
namespace qre::config{
    //Genuinely just hardcoding numbers that would be hardcoded and duplicated across files regardless
    inline constexpr double kDefTolerance = 1e-8;
    inline constexpr int kDefMaxIterations = 100;
    inline constexpr uint64_t kDefSeed = 42;
}