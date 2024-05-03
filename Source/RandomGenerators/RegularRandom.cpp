#include "RegularRandom.h"

namespace RandomGenerators {
    std::mt19937 RegularRandom::s_randomGenerator(std::random_device{}());
    RegularRandom RegularRandom::Generator{};
}