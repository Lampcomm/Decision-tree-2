#ifndef DECISION_TREE_2_REGULARRANDOM_H
#define DECISION_TREE_2_REGULARRANDOM_H

#include <random>
namespace RandomGenerators {
    class RegularRandom {
    public:
        using result_type = std::mt19937::result_type;

        static RegularRandom Generator;

        auto operator()() { return s_randomGenerator(); }
        static constexpr auto min() { return std::mt19937::min(); }
        static constexpr auto max() { return std::mt19937::max(); }

        RegularRandom(const RegularRandom&) = delete;
        RegularRandom& operator=(const RegularRandom&) = delete;

        RegularRandom(RegularRandom&&) = delete;
        RegularRandom& operator=(RegularRandom&&) = delete;

    private:
        RegularRandom() = default;

    private:
        static std::mt19937 s_randomGenerator;
    };
}

#endif