#ifndef DECISION_TREE_2_THREADSAFERANDOM_H
#define DECISION_TREE_2_THREADSAFERANDOM_H

#include <random>
namespace RandomGenerators {
    class ThreadSafeRandom {
    public:
        using result_type = std::mt19937::result_type;

        static ThreadSafeRandom Generator;

        auto operator()() { return s_randomGenerator(); }
        static constexpr auto min() { return std::mt19937::min(); };
        static constexpr auto max() { return std::mt19937::max(); }

        ThreadSafeRandom(const ThreadSafeRandom&) = delete;
        ThreadSafeRandom& operator=(const ThreadSafeRandom&) = delete;

        ThreadSafeRandom(ThreadSafeRandom&&) = delete;
        ThreadSafeRandom& operator=(ThreadSafeRandom&&) = delete;

    private:
        ThreadSafeRandom() = default;

    private:
        static thread_local std::mt19937 s_randomGenerator;
    };
}

#endif
