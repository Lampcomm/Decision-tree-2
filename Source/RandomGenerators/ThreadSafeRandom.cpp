#include "ThreadSafeRandom.h"

namespace RandomGenerators {
    thread_local std::mt19937 ThreadSafeRandom::s_randomGenerator(std::random_device{}());
    ThreadSafeRandom ThreadSafeRandom::Generator{};
}