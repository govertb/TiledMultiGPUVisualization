/*
 ==============================================================================

 benchmark_setup.hpp
 Author: Govert Brinkmann, unless a 'due' is given.

 This code was developed as part of research at the Leiden Institute of
 Advanced Computer Science (https://liacs.leidenuniv.nl).

 ==============================================================================
*/

#ifndef benchmark_setup_hpp
#define benchmark_setup_hpp

#include <chrono>
#include <ratio>

// select clock type used for benchmarks
typedef std::chrono::steady_clock benchmark_clock_t;

// assert it is steady...
static_assert(benchmark_clock_t::is_steady,
              "benchmark_clock_t should be steady");

// and has sufficient resolution
static_assert(std::ratio_less_equal<benchmark_clock_t::period, std::micro>::value,
              "benchmark_clock_t resolution should be ≤ 1us");

// convert a benchmark_clock_t::duration to milliseconds
// due to: https://en.cppreference.com/w/cpp/chrono/duration/duration_cast
float to_ms(benchmark_clock_t::duration d)
{
   return std::chrono::duration<double, std::milli>(d).count();
}

#endif
