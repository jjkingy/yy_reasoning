#ifndef YY_REASONING_INCLUDE_TICK_HPPH
#define YY_REASONING_INCLUDE_TICK_HPPH
#include <chrono>
#include <iostream>

#ifndef __ycm__
#define TICK(x) auto bench_##x = std::chrono::steady_clock::now();
#define TOCK(x)                                                     \
  printf("%s: %lfs\n", #x,                                          \
         std::chrono::duration_cast<std::chrono::duration<double>>( \
             std::chrono::steady_clock::now() - bench_##x)          \
             .count());
#else
#define TICK(x)
#define TOCK(x)
#endif

#endif