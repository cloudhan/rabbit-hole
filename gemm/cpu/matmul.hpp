#pragma once

#include <cstdint>
#include <iostream>
#include <sstream>

namespace detail {
template <typename T>
void make_string_impl(std::ostringstream& oss, const T& arg) {
  oss << arg;
}

template <typename T, typename... Ts>
void make_string_impl(std::ostringstream& oss, const T& head, Ts... tail) {
  oss << head;
  make_string_impl(oss, tail...);
}
}  // namespace detail

template <typename... Ts>
std::string make_string(Ts... args) {
  std::ostringstream oss;
  detail::make_string_impl(oss, args...);
  return oss.str();
}

#define ENFORCE(condition, ...)          \
  if (!(condition)) {                    \
    auto msg = make_string(__VA_ARGS__); \
    std::cerr << msg << std::endl;       \
    throw msg;                           \
  }

// define simple matmul without considering considering
#define MATMUL_SIGNATURE(name)                                                                                         \
  void name(                                                                                                           \
      int64_t m, int64_t n, int64_t k, const float* a, int64_t lda, const float* b, int64_t ldb, float* c, int64_t ldc \
  )
