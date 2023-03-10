#pragma once

#include <immintrin.h>
#include "cpu/matmul.hpp"

namespace column_major {

#define C(i, j) c[(j)*ldc + (i)]

namespace microkernels {
namespace avx2 {

template <int _MR, int _NR>
struct packed_microkernel {
  void operator()(int k, const float *a, const float *b, float *c, int ldc) const;
};

template <>
struct packed_microkernel<16, 6> {
  constexpr static size_t MR = 16;
  constexpr static size_t NR = 6;

  void operator()(int k, const float *a, const float *b, float *c, int ldc) const {
    // Declare vector registers to hold 16x6 C and load them
    __m256 c_01234567_0 = _mm256_loadu_ps(&C(0, 0));
    __m256 c_89abcdef_0 = _mm256_loadu_ps(&C(8, 0));
    __m256 c_01234567_1 = _mm256_loadu_ps(&C(0, 1));
    __m256 c_89abcdef_1 = _mm256_loadu_ps(&C(8, 1));
    __m256 c_01234567_2 = _mm256_loadu_ps(&C(0, 2));
    __m256 c_89abcdef_2 = _mm256_loadu_ps(&C(8, 2));
    __m256 c_01234567_3 = _mm256_loadu_ps(&C(0, 3));
    __m256 c_89abcdef_3 = _mm256_loadu_ps(&C(8, 3));
    __m256 c_01234567_4 = _mm256_loadu_ps(&C(0, 4));
    __m256 c_89abcdef_4 = _mm256_loadu_ps(&C(8, 4));
    __m256 c_01234567_5 = _mm256_loadu_ps(&C(0, 5));
    __m256 c_89abcdef_5 = _mm256_loadu_ps(&C(8, 5));

    for (int p = 0; p < k; p++) {
      // Declare vector register for load/broadcasting B(p,j)
      __m256 b_p_j;

      // Declare a vector register to hold the current column of A and load
      // it with the four elements of that column.
      __m256 a_01234567_p = _mm256_loadu_ps(a + 0);
      __m256 a_89abcdef_p = _mm256_loadu_ps(a + 8);

      // Load/broadcast B(p,0).
      b_p_j = _mm256_broadcast_ss(b + 0);
      // update the first column of C with the current column of A times B(p,0)
      c_01234567_0 = _mm256_fmadd_ps(a_01234567_p, b_p_j, c_01234567_0);
      c_89abcdef_0 = _mm256_fmadd_ps(a_89abcdef_p, b_p_j, c_89abcdef_0);

      // REPEAT.
      b_p_j = _mm256_broadcast_ss(b + 1);
      c_01234567_1 = _mm256_fmadd_ps(a_01234567_p, b_p_j, c_01234567_1);
      c_89abcdef_1 = _mm256_fmadd_ps(a_89abcdef_p, b_p_j, c_89abcdef_1);

      b_p_j = _mm256_broadcast_ss(b + 2);
      c_01234567_2 = _mm256_fmadd_ps(a_01234567_p, b_p_j, c_01234567_2);
      c_89abcdef_2 = _mm256_fmadd_ps(a_89abcdef_p, b_p_j, c_89abcdef_2);

      b_p_j = _mm256_broadcast_ss(b + 3);
      c_01234567_3 = _mm256_fmadd_ps(a_01234567_p, b_p_j, c_01234567_3);
      c_89abcdef_3 = _mm256_fmadd_ps(a_89abcdef_p, b_p_j, c_89abcdef_3);

      b_p_j = _mm256_broadcast_ss(b + 4);
      c_01234567_4 = _mm256_fmadd_ps(a_01234567_p, b_p_j, c_01234567_4);
      c_89abcdef_4 = _mm256_fmadd_ps(a_89abcdef_p, b_p_j, c_89abcdef_4);

      b_p_j = _mm256_broadcast_ss(b + 5);
      c_01234567_5 = _mm256_fmadd_ps(a_01234567_p, b_p_j, c_01234567_5);
      c_89abcdef_5 = _mm256_fmadd_ps(a_89abcdef_p, b_p_j, c_89abcdef_5);

      a += MR;
      b += NR;
    }

    // Store the updated results
    _mm256_storeu_ps(&C(0, 0), c_01234567_0);
    _mm256_storeu_ps(&C(8, 0), c_89abcdef_0);
    _mm256_storeu_ps(&C(0, 1), c_01234567_1);
    _mm256_storeu_ps(&C(8, 1), c_89abcdef_1);
    _mm256_storeu_ps(&C(0, 2), c_01234567_2);
    _mm256_storeu_ps(&C(8, 2), c_89abcdef_2);
    _mm256_storeu_ps(&C(0, 3), c_01234567_3);
    _mm256_storeu_ps(&C(8, 3), c_89abcdef_3);
    _mm256_storeu_ps(&C(0, 4), c_01234567_4);
    _mm256_storeu_ps(&C(8, 4), c_89abcdef_4);
    _mm256_storeu_ps(&C(0, 5), c_01234567_5);
    _mm256_storeu_ps(&C(8, 5), c_89abcdef_5);
  }
};

template <>
struct packed_microkernel<16, 8> {
  constexpr static size_t MR = 16;
  constexpr static size_t NR = 8;

  void operator()(int k, const float *a, const float *b, float *c, int ldc) const {
    // Declare vector registers to hold 16x8 C and load them
    __m256 c_01234567_0 = _mm256_loadu_ps(&C(0, 0));
    __m256 c_89abcdef_0 = _mm256_loadu_ps(&C(8, 0));
    __m256 c_01234567_1 = _mm256_loadu_ps(&C(0, 1));
    __m256 c_89abcdef_1 = _mm256_loadu_ps(&C(8, 1));
    __m256 c_01234567_2 = _mm256_loadu_ps(&C(0, 2));
    __m256 c_89abcdef_2 = _mm256_loadu_ps(&C(8, 2));
    __m256 c_01234567_3 = _mm256_loadu_ps(&C(0, 3));
    __m256 c_89abcdef_3 = _mm256_loadu_ps(&C(8, 3));
    __m256 c_01234567_4 = _mm256_loadu_ps(&C(0, 4));
    __m256 c_89abcdef_4 = _mm256_loadu_ps(&C(8, 4));
    __m256 c_01234567_5 = _mm256_loadu_ps(&C(0, 5));
    __m256 c_89abcdef_5 = _mm256_loadu_ps(&C(8, 5));
    __m256 c_01234567_6 = _mm256_loadu_ps(&C(0, 6));
    __m256 c_89abcdef_6 = _mm256_loadu_ps(&C(8, 6));
    __m256 c_01234567_7 = _mm256_loadu_ps(&C(0, 7));
    __m256 c_89abcdef_7 = _mm256_loadu_ps(&C(8, 7));

    for (int p = 0; p < k; p++) {
      // Declare vector register for load/broadcasting B(p,j)
      __m256 b_p_j;

      // Declare a vector register to hold the current column of A and load
      // it with the four elements of that column.
      __m256 a_01234567_p = _mm256_loadu_ps(a + 0);
      __m256 a_89abcdef_p = _mm256_loadu_ps(a + 8);

      // Load/broadcast B(p,0).
      b_p_j = _mm256_broadcast_ss(b + 0);
      // update the first column of C with the current column of A times B(p,0)
      c_01234567_0 = _mm256_fmadd_ps(a_01234567_p, b_p_j, c_01234567_0);
      c_89abcdef_0 = _mm256_fmadd_ps(a_89abcdef_p, b_p_j, c_89abcdef_0);

      // REPEAT.
      b_p_j = _mm256_broadcast_ss(b + 1);
      c_01234567_1 = _mm256_fmadd_ps(a_01234567_p, b_p_j, c_01234567_1);
      c_89abcdef_1 = _mm256_fmadd_ps(a_89abcdef_p, b_p_j, c_89abcdef_1);

      b_p_j = _mm256_broadcast_ss(b + 2);
      c_01234567_2 = _mm256_fmadd_ps(a_01234567_p, b_p_j, c_01234567_2);
      c_89abcdef_2 = _mm256_fmadd_ps(a_89abcdef_p, b_p_j, c_89abcdef_2);

      b_p_j = _mm256_broadcast_ss(b + 3);
      c_01234567_3 = _mm256_fmadd_ps(a_01234567_p, b_p_j, c_01234567_3);
      c_89abcdef_3 = _mm256_fmadd_ps(a_89abcdef_p, b_p_j, c_89abcdef_3);

      b_p_j = _mm256_broadcast_ss(b + 4);
      c_01234567_4 = _mm256_fmadd_ps(a_01234567_p, b_p_j, c_01234567_4);
      c_89abcdef_4 = _mm256_fmadd_ps(a_89abcdef_p, b_p_j, c_89abcdef_4);

      b_p_j = _mm256_broadcast_ss(b + 5);
      c_01234567_5 = _mm256_fmadd_ps(a_01234567_p, b_p_j, c_01234567_5);
      c_89abcdef_5 = _mm256_fmadd_ps(a_89abcdef_p, b_p_j, c_89abcdef_5);

      b_p_j = _mm256_broadcast_ss(b + 6);
      c_01234567_6 = _mm256_fmadd_ps(a_01234567_p, b_p_j, c_01234567_6);
      c_89abcdef_6 = _mm256_fmadd_ps(a_89abcdef_p, b_p_j, c_89abcdef_6);

      b_p_j = _mm256_broadcast_ss(b + 7);
      c_01234567_7 = _mm256_fmadd_ps(a_01234567_p, b_p_j, c_01234567_7);
      c_89abcdef_7 = _mm256_fmadd_ps(a_89abcdef_p, b_p_j, c_89abcdef_7);

      a += MR;
      b += NR;
    }

    // Store the updated results
    _mm256_storeu_ps(&C(0, 0), c_01234567_0);
    _mm256_storeu_ps(&C(8, 0), c_89abcdef_0);
    _mm256_storeu_ps(&C(0, 1), c_01234567_1);
    _mm256_storeu_ps(&C(8, 1), c_89abcdef_1);
    _mm256_storeu_ps(&C(0, 2), c_01234567_2);
    _mm256_storeu_ps(&C(8, 2), c_89abcdef_2);
    _mm256_storeu_ps(&C(0, 3), c_01234567_3);
    _mm256_storeu_ps(&C(8, 3), c_89abcdef_3);
    _mm256_storeu_ps(&C(0, 4), c_01234567_4);
    _mm256_storeu_ps(&C(8, 4), c_89abcdef_4);
    _mm256_storeu_ps(&C(0, 5), c_01234567_5);
    _mm256_storeu_ps(&C(8, 5), c_89abcdef_5);
    _mm256_storeu_ps(&C(0, 6), c_01234567_6);
    _mm256_storeu_ps(&C(8, 6), c_89abcdef_6);
    _mm256_storeu_ps(&C(0, 7), c_01234567_7);
    _mm256_storeu_ps(&C(8, 7), c_89abcdef_7);
  }
};

}  // namespace avx2
}  // namespace microkernels
}  // namespace column_major
