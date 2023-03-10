#include <immintrin.h>
#include "cpu/matmul.hpp"

namespace column_major {

#define A(i, j) a[(j)*lda + (i)]
#define B(i, j) b[(j)*ldb + (i)]
#define C(i, j) c[(j)*ldc + (i)]

namespace microkernels {
namespace adhoc {
inline void naive_4x4(int k, const float *a, int lda, const float *b, int ldb, float *c, int ldc) {
  // ordered as o[j][i], default initialized to zero
  float o[4][4] = {};

  // manually unrolled inner loop
  for (int p = 0; p < k; p++) {
    o[0][0] += A(0, p) * B(p, 0);
    o[0][1] += A(1, p) * B(p, 0);
    o[0][2] += A(2, p) * B(p, 0);
    o[0][3] += A(3, p) * B(p, 0);

    o[1][0] += A(0, p) * B(p, 1);
    o[1][1] += A(1, p) * B(p, 1);
    o[1][2] += A(2, p) * B(p, 1);
    o[1][3] += A(3, p) * B(p, 1);

    o[2][0] += A(0, p) * B(p, 2);
    o[2][1] += A(1, p) * B(p, 2);
    o[2][2] += A(2, p) * B(p, 2);
    o[2][3] += A(3, p) * B(p, 2);

    o[3][0] += A(0, p) * B(p, 3);
    o[3][1] += A(1, p) * B(p, 3);
    o[3][2] += A(2, p) * B(p, 3);
    o[3][3] += A(3, p) * B(p, 3);
  }

  for (int j = 0; j < 4; j++) {
    for (int i = 0; i < 4; i++) {
      C(i, j) = o[j][i];
    }
  }
}
}  // namespace adhoc
}  // namespace microkernels

MATMUL_SIGNATURE(matmul_tile_IJ_naive_4x4) {
  ENFORCE(m % 4 == 0, "m must be divisible by 4");
  ENFORCE(n % 4 == 0, "n must be divisible by 4");

  for (int i = 0; i < m; i += 4) {
    for (int j = 0; j < n; j += 4) {
      microkernels::adhoc::naive_4x4(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc);
    }
  }
}

MATMUL_SIGNATURE(matmul_tile_JI_naive_4x4) {
  ENFORCE(m % 4 == 0, "m must be divisible by 4");
  ENFORCE(n % 4 == 0, "n must be divisible by 4");

  for (int j = 0; j < n; j += 4) {
    for (int i = 0; i < m; i += 4) {
      microkernels::adhoc::naive_4x4(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc);
    }
  }
}

namespace microkernels {
namespace adhoc {
inline void avx_4x4(int k, const float *a, int lda, const float *b, int ldb, float *c, int ldc) {
  // Declare vector registers to hold 4x4 C and load them
  __m128 c_0123_0 = _mm_loadu_ps(&C(0, 0));
  __m128 c_0123_1 = _mm_loadu_ps(&C(0, 1));
  __m128 c_0123_2 = _mm_loadu_ps(&C(0, 2));
  __m128 c_0123_3 = _mm_loadu_ps(&C(0, 3));

  for (int p = 0; p < k; p++) {
    // Declare vector register for load/broadcasting B(p,j)
    __m128 b_p_j;

    // Declare a vector register to hold the current column of A and load it with the 4 elements of that column.
    __m128 a_0123_p = _mm_loadu_ps(&A(0, p));

    // Load/broadcast B(p,0).
    b_p_j = _mm_broadcast_ss(&B(p, 0));

    // Update the first column of C with the current column of A times B(p,0).
    c_0123_0 = _mm_fmadd_ps(a_0123_p, b_p_j, c_0123_0);

    // REPEAT for second, third, and fourth columns of C. Notice that the current column of A is reused.
    b_p_j = _mm_broadcast_ss(&B(p, 1));
    c_0123_1 = _mm_fmadd_ps(a_0123_p, b_p_j, c_0123_1);

    b_p_j = _mm_broadcast_ss(&B(p, 2));
    c_0123_2 = _mm_fmadd_ps(a_0123_p, b_p_j, c_0123_2);

    b_p_j = _mm_broadcast_ss(&B(p, 3));
    c_0123_3 = _mm_fmadd_ps(a_0123_p, b_p_j, c_0123_3);
  }

  // Store the updated results.
  _mm_storeu_ps(&C(0, 0), c_0123_0);
  _mm_storeu_ps(&C(0, 1), c_0123_1);
  _mm_storeu_ps(&C(0, 2), c_0123_2);
  _mm_storeu_ps(&C(0, 3), c_0123_3);
}
}  // namespace adhoc
}  // namespace microkernels

MATMUL_SIGNATURE(matmul_tile_IJ_avx_4x4) {
  ENFORCE(m % 4 == 0, "m must be divisible by 4");
  ENFORCE(n % 4 == 0, "n must be divisible by 4");

  for (int i = 0; i < m; i += 4) {
    for (int j = 0; j < n; j += 4) {
      microkernels::adhoc::avx_4x4(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc);
    }
  }
}

MATMUL_SIGNATURE(matmul_tile_JI_avx_4x4) {
  ENFORCE(m % 4 == 0, "m must be divisible by 4");
  ENFORCE(n % 4 == 0, "n must be divisible by 4");

  for (int j = 0; j < n; j += 4) {
    for (int i = 0; i < m; i += 4) {
      microkernels::adhoc::avx_4x4(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc);
    }
  }
}

namespace microkernel {
inline void avx2_8x4(int k, const float *a, int lda, const float *b, int ldb, float *c, int ldc) {
  // Declare vector registers to hold 8x4 C and load them
  __m256 c_01234567_0 = _mm256_loadu_ps(&C(0, 0));
  __m256 c_01234567_1 = _mm256_loadu_ps(&C(0, 1));
  __m256 c_01234567_2 = _mm256_loadu_ps(&C(0, 2));
  __m256 c_01234567_3 = _mm256_loadu_ps(&C(0, 3));

  for (int p = 0; p < k; p++) {
    // Declare vector register for load/broadcasting B(p,j)
    __m256 b_p_j;

    // Declare a vector register to hold the current column of A and load it with the 8 elements of that column.
    __m256 a_01234567_p = _mm256_loadu_ps(&A(0, p));

    // Load/broadcast B(p,0).
    b_p_j = _mm256_broadcast_ss(&B(p, 0));

    // Update the first column of C with the current column of A times B(p,0).
    c_01234567_0 = _mm256_fmadd_ps(a_01234567_p, b_p_j, c_01234567_0);

    // REPEAT for second, third, and fourth columns of C. Notice that the current column of A is reused.
    b_p_j = _mm256_broadcast_ss(&B(p, 1));
    c_01234567_1 = _mm256_fmadd_ps(a_01234567_p, b_p_j, c_01234567_1);

    b_p_j = _mm256_broadcast_ss(&B(p, 2));
    c_01234567_2 = _mm256_fmadd_ps(a_01234567_p, b_p_j, c_01234567_2);

    b_p_j = _mm256_broadcast_ss(&B(p, 3));
    c_01234567_3 = _mm256_fmadd_ps(a_01234567_p, b_p_j, c_01234567_3);
  }

  // Store the updated results.
  _mm256_storeu_ps(&C(0, 0), c_01234567_0);
  _mm256_storeu_ps(&C(0, 1), c_01234567_1);
  _mm256_storeu_ps(&C(0, 2), c_01234567_2);
  _mm256_storeu_ps(&C(0, 3), c_01234567_3);
}
}  // namespace microkernel

MATMUL_SIGNATURE(matmul_tile_IJ_avx2_8x4) {
  ENFORCE(m % 8 == 0, "m must be divisible by 8");
  ENFORCE(n % 4 == 0, "n must be divisible by 4");

  for (int i = 0; i < m; i += 8) {
    for (int j = 0; j < n; j += 4) {
      microkernel::avx2_8x4(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc);
    }
  }
}

MATMUL_SIGNATURE(matmul_tile_JI_avx2_8x4) {
  ENFORCE(m % 8 == 0, "m must be divisible by 8");
  ENFORCE(n % 4 == 0, "n must be divisible by 4");

  for (int j = 0; j < n; j += 4) {
    for (int i = 0; i < m; i += 8) {
      microkernel::avx2_8x4(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc);
    }
  }
}

}  // namespace column_major
