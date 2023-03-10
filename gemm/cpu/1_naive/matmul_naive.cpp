#include "cpu/matmul.hpp"

namespace column_major {

#define A(i, j) a[(j)*lda + (i)]
#define B(i, j) b[(j)*ldb + (i)]
#define C(i, j) c[(j)*ldc + (i)]

MATMUL_SIGNATURE(matmul_IJP) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      for (int p = 0; p < k; ++p) {
        C(i, j) = C(i, j) + A(i, p) * B(p, j);
      }
    }
  }
}

MATMUL_SIGNATURE(matmul_JIP) {
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < m; ++i) {
      for (int p = 0; p < k; ++p) {
        C(i, j) = C(i, j) + A(i, p) * B(p, j);
      }
    }
  }
}

MATMUL_SIGNATURE(matmul_IPJ) {
  for (int i = 0; i < m; ++i) {
    for (int p = 0; p < k; ++p) {
      for (int j = 0; j < n; ++j) {
        C(i, j) = C(i, j) + A(i, p) * B(p, j);
      }
    }
  }
}
MATMUL_SIGNATURE(matmul_JPI) {
  for (int j = 0; j < n; ++j) {
    for (int p = 0; p < k; ++p) {
      for (int i = 0; i < m; ++i) {
        C(i, j) = C(i, j) + A(i, p) * B(p, j);
      }
    }
  }
}

MATMUL_SIGNATURE(matmul_PIJ) {
  for (int p = 0; p < k; ++p) {
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        C(i, j) = C(i, j) + A(i, p) * B(p, j);
      }
    }
  }
}

MATMUL_SIGNATURE(matmul_PJI) {
  for (int p = 0; p < k; ++p) {
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < m; ++i) {
        C(i, j) = C(i, j) + A(i, p) * B(p, j);
      }
    }
  }
}

}  // namespace column_major
