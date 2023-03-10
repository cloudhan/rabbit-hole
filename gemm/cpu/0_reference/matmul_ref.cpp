#include "cpu/matmul.hpp"

#include "blis/blis.h"

namespace column_major {

MATMUL_SIGNATURE(matmul_reference) {
  float one = 1.0f;
  float zero = 0.0f;
  // clang-format onff
  bli_sgemm(
      trans_t::BLIS_NO_TRANSPOSE, trans_t::BLIS_NO_TRANSPOSE,
      m, n, k,
      &one,
      (float*)a, 1, lda,
      (float*)b, 1, ldb,
      &zero,
      c, 1, ldc
  );
  // clang-format on
}

}  // namespace column_major
