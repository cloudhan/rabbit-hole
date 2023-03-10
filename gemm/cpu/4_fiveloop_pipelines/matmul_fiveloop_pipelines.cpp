#include "cpu/matmul.hpp"
#include "cpu/3_simple_pipelines/matmul_microkernels.hpp"
#include <string>
#include <iostream>

namespace column_major {

using namespace microkernels;

template <typename MicroKernel>
MATMUL_SIGNATURE(matmul_loop_over_register_tile) {
  MicroKernel microkernel{};

  // MR (NR) is basically register blocking for m (n).
  ENFORCE(m % MicroKernel::MR == 0, "m must be divisible by " + std::to_string(MicroKernel::MR));
  ENFORCE(n % MicroKernel::NR == 0, "n must be divisible by " + std::to_string(MicroKernel::NR));

  for (int j = 0; j < n; j += MicroKernel::NR) {
    for (int i = 0; i < m; i += MicroKernel::MR) {
      microkernel(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc);
    }
  }
}

template <int MC, int NC, int KC, typename MicroKernel>
MATMUL_SIGNATURE(matmul_five_loop_pipeline) {
  // MC, NC, KC is basically cache blocking for m, n, k
  ENFORCE(m % MC == 0, "m must be divisible by " + std::to_string(MC));
  ENFORCE(n % NC == 0, "n must be divisible by " + std::to_string(NC));
  ENFORCE(k % KC == 0, "n must be divisible by " + std::to_string(KC));

  for (int j = 0; j < n; j += NC) {
    for (int p = 0; p < k; p += KC) {
      for (int i = 0; i < m; i += MC) {
        matmul_loop_over_register_tile<MicroKernel>(MC, NC, KC, &A(i, p), lda, &B(p, j), ldb, &C(i, j), ldc);
      }
    }
  }
}

MATMUL_SIGNATURE(matmul_fiveloop_pipeline_JPI_96_96_96_16x6) {
  matmul_five_loop_pipeline<96, 96, 96, avx2::microkernel<16, 6>>(m, n, k, a, lda, b, ldb, c, ldc);
}

MATMUL_SIGNATURE(matmul_fiveloop_pipeline_JPI_144_144_144_16x6) {
  matmul_five_loop_pipeline<144, 144, 144, avx2::microkernel<16, 6>>(m, n, k, a, lda, b, ldb, c, ldc);
}

}  // namespace column_major
