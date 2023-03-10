#include "cpu/matmul.hpp"
#include "cpu/3_simple_pipelines/matmul_microkernels.hpp"
#include "cpu/5_fiveloop_pipelines_copied/matmul_direct_copy.hpp"
#include <string>
#include <iostream>

namespace column_major {

using namespace microkernels;
using namespace packers;

// Note this can be reused from previous version
template <typename MicroKernel>
MATMUL_SIGNATURE(matmul_loop_over_register_tile) {
  MicroKernel microkernel{};

  ENFORCE(m % MicroKernel::MR == 0, "m must be divisible by " + std::to_string(MicroKernel::MR));
  ENFORCE(n % MicroKernel::NR == 0, "n must be divisible by " + std::to_string(MicroKernel::NR));

  for (int j = 0; j < n; j += MicroKernel::NR) {
    for (int i = 0; i < m; i += MicroKernel::MR) {
      microkernel(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc);
    }
  }
}

template <int MC, int NC, int KC, typename MicroKernel>
MATMUL_SIGNATURE(matmul_five_loop_pipeline_copied) {
  auto a_copy = direct_copy<MC, KC>{};
  auto b_copy = direct_copy<KC, NC>{};
  for (int j = 0; j < n; j += NC) {
    int64_t jb = std::min<int64_t>(NC, n - j);
    for (int p = 0; p < k; p += KC) {
      int64_t kb = std::min<int64_t>(KC, k - p);
      b_copy.load(&B(p, j), 1, ldb, kb, jb);
      for (int i = 0; i < m; i += MC) {
        int64_t ib = std::min<int64_t>(MC, m - i);
        a_copy.load(&A(i, p), 1, lda, ib, kb);
        matmul_loop_over_register_tile<MicroKernel>(
            ib, jb, kb, a_copy.data(), a_copy.stride(), b_copy.data(), b_copy.stride(), &C(i, j), ldc
        );
      }
    }
  }
}

MATMUL_SIGNATURE(matmul_fiveloop_pipeline_copied_JPI_144_96_144_16x6) {
  matmul_five_loop_pipeline_copied<144, 96, 144, avx2::microkernel<16, 6>>(
      m, n, k, a, lda, b, ldb, c, ldc
  );
}

MATMUL_SIGNATURE(matmul_fiveloop_pipeline_copied_JPI_320_2016_60_16x6) {
  matmul_five_loop_pipeline_copied<320, 2016, 60, avx2::microkernel<16, 6>>(
      m, n, k, a, lda, b, ldb, c, ldc
  );
}

}  // namespace column_major
