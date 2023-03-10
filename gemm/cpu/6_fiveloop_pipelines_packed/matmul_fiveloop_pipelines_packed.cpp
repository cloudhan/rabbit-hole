#include "cpu/matmul.hpp"
#include "cpu/6_fiveloop_pipelines_packed/matmul_microkernels.hpp"
#include "cpu/6_fiveloop_pipelines_packed/matmul_pack.hpp"
#include <string>
#include <iostream>

namespace column_major {

using namespace microkernels;
using namespace packers;

#define A(i, j) a[(j)*lda + (i)]
#define B(i, j) b[(j)*ldb + (i)]

template <typename MicroKernel>
void matmul_loop_over_register_tile_packed(
    int64_t m, int64_t n, int64_t k,
    const float* a, const float* b, float* c, int64_t ldc
) {
  MicroKernel microkernel{};

  // NOTE: Due to packing, we can handle non-divisible matrices size, because of the implicit padding.
  // ENFORCE(m % MicroKernel::MR == 0, "m must be divisible by " + std::to_string(MicroKernel::MR));
  // ENFORCE(n % MicroKernel::NR == 0, "n must be divisible by " + std::to_string(MicroKernel::NR));

  for (int j = 0; j < n; j += MicroKernel::NR) {
    for (int i = 0; i < m; i += MicroKernel::MR) {
      microkernel(k, &a[i * k], &b[j * k], &C(i, j), ldc);
    }
  }
}

template <int MC, int NC, int KC, typename MicroKernel>
MATMUL_SIGNATURE(matmul_five_loop_pipeline_packed) {
  auto a_pack = a_packer<MC, KC, MicroKernel::MR, float>{};
  auto b_pack = b_packer<KC, NC, MicroKernel::NR, float>{};
  for (int j = 0; j < n; j += NC) {
    int64_t jb = std::min<int64_t>(NC, n - j);
    for (int p = 0; p < k; p += KC) {
      int64_t kb = std::min<int64_t>(KC, k - p);
      b_pack.load(&B(p, j), 1, ldb, kb, jb);
      for (int i = 0; i < m; i += MC) {
        int64_t ib = std::min<int64_t>(MC, m - i);
        a_pack.load(&A(i, p), 1, lda, ib, kb);
        matmul_loop_over_register_tile_packed<MicroKernel>(
            ib, jb, kb, a_pack.data(), b_pack.data(), &C(i, j), ldc
        );
      }
    }
  }
}

MATMUL_SIGNATURE(matmul_fiveloop_pipeline_packed_JPI_144_96_144_16x6) {
  matmul_five_loop_pipeline_packed<144, 96, 144, avx2::packed_microkernel<16, 6>>(
      m, n, k, a, lda, b, ldb, c, ldc
  );
}

MATMUL_SIGNATURE(matmul_fiveloop_pipeline_packed_JPI_320_2016_60_16x6) {
  matmul_five_loop_pipeline_packed<320, 2016, 60, avx2::packed_microkernel<16, 6>>(
      m, n, k, a, lda, b, ldb, c, ldc
  );
}

}  // namespace column_major
