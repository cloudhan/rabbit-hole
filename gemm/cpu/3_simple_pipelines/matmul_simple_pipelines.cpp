#include "cpu/matmul.hpp"
#include "cpu/3_simple_pipelines/matmul_microkernels.hpp"
#include <string>

namespace column_major {

using namespace microkernels;

template <typename MicroKernel>
MATMUL_SIGNATURE(matmul_simple_pipeline) {
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

MATMUL_SIGNATURE(matmul_simple_pipeline_8x4) {
  matmul_simple_pipeline<avx2::microkernel<8, 4>>(m, n, k, a, lda, b, ldb, c, ldc);
}

MATMUL_SIGNATURE(matmul_simple_pipeline_16x4) {
  matmul_simple_pipeline<avx2::microkernel<16, 4>>(m, n, k, a, lda, b, ldb, c, ldc);
}

MATMUL_SIGNATURE(matmul_simple_pipeline_16x6) {
  matmul_simple_pipeline<avx2::microkernel<16, 6>>(m, n, k, a, lda, b, ldb, c, ldc);
}

MATMUL_SIGNATURE(matmul_simple_pipeline_16x8) {
  matmul_simple_pipeline<avx2::microkernel<16, 8>>(m, n, k, a, lda, b, ldb, c, ldc);
}

}  // namespace column_major
