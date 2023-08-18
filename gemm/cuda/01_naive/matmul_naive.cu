#include "cuda/matmul.cuh"

namespace column_major {

MATMUL_KERNEL_SIGNATURE(matmul_kernel_naive) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;

  if (i < m && j < n) {
    float acc = 0.0;
    auto A_i_p = &a[i * 1 + 0 * lda];  // A(i, p) where p = 0
    auto B_p_j = &b[0 * 1 + j * ldb];  // B(p, j) where p = 0

    for (int p = 0; p < k; p++) {
      acc += (*A_i_p) * (*B_p_j);
      A_i_p += lda;
      B_p_j += 1;
    }
    c[i * 1 + j * ldc] = acc;
  }
}

#define MATMUL_KERNEL_LAUNCH_NAIVE(name, Mcta, Ncta)                        \
  MATMUL_SIGNATURE(launch_##name##_cta##Mcta##x##Ncta) {                    \
    dim3 threads(Mcta, Ncta);                                               \
    dim3 blocks((m - 1) / Mcta + 1, (n - 1) / Ncta + 1);                    \
    name<<<blocks, threads, 0, nullptr>>>(m, n, k, a, lda, b, ldb, c, ldc); \
    CUDA_CHECK(cudaGetLastError());                                         \
  }

MATMUL_KERNEL_LAUNCH_NAIVE(matmul_kernel_naive, 16, 16);
MATMUL_KERNEL_LAUNCH_NAIVE(matmul_kernel_naive, 16, 32);
MATMUL_KERNEL_LAUNCH_NAIVE(matmul_kernel_naive, 32, 16);
MATMUL_KERNEL_LAUNCH_NAIVE(matmul_kernel_naive, 32, 32);
// They are not callable, CUDA only support max 1024 threads per block. We will solve this latter.
// MATMUL_KERNEL_LAUNCH_NAIVE(matmul_kernel_naive, 32, 64);
// MATMUL_KERNEL_LAUNCH_NAIVE(matmul_kernel_naive, 64, 32);
// MATMUL_KERNEL_LAUNCH_NAIVE(matmul_kernel_naive, 64, 64);
// MATMUL_KERNEL_LAUNCH_NAIVE(matmul_kernel_naive, 64, 128);
// MATMUL_KERNEL_LAUNCH_NAIVE(matmul_kernel_naive, 128, 64);
// MATMUL_KERNEL_LAUNCH_NAIVE(matmul_kernel_naive, 64, 256);
// MATMUL_KERNEL_LAUNCH_NAIVE(matmul_kernel_naive, 256, 64);

MATMUL_DMODULE(m) {
  REGISTER(launch_matmul_kernel_naive_cta16x16);
  REGISTER(launch_matmul_kernel_naive_cta16x32);
  REGISTER(launch_matmul_kernel_naive_cta32x16);
  REGISTER(launch_matmul_kernel_naive_cta32x32);
}

}  // namespace column_major
