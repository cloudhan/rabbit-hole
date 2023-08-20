#include "cuda/matmul.cuh"

namespace column_major {

// Laucnh a 1D CTA(threadblock), otherwise, the logic is the same as 01_naive
template <int CtaShapeM, int CtaShapeN>
MATMUL_KERNEL_SIGNATURE(matmul_kernel_1d_cta) {
  // "column major" thread mapping
  int i = CtaShapeM * blockIdx.x + threadIdx.x % CtaShapeM;  // thus we align the thread layout with memory layout, in terms of Matrix C
  int j = CtaShapeN * blockIdx.y + threadIdx.x / CtaShapeM;

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

#define MATMUL_KERNEL_LAUNCH_1D(name, num_threads, cta_shape_m, cta_shape_n)                          \
  MATMUL_SIGNATURE(launch_##name##_##num_threads##t_cta##cta_shape_m##x##cta_shape_n) {               \
    dim3 threads(num_threads);                                                                        \
    dim3 blocks(ceil_div<int64_t>(m, cta_shape_m), ceil_div<int64_t>(n, cta_shape_n));                \
    name<cta_shape_m, cta_shape_n><<<blocks, threads, 0, nullptr>>>(m, n, k, a, lda, b, ldb, c, ldc); \
    CUDA_CHECK(cudaGetLastError());                                                                   \
  }

MATMUL_KERNEL_LAUNCH_1D(matmul_kernel_1d_cta, 256, 16, 16);
MATMUL_KERNEL_LAUNCH_1D(matmul_kernel_1d_cta, 512, 16, 32);
MATMUL_KERNEL_LAUNCH_1D(matmul_kernel_1d_cta, 512, 32, 16);
MATMUL_KERNEL_LAUNCH_1D(matmul_kernel_1d_cta, 1024, 32, 32);

MATMUL_DMODULE(m) {
  REGISTER(launch_matmul_kernel_1d_cta_256t_cta16x16);
  REGISTER(launch_matmul_kernel_1d_cta_512t_cta16x32);
  REGISTER(launch_matmul_kernel_1d_cta_512t_cta32x16);
  REGISTER(launch_matmul_kernel_1d_cta_1024t_cta32x32);
}

}  // namespace column_major
