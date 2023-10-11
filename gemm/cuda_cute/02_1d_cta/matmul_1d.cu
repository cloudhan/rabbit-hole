#include <cute/layout.hpp>                     // make_shape, make_stride, make_layout
#include <cute/numeric/integral_constant.hpp>  // _1
#include "cuda/matmul.cuh"

using namespace cute;

namespace column_major {
template <int CtaShapeM, int CtaShapeN>
MATMUL_KERNEL_SIGNATURE(matmul_kernel_1d_cta) {
  const auto CtaCShape = make_shape(Int<CtaShapeM>{}, Int<CtaShapeN>{});
  const auto CtaCStride = make_stride(_1{}, n);
  const auto CtaCLayout = make_layout(CtaCShape, CtaCStride);

  if (cute::thread0()) {
    cute::print(CtaCLayout);
  }
}

#define MATMUL_KERNEL_LAUNCH(name, num_threads, cta_shape_m, cta_shape_n)                             \
  MATMUL_SIGNATURE(launch_##name##_##num_threads##t_cta##cta_shape_m##x##cta_shape_n) {               \
    dim3 threads(num_threads);                                                                        \
    dim3 blocks(ceil_div<int64_t>(m, cta_shape_m), ceil_div<int64_t>(n, cta_shape_n));                \
    name<cta_shape_m, cta_shape_n><<<blocks, threads, 0, nullptr>>>(m, n, k, a, lda, b, ldb, c, ldc); \
    CUDA_CHECK(cudaGetLastError());                                                                   \
  }

MATMUL_KERNEL_LAUNCH(matmul_kernel_1d_cta, 256, 16, 16);
MATMUL_KERNEL_LAUNCH(matmul_kernel_1d_cta, 512, 16, 32);
MATMUL_KERNEL_LAUNCH(matmul_kernel_1d_cta, 512, 32, 16);
MATMUL_KERNEL_LAUNCH(matmul_kernel_1d_cta, 1024, 32, 32);

MATMUL_DMODULE(m) {
  REGISTER(launch_matmul_kernel_1d_cta_256t_cta16x16);
  REGISTER(launch_matmul_kernel_1d_cta_512t_cta16x32);
  REGISTER(launch_matmul_kernel_1d_cta_512t_cta32x16);
  REGISTER(launch_matmul_kernel_1d_cta_1024t_cta32x32);
}

}  // namespace column_major
