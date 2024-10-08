#include <cute/layout.hpp>                     // make_shape, make_stride, make_layout
#include <cute/pointer.hpp>                    // make_gmem_ptr
#include <cute/tensor.hpp>                     // make_tensor
#include <cute/numeric/integral_constant.hpp>  // _1

#include "cuda/matmul.cuh"

using namespace cute;

namespace column_major {
template <int CtaShapeM, int CtaShapeN>
MATMUL_KERNEL_SIGNATURE(matmul_kernel_1d_cta) {
  // original matrix
  const auto mA = make_tensor(make_gmem_ptr(a), make_layout(make_shape(m, k), make_stride(_1{}, lda)));  // col-major, indexed as (m, k)
  const auto mB = make_tensor(make_gmem_ptr(b), make_layout(make_shape(k, n), make_stride(_1{}, ldb)));  // col-major, indexed as (k, n)
  auto mC = make_tensor(make_gmem_ptr(c), make_layout(make_shape(m, n), make_stride(_1{}, ldc)));        // col-major, indexed as (m, n)

  // a local view (in CuTe term, local tile) this cta will need to process
  const auto ctaA = local_tile(mA, make_tile(Int<CtaShapeM>{}, k), make_coord(blockIdx.x, _));  // in form of numpy: A[blockIdx.x*CtaShapeM:(blockIdx.x+1)*CtaShapeM, :]
  const auto ctaB = local_tile(mB, make_tile(k, Int<CtaShapeN>{}), make_coord(_, blockIdx.y));  // in form of numpy: B[:, blockIdx.y*CtaShapeN:(blockIdx.y+1)*CtaShapeN]
  auto ctaC = local_tile(mC, make_tile(Int<CtaShapeM>{}, Int<CtaShapeN>{}), make_coord(blockIdx.x, blockIdx.y));

  const auto CtaLayout = make_layout(make_shape(Int<CtaShapeM>{}, Int<CtaShapeN>{}));
  const auto [cta_thread_i, cta_thread_j] = idx2crd(make_tuple(threadIdx.x, threadIdx.x), CtaLayout.shape(), CtaLayout.stride());

  if (blockIdx.x * CtaShapeM + cta_thread_i < m && blockIdx.y * CtaShapeN + cta_thread_j < n) {
    const auto A_i_p = local_tile(ctaA, make_tile(_1{}, k), make_coord(cta_thread_i, _));  // A(i, _)
    const auto B_p_j = local_tile(ctaB, make_tile(k, _1{}), make_coord(_, cta_thread_j));  // B(_, j)

    float acc = 0.0;
    for (int p = 0; p < k; p++) {
      acc += A_i_p(p) * B_p_j(p);
    }
    ctaC(cta_thread_i, cta_thread_j) = acc;
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
