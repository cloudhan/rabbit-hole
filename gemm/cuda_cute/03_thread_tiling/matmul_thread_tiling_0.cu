#include <cute/layout.hpp>                     // make_shape, make_stride, make_layout
#include <cute/pointer.hpp>                    // make_gmem_ptr
#include <cute/tensor.hpp>                     // make_tensor
#include <cute/numeric/integral_constant.hpp>  // _1

#include "cuda/matmul.cuh"

using namespace cute;

template <int ThreadShapeM, int ThreadShapeN, typename AccT, typename FragAT, typename FragBT>
__device__ void rank1_update(const FragAT& fragA, const FragBT& fragB, AccT& acc) {
  // rank-1 update to acc registers
#pragma unroll
  for (int j = 0; j < ThreadShapeN; j++) {
#pragma unroll
    for (int i = 0; i < ThreadShapeM; i++) {
      acc(i, j) += fragA(i) * fragB(j);
    }
  }
}

template <int ThreadShapeM, int ThreadShapeN, typename AccT, typename PredAT, typename PredBT, typename CtaCT>
__device__ void acc_store(const AccT& acc, const PredAT& predA, const PredBT& predB, CtaCT& ctaC, int cta_thread_i, int cta_thread_j) {
#pragma unroll
  for (int j = 0; j < ThreadShapeN; j++) {
#pragma unroll
    for (int i = 0; i < ThreadShapeM; i++) {
      if (predA(i) && predB(j)) {
        // printf("%03d store %d,%d %f\n", threadIdx.x, cta_thread_i + i, cta_thread_j + j, acc(i, j));
        ctaC(cta_thread_i + i, cta_thread_j + j) = acc(i, j);
      }
    }
  }
}

namespace column_major {
template <int NumThreads, int CtaShapeM, int CtaShapeN, int ThreadShapeM, int ThreadShapeN>
MATMUL_KERNEL_SIGNATURE(matmul_kernel_thread_tiling_0) {
  // original matrix
  const auto mA = make_tensor(make_gmem_ptr(a), make_layout(make_shape(m, k), make_stride(_1{}, lda)));  // col-major, indexed as (m, k)
  const auto mB = make_tensor(make_gmem_ptr(b), make_layout(make_shape(n, k), make_stride(ldb, _1{})));  // col-major storage, row-major indexing, indexed as (n, k), effectively a "transposed view"
  auto mC = make_tensor(make_gmem_ptr(c), make_layout(make_shape(m, n), make_stride(_1{}, ldc)));        // col-major, indexed as (m, n)

  const auto CtaShape = make_shape(Int<CtaShapeM>{}, Int<CtaShapeN>{}, k);
  const auto cta_coord = make_coord(blockIdx.x, blockIdx.y, 0);

  // a local view (in CuTe term, local tile) this CTA will need to process
  const auto ctaA = local_tile(mA, CtaShape, cta_coord, make_step(_1{}, _, _1{}));
  const auto ctaB = local_tile(mB, CtaShape, cta_coord, make_step(_, _1{}, _1{}));
  auto ctaC = local_tile(mC, CtaShape, cta_coord, make_step(_1{}, _1{}, _));

  // FIXME: We are still manually writing the mapping of index to coordinate, how to derive it?
  // constexpr const auto CtaLayout = make_layout(make_shape(Int<CtaShapeM>{}, Int<CtaShapeN>{}));
  // constexpr const auto ThreadTile = make_tile(Int<ThreadShapeM>{}, Int<ThreadShapeN>{});
  // constexpr const auto MapToThread = tiled_divide(CtaLayout, ThreadTile)(Int<0>{}, _, _);
  constexpr const auto MapToThread = make_layout(make_shape(Int<CtaShapeM / ThreadShapeM>{}, Int<CtaShapeN / ThreadShapeN>{}));
  auto [cta_thread_i, cta_thread_j] = idx2crd(MapToThread(threadIdx.x), MapToThread.shape(), MapToThread.stride());
  cta_thread_i *= ThreadShapeM;
  cta_thread_j *= ThreadShapeN;

  // whether are we in bound
  auto predA = make_tensor<bool>(Int<ThreadShapeM>{});
  for (int i = 0; i < size<0>(predA); i++) {
    predA(i) = blockIdx.x * CtaShapeM + cta_thread_i + i < m;
  }
  auto predB = make_tensor<bool>(Int<ThreadShapeN>{});
  for (int j = 0; j < size<0>(predB); j++) {
    predB(j) = blockIdx.y * CtaShapeN + cta_thread_j + j < n;
  }

  // data view that this thread will process
  const auto A_i_p = local_tile(ctaA, make_tile(Int<ThreadShapeM>{}, k), make_coord(cta_thread_i / ThreadShapeM, 0));  // A(i, _)
  const auto B_j_p = local_tile(ctaB, make_tile(Int<ThreadShapeN>{}, k), make_coord(cta_thread_j / ThreadShapeN, 0));  // B(j, _)

  auto fragA = make_fragment_like<float>(Int<ThreadShapeM>{});
  auto fragB = make_fragment_like<float>(Int<ThreadShapeN>{});
  auto acc = make_fragment_like<float>(make_shape(Int<ThreadShapeM>{}, Int<ThreadShapeN>{}));
  clear(acc);

  for (int p = 0; p < k; p++) {
    clear(fragA);
    copy_if(predA, A_i_p(_, p), fragA);

    clear(fragB);
    copy_if(predB, B_j_p(_, p), fragB);

    rank1_update<ThreadShapeM, ThreadShapeN>(fragA, fragB, acc);
  }

  acc_store<ThreadShapeM, ThreadShapeN>(acc, predA, predB, ctaC, cta_thread_i, cta_thread_j);
}

#define MATMUL_KERNEL_LAUNCH(name, num_threads, cta_shape_m, cta_shape_n, thread_shape_m, thread_shape_n)                                          \
  MATMUL_SIGNATURE(launch_##name##_##num_threads##t_cta##cta_shape_m##x##cta_shape_n##_thread##thread_shape_m##x##thread_shape_n) {                \
    dim3 threads(num_threads);                                                                                                                     \
    dim3 blocks(ceil_div<int64_t>(m, cta_shape_m), ceil_div<int64_t>(n, cta_shape_n));                                                             \
    name<num_threads, cta_shape_m, cta_shape_n, thread_shape_m, thread_shape_n><<<blocks, threads, 0, nullptr>>>(m, n, k, a, lda, b, ldb, c, ldc); \
    CUDA_CHECK(cudaGetLastError());                                                                                                                \
  }

MATMUL_KERNEL_LAUNCH(matmul_kernel_thread_tiling_0, 64, 32, 32, 4, 4);
MATMUL_KERNEL_LAUNCH(matmul_kernel_thread_tiling_0, 64, 64, 32, 8, 4);
MATMUL_KERNEL_LAUNCH(matmul_kernel_thread_tiling_0, 64, 64, 64, 8, 8);
MATMUL_KERNEL_LAUNCH(matmul_kernel_thread_tiling_0, 128, 32, 32, 4, 2);
MATMUL_KERNEL_LAUNCH(matmul_kernel_thread_tiling_0, 128, 64, 32, 4, 4);
MATMUL_KERNEL_LAUNCH(matmul_kernel_thread_tiling_0, 256, 32, 32, 2, 2);
MATMUL_KERNEL_LAUNCH(matmul_kernel_thread_tiling_0, 256, 64, 32, 4, 2);
MATMUL_KERNEL_LAUNCH(matmul_kernel_thread_tiling_0, 256, 64, 64, 4, 4);
MATMUL_KERNEL_LAUNCH(matmul_kernel_thread_tiling_0, 256, 128, 64, 8, 4);
MATMUL_KERNEL_LAUNCH(matmul_kernel_thread_tiling_0, 256, 128, 128, 8, 8);

MATMUL_DMODULE(m) {
  REGISTER(launch_matmul_kernel_thread_tiling_0_64t_cta32x32_thread4x4);
  REGISTER(launch_matmul_kernel_thread_tiling_0_64t_cta64x32_thread8x4);
  REGISTER(launch_matmul_kernel_thread_tiling_0_64t_cta64x64_thread8x8);
  REGISTER(launch_matmul_kernel_thread_tiling_0_128t_cta32x32_thread4x2);
  REGISTER(launch_matmul_kernel_thread_tiling_0_128t_cta64x32_thread4x4);
  REGISTER(launch_matmul_kernel_thread_tiling_0_256t_cta32x32_thread2x2);
  REGISTER(launch_matmul_kernel_thread_tiling_0_256t_cta64x32_thread4x2);
  REGISTER(launch_matmul_kernel_thread_tiling_0_256t_cta64x64_thread4x4);
  REGISTER(launch_matmul_kernel_thread_tiling_0_256t_cta128x64_thread8x4);
  REGISTER(launch_matmul_kernel_thread_tiling_0_256t_cta128x128_thread8x8);
}

}  // namespace column_major
