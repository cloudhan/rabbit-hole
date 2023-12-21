#include <cute/layout.hpp>                     // make_shape, make_stride, make_layout
#include <cute/pointer.hpp>                    // make_gmem_ptr
#include <cute/tensor.hpp>                     // make_tensor
#include <cute/numeric/integral_constant.hpp>  // _1

#include "cuda/matmul.cuh"

using namespace cute;

// copy/store with multiple batches in this kernel is automatically achieved via TiledCopy
template <typename TiledCopy, typename ThrCopy, typename GmemT, typename CoordT, typename BoundT, typename RegT>
__device__ void ldg(
    const TiledCopy& tiled_copy, const ThrCopy& thr_copy,
    const GmemT& tensor, const CoordT& coord, const BoundT& coord_bound, RegT& reg
) {
  static_assert(is_gmem<typename GmemT::engine_type>());
  static_assert(is_rmem<typename RegT::engine_type>());
  auto tv = thr_copy.partition_S(tensor);
  auto tc = thr_copy.partition_S(coord);
  if (elem_less(tc(size(tv)), coord_bound)) {
    // https://github.com/NVIDIA/cutlass/issues/1272
    copy_vec<float>(tv, reg);
  } else {
#pragma unroll
    for (int i = 0; i < size(tv); i++) {
      reg(i) = elem_less(tc(i), coord_bound) ? tv(i) : 0;
    }
  }
}

template <int SubTileStepM, int SubTileStepN, typename AccT, typename CtaCT, typename CoordT>
__device__ void acc_store(const AccT& acc, CtaCT& threadC, CoordT thread_cC, int m, int n) {
  if (elem_less(thread_cC(size<0>(thread_cC) - 1, size<1>(thread_cC) - 1), make_coord(m, n))) {  // fast path
    copy(acc, threadC);
  } else {
    auto predA = make_tensor<bool>(get<0>(thread_cC.shape()));
    for (int i = 0; i < size(predA); i++) {
      predA(i) = get<0>(thread_cC(i, 0)) < m;
    }
#pragma unroll
    for (int j = 0; j < size<1>(typename AccT::layout_type{}); j++) {
      if (get<0>(thread_cC(0, j)) < n) {
        copy_if(predA, acc(_, j), threadC(_, j));
      }
    }
  }
}

__forceinline__ __device__ auto lane_id() {
  uint32_t laneid;
  asm("mov.u32 %0, %%laneid;" : "=r"(laneid) :);
  return laneid;
}

__forceinline__ __device__ auto warp_id() {
  uint32_t warpid;
  asm("mov.u32 %0, %%warpid;" : "=r"(warpid) :);
  return warpid;
}

namespace column_major {
template <int NumThreads, int CtaShapeM, int CtaShapeN, int SmemShapeK, int WarpShapeM, int WarpShapeN, int ThreadShapeM, int ThreadShapeN>
__launch_bounds__(NumThreads, 2)
    MATMUL_KERNEL_SIGNATURE(matmul_kernel_copy_1_ldg) {
  constexpr const auto SmemShapeM = CtaShapeM;
  constexpr const auto SmemShapeN = CtaShapeN;

  // original matrix
  const auto mA = make_tensor(make_gmem_ptr(a), make_layout(make_shape(m, k), make_stride(_1{}, lda)));  // col-major, indexed as (m, k)
  const auto mB = make_tensor(make_gmem_ptr(b), make_layout(make_shape(n, k), make_stride(ldb, _1{})));  // col-major storage, row-major indexing, indexed as (n, k), effectively a "transposed view"
  auto mC = make_tensor(make_gmem_ptr(c), make_layout(make_shape(m, n), make_stride(_1{}, ldc)));        // col-major, indexed as (m, n)

  // coordinate matrix
  const auto cA = make_identity_tensor(make_shape(m, k));
  const auto cB = make_identity_tensor(make_shape(n, k));
  const auto cC = make_identity_tensor(make_shape(m, n));

  const auto CtaShape = make_shape(Int<SmemShapeM>{}, Int<SmemShapeN>{}, Int<SmemShapeK>{});
  const auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);

  // a local view (in CuTe term, local tile) this CTA will need to process
  const auto ctaA = local_tile(mA, CtaShape, cta_coord, make_step(_1{}, _, _1{}));
  const auto ctaB = local_tile(mB, CtaShape, cta_coord, make_step(_, _1{}, _1{}));
  auto ctaC = local_tile(mC, CtaShape, cta_coord, make_step(_1{}, _1{}, _));

  const auto cta_cA = local_tile(cA, CtaShape, cta_coord, make_step(_1{}, _, _1{}));
  const auto cta_cB = local_tile(cB, CtaShape, cta_coord, make_step(_, _1{}, _1{}));
  const auto cta_cC = local_tile(cC, CtaShape, cta_coord, make_step(_1{}, _1{}, _));

  constexpr const int SmemAPad = 0;
  constexpr const int SmemBPad = 4;
  constexpr const auto SmemALayout = make_layout(make_shape(Int<SmemShapeM>{}, Int<SmemShapeK>{}), make_stride(_1{}, Int<SmemShapeM + SmemAPad>{}));
  constexpr const auto SmemBLayout = make_layout(make_shape(Int<SmemShapeN>{}, Int<SmemShapeK>{}), make_stride(_1{}, Int<SmemShapeN + SmemBPad>{}));
  // double buffering
  __shared__ float smemA[cosize(SmemALayout)];
  __shared__ float smemB[cosize(SmemBLayout)];
  auto sA = make_tensor(make_smem_ptr(smemA), SmemALayout);  // (m, k)
  auto sB = make_tensor(make_smem_ptr(smemB), SmemBLayout);  // (n, k)

  const auto stripe_gA = local_tile(ctaA, make_tile(Int<SmemShapeM>{}, Int<SmemShapeK>{}), make_coord(blockIdx.x, _));  // A(blockIdx.x*SmemShapeM:(blockIdx.x+1)*SmemShapeM, :)
  const auto stripe_gB = local_tile(ctaB, make_tile(Int<SmemShapeN>{}, Int<SmemShapeK>{}), make_coord(blockIdx.y, _));  // B(blockIdx.y*SmemShapeN:(blockIdx.y+1)*SmemShapeN, :)

  const auto stripe_cA = local_tile(cta_cA, make_tile(Int<SmemShapeM>{}, Int<SmemShapeK>{}), make_coord(blockIdx.x, _));
  const auto stripe_cB = local_tile(cta_cB, make_tile(Int<SmemShapeN>{}, Int<SmemShapeK>{}), make_coord(blockIdx.y, _));

  auto fragA = make_fragment_like<float>(make_shape(Int<ThreadShapeM / 2>{}, _2{}, _2{}));                                                 // (frag_idx, sub_tile_idx, buffering_idx)
  auto fragB = make_fragment_like<float>(make_shape(Int<ThreadShapeN / 2>{}, _2{}, _2{}));                                                 // (frag_idx, sub_tile_idx, buffering_idx)
  auto acc = make_fragment_like<float>(make_shape(make_shape(Int<ThreadShapeM / 2>{}, _2{}), make_shape(Int<ThreadShapeN / 2>{}, _2{})));  // ((i, ii), (j, jj))
  clear(acc);

  constexpr const auto SmemALoadStoreVec = std::min(4, (SmemShapeM * SmemShapeK) / NumThreads);
  constexpr const auto SmemBLoadStoreVec = std::min(4, (SmemShapeN * SmemShapeK) / NumThreads);
  constexpr const auto SmemALoadStoreBatch = (SmemShapeM * SmemShapeK) / (NumThreads * SmemALoadStoreVec);
  constexpr const auto SmemBLoadStoreBatch = (SmemShapeN * SmemShapeK) / (NumThreads * SmemBLoadStoreVec);
  static_assert(SmemShapeM % (SmemALoadStoreVec * SmemALoadStoreBatch) == 0);
  static_assert(SmemShapeN % (SmemBLoadStoreVec * SmemBLoadStoreBatch) == 0);

  const auto tiled_copy_smem_a = make_tiled_copy(
      Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<32>, float>{},  // For store only
      make_layout(make_shape(Int<SmemShapeM / (SmemALoadStoreVec * SmemALoadStoreBatch)>{}, Int<SmemShapeK>{})),
      make_layout(make_shape(Int<SmemALoadStoreVec>{}))
  );
  const auto tiled_copy_smem_b = make_tiled_copy(
      Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<32>, float>{},  // For store only
      make_layout(make_shape(Int<SmemShapeN / (SmemBLoadStoreVec * SmemBLoadStoreBatch)>{}, Int<SmemShapeK>{}), LayoutRight{}),
      make_layout(make_shape(Int<SmemBLoadStoreVec>{}))
  );
  auto thr_copy_smem_a = tiled_copy_smem_a.get_thread_slice(threadIdx.x);
  auto thr_copy_smem_b = tiled_copy_smem_b.get_thread_slice(threadIdx.x);
  auto thr_store_smem_a_view = thr_copy_smem_a.partition_D(sA);
  auto thr_store_smem_b_view = thr_copy_smem_b.partition_D(sB);

  auto staging_a = make_fragment_like<float>(shape(thr_store_smem_a_view));
  auto staging_b = make_fragment_like<float>(shape(thr_store_smem_b_view));

  int p_tile_count = (k - 1) / SmemShapeK + 1;
  int p_tile_next = 0;

  ldg(tiled_copy_smem_a, thr_copy_smem_a, stripe_gA(_, _, _0{}, p_tile_next), stripe_cA(_, _, _0{}, p_tile_next), shape(mA), staging_a);
  ldg(tiled_copy_smem_b, thr_copy_smem_b, stripe_gB(_, _, _0{}, p_tile_next), stripe_cB(_, _, _0{}, p_tile_next), shape(mB), staging_b);
  if (--p_tile_count > 0) {
    ++p_tile_next;
  }
  copy(tiled_copy_smem_a, staging_a, thr_store_smem_a_view);
  copy(tiled_copy_smem_b, staging_b, thr_store_smem_b_view);
  __syncthreads();

  constexpr const auto CtaLayout = make_layout(make_shape(Int<CtaShapeM / ThreadShapeM>{}, Int<CtaShapeN / ThreadShapeN>{}));
  constexpr const auto WarpTile = make_tile(Int<WarpShapeM / ThreadShapeM>{}, Int<WarpShapeN / ThreadShapeN>{});
  constexpr const auto LaneWarp = zipped_divide(CtaLayout, WarpTile);  // ((lane),(warp)):(...), map from threadIdx.x to warp tiled index
  // constexpr const auto MN = logical_divide(CtaLayout, WarpTile);       // ((CtaM),(CtaN)):(...)
  // const auto remapped_coord = CtaLayout[LaneWarp(lane_id(), warp_id())];  // map to then unmap from warp tiled index, to get i,j coord.

  const auto remapped_coord = CtaLayout[LaneWarp(threadIdx.x)];     // NOTE: it is weird that LaneWarp(threadIdx.x) is faster than LaneWarp(lane_id(), warp_id())
  const auto cta_thread_i = get<0>(remapped_coord) * ThreadShapeM;  // Again, scale by ThreadShapeM not necessary,
  const auto cta_thread_j = get<1>(remapped_coord) * ThreadShapeN;  // just to keep the semantics identical with cuda impls
  constexpr const auto SubTileStepM = WarpShapeM / 2;
  constexpr const auto SubTileStepN = WarpShapeN / 2;
  // This time, for ThreadShape 8x8, we slice 2x2 (with stride in between) tiles of 4x4 of data tile. This removes all bank conflicts.
  const auto stripe_sA = local_tile(sA, make_tile(make_layout(make_shape(Int<ThreadShapeM / 2>{}, _2{}), make_stride(_1{}, Int<SubTileStepM>{})), Int<SmemShapeK>{}), make_coord(cta_thread_i / ThreadShapeM));
  const auto stripe_sB = local_tile(sB, make_tile(make_layout(make_shape(Int<ThreadShapeN / 2>{}, _2{}), make_stride(_1{}, Int<SubTileStepN>{})), Int<SmemShapeK>{}), make_coord(cta_thread_j / ThreadShapeN));

  copy(stripe_sA(_, 0, _0{}), fragA(_, _, 0));  // load_fragment a
  copy(stripe_sB(_, 0, _0{}), fragB(_, _, 0));  // load_fragment b

  const auto num_smem_block = size<3>(stripe_gA);
#pragma unroll 1  // no unroll
  for (int block_p = 0; block_p < num_smem_block; block_p++) {
#pragma unroll
    for (int smem_AB_thread_p = 0; smem_AB_thread_p < SmemShapeK; smem_AB_thread_p++) {
      if (smem_AB_thread_p == SmemShapeK - 1) {
        __syncthreads();
        copy(tiled_copy_smem_a, staging_a, thr_store_smem_a_view);
        copy(tiled_copy_smem_b, staging_b, thr_store_smem_b_view);
        __syncthreads();
        if (--p_tile_count > 0) {
          ++p_tile_next;
        }
      }

      copy(stripe_sA(_, (smem_AB_thread_p + 1) % SmemShapeK, _0{}), fragA(_, _, (smem_AB_thread_p + 1) % 2));  // load_fragment a
      copy(stripe_sB(_, (smem_AB_thread_p + 1) % SmemShapeK, _0{}), fragB(_, _, (smem_AB_thread_p + 1) % 2));  // load_fragment b

      if (smem_AB_thread_p == 0) {
        ldg(tiled_copy_smem_a, thr_copy_smem_a, stripe_gA(_, _, _0{}, p_tile_next), stripe_cA(_, _, _0{}, p_tile_next), shape(mA), staging_a);
        ldg(tiled_copy_smem_b, thr_copy_smem_b, stripe_gB(_, _, _0{}, p_tile_next), stripe_cB(_, _, _0{}, p_tile_next), shape(mB), staging_b);
      }

      // Just a simple reorganization, it allows us to dispatch to (M) x (N) => (M,N) version of gemm defined
      // include/cute/algorithm/gemm.hpp
      auto fa = group_modes<0, 2>(fragA(_, _, smem_AB_thread_p % 2));  // ((ThreadShapeM/2,2)):...
      auto fb = group_modes<0, 2>(fragB(_, _, smem_AB_thread_p % 2));  // ((ThreadShapeN/2,2)):...
      gemm(fa, fb, acc);
    }
  }

  constexpr const auto Tiler = make_tile(
      make_layout(make_shape(Int<ThreadShapeM / 2>{}, _2{}), make_stride(_1{}, Int<SubTileStepM>{})),
      make_layout(make_shape(Int<ThreadShapeN / 2>{}, _2{}), make_stride(_1{}, Int<SubTileStepN>{}))
  );
  auto threadC = local_tile(ctaC, Tiler, make_coord(cta_thread_i / ThreadShapeM, cta_thread_j / ThreadShapeN));
  auto thread_cC = local_tile(cta_cC, Tiler, make_coord(cta_thread_i / ThreadShapeM, cta_thread_j / ThreadShapeN));
  acc_store<SubTileStepM, SubTileStepN>(acc, threadC, thread_cC, m, n);
}

#define MATMUL_KERNEL_LAUNCH(name, num_threads, cta_shape_m, cta_shape_n, smem_shape_k, warp_shape_m, warp_shape_n, thread_shape_m, thread_shape_n)                                            \
  MATMUL_SIGNATURE(launch_##name##_##num_threads##t_cta##cta_shape_m##x##cta_shape_n##_smem##smem_shape_k##_warp##warp_shape_m##x##warp_shape_n##_thread##thread_shape_m##x##thread_shape_n) { \
    dim3 threads(num_threads);                                                                                                                                                                 \
    dim3 blocks(ceil_div<int64_t>(m, cta_shape_m), ceil_div<int64_t>(n, cta_shape_n));                                                                                                         \
    name<num_threads, cta_shape_m, cta_shape_n, smem_shape_k, warp_shape_m, warp_shape_n, thread_shape_m, thread_shape_n><<<                                                                   \
        blocks, threads, 0, nullptr>>>(m, n, k, a, lda, b, ldb, c, ldc);                                                                                                                       \
    CUDA_CHECK(cudaGetLastError());                                                                                                                                                            \
  }

// Iter mode from TiledCopy allows us to finish a smem copy in multiple batches. Then SmemShapeK == 16 is possible now.
// But SmemShapeK == 24 is not possible, we are vectorizing and batching over M (or N for b), but SmemShapeM % (VecSize * Batch) != 0
// Vectorizing over M and batching over K is possible, but is not performant.
MATMUL_KERNEL_LAUNCH(matmul_kernel_copy_1_ldg, 256, 128, 128, 8, 32, 64, 8, 8);
MATMUL_KERNEL_LAUNCH(matmul_kernel_copy_1_ldg, 256, 128, 128, 8, 64, 32, 8, 8);
MATMUL_KERNEL_LAUNCH(matmul_kernel_copy_1_ldg, 256, 128, 128, 16, 32, 64, 8, 8);
MATMUL_KERNEL_LAUNCH(matmul_kernel_copy_1_ldg, 256, 128, 128, 16, 64, 32, 8, 8);

MATMUL_DMODULE(m) {
  REGISTER(launch_matmul_kernel_copy_1_ldg_256t_cta128x128_smem8_warp32x64_thread8x8);
  REGISTER(launch_matmul_kernel_copy_1_ldg_256t_cta128x128_smem8_warp64x32_thread8x8);
  REGISTER(launch_matmul_kernel_copy_1_ldg_256t_cta128x128_smem16_warp32x64_thread8x8);
  REGISTER(launch_matmul_kernel_copy_1_ldg_256t_cta128x128_smem16_warp64x32_thread8x8);
}

}  // namespace column_major
