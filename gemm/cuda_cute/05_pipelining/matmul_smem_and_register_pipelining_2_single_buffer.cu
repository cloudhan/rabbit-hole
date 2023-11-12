#include <cute/layout.hpp>                     // make_shape, make_stride, make_layout
#include <cute/pointer.hpp>                    // make_gmem_ptr
#include <cute/tensor.hpp>                     // make_tensor
#include <cute/numeric/integral_constant.hpp>  // _1

#include "cuda/matmul.cuh"

using namespace cute;

// not going to implement copy/store with multiple batches in this kernel.
template <int NumThreads, int SmemShapeM, int SmemShapeK, int VecSize, typename RegT, typename GmemT, typename CoordT>
__device__ void load_global_a(RegT& rA, const GmemT& gA, const CoordT& cA, int m, int k) {
  static_assert(is_rmem<typename RegT::engine_type>());
  static_assert(VecSize == 1 || VecSize == 2 || VecSize == 4);
  constexpr const auto ThrVal = make_layout(make_layout(Int<NumThreads>{}, Int<VecSize>{}), make_layout(Int<VecSize>{}));
  const auto ld_gA = gA.compose(ThrVal)(threadIdx.x, _);
  const auto ld_cA = cA.compose(ThrVal);
  clear(rA);
#pragma unroll
  for (int i = 0; i < VecSize; i++) {
    if (elem_less(ld_cA(threadIdx.x, i), make_coord(m, k))) {
      rA(i) = ld_gA(i);
    }
  }
}

template <int NumThreads, int SmemShapeN, int SmemShapeK, int VecSize, typename RegT, typename GmemT, typename CoordT>
__device__ void load_global_b(RegT& rB, const GmemT& gB, const CoordT& cB, int n, int k) {
  static_assert(is_rmem<typename RegT::engine_type>());
  static_assert(VecSize == 1 || VecSize == 2 || VecSize == 4);
  constexpr const auto NumLoadK = SmemShapeK / VecSize; // number of load along k
  static_assert(NumLoadK > 0);
  constexpr const auto ThrVal = make_layout(
      make_layout(make_shape(Int<NumThreads/NumLoadK>{}, Int<NumLoadK>{}), make_stride(Int<SmemShapeK>{}, Int<VecSize>{})),
      make_layout(Int<VecSize>{})
      );
  const auto ld_gB = gB.compose(ThrVal)(threadIdx.x, _);
  const auto ld_cB = cB.compose(ThrVal);
  clear(rB);
#pragma unroll
  for (int i = 0; i < VecSize; i++) {
    if (elem_less(ld_cB(threadIdx.x, i), make_coord(n, k))) {
      rB(i) = ld_gB(i);
    }
  }
}

template <int NumThreads, int SmemShapeM, int SmemShapeK, int VecSize, typename SmemT, typename RegT>
__device__ void store_smem_a(SmemT& sA, const RegT& rA, int m, int k) {
  static_assert(is_rmem<typename RegT::engine_type>());
  static_assert(VecSize == 1 || VecSize == 2 || VecSize == 4);
  constexpr const auto ThrVal = make_layout(make_layout(Int<NumThreads>{}, Int<VecSize>{}), make_layout(Int<VecSize>{}));
  auto st_sA = sA.compose(ThrVal)(threadIdx.x, _);
  copy(rA, st_sA);
}

template <int NumThreads, int SmemShapeN, int SmemShapeK, int VecSize, typename SmemT, typename RegT>
__device__ void store_smem_b(SmemT& sB, const RegT& rB, int n, int k) {
  static_assert(is_rmem<typename RegT::engine_type>());
  static_assert(VecSize == 1 || VecSize == 2 || VecSize == 4);
  constexpr const auto NumLoadK = SmemShapeK / VecSize; // number of load along k
  static_assert(NumLoadK > 0);
  constexpr const auto ThrVal = make_layout(
      make_layout(make_shape(Int<NumThreads/NumLoadK>{}, Int<NumLoadK>{}), make_stride(Int<SmemShapeK>{}, Int<VecSize>{})),
      make_layout(Int<VecSize>{})
      );
  auto st_sB = sB.compose(ThrVal)(threadIdx.x, _);
  copy(rB, st_sB);
}

template <typename AccT, typename FragAT, typename FragBT>
__device__ void rank1_update(const FragAT& fragA, const FragBT& fragB, AccT& acc) {
  static_assert(is_rmem<typename AccT::engine_type>() && is_rmem<typename FragAT::engine_type>() && is_rmem<typename FragBT::engine_type>());
  constexpr const auto ThreadShapeM = size<0>(typename AccT::layout_type{});
  constexpr const auto ThreadShapeN = size<1>(typename AccT::layout_type{});
#pragma unroll
  for (int j = 0; j < ThreadShapeN; j++) {
#pragma unroll
    for (int i = 0; i < ThreadShapeM; i++) {
      acc(i, j) += fragA(i) * fragB(j);
    }
  }
}

template <typename AccT, typename CtaCT, typename CoordT>
__device__ void acc_store(const AccT& acc, CtaCT& threadC, CoordT thread_cC, int m, int n) {
  if (elem_less(thread_cC(size<0>(thread_cC) - 1, size<1>(thread_cC) - 1), make_coord(m, n))) {  // fast path
    copy(acc, threadC);
  } else {
    const auto [thread_i, thread_j] = thread_cC(_0{}, _0{});
    auto predA = make_tensor<bool>(Int<size<0>(thread_cC)>{});
    for (int i = 0; i < size<0>(predA); i++) {
      predA(i) = thread_i + i < m;
    }
    constexpr const auto ThreadShapeN = size<1>(typename AccT::layout_type{});
#pragma unroll
    for (int j = 0; j < ThreadShapeN; j++) {
      if (thread_j + j < n) {
        copy_if(predA, acc(_, j), threadC(_, j));
      }
    }
  }
}

namespace column_major {
template <int NumThreads, int CtaShapeM, int CtaShapeN, int SmemShapeK, int ThreadShapeM, int ThreadShapeN>
__launch_bounds__(NumThreads, 2)
MATMUL_KERNEL_SIGNATURE(matmul_smem_and_register_pipelining_2) {
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

  using fragA_t = decltype(make_fragment_like<float>(Int<ThreadShapeM>{}));
  using fragB_t = decltype(make_fragment_like<float>(Int<ThreadShapeN>{}));
  fragA_t fragA[2] = {make_fragment_like<float>(Int<ThreadShapeM>{}), make_fragment_like<float>(Int<ThreadShapeM>{})};
  fragB_t fragB[2] = {make_fragment_like<float>(Int<ThreadShapeN>{}), make_fragment_like<float>(Int<ThreadShapeN>{})};
  auto acc = make_fragment_like<float>(make_shape(Int<ThreadShapeM>{}, Int<ThreadShapeN>{}));
  clear(acc);

  constexpr const auto SmemALoadStoreVec = (SmemShapeM * SmemShapeK) / NumThreads;  // VecSize
  constexpr const auto SmemBLoadStoreVec = (SmemShapeN * SmemShapeK) / NumThreads;  // VecSize
  auto staging_a = make_fragment_like<float>(Int<SmemALoadStoreVec>{});
  auto staging_b = make_fragment_like<float>(Int<SmemBLoadStoreVec>{});

  int p_tile_count = (k - 1) / SmemShapeK + 1;
  int p_tile_next = 0;

  load_global_a<NumThreads, SmemShapeM, SmemShapeK, SmemALoadStoreVec>(staging_a, stripe_gA(_, _, _0{}, p_tile_next), stripe_cA(_, _, _0{}, p_tile_next), m, k);
  load_global_b<NumThreads, SmemShapeN, SmemShapeK, SmemBLoadStoreVec>(staging_b, stripe_gB(_, _, _0{}, p_tile_next), stripe_cB(_, _, _0{}, p_tile_next), n, k);
  if (--p_tile_count > 0) {
    ++p_tile_next;
  }
  store_smem_a<NumThreads, SmemShapeM, SmemShapeK, SmemALoadStoreVec>(sA, staging_a, m, k);
  store_smem_b<NumThreads, SmemShapeN, SmemShapeK, SmemBLoadStoreVec>(sB, staging_b, n, k);
  __syncthreads();

  const auto stripe_sA = local_tile(sA, make_tile(Int<ThreadShapeM>{}, Int<SmemShapeK>{}), threadIdx.x % (CtaShapeM / ThreadShapeM));
  const auto stripe_sB = local_tile(sB, make_tile(Int<ThreadShapeN>{}, Int<SmemShapeK>{}), threadIdx.x / (CtaShapeM / ThreadShapeM));

  copy(stripe_sA(_, 0, _0{}), fragA[0]);  // load_fragment a
  copy(stripe_sB(_, 0, _0{}), fragB[0]);  // load_fragment b

  const auto num_smem_block = size<3>(stripe_gA);
#pragma unroll 1  // no unroll
  for (int block_p = 0; block_p < num_smem_block; block_p++) {
#pragma unroll
    for (int smem_AB_thread_p = 0; smem_AB_thread_p < SmemShapeK; smem_AB_thread_p++) {
      if (smem_AB_thread_p == SmemShapeK - 1) {
        __syncthreads();
        store_smem_a<NumThreads, SmemShapeM, SmemShapeK, SmemALoadStoreVec>(sA, staging_a, m, k);
        store_smem_b<NumThreads, SmemShapeN, SmemShapeK, SmemBLoadStoreVec>(sB, staging_b, n, k);
        __syncthreads();
        if (--p_tile_count > 0) {
          ++p_tile_next;
        }
      }

      copy(stripe_sA(_, (smem_AB_thread_p + 1) % SmemShapeK, _0{}), fragA[(smem_AB_thread_p + 1) % 2]);  // load_fragment a
      copy(stripe_sB(_, (smem_AB_thread_p + 1) % SmemShapeK, _0{}), fragB[(smem_AB_thread_p + 1) % 2]);  // load_fragment b

      if (smem_AB_thread_p == 0) {
        load_global_a<NumThreads, SmemShapeM, SmemShapeK, SmemALoadStoreVec>(staging_a, stripe_gA(_, _, _0{}, p_tile_next), stripe_cA(_, _, _0{}, p_tile_next), m, k);
        load_global_b<NumThreads, SmemShapeN, SmemShapeK, SmemBLoadStoreVec>(staging_b, stripe_gB(_, _, _0{}, p_tile_next), stripe_cB(_, _, _0{}, p_tile_next), n, k);
      }

      rank1_update(fragA[smem_AB_thread_p % 2], fragB[smem_AB_thread_p % 2], acc);
    }
  }

  constexpr const auto MapToThread = make_layout(make_shape(Int<CtaShapeM / ThreadShapeM>{}, Int<CtaShapeN / ThreadShapeN>{}));
  auto [cta_thread_i, cta_thread_j] = idx2crd(MapToThread(threadIdx.x), MapToThread.shape(), MapToThread.stride());
  cta_thread_i *= ThreadShapeM;
  cta_thread_j *= ThreadShapeN;

  auto threadC = local_tile(ctaC, make_tile(Int<ThreadShapeM>{}, Int<ThreadShapeN>{}), make_coord(cta_thread_i / ThreadShapeM, cta_thread_j / ThreadShapeN));
  auto thread_cC = local_tile(cta_cC, make_tile(Int<ThreadShapeM>{}, Int<ThreadShapeN>{}), make_coord(cta_thread_i / ThreadShapeM, cta_thread_j / ThreadShapeN));
  acc_store(acc, threadC, thread_cC, m, n);
}

#define MATMUL_KERNEL_LAUNCH(name, num_threads, cta_shape_m, cta_shape_n, smem_shape_k, thread_shape_m, thread_shape_n)                                  \
  MATMUL_SIGNATURE(launch_##name##_##num_threads##t_cta##cta_shape_m##x##cta_shape_n##_smem##smem_shape_k##_thread##thread_shape_m##x##thread_shape_n) { \
    dim3 threads(num_threads);                                                                                                                           \
    dim3 blocks(ceil_div<int64_t>(m, cta_shape_m), ceil_div<int64_t>(n, cta_shape_n));                                                                   \
    name<num_threads, cta_shape_m, cta_shape_n, smem_shape_k, thread_shape_m, thread_shape_n><<<                                                         \
        blocks, threads, 0, nullptr>>>(m, n, k, a, lda, b, ldb, c, ldc);                                                                                 \
    CUDA_CHECK(cudaGetLastError());                                                                                                                      \
  }

MATMUL_KERNEL_LAUNCH(matmul_smem_and_register_pipelining_2, 256, 128, 128, 4, 8, 8);
MATMUL_KERNEL_LAUNCH(matmul_smem_and_register_pipelining_2, 256, 128, 128, 8, 8, 8);
// MATMUL_KERNEL_LAUNCH(matmul_smem_and_register_pipelining_2, 256, 128, 128, 16, 8, 8);
// MATMUL_KERNEL_LAUNCH(matmul_smem_and_register_pipelining_2, 256, 128, 128, 24, 8, 8);
// MATMUL_KERNEL_LAUNCH(matmul_smem_and_register_pipelining_2, 256, 128, 128, 32, 8, 8);
// MATMUL_KERNEL_LAUNCH(matmul_smem_and_register_pipelining_2, 256, 128, 128, 40, 8, 8);
// MATMUL_KERNEL_LAUNCH(matmul_smem_and_register_pipelining_2, 256, 128, 128, 48, 8, 8);

MATMUL_DMODULE(m) {
  REGISTER(launch_matmul_smem_and_register_pipelining_2_256t_cta128x128_smem4_thread8x8);
  REGISTER(launch_matmul_smem_and_register_pipelining_2_256t_cta128x128_smem8_thread8x8);
  // REGISTER(launch_matmul_smem_and_register_pipelining_2_256t_cta128x128_smem16_thread8x8);
  // REGISTER(launch_matmul_smem_and_register_pipelining_2_256t_cta128x128_smem24_thread8x8);
  // REGISTER(launch_matmul_smem_and_register_pipelining_2_256t_cta128x128_smem32_thread8x8);
  // REGISTER(launch_matmul_smem_and_register_pipelining_2_256t_cta128x128_smem40_thread8x8);
  // REGISTER(launch_matmul_smem_and_register_pipelining_2_256t_cta128x128_smem48_thread8x8);
}

}  // namespace column_major
