#include "cuda/matmul.cuh"

namespace column_major {

template <int M, int N>
using Acc = float[N][M];

template <int Size>
struct Registers {
  float reg[Size];
};

template <int Size>
using Fragment = float[Size];

// Since we loop along k-axis for rank-1 update. Smem of a and b matrix should have the same K dimension
// template <int K, int Size> using Smem = float[K][Size];  // this cause ICE with nvcc 12.2
template <int K, int Size>
struct Array {
  float mem[K][Size];
};

// | ^ ^ ^  The transverse order of A and B.
// |/|/|/|  The A and B are assumed to be column majored.
// v v v |  The order ensure the access to global memory is coalesced.
// Threads cooperatively load from global memory to shared memory
// if number of elements in shared memory, then split the loading into multiple batches, along k-axis.
template <int NumThreads, int SmemShapeM, int SmemShapeK, int SmemANumBatch = (SmemShapeM * SmemShapeK) / NumThreads>
__device__ void load_global_a(
    Registers<SmemANumBatch>& reg_a,
    int m,
    int k,
    const float* a,
    int lda,
    int a_thread_p
) {
  // Ensure the threads fill the column of A and the row of B. That is, when split only split along k-axis.
  // Otherwise, some elements will not be correctly handled.
  static_assert(NumThreads % SmemShapeM == 0);
  constexpr const auto SmemABatchShapeK = SmemShapeK / SmemANumBatch;

  const int A_i = SmemShapeM * blockIdx.x + threadIdx.x % SmemShapeM;
  const int A_batchp = a_thread_p + threadIdx.x / SmemShapeM;
#pragma unroll
  for (int batch = 0; batch < SmemANumBatch; batch++) {
    const auto A_p = A_batchp + batch * SmemABatchShapeK;
    reg_a.reg[batch] = A_i >= m || A_p >= k ? 0 : a[A_i * 1 + A_p * lda];
  }
}

template <int NumThreads, int SmemShapeK, int SmemShapeN, int SmemBNumBatch = (SmemShapeN * SmemShapeK) / NumThreads>
__device__ void load_global_b(
    Registers<SmemBNumBatch>& reg_b,
    int k,
    int n,
    const float* b,
    int ldb,
    int b_thread_p
) {
  static_assert(NumThreads % SmemShapeN == 0);
  constexpr const auto SmemBBatchShapeK = SmemShapeK / SmemBNumBatch;

  const int B_batchp = b_thread_p + threadIdx.x % SmemBBatchShapeK;
  const int B_j = SmemShapeN * blockIdx.y + threadIdx.x / SmemBBatchShapeK;
#pragma unroll
  for (int batch = 0; batch < SmemShapeN * SmemShapeK / NumThreads; batch++) {
    const auto B_p = B_batchp + batch * SmemBBatchShapeK;
    reg_b.reg[batch] = B_p >= k || B_j >= n ? 0 : b[B_p * 1 + B_j * ldb];
  }
}

template <int NumThreads, int SmemShapeM, int SmemShapeK, int SmemANumBatch = (SmemShapeM * SmemShapeK) / NumThreads>
__device__ void store_smem_a(
    Array<SmemShapeK, SmemShapeM>& smem_a,
    const Registers<SmemANumBatch>& reg_a
) {
  static_assert(NumThreads % SmemShapeM == 0);
  constexpr const auto SmemABatchShapeK = SmemShapeK / SmemANumBatch;
#pragma unroll
  for (int batch = 0; batch < SmemANumBatch; batch++) {
    const auto smem_A_thread_i = threadIdx.x % SmemShapeM;
    const auto smem_A_thread_p = threadIdx.x / SmemShapeM + batch * SmemABatchShapeK;
    smem_a.mem[smem_A_thread_p][smem_A_thread_i] = reg_a.reg[batch];
  }
}

template <int NumThreads, int SmemShapeK, int SmemShapeN, int SmemBNumBatch = (SmemShapeN * SmemShapeK) / NumThreads>
__device__ void store_smem_b(
    Array<SmemShapeK, SmemShapeN>& smem_b,
    const Registers<SmemBNumBatch>& reg_b
) {
  static_assert(NumThreads % SmemShapeN == 0);
  constexpr const auto SmemBBatchShapeK = SmemShapeK / SmemBNumBatch;
#pragma unroll
  for (int batch = 0; batch < SmemShapeN * SmemShapeK / NumThreads; batch++) {
    const auto smem_B_thread_p = threadIdx.x % SmemBBatchShapeK + batch * SmemBBatchShapeK;
    const auto smem_B_thread_j = threadIdx.x / SmemBBatchShapeK;
    smem_b.mem[smem_B_thread_p][smem_B_thread_j] = reg_b.reg[batch];
  }
}

template <int FragmentSize, int Step, int SmemShapeK, int SmemShapeM /*or SmemShapeN*/>
__device__ void load_fragment(
    Fragment<FragmentSize>& frag_a,               // or frag_b
    const Array<SmemShapeK, SmemShapeM>& smem_a,  // or smem_b
    int smem_a_thread_p,                          // or smem_b_thread_p
    int smem_a_thread_i                           // or smem_b_thread_j
) {
  static_assert(SmemShapeM % FragmentSize == 0);
  static_assert(FragmentSize % 4 == 0);
  auto base = &smem_a.mem[smem_a_thread_p][smem_a_thread_i];
#pragma unroll
  for (int f = 0; f < FragmentSize / 4; f++) {
    auto ptr = base + f * Step;
#pragma unroll
    for (int n = 0; n < 4; n++) {
      frag_a[f * 4 + n] = *ptr++;
    }
  }
}

template <int ThreadShapeM, int ThreadShapeN>
__device__ void rank1_update(Acc<ThreadShapeM, ThreadShapeN>& acc, const Fragment<ThreadShapeM>& frag_a, const Fragment<ThreadShapeN>& frag_b) {
  // rank-1 update to acc registers
#pragma unroll
  for (int jj = 0; jj < ThreadShapeN; jj += 4) {
#pragma unroll
    for (int ii = 0; ii < ThreadShapeM; ii += 4) {
#pragma unroll
      for (int j = 0; j < 4; j++) {
#pragma unroll
        for (int i = 0; i < 4; i++) {
          acc[jj + j][ii + i] += frag_a[ii + i] * frag_b[jj + j];
        }
      }
    }
  }
}

// FIXME: change for 2x2
template <int ThreadShapeM, int ThreadShapeN, int StepM, int StepN>
__device__ void acc_store(int m, int n, float* C, int ldc, Acc<ThreadShapeM, ThreadShapeN>& acc, int thread_i, int thread_j) {
  // store acc registers results to C
  float* thread_c = &C[thread_i * 1 + thread_j * ldc];
#pragma unroll
  for (int bb = 0; bb < ThreadShapeN / 4; bb++) {
#pragma unroll
    for (int aa = 0; aa < ThreadShapeM / 4; aa++) {
#pragma unroll
      for (int b = 0; b < 4; b++) {
#pragma unroll
        for (int a = 0; a < 4; a++) {
          if (thread_j + (bb * StepN + b) < n) {
            if (thread_i + (aa * StepM + a) < m) {
              thread_c[(aa * StepM + a) * 1 + (bb * StepN + b) * ldc] = acc[bb * 4 + b][aa * 4 + a];
            }
          }
        }
      }
    }
  }
}

#if 0
#define PRINTF(...)                                           \
  if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) \
  printf(__VA_ARGS__)
#else
#define PRINTF(...)
#endif

// Laucnh a 1D CTA(threadblock)
// Each CTA process CtaShapeM x CtaShapeN tile of C
// Preload a block of shared memory
// In the loop
// CTA load SmemShapeM x SmemShapeK and SmemShapeK x SmemShapeN of A and B from global memory to shared memory
// Each thread process ThreadShapeM x ThreadShapeN of **collocated** data
// Each thread then load (ThreadShapeM + ThreadShapeN) of elements, and do ThreadShapeM * ThreadShapeN of FMAs.
template <int NumThreads, int CtaShapeM, int CtaShapeN, int SmemShapeK, int WarpShapeM, int WarpShapeN, int ThreadShapeM, int ThreadShapeN>
MATMUL_KERNEL_SIGNATURE(matmul_kernel_pipelining_and_warp_tiling_3) {
  constexpr const auto SmemShapeM = CtaShapeM;
  constexpr const auto SmemShapeN = CtaShapeN;
  static_assert((SmemShapeM * SmemShapeK) % NumThreads == 0 && (SmemShapeN * SmemShapeK) % NumThreads == 0);
  static_assert((CtaShapeM * CtaShapeN / (ThreadShapeM * ThreadShapeN)) == NumThreads);
  static_assert(CtaShapeM % ThreadShapeM == 0 && CtaShapeN % ThreadShapeN == 0);

  __shared__ Array<SmemShapeK, SmemShapeM> smem_a[2];
  __shared__ Array<SmemShapeK, SmemShapeN> smem_b[2];
  Registers<SmemShapeM * SmemShapeK / NumThreads> staging_a;
  Registers<SmemShapeN * SmemShapeK / NumThreads> staging_b;

  constexpr const int LoadFragmentNumBatchM = ThreadShapeM / 4;
  constexpr const int LoadFragmentNumBatchN = ThreadShapeN / 4;
  constexpr const int WarpStepM = WarpShapeM / LoadFragmentNumBatchM;
  constexpr const int WarpStepN = WarpShapeN / LoadFragmentNumBatchN;

  const int warp_idx = threadIdx.x / warpSize;
  const int warp_part_i = (warp_idx % (CtaShapeM / WarpShapeM)) * WarpShapeM;
  const int warp_part_j = (warp_idx / (CtaShapeM / WarpShapeM)) * WarpShapeN;
  const int warp_i = warp_part_i + ((threadIdx.x % warpSize) % (WarpShapeM / ThreadShapeM)) * 4;
  const int warp_j = warp_part_j + ((threadIdx.x % warpSize) / (WarpShapeM / ThreadShapeM)) * 4;

  Acc<ThreadShapeM, ThreadShapeN> acc{};
  Fragment<ThreadShapeM> frag_a[2];
  Fragment<ThreadShapeN> frag_b[2];

  // each thread then load from shared memory to register and perform the rank-1 update
  // threads are not organized naively as previous kernel, instead, each warp now have a shape.
  const auto smem_A_thread_i = warp_i;
  const auto smem_B_thread_j = warp_j;

  int p_tile_count = (k - 1) / SmemShapeK + 1;
  int p_tile_curr = 0;
  int p_tile_next = 0;

  // pre-load first block of A, B to smem_A, smem_B
  PRINTF("%4d g->r    0\n", __LINE__);
  load_global_a<NumThreads, SmemShapeM, SmemShapeK>(staging_a, m, k, a, lda, p_tile_next);
  load_global_b<NumThreads, SmemShapeK, SmemShapeN>(staging_b, k, n, b, ldb, p_tile_next);
  if (--p_tile_count > 0) {
    ++p_tile_next;
  }

  PRINTF("%4d r->s    0\n", __LINE__);
  store_smem_a<NumThreads, SmemShapeM, SmemShapeK>(smem_a[0], staging_a);
  store_smem_b<NumThreads, SmemShapeK, SmemShapeN>(smem_b[0], staging_b);
  __syncthreads();
  // pre-load first fragment of registers
  PRINTF("%4d s->r    0    0\n", __LINE__);
  load_fragment<ThreadShapeM, WarpStepM, SmemShapeK, SmemShapeM>(frag_a[0], smem_a[0], 0, smem_A_thread_i);
  load_fragment<ThreadShapeN, WarpStepN, SmemShapeK, SmemShapeN>(frag_b[0], smem_b[0], 0, smem_B_thread_j);

#pragma unroll 1  // no unroll
  while (p_tile_count > -1) {
#pragma unroll
    for (int smem_AB_thread_p = 0; smem_AB_thread_p < SmemShapeK; smem_AB_thread_p++) {
      if (smem_AB_thread_p == SmemShapeK - 1) {
        // __syncthreads();
        PRINTF("%4d r->s %4d\n", __LINE__, p_tile_next * SmemShapeK);
        store_smem_a<NumThreads, SmemShapeM, SmemShapeK>(smem_a[p_tile_next % 2], staging_a);
        store_smem_b<NumThreads, SmemShapeK, SmemShapeN>(smem_b[p_tile_next % 2], staging_b);
        __syncthreads();
        if (--p_tile_count > 0) {
          ++p_tile_next;
        }
        ++p_tile_curr;
      }

      PRINTF("%4d s->r %4d %4d\n", __LINE__, p_tile_curr * SmemShapeK, (smem_AB_thread_p + 1) % SmemShapeK);
      load_fragment<ThreadShapeM, WarpStepM, SmemShapeK, SmemShapeM>(frag_a[(smem_AB_thread_p + 1) % 2], smem_a[p_tile_curr % 2], (smem_AB_thread_p + 1) % SmemShapeK, smem_A_thread_i);
      load_fragment<ThreadShapeN, WarpStepN, SmemShapeK, SmemShapeN>(frag_b[(smem_AB_thread_p + 1) % 2], smem_b[p_tile_curr % 2], (smem_AB_thread_p + 1) % SmemShapeK, smem_B_thread_j);

      if (smem_AB_thread_p == 0) {
        PRINTF("%4d g->r %4d\n", __LINE__, p_tile_next * SmemShapeK);
        load_global_a<NumThreads, SmemShapeM, SmemShapeK>(staging_a, m, k, a, lda, p_tile_next * SmemShapeK);
        load_global_b<NumThreads, SmemShapeK, SmemShapeN>(staging_b, k, n, b, ldb, p_tile_next * SmemShapeK);
      }

      PRINTF("%4d acc       %4d\n", __LINE__, smem_AB_thread_p);
      rank1_update<ThreadShapeM, ThreadShapeN>(acc, frag_a[smem_AB_thread_p % 2], frag_b[smem_AB_thread_p % 2]);
    }
  }

  // store acc registers results to C
  const int cta_part_i = CtaShapeM * blockIdx.x;
  const int cta_part_j = CtaShapeN * blockIdx.y;
  const int thread_i = cta_part_i + warp_i;
  const int thread_j = cta_part_j + warp_j;
  acc_store<ThreadShapeM, ThreadShapeN, WarpStepM, WarpStepN>(m, n, c, ldc, acc, thread_i, thread_j);
}

#define MATMUL_KERNEL_LAUNCH(name, num_threads, cta_shape_m, cta_shape_n, smem_shape_k, warp_shape_m, warp_shape_n, thread_shape_m, thread_shape_n)                                            \
  MATMUL_SIGNATURE(launch_##name##_##num_threads##t_cta##cta_shape_m##x##cta_shape_n##_smem##smem_shape_k##_warp##warp_shape_m##x##warp_shape_n##_thread##thread_shape_m##x##thread_shape_n) { \
    dim3 threads(num_threads);                                                                                                                                                                 \
    dim3 blocks(ceil_div<int64_t>(m, cta_shape_m), ceil_div<int64_t>(n, cta_shape_n));                                                                                                         \
    name<num_threads, cta_shape_m, cta_shape_n, smem_shape_k, warp_shape_m, warp_shape_n, thread_shape_m, thread_shape_n><<<                                                                   \
        blocks, threads, 0, nullptr>>>(m, n, k, a, lda, b, ldb, c, ldc);                                                                                                                       \
    CUDA_CHECK(cudaGetLastError());                                                                                                                                                            \
  }

MATMUL_KERNEL_LAUNCH(matmul_kernel_pipelining_and_warp_tiling_3, 256, 128, 128, 8, 64, 32, 8, 8);
MATMUL_KERNEL_LAUNCH(matmul_kernel_pipelining_and_warp_tiling_3, 256, 128, 128, 8, 32, 64, 8, 8);
MATMUL_KERNEL_LAUNCH(matmul_kernel_pipelining_and_warp_tiling_3, 256, 128, 128, 16, 64, 32, 8, 8);
MATMUL_KERNEL_LAUNCH(matmul_kernel_pipelining_and_warp_tiling_3, 256, 128, 128, 16, 32, 64, 8, 8);

MATMUL_DMODULE(m) {
  REGISTER(launch_matmul_kernel_pipelining_and_warp_tiling_3_256t_cta128x128_smem8_warp64x32_thread8x8);
  REGISTER(launch_matmul_kernel_pipelining_and_warp_tiling_3_256t_cta128x128_smem8_warp32x64_thread8x8);
  REGISTER(launch_matmul_kernel_pipelining_and_warp_tiling_3_256t_cta128x128_smem16_warp64x32_thread8x8);
  REGISTER(launch_matmul_kernel_pipelining_and_warp_tiling_3_256t_cta128x128_smem16_warp32x64_thread8x8);
}

}  // namespace column_major
