#include "cuda/matmul.cuh"

namespace column_major {

template <int M, int N>
using Acc = float[N][M];

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
template <int NumThreads, int SmemShapeM, int SmemShapeK>
__device__ void store_smem_load_global_a(
    Array<SmemShapeK, SmemShapeM>& smem_a,
    int m,
    int k,
    const float* a,
    int lda,
    int a_basep
) {
  // Ensure the threads fill the column of A and the row of B. That is, when split only split along k-axis.
  // Otherwise, some elements will not be correctly handled.
  static_assert(NumThreads % SmemShapeM == 0);
  constexpr const auto SmemANumBatch = (SmemShapeM * SmemShapeK) / NumThreads;
  constexpr const auto SmemABatchShapeK = SmemShapeK / SmemANumBatch;

  const int A_i = SmemShapeM * blockIdx.x + threadIdx.x % SmemShapeM;
  const int A_batchp = a_basep + threadIdx.x / SmemShapeM;
#pragma unroll
  for (int smem_batch = 0; smem_batch < SmemShapeM * SmemShapeK / NumThreads; smem_batch++) {
    const auto smem_A_thread_i = threadIdx.x % SmemShapeM;
    const auto smem_A_thread_p = threadIdx.x / SmemShapeM + smem_batch * SmemABatchShapeK;
    const auto A_p = A_batchp + smem_batch * SmemABatchShapeK;
    smem_a.mem[smem_A_thread_p][smem_A_thread_i] = A_i >= m || A_p >= k ? 0 : a[A_i * 1 + A_p * lda];
  }
}

template <int NumThreads, int SmemShapeK, int SmemShapeN>
__device__ void store_smem_load_global_b(
    Array<SmemShapeK, SmemShapeN>& smem_b,
    int k,
    int n,
    const float* b,
    int ldb,
    int b_basep
) {
  static_assert(NumThreads % SmemShapeN == 0);
  constexpr const auto SmemBNumBatch = (SmemShapeN * SmemShapeK) / NumThreads;
  constexpr const auto SmemBBatchShapeK = SmemShapeK / SmemBNumBatch;

  const int B_batchp = b_basep + threadIdx.x % SmemBBatchShapeK;
  const int B_j = SmemShapeN * blockIdx.y + threadIdx.x / SmemBBatchShapeK;
#pragma unroll
  for (int smem_batch = 0; smem_batch < SmemShapeN * SmemShapeK / NumThreads; smem_batch++) {
    const auto smem_B_thread_p = threadIdx.x % SmemBBatchShapeK + smem_batch * SmemBBatchShapeK;
    const auto smem_B_thread_j = threadIdx.x / SmemBBatchShapeK;
    const auto B_p = B_batchp + smem_batch * SmemBBatchShapeK;
    smem_b.mem[smem_B_thread_p][smem_B_thread_j] = B_p >= k || B_j >= n ? 0 : b[B_p * 1 + B_j * ldb];
  }
}

template <int FragmentSize, int SmemShapeK, int SmemShapeM /*or SmemShapeN*/>
__device__ void load_fragment(
    Fragment<FragmentSize>& frag_a,               // or frag_b
    const Array<SmemShapeK, SmemShapeM>& smem_a,  // or smem_b
    int smem_a_thread_p,                          // or smem_b_thread_p
    int smem_a_thread_i                           // or smem_b_thread_j
) {
  static_assert(SmemShapeM % FragmentSize == 0);
  auto ptr = &smem_a.mem[smem_a_thread_p][smem_a_thread_i];
#pragma unroll
  for (int f = 0; f < FragmentSize; f++, ptr += 1) {
    frag_a[f] = *ptr;
  }
}

template <int ThreadShapeM, int ThreadShapeN>
__device__ void rank1_update(Acc<ThreadShapeM, ThreadShapeN>& acc, const Fragment<ThreadShapeM>& frag_a, const Fragment<ThreadShapeN>& frag_b) {
  // rank-1 update to acc registers
#pragma unroll
  for (int j = 0; j < ThreadShapeN; j++) {
#pragma unroll
    for (int i = 0; i < ThreadShapeM; i++) {
      acc[j][i] += frag_a[i] * frag_b[j];
    }
  }
}

template <int ThreadShapeM, int ThreadShapeN>
__device__ void acc_store(int m, int n, float* C, int ldc, Acc<ThreadShapeM, ThreadShapeN>& acc, int thread_i, int thread_j) {
  // store acc registers results to C
  float* thread_c = &C[thread_i * 1 + thread_j * ldc];
#pragma unroll
  for (int b = 0; b < ThreadShapeN; b++) {
    if (thread_j + b < n) {
#pragma unroll
      for (int a = 0; a < ThreadShapeM; a++) {
        if (thread_i + a < m) {
          thread_c[a * 1 + b * ldc] = acc[b][a];
        }
      }
    }
  }
}

// Laucnh a 1D CTA(threadblock)
// Each CTA process CtaShapeM x CtaShapeN tile of C
// CTA load SmemShapeM x SmemShapeK and SmemShapeK x SmemShapeN of A and B from global memory to shared memory
// CTA contains NumThreads/32 of warps. Each warp is then organized to process WarpShapeM x WarpShapeN of data of C,
//     threads in a warp are layouted as column major and warps in a CTA are layouted in column major.
// Each thread process ThreadShapeM x ThreadShapeN of **collocated** data
// Each thread then load (ThreadShapeM + ThreadShapeN) of elements, and do ThreadShapeM * ThreadShapeN of FMAs.
template <int NumThreads, int CtaShapeM, int CtaShapeN, int SmemShapeK, int WarpShapeM, int WarpShapeN, int ThreadShapeM, int ThreadShapeN>
MATMUL_KERNEL_SIGNATURE(matmul_kernel_warp_tiling) {
  constexpr const auto SmemShapeM = CtaShapeM;
  constexpr const auto SmemShapeN = CtaShapeN;
  static_assert((SmemShapeM * SmemShapeK) % NumThreads == 0 && (SmemShapeN * SmemShapeK) % NumThreads == 0);
  static_assert((CtaShapeM * CtaShapeN / (ThreadShapeM * ThreadShapeN)) == NumThreads);
  // static_assert((WarpShapeM * WarpShapeN / (ThreadShapeM * ThreadShapeN)) == warpSize);
  static_assert(CtaShapeM % WarpShapeM == 0 && CtaShapeN % WarpShapeN == 0);
  static_assert(WarpShapeM % ThreadShapeM == 0 && WarpShapeN % ThreadShapeN == 0);

  __shared__ Array<SmemShapeK, SmemShapeM> smem_a;
  __shared__ Array<SmemShapeK, SmemShapeN> smem_b;

  const int warp_id = threadIdx.x / warpSize;  // can be accessed with special registers %warpid in PTX
  const int lane_id = threadIdx.x % warpSize;  // can be accessed with special registers %laneid in PTX
  // from cta to warp
  const int cta_warp_i = (warp_id % (CtaShapeM / WarpShapeM)) * WarpShapeM;
  const int cta_warp_j = (warp_id / (CtaShapeM / WarpShapeM)) * WarpShapeN;
  // from cta to thread
  const int cta_thread_i = cta_warp_i + (lane_id % (WarpShapeM / ThreadShapeM)) * ThreadShapeM;
  const int cta_thread_j = cta_warp_j + (lane_id / (WarpShapeM / ThreadShapeM)) * ThreadShapeN;

  Acc<ThreadShapeM, ThreadShapeN> acc{};
  Fragment<ThreadShapeM> frag_a;
  Fragment<ThreadShapeN> frag_b;

  for (int p = 0; p < k; p += SmemShapeK) {
    // load A, B to smem_A, smem_B
    store_smem_load_global_a<NumThreads, SmemShapeM, SmemShapeK>(smem_a, m, k, a, lda, p);
    store_smem_load_global_b<NumThreads, SmemShapeK, SmemShapeN>(smem_b, k, n, b, ldb, p);
    __syncthreads();

    // each thread then load from shared memory to register and perform the rank-1 update
    // threads are not organized naively as previous kernel, instead, each warp now have a shape.
    const auto smem_A_thread_i = cta_thread_i;
    const auto smem_B_thread_j = cta_thread_j;
    // #pragma unroll
    for (int smem_AB_thread_p = 0; smem_AB_thread_p < SmemShapeK; smem_AB_thread_p++) {
      // register load
      load_fragment<ThreadShapeM, SmemShapeK, SmemShapeM>(frag_a, smem_a, smem_AB_thread_p, smem_A_thread_i);
      load_fragment<ThreadShapeN, SmemShapeK, SmemShapeN>(frag_b, smem_b, smem_AB_thread_p, smem_B_thread_j);

      rank1_update<ThreadShapeM, ThreadShapeN>(acc, frag_a, frag_b);
    }
    __syncthreads();
  }

  // store acc registers results to C
  const int cta_i = CtaShapeM * blockIdx.x;
  const int cta_j = CtaShapeN * blockIdx.y;
  const int thread_i = cta_i + cta_thread_i;
  const int thread_j = cta_j + cta_thread_j;
  acc_store(m, n, c, ldc, acc, thread_i, thread_j);
}

#define MATMUL_KERNEL_LAUNCH(name, num_threads, cta_shape_m, cta_shape_n, smem_shape_k, warp_shape_m, warp_shape_n, thread_shape_m, thread_shape_n)                                            \
  MATMUL_SIGNATURE(launch_##name##_##num_threads##t_cta##cta_shape_m##x##cta_shape_n##_smem##smem_shape_k##_warp##warp_shape_m##x##warp_shape_n##_thread##thread_shape_m##x##thread_shape_n) { \
    dim3 threads(num_threads);                                                                                                                                                                 \
    dim3 blocks(ceil_div<int64_t>(m, cta_shape_m), ceil_div<int64_t>(n, cta_shape_n));                                                                                                         \
    name<num_threads, cta_shape_m, cta_shape_n, smem_shape_k, warp_shape_m, warp_shape_n, thread_shape_m, thread_shape_n><<<                                                                   \
        blocks, threads, 0, nullptr>>>(m, n, k, a, lda, b, ldb, c, ldc);                                                                                                                       \
    CUDA_CHECK(cudaGetLastError());                                                                                                                                                            \
  }

MATMUL_KERNEL_LAUNCH(matmul_kernel_warp_tiling, 256, 128, 128, 8, 128, 16, 8, 8);
MATMUL_KERNEL_LAUNCH(matmul_kernel_warp_tiling, 256, 128, 128, 8, 64, 32, 8, 8);
MATMUL_KERNEL_LAUNCH(matmul_kernel_warp_tiling, 256, 128, 128, 8, 32, 64, 8, 8);
MATMUL_KERNEL_LAUNCH(matmul_kernel_warp_tiling, 256, 128, 128, 8, 16, 128, 8, 8);
MATMUL_KERNEL_LAUNCH(matmul_kernel_warp_tiling, 256, 128, 128, 16, 128, 16, 8, 8);
MATMUL_KERNEL_LAUNCH(matmul_kernel_warp_tiling, 256, 128, 128, 16, 64, 32, 8, 8);
MATMUL_KERNEL_LAUNCH(matmul_kernel_warp_tiling, 256, 128, 128, 16, 32, 64, 8, 8);
MATMUL_KERNEL_LAUNCH(matmul_kernel_warp_tiling, 256, 128, 128, 16, 16, 128, 8, 8);

MATMUL_DMODULE(m) {
  REGISTER(launch_matmul_kernel_warp_tiling_256t_cta128x128_smem8_warp128x16_thread8x8);
  REGISTER(launch_matmul_kernel_warp_tiling_256t_cta128x128_smem8_warp64x32_thread8x8);
  REGISTER(launch_matmul_kernel_warp_tiling_256t_cta128x128_smem8_warp32x64_thread8x8);
  REGISTER(launch_matmul_kernel_warp_tiling_256t_cta128x128_smem8_warp16x128_thread8x8);
  REGISTER(launch_matmul_kernel_warp_tiling_256t_cta128x128_smem16_warp128x16_thread8x8);
  REGISTER(launch_matmul_kernel_warp_tiling_256t_cta128x128_smem16_warp64x32_thread8x8);
  REGISTER(launch_matmul_kernel_warp_tiling_256t_cta128x128_smem16_warp32x64_thread8x8);
  REGISTER(launch_matmul_kernel_warp_tiling_256t_cta128x128_smem16_warp16x128_thread8x8);
}

}  // namespace column_major
