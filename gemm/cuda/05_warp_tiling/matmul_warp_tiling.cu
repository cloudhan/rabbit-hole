#include "cuda/matmul.cuh"

namespace column_major {

// Laucnh a 1D CTA(threadblock)
// Each CTA process CtaShapeM x CtaShapeN tile of C
// CTA load SmemShapeM x SmemShapeK and SmemShapeK x SmemShapeN of A and B from global memory to shared memory
// CTA contains NumThreads/32 of warps. The each warp is then organized to process WarpShapeM x WarpShapeN of data of C,
//     threads in a warp are layouted as column major and warps in a CTA are layouted in column major.
// Each thread process ThreadShapeM x ThreadShapeN of **collocated** data
// Each thread then load (ThreadShapeM + ThreadShapeN) of elements, and do ThreadShapeM * ThreadShapeN of FMAs.
template <int NumThreads, int CtaShapeM, int CtaShapeN, int SmemShapeK, int WarpShapeM, int WarpShapeN, int ThreadShapeM, int ThreadShapeN>
__launch_bounds__(NumThreads) MATMUL_KERNEL_SIGNATURE(matmul_kernel_warp_tiling) {
  constexpr const auto SmemShapeM = CtaShapeM;
  constexpr const auto SmemShapeN = CtaShapeN;
  static_assert((SmemShapeM * SmemShapeK) % NumThreads == 0 && (SmemShapeN * SmemShapeK) % NumThreads == 0);
  static_assert((CtaShapeM * CtaShapeN / (ThreadShapeM * ThreadShapeN)) == NumThreads);
  // static_assert((WarpShapeM * WarpShapeN / (ThreadShapeM * ThreadShapeN)) == warpSize);
  static_assert(CtaShapeM % WarpShapeM == 0 && CtaShapeN % WarpShapeN == 0);
  static_assert(WarpShapeM % ThreadShapeM == 0 && WarpShapeN % ThreadShapeN == 0);

  __shared__ float smem_A[SmemShapeK][SmemShapeM];
  __shared__ float smem_B[SmemShapeK][SmemShapeN];

  const int cta_part_i = CtaShapeM * blockIdx.x;
  const int cta_part_j = CtaShapeN * blockIdx.y;
  const int warp_idx = threadIdx.x / warpSize;
  const int warp_part_i = (warp_idx % (CtaShapeM / WarpShapeM)) * WarpShapeM;
  const int warp_part_j = (warp_idx / (CtaShapeM / WarpShapeM)) * WarpShapeN;
  const int thread_i = cta_part_i + warp_part_i + ((threadIdx.x % warpSize) % (WarpShapeM / ThreadShapeM)) * ThreadShapeM;
  const int thread_j = cta_part_j + warp_part_j + ((threadIdx.x % warpSize) / (WarpShapeM / ThreadShapeM)) * ThreadShapeN;

  float acc[ThreadShapeN][ThreadShapeM]{};  // zeroing out

  float reg_A[ThreadShapeM];  // a column register for rank-1 update
  float reg_B[ThreadShapeN];  // a row    register for rank-1 update

  for (int p = 0; p < k; p += SmemShapeK) {
    // load A, B to smem_A, smem_B
    {
      // | ^ ^ ^  The transverse order of A and B.
      // |/|/|/|  The A and B are assumed to be column majored.
      // v v v |  The order ensure the access to global memory is coalesced.
      // Threads cooperatively load from global memory to shared memory
      // if number of elements in shared memory, the split the loading into multiple batches, along k-axis.
      constexpr const auto SmemANumBatch = (SmemShapeM * SmemShapeK) / NumThreads;
      constexpr const auto SmemBNumBatch = (SmemShapeN * SmemShapeK) / NumThreads;
      constexpr const auto SmemABatchShapeK = SmemShapeK / SmemANumBatch;
      constexpr const auto SmemBBatchShapeK = SmemShapeK / SmemBNumBatch;

      // Ensure the threads fill the column of A and the row of B. That is, when split only split along k-axis.
      // Otherwise, some elements will not be correctly handled.
      static_assert(NumThreads % SmemShapeM == 0);
      static_assert(NumThreads % SmemShapeN == 0);

      const int A_i = SmemShapeM * blockIdx.x + threadIdx.x % SmemShapeM;
      const int A_batchp = p + threadIdx.x / SmemShapeM;
#pragma unroll
      for (int smem_batch = 0; smem_batch < SmemANumBatch; smem_batch++) {
        const auto smem_A_thread_i = threadIdx.x % SmemShapeM;
        const auto smem_A_thread_p = threadIdx.x / SmemShapeM + smem_batch * SmemABatchShapeK;
        const auto A_p = A_batchp + smem_batch * SmemABatchShapeK;
        smem_A[smem_A_thread_p][smem_A_thread_i] = A_i >= m || A_p >= k ? 0 : a[A_i * 1 + A_p * lda];
        // printf("A(%d,%d) -> smem_A(%d, %d) %d %d %d\n", A_i, A_p, smem_A_thread_i, smem_A_thread_p, smem_batch, A_i >= m, A_p >= k);
      }

      const int B_batchp = p + threadIdx.x % SmemBBatchShapeK;
      const int B_j = SmemShapeN * blockIdx.y + threadIdx.x / SmemBBatchShapeK;
#pragma unroll
      for (int smem_batch = 0; smem_batch < SmemBNumBatch; smem_batch++) {
        const auto smem_B_thread_p = threadIdx.x % SmemBBatchShapeK + smem_batch * SmemBBatchShapeK;
        const auto smem_B_thread_j = threadIdx.x / SmemBBatchShapeK;
        const auto B_p = B_batchp + smem_batch * SmemBBatchShapeK;
        smem_B[smem_B_thread_p][smem_B_thread_j] = B_p >= k || B_j >= n ? 0 : b[B_p * 1 + B_j * ldb];
        // printf("B(%d,%d) -> smem_B(%d, %d) %d %d %d\n", B_p, B_j, smem_B_thread_p, smem_B_thread_j, smem_batch, B_p >= k, B_j >= n);
      }
    }
    __syncthreads();

    // each thread then load from shared memory to register and perform the rank-1 update
    // threads are not organized naively as previous kernel, instead, each warp now have a shape.
    {
      static_assert(SmemShapeM % ThreadShapeM == 0);
      static_assert(SmemShapeN % ThreadShapeN == 0);
      const auto smem_A_thread_i = thread_i - cta_part_i;
      const auto smem_B_thread_j = thread_j - cta_part_j;
      // printf("%d %d %d %d %d\n", threadIdx.x, warp_part_i, warp_part_j, smem_A_thread_i, smem_B_thread_j);
#pragma unroll
      for (int smem_AB_thread_p = 0; smem_AB_thread_p < SmemShapeK; smem_AB_thread_p++) {
        // register load
#pragma unroll
        for (int a = 0; a < ThreadShapeM; a++) {
          reg_A[a] = smem_A[smem_AB_thread_p][smem_A_thread_i + a];
        }

        // register load
#pragma unroll
        for (int b = 0; b < ThreadShapeN; b++) {
          reg_B[b] = smem_B[smem_AB_thread_p][smem_B_thread_j + b];
        }

        // rank-1 update to acc registers
#pragma unroll
        for (int b = 0; b < ThreadShapeN; b++) {
#pragma unroll
          for (int a = 0; a < ThreadShapeM; a++) {
            acc[b][a] += reg_A[a] * reg_B[b];
          }
        }
      }
    }
    __syncthreads();
  }

  // store acc registers results to C
#pragma unroll 1
  for (int b = 0; b < ThreadShapeN; b++) {
#pragma unroll 1
    for (int a = 0; a < ThreadShapeM; a++) {
      if (thread_i + a < m && thread_j + b < n) {
        c[(thread_i + a) * 1 + (thread_j + b) * ldc] = acc[b][a];
      }
    }
  }
}

#define MATMUL_KERNEL_LAUNCH_THREAD_TILING(name, num_threads, cta_shape_m, cta_shape_n, smem_shape_k, warp_shape_m, warp_shape_n, thread_shape_m, thread_shape_n)                              \
  MATMUL_SIGNATURE(launch_##name##_##num_threads##t_cta##cta_shape_m##x##cta_shape_n##_smem##smem_shape_k##_warp##warp_shape_m##x##warp_shape_n##_thread##thread_shape_m##x##thread_shape_n) { \
    dim3 threads(num_threads);                                                                                                                                                                 \
    dim3 blocks(ceil_div<int64_t>(m, cta_shape_m), ceil_div<int64_t>(n, cta_shape_n));                                                                                                         \
    name<num_threads, cta_shape_m, cta_shape_n, smem_shape_k, warp_shape_m, warp_shape_n, thread_shape_m, thread_shape_n><<<                                                                   \
        blocks, threads, 0, nullptr>>>(m, n, k, a, lda, b, ldb, c, ldc);                                                                                                                       \
    CUDA_CHECK(cudaGetLastError());                                                                                                                                                            \
  }

MATMUL_KERNEL_LAUNCH_THREAD_TILING(matmul_kernel_warp_tiling, 256, 128, 128, 8, 128, 16, 8, 8);
MATMUL_KERNEL_LAUNCH_THREAD_TILING(matmul_kernel_warp_tiling, 256, 128, 128, 8, 64, 32, 8, 8);
MATMUL_KERNEL_LAUNCH_THREAD_TILING(matmul_kernel_warp_tiling, 256, 128, 128, 8, 32, 64, 8, 8);
MATMUL_KERNEL_LAUNCH_THREAD_TILING(matmul_kernel_warp_tiling, 256, 128, 128, 8, 16, 128, 8, 8);

MATMUL_DMODULE(m) {
  REGISTER(launch_matmul_kernel_warp_tiling_256t_cta128x128_smem8_warp128x16_thread8x8);
  REGISTER(launch_matmul_kernel_warp_tiling_256t_cta128x128_smem8_warp64x32_thread8x8);
  REGISTER(launch_matmul_kernel_warp_tiling_256t_cta128x128_smem8_warp32x64_thread8x8);
  REGISTER(launch_matmul_kernel_warp_tiling_256t_cta128x128_smem8_warp16x128_thread8x8);
}

}  // namespace column_major
