#include "cuda/matmul.cuh"

namespace column_major {

// Laucnh a 1D CTA(threadblock)
// Each CTA process CtaShapeM x CtaShapeN tile of C
// CTA load SmemShapeM x SmemShapeK and SmemShapeK x SmemShapeN of A and B from global memory to shared memory
// Each thread process ThreadShapeM x ThreadShapeN of **collocated** data
// Each thread then load (ThreadShapeM + ThreadShapeN) of elements, and do ThreadShapeM * ThreadShapeN of FMAs.
template <int NumThreads, int CtaShapeM, int CtaShapeN, int SmemShapeK, int ThreadShapeM, int ThreadShapeN>
__launch_bounds__(NumThreads) MATMUL_KERNEL_SIGNATURE(matmul_kernel_register_pipelining) {
  constexpr const auto SmemShapeM = CtaShapeM;
  constexpr const auto SmemShapeN = CtaShapeN;
  static_assert((SmemShapeM * SmemShapeK) % NumThreads == 0 && (SmemShapeN * SmemShapeK) % NumThreads == 0);
  static_assert((CtaShapeM * CtaShapeN / (ThreadShapeM * ThreadShapeN)) == NumThreads);
  static_assert(CtaShapeM % ThreadShapeM == 0 && CtaShapeN % ThreadShapeN == 0);

  __shared__ float smem_A[SmemShapeK][SmemShapeM];
  __shared__ float smem_B[SmemShapeK][SmemShapeN];

  const int cta_i = CtaShapeM * blockIdx.x;
  const int cta_j = CtaShapeN * blockIdx.y;
  const int thread_i = cta_i + (threadIdx.x % (CtaShapeM / ThreadShapeM)) * ThreadShapeM;
  const int thread_j = cta_j + (threadIdx.x / (CtaShapeM / ThreadShapeM)) * ThreadShapeN;

  float acc[ThreadShapeN][ThreadShapeM]{};  // zeroing out

  float reg_A[2][ThreadShapeM];  // a column registers for rank-1 update
  float reg_B[2][ThreadShapeN];  // a row    registers for rank-1 update

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
      for (int smem_batch = 0; smem_batch < SmemShapeM * SmemShapeK / NumThreads; smem_batch++) {
        const auto smem_A_thread_i = threadIdx.x % SmemShapeM;
        const auto smem_A_thread_p = threadIdx.x / SmemShapeM + smem_batch * SmemABatchShapeK;
        const auto A_p = A_batchp + smem_batch * SmemABatchShapeK;
        smem_A[smem_A_thread_p][smem_A_thread_i] = A_i >= m || A_p >= k ? 0 : a[A_i * 1 + A_p * lda];
        // printf("A(%d,%d) -> smem_A(%d, %d) %d %d %d\n", A_i, A_p, smem_A_thread_i, smem_A_thread_p, smem_batch, A_i >= m, A_p >= k);
      }

      const int B_batchp = p + threadIdx.x % SmemBBatchShapeK;
      const int B_j = SmemShapeN * blockIdx.y + threadIdx.x / SmemBBatchShapeK;
#pragma unroll
      for (int smem_batch = 0; smem_batch < SmemShapeN * SmemShapeK / NumThreads; smem_batch++) {
        const auto smem_B_thread_p = threadIdx.x % SmemBBatchShapeK + smem_batch * SmemBBatchShapeK;
        const auto smem_B_thread_j = threadIdx.x / SmemBBatchShapeK;
        const auto B_p = B_batchp + smem_batch * SmemBBatchShapeK;
        smem_B[smem_B_thread_p][smem_B_thread_j] = B_p >= k || B_j >= n ? 0 : b[B_p * 1 + B_j * ldb];
        // printf("B(%d,%d) -> smem_B(%d, %d) %d %d %d\n", B_p, B_j, smem_B_thread_p, smem_B_thread_j, smem_batch, B_p >= k, B_j >= n);
      }
    }
    __syncthreads();

    // each thread then load from shared memory to register and perform the rank-1 update
    {
      static_assert(SmemShapeM % ThreadShapeM == 0);
      // register pre-load of first fragment of registers
      const auto smem_A_thread_i = threadIdx.x % (CtaShapeM / ThreadShapeM) * ThreadShapeM;
      const auto smem_B_thread_j = threadIdx.x / (CtaShapeM / ThreadShapeM) * ThreadShapeN;
      // clang-format off
#pragma unroll
      for (int a = 0; a < ThreadShapeM; a++) reg_A[0][a] = smem_A[0][smem_A_thread_i + a];
#pragma unroll
      for (int b = 0; b < ThreadShapeN; b++) reg_B[0][b] = smem_B[0][smem_B_thread_j + b];
      static_assert(SmemShapeN % ThreadShapeN == 0);
#pragma unroll
      for (int smem_AB_thread_p = 0; smem_AB_thread_p < SmemShapeK; smem_AB_thread_p++) {
        // register pre-load of next fragment of registers
        if (smem_AB_thread_p + 1 <= SmemShapeK) {
#pragma unroll
          for (int a = 0; a < ThreadShapeM; a++) reg_A[(smem_AB_thread_p + 1) % 2][a] = smem_A[smem_AB_thread_p + 1][smem_A_thread_i + a];
#pragma unroll
          for (int b = 0; b < ThreadShapeN; b++) reg_B[(smem_AB_thread_p + 1) % 2][b] = smem_B[smem_AB_thread_p + 1][smem_B_thread_j + b];
        }

        // rank-1 update to acc registers
#pragma unroll
        for (int b = 0; b < ThreadShapeN; b++) {
#pragma unroll
          for (int a = 0; a < ThreadShapeM; a++) {
            acc[b][a] += reg_A[smem_AB_thread_p %2][a] * reg_B[smem_AB_thread_p %2][b];
          }
        }
      }
      // clang-format on
    }
    __syncthreads();
  }

  // store acc registers results to C
#pragma unroll
  for (int b = 0; b < ThreadShapeN; b++) {
#pragma unroll
    for (int a = 0; a < ThreadShapeM; a++) {
      if (thread_i + a < m && thread_j + b < n) {
        c[(thread_i + a) * 1 + (thread_j + b) * ldc] = acc[b][a];
      }
    }
  }
}

#define MATMUL_KERNEL_LAUNCH_THREAD_TILING(name, num_threads, cta_shape_m, cta_shape_n, smem_shape_k, thread_shape_m, thread_shape_n)                    \
  MATMUL_SIGNATURE(launch_##name##_##num_threads##t_cta##cta_shape_m##x##cta_shape_n##_smem##smem_shape_k##_thread##thread_shape_m##x##thread_shape_n) { \
    dim3 threads(num_threads);                                                                                                                           \
    dim3 blocks(ceil_div<int64_t>(m, cta_shape_m), ceil_div<int64_t>(n, cta_shape_n));                                                                   \
    name<num_threads, cta_shape_m, cta_shape_n, smem_shape_k, thread_shape_m, thread_shape_n><<<                                                         \
        blocks, threads, 0, nullptr>>>(m, n, k, a, lda, b, ldb, c, ldc);                                                                                 \
    CUDA_CHECK(cudaGetLastError());                                                                                                                      \
  }

MATMUL_KERNEL_LAUNCH_THREAD_TILING(matmul_kernel_register_pipelining, 256, 128, 128, 4, 8, 8);
MATMUL_KERNEL_LAUNCH_THREAD_TILING(matmul_kernel_register_pipelining, 256, 128, 128, 8, 8, 8);
MATMUL_KERNEL_LAUNCH_THREAD_TILING(matmul_kernel_register_pipelining, 256, 128, 128, 16, 8, 8);
MATMUL_KERNEL_LAUNCH_THREAD_TILING(matmul_kernel_register_pipelining, 256, 128, 128, 24, 8, 8);
MATMUL_KERNEL_LAUNCH_THREAD_TILING(matmul_kernel_register_pipelining, 256, 128, 128, 32, 8, 8);
MATMUL_KERNEL_LAUNCH_THREAD_TILING(matmul_kernel_register_pipelining, 256, 128, 128, 40, 8, 8);
MATMUL_KERNEL_LAUNCH_THREAD_TILING(matmul_kernel_register_pipelining, 256, 128, 128, 48, 8, 8);

MATMUL_DMODULE(m) {
  REGISTER(launch_matmul_kernel_register_pipelining_256t_cta128x128_smem4_thread8x8);
  REGISTER(launch_matmul_kernel_register_pipelining_256t_cta128x128_smem8_thread8x8);
  REGISTER(launch_matmul_kernel_register_pipelining_256t_cta128x128_smem16_thread8x8);
  REGISTER(launch_matmul_kernel_register_pipelining_256t_cta128x128_smem24_thread8x8);
  REGISTER(launch_matmul_kernel_register_pipelining_256t_cta128x128_smem32_thread8x8);
  REGISTER(launch_matmul_kernel_register_pipelining_256t_cta128x128_smem40_thread8x8);
  REGISTER(launch_matmul_kernel_register_pipelining_256t_cta128x128_smem48_thread8x8);
}

}  // namespace column_major
