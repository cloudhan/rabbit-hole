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
// Preload a block of shared memory
// In the loop
// CTA load SmemShapeM x SmemShapeK and SmemShapeK x SmemShapeN of A and B from global memory to shared memory
// Each thread process ThreadShapeM x ThreadShapeN of **collocated** data
// Each thread then load (ThreadShapeM + ThreadShapeN) of elements, and do ThreadShapeM * ThreadShapeN of FMAs.
template <int NumThreads, int CtaShapeM, int CtaShapeN, int SmemShapeK, int ThreadShapeM, int ThreadShapeN>
MATMUL_KERNEL_SIGNATURE(matmul_kernel_pipelining) {
  constexpr const auto SmemShapeM = CtaShapeM;
  constexpr const auto SmemShapeN = CtaShapeN;
  static_assert((SmemShapeM * SmemShapeK) % NumThreads == 0 && (SmemShapeN * SmemShapeK) % NumThreads == 0);
  static_assert((CtaShapeM * CtaShapeN / (ThreadShapeM * ThreadShapeN)) == NumThreads);
  static_assert(CtaShapeM % ThreadShapeM == 0 && CtaShapeN % ThreadShapeN == 0);

  __shared__ Array<SmemShapeK, SmemShapeM> smem_a[2];
  __shared__ Array<SmemShapeK, SmemShapeN> smem_b[2];
  Registers<SmemShapeM * SmemShapeK / NumThreads> staging_a;
  Registers<SmemShapeN * SmemShapeK / NumThreads> staging_b;

  const int cta_i = CtaShapeM * blockIdx.x;
  const int cta_j = CtaShapeN * blockIdx.y;
  const int thread_i = cta_i + (threadIdx.x % (CtaShapeM / ThreadShapeM)) * ThreadShapeM;
  const int thread_j = cta_j + (threadIdx.x / (CtaShapeM / ThreadShapeM)) * ThreadShapeN;

  Acc<ThreadShapeM, ThreadShapeN> acc{};
  Fragment<ThreadShapeM> frag_a[2];
  Fragment<ThreadShapeN> frag_b[2];

  // pre-load first block of A, B to smem_A, smem_B
  load_global_a<NumThreads, SmemShapeM, SmemShapeK>(staging_a, m, k, a, lda, 0);
  load_global_b<NumThreads, SmemShapeK, SmemShapeN>(staging_b, k, n, b, ldb, 0);
  for (int p = 0; p < k; p += SmemShapeK) {
    store_smem_a<NumThreads, SmemShapeM, SmemShapeK>(smem_a[(p / SmemShapeK) % 2], staging_a);
    store_smem_b<NumThreads, SmemShapeK, SmemShapeN>(smem_b[(p / SmemShapeK) % 2], staging_b);
    __syncthreads();
    if (p + SmemShapeK < k) {
      load_global_a<NumThreads, SmemShapeM, SmemShapeK>(staging_a, m, k, a, lda, p + SmemShapeK);
      load_global_b<NumThreads, SmemShapeK, SmemShapeN>(staging_b, k, n, b, ldb, p + SmemShapeK);
    }

    // each thread then load from shared memory to register and perform the rank-1 update
    const auto smem_A_thread_i = threadIdx.x % (CtaShapeM / ThreadShapeM) * ThreadShapeM;
    const auto smem_B_thread_j = threadIdx.x / (CtaShapeM / ThreadShapeM) * ThreadShapeN;

    // pre-load first fragment of registers
    load_fragment<ThreadShapeM, SmemShapeK, SmemShapeM>(frag_a[0], smem_a[(p / SmemShapeK) % 2], 0, smem_A_thread_i);
    load_fragment<ThreadShapeN, SmemShapeK, SmemShapeN>(frag_b[0], smem_b[(p / SmemShapeK) % 2], 0, smem_B_thread_j);
#pragma unroll
    for (int smem_AB_thread_p = 0; smem_AB_thread_p < SmemShapeK; smem_AB_thread_p++) {
      // pre-load next fragment of registers
      if (smem_AB_thread_p + 1 < SmemShapeK) {
        load_fragment<ThreadShapeM, SmemShapeK, SmemShapeM>(frag_a[(smem_AB_thread_p + 1) % 2], smem_a[(p / SmemShapeK) % 2], smem_AB_thread_p + 1, smem_A_thread_i);
        load_fragment<ThreadShapeN, SmemShapeK, SmemShapeN>(frag_b[(smem_AB_thread_p + 1) % 2], smem_b[(p / SmemShapeK) % 2], smem_AB_thread_p + 1, smem_B_thread_j);
      }

      rank1_update<ThreadShapeM, ThreadShapeN>(acc, frag_a[smem_AB_thread_p % 2], frag_b[smem_AB_thread_p % 2]);
    }
  }

  // store acc registers results to C
  acc_store(m, n, c, ldc, acc, thread_i, thread_j);
}

#define MATMUL_KERNEL_LAUNCH(name, num_threads, cta_shape_m, cta_shape_n, smem_shape_k, thread_shape_m, thread_shape_n)                                  \
  MATMUL_SIGNATURE(launch_##name##_##num_threads##t_cta##cta_shape_m##x##cta_shape_n##_smem##smem_shape_k##_thread##thread_shape_m##x##thread_shape_n) { \
    dim3 threads(num_threads);                                                                                                                           \
    dim3 blocks(ceil_div<int64_t>(m, cta_shape_m), ceil_div<int64_t>(n, cta_shape_n));                                                                   \
    name<num_threads, cta_shape_m, cta_shape_n, smem_shape_k, thread_shape_m, thread_shape_n><<<                                                         \
        blocks, threads, 0, nullptr>>>(m, n, k, a, lda, b, ldb, c, ldc);                                                                                 \
    CUDA_CHECK(cudaGetLastError());                                                                                                                      \
  }

MATMUL_KERNEL_LAUNCH(matmul_kernel_pipelining, 256, 128, 128, 4, 8, 8);
MATMUL_KERNEL_LAUNCH(matmul_kernel_pipelining, 256, 128, 128, 8, 8, 8);
MATMUL_KERNEL_LAUNCH(matmul_kernel_pipelining, 256, 128, 128, 16, 8, 8);
MATMUL_KERNEL_LAUNCH(matmul_kernel_pipelining, 256, 128, 128, 24, 8, 8);

MATMUL_DMODULE(m) {
  REGISTER(launch_matmul_kernel_pipelining_256t_cta128x128_smem4_thread8x8);
  REGISTER(launch_matmul_kernel_pipelining_256t_cta128x128_smem8_thread8x8);
  REGISTER(launch_matmul_kernel_pipelining_256t_cta128x128_smem16_thread8x8);
  REGISTER(launch_matmul_kernel_pipelining_256t_cta128x128_smem24_thread8x8);
}

}  // namespace column_major
