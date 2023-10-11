#include "cuda/matmul.cuh"

namespace column_major {

template <int M, int N>
using Acc = float[N][M];

template <int Size>
using Fragment = float[Size];

template <int Size>
__device__ void load_fragment(
    Fragment<Size>& frag_a,  // or frag_b,
    int A_thread_i,          // or B_thread_j,
    int m,                   // or n,
    const float* A_i_p,      // or B_p_j,
    int step
) {
#pragma unroll
  for (int i = 0; i < Size; i++, A_i_p += step) {
    frag_a[i] = A_thread_i + i >= m ? 0 : *A_i_p;
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
// Each thread process ThreadShapeM x ThreadShapeN of **collocated** data
// Each thread then load (ThreadShapeM + ThreadShapeN) of elements, and do ThreadShapeM * ThreadShapeN of FMAs.
template <int NumThreads, int CtaShapeM, int CtaShapeN, int ThreadShapeM, int ThreadShapeN>
MATMUL_KERNEL_SIGNATURE(matmul_kernel_thread_tiling_abstract) {
  static_assert((CtaShapeM * CtaShapeN / (ThreadShapeM * ThreadShapeN)) == NumThreads);
  static_assert(CtaShapeM % ThreadShapeM == 0 && CtaShapeN % ThreadShapeN == 0);

  // The indexing will getting more and more complex as we go, so we establish an convention for it:
  // [object]_[from_level]_<to_level>, for example C_warp_thread_i means index C tile at warp level to thread level
  // if object is omitted, the it defaults to C in global memory
  // if from_level is omitted, it defaults to the object's storage level
  // this allows us, for example, to write C_cta_i + C_cta_warp_i + C_warp_lane_i + C_lane_thread_i
  int thread_i = CtaShapeM * blockIdx.x + (threadIdx.x % (CtaShapeM / ThreadShapeM)) * ThreadShapeM;
  int thread_j = CtaShapeN * blockIdx.y + (threadIdx.x / (CtaShapeM / ThreadShapeM)) * ThreadShapeN;

  // i (or j) part of the index for A (or B) is the same as C
  const auto& A_thread_i = thread_i;
  const auto& B_thread_j = thread_j;

  if (thread_i < m && thread_j < n) {
    Acc<ThreadShapeM, ThreadShapeN> acc{};
    Fragment<ThreadShapeM> frag_a;  // a col register for rank-1 update
    Fragment<ThreadShapeN> frag_b;  // a row register for rank-1 update

    auto A_i_p = &a[A_thread_i * 1 + 0 * lda];  // A(i, p) where p = 0
    auto B_p_j = &b[0 * 1 + B_thread_j * ldb];  // B(p, j) where p = 0

    for (int p = 0; p < k; p++) {
      // register load, since we load directly from gmem, pass A_thread_i (or B_thread_j) for out of bound checking
      load_fragment<ThreadShapeM>(frag_a, A_thread_i, m, A_i_p, 1);    // load a col fragment from A, thus, step is 1
      load_fragment<ThreadShapeN>(frag_b, B_thread_j, n, B_p_j, ldb);  // load a row fragment from B, thus, step is ldb

      rank1_update<ThreadShapeM, ThreadShapeN>(acc, frag_a, frag_b);

      A_i_p += lda;  // advance to A(i, p+1)
      B_p_j += 1;    // advance to B(p+1, j)
    }

    acc_store(m, n, c, ldc, acc, thread_i, thread_j);
  }
}

#define MATMUL_KERNEL_LAUNCH(name, num_threads, cta_shape_m, cta_shape_n, thread_shape_m, thread_shape_n)                                          \
  MATMUL_SIGNATURE(launch_##name##_##num_threads##t_cta##cta_shape_m##x##cta_shape_n##_thread##thread_shape_m##x##thread_shape_n) {                \
    dim3 threads(num_threads);                                                                                                                     \
    dim3 blocks(ceil_div<int64_t>(m, cta_shape_m), ceil_div<int64_t>(n, cta_shape_n));                                                             \
    name<num_threads, cta_shape_m, cta_shape_n, thread_shape_m, thread_shape_n><<<blocks, threads, 0, nullptr>>>(m, n, k, a, lda, b, ldb, c, ldc); \
    CUDA_CHECK(cudaGetLastError());                                                                                                                \
  }

MATMUL_KERNEL_LAUNCH(matmul_kernel_thread_tiling_abstract, 64, 32, 32, 4, 4);
MATMUL_KERNEL_LAUNCH(matmul_kernel_thread_tiling_abstract, 64, 64, 32, 8, 4);
MATMUL_KERNEL_LAUNCH(matmul_kernel_thread_tiling_abstract, 64, 64, 64, 8, 8);
MATMUL_KERNEL_LAUNCH(matmul_kernel_thread_tiling_abstract, 128, 32, 32, 4, 2);
MATMUL_KERNEL_LAUNCH(matmul_kernel_thread_tiling_abstract, 128, 64, 32, 4, 4);
MATMUL_KERNEL_LAUNCH(matmul_kernel_thread_tiling_abstract, 256, 32, 32, 2, 2);
MATMUL_KERNEL_LAUNCH(matmul_kernel_thread_tiling_abstract, 256, 64, 32, 4, 2);
MATMUL_KERNEL_LAUNCH(matmul_kernel_thread_tiling_abstract, 256, 64, 64, 4, 4);
MATMUL_KERNEL_LAUNCH(matmul_kernel_thread_tiling_abstract, 256, 128, 64, 8, 4);
MATMUL_KERNEL_LAUNCH(matmul_kernel_thread_tiling_abstract, 256, 128, 128, 8, 8);

MATMUL_DMODULE(m) {
  REGISTER(launch_matmul_kernel_thread_tiling_abstract_64t_cta32x32_thread4x4);
  REGISTER(launch_matmul_kernel_thread_tiling_abstract_64t_cta64x32_thread8x4);
  REGISTER(launch_matmul_kernel_thread_tiling_abstract_64t_cta64x64_thread8x8);
  REGISTER(launch_matmul_kernel_thread_tiling_abstract_128t_cta32x32_thread4x2);
  REGISTER(launch_matmul_kernel_thread_tiling_abstract_128t_cta64x32_thread4x4);
  REGISTER(launch_matmul_kernel_thread_tiling_abstract_256t_cta32x32_thread2x2);
  REGISTER(launch_matmul_kernel_thread_tiling_abstract_256t_cta64x32_thread4x2);
  REGISTER(launch_matmul_kernel_thread_tiling_abstract_256t_cta64x64_thread4x4);
  REGISTER(launch_matmul_kernel_thread_tiling_abstract_256t_cta128x64_thread8x4);
  REGISTER(launch_matmul_kernel_thread_tiling_abstract_256t_cta128x128_thread8x8);
}

}  // namespace column_major
