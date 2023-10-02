#include "cuda/matmul.cuh"

namespace column_major {

// Laucnh a 1D CTA(threadblock)
// Each CTA process CtaShapeM x CtaShapeN tile of C
// Each thread process ThreadShapeM x ThreadShapeN of **collocated** data
// Each thread then load (ThreadShapeM + ThreadShapeN) of elements, and do ThreadShapeM * ThreadShapeN of FMAs.
template <int NumThreads, int CtaShapeM, int CtaShapeN, int ThreadShapeM, int ThreadShapeN>
MATMUL_KERNEL_SIGNATURE(matmul_kernel_thread_tiling) {
  static_assert((CtaShapeM * CtaShapeN / (ThreadShapeM * ThreadShapeN)) == NumThreads);
  static_assert(CtaShapeM % ThreadShapeM == 0 && CtaShapeN % ThreadShapeN == 0);

  // The indexing will getting more and more complex as we go, so we establish an convention for it:
  // [from_level]_[object]_<to_level>, for example warp_C_thread_i means index C tile at warp level to thread level
  // if object is omitted, the it defaults to C in global memory from original matrix level
  int thread_i = CtaShapeM * blockIdx.x + (threadIdx.x % (CtaShapeM / ThreadShapeM)) * ThreadShapeM;
  int thread_j = CtaShapeN * blockIdx.y + (threadIdx.x / (CtaShapeM / ThreadShapeM)) * ThreadShapeN;

  if (thread_i < m && thread_j < n) {
    float acc[ThreadShapeN][ThreadShapeM]{};  // zeroing out

    float reg_A[ThreadShapeM];  // a column register for rank-1 update
    float reg_B[ThreadShapeN];  // a row    register for rank-1 update

    auto A_i_p = &a[thread_i * 1 + 0 * lda];  // A(i, p) where p = 0
    auto B_p_j = &b[0 * 1 + thread_j * ldb];  // B(p, j) where p = 0

    for (int p = 0; p < k; p++) {
      // register load
#pragma unroll
      for (int a = 0; a < ThreadShapeM; a++) {
        reg_A[a] = thread_i + a < m ? *(A_i_p + a) : 0;
      }

      // register load
#pragma unroll
      for (int b = 0; b < ThreadShapeN; b++) {
        reg_B[b] = thread_j + b < n ? *(B_p_j + b * ldb) : 0;
      }

      // rank-1 update to acc registers
#pragma unroll
      for (int b = 0; b < ThreadShapeN; b++) {
#pragma unroll
        for (int a = 0; a < ThreadShapeM; a++) {
          acc[b][a] += reg_A[a] * reg_B[b];
        }
      }

      A_i_p += lda;
      B_p_j += 1;
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
}

#define MATMUL_KERNEL_LAUNCH(name, num_threads, cta_shape_m, cta_shape_n, thread_shape_m, thread_shape_n)                            \
  MATMUL_SIGNATURE(launch_##name##_##num_threads##t_cta##cta_shape_m##x##cta_shape_n##_thread##thread_shape_m##x##thread_shape_n) {                \
    dim3 threads(num_threads);                                                                                                                     \
    dim3 blocks(ceil_div<int64_t>(m, cta_shape_m), ceil_div<int64_t>(n, cta_shape_n));                                                             \
    name<num_threads, cta_shape_m, cta_shape_n, thread_shape_m, thread_shape_n><<<blocks, threads, 0, nullptr>>>(m, n, k, a, lda, b, ldb, c, ldc); \
    CUDA_CHECK(cudaGetLastError());                                                                                                                \
  }

MATMUL_KERNEL_LAUNCH(matmul_kernel_thread_tiling, 64, 32, 32, 4, 4);
MATMUL_KERNEL_LAUNCH(matmul_kernel_thread_tiling, 64, 64, 32, 8, 4);
MATMUL_KERNEL_LAUNCH(matmul_kernel_thread_tiling, 64, 64, 64, 8, 8);
MATMUL_KERNEL_LAUNCH(matmul_kernel_thread_tiling, 128, 32, 32, 4, 2);
MATMUL_KERNEL_LAUNCH(matmul_kernel_thread_tiling, 128, 64, 32, 4, 4);
MATMUL_KERNEL_LAUNCH(matmul_kernel_thread_tiling, 256, 32, 32, 2, 2);
MATMUL_KERNEL_LAUNCH(matmul_kernel_thread_tiling, 256, 64, 32, 4, 2);
MATMUL_KERNEL_LAUNCH(matmul_kernel_thread_tiling, 256, 64, 64, 4, 4);
MATMUL_KERNEL_LAUNCH(matmul_kernel_thread_tiling, 256, 128, 64, 8, 4);
MATMUL_KERNEL_LAUNCH(matmul_kernel_thread_tiling, 256, 128, 128, 8, 8);

MATMUL_DMODULE(m) {
  REGISTER(launch_matmul_kernel_thread_tiling_64t_cta32x32_thread4x4);
  REGISTER(launch_matmul_kernel_thread_tiling_64t_cta64x32_thread8x4);
  REGISTER(launch_matmul_kernel_thread_tiling_64t_cta64x64_thread8x8);
  REGISTER(launch_matmul_kernel_thread_tiling_128t_cta32x32_thread4x2);
  REGISTER(launch_matmul_kernel_thread_tiling_128t_cta64x32_thread4x4);
  REGISTER(launch_matmul_kernel_thread_tiling_256t_cta32x32_thread2x2);
  REGISTER(launch_matmul_kernel_thread_tiling_256t_cta64x32_thread4x2);
  REGISTER(launch_matmul_kernel_thread_tiling_256t_cta64x64_thread4x4);
  REGISTER(launch_matmul_kernel_thread_tiling_256t_cta128x64_thread8x4);
  REGISTER(launch_matmul_kernel_thread_tiling_256t_cta128x128_thread8x8);
}

}  // namespace column_major
