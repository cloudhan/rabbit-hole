#pragma once

#include "cpu/matmul.hpp"

#define CUDA_CHECK(expr)                                                                               \
  do {                                                                                                 \
    cudaError_t err = (expr);                                                                          \
    if (err != cudaSuccess) {                                                                          \
      fprintf(stderr, "CUDA Error on %s:%d\n", __FILE__, __LINE__);                                    \
      fprintf(stderr, "CUDA Error Code  : %d\n     Error String: %s\n", err, cudaGetErrorString(err)); \
      exit(err);                                                                                       \
    }                                                                                                  \
  } while (0)

#define MATMUL_KERNEL_SIGNATURE(name)           \
  __global__ void name(                         \
      int64_t m, int64_t n, int64_t k,          \
      const float* __restrict__ a, int64_t lda, \
      const float* __restrict__ b, int64_t ldb, \
      float* __restrict__ c, int64_t ldc        \
  )
