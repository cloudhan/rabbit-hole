#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <thrust/device_vector.h>

#include "cpu/matmul.hpp"
#include "cuda/pybind_matmul.hpp"

#define CUDA_CHECK(expr)                                                                               \
  do {                                                                                                 \
    cudaError_t err = (expr);                                                                          \
    if (err != cudaSuccess) {                                                                          \
      fprintf(stderr, "CUDA Error on %s:%d\n", __FILE__, __LINE__);                                    \
      fprintf(stderr, "CUDA Error Code  : %d\n     Error String: %s\n", err, cudaGetErrorString(err)); \
      exit(err);                                                                                       \
    }                                                                                                  \
  } while (0)

template <typename T, typename E = std::enable_if<std::is_integral_v<T>>>
T ceil_div(T a, T b) {
  return (a - 1) / b + 1;
}

#define MATMUL_KERNEL_SIGNATURE(name)           \
  __global__ void name(                         \
      int64_t m, int64_t n, int64_t k,          \
      const float* __restrict__ a, int64_t lda, \
      const float* __restrict__ b, int64_t ldb, \
      float* __restrict__ c, int64_t ldc        \
  )

#define REGISTER(matmul_func)                                                        \
  m.def(                                                                             \
      #matmul_func,                                                                  \
      [&](pybind11::array_t<float> a,                                                \
          pybind11::array_t<float> b,                                                \
          pybind11::array_t<float> c,                                                \
          int repeats = 1,                                                           \
          int warmup = 0) {                                                          \
        ENFORCE(a.ndim() == 2, "a must be a matrix");                                \
        ENFORCE(b.ndim() == 2, "b must be a matrix");                                \
        ENFORCE(c.ndim() == 2, "c must be a matrix");                                \
        int64_t m = c.shape(0);                                                      \
        int64_t n = c.shape(1);                                                      \
        int64_t k = a.shape(1);                                                      \
        ENFORCE(m == a.shape(0), "a should be of " + std::to_string(m) + "rows");    \
        ENFORCE(k == b.shape(0), "b should be of " + std::to_string(k) + "rows");    \
        ENFORCE(n == b.shape(1), "b should be of " + std::to_string(n) + "columns"); \
                                                                                     \
        thrust::device_vector<float> dev_a(a.data(), a.data() + m * k);              \
        thrust::device_vector<float> dev_b(b.data(), b.data() + k * n);              \
        thrust::device_vector<float> dev_c(c.data(), c.data() + m * n);              \
                                                                                     \
        cudaStream_t default_stream = 0;                                             \
        cudaEvent_t start_event;                                                     \
        cudaEvent_t end_event;                                                       \
        CUDA_CHECK(cudaEventCreate(&start_event));                                   \
        CUDA_CHECK(cudaEventCreate(&end_event));                                     \
        for (int i = 0; i < warmup; i++) {                                           \
          matmul_func(                                                               \
              n, m, k,                                                               \
              thrust::raw_pointer_cast(dev_b.data()), n,                             \
              thrust::raw_pointer_cast(dev_a.data()), k,                             \
              thrust::raw_pointer_cast(dev_c.data()), n                              \
          );                                                                         \
        }                                                                            \
        CUDA_CHECK(cudaEventRecord(start_event, default_stream));                    \
        for (int i = 0; i < repeats; i++) {                                          \
          matmul_func(                                                               \
              n, m, k,                                                               \
              thrust::raw_pointer_cast(dev_b.data()), n,                             \
              thrust::raw_pointer_cast(dev_a.data()), k,                             \
              thrust::raw_pointer_cast(dev_c.data()), n                              \
          );                                                                         \
        }                                                                            \
        CUDA_CHECK(cudaEventRecord(end_event, default_stream));                      \
        CUDA_CHECK(cudaEventSynchronize(end_event));                                 \
        float duration_ms;                                                           \
        CUDA_CHECK(cudaEventElapsedTime(&duration_ms, start_event, end_event));      \
        CUDA_CHECK(cudaMemcpy(                                                       \
            c.mutable_data(),                                                        \
            thrust::raw_pointer_cast(dev_c.data()),                                  \
            dev_c.size() * sizeof(float), cudaMemcpyDeviceToHost                     \
        ));                                                                          \
        return duration_ms / repeats;                                                \
      },                                                                             \
      pybind11::arg("a"),                                                            \
      pybind11::arg("b"),                                                            \
      pybind11::arg("c"),                                                            \
      pybind11::kw_only(),                                                           \
      pybind11::arg("repeats") = 1,                                                  \
      pybind11::arg("warmup") = 0                                                    \
  );
