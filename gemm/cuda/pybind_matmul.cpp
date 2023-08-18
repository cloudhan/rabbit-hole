#include "cuda/matmul.cuh"

#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"

#include <iostream>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>

namespace py = pybind11;

#define REGISTER(name)                                                                         \
  m.def(                                                                                       \
      #name,                                                                                   \
      [&](py::array_t<float> a, py::array_t<float> b, py::array_t<float> c, int repeats = 1) { \
        ENFORCE(a.ndim() == 2, "a must be a matrix");                                          \
        ENFORCE(b.ndim() == 2, "b must be a matrix");                                          \
        ENFORCE(c.ndim() == 2, "c must be a matrix");                                          \
        int64_t m = c.shape(0);                                                                \
        int64_t n = c.shape(1);                                                                \
        int64_t k = a.shape(1);                                                                \
        ENFORCE(m == a.shape(0), "a should be of " + std::to_string(m) + "rows");              \
        ENFORCE(k == b.shape(0), "b should be of " + std::to_string(k) + "rows");              \
        ENFORCE(n == b.shape(1), "b should be of " + std::to_string(n) + "columns");           \
                                                                                               \
        thrust::device_vector<float> dev_a(a.data(), a.data() + m * k);                        \
        thrust::device_vector<float> dev_b(b.data(), b.data() + k * n);                        \
        thrust::device_vector<float> dev_c(c.data(), c.data() + m * n);                        \
                                                                                               \
        cudaStream_t default_stream = 0;                                                       \
        cudaEvent_t start_event;                                                               \
        cudaEvent_t end_event;                                                                 \
        CUDA_CHECK(cudaEventCreate(&start_event));                                             \
        CUDA_CHECK(cudaEventCreate(&end_event));                                               \
        CUDA_CHECK(cudaEventRecord(start_event, default_stream));                              \
        for (int i = 0; i < repeats; i++) {                                                    \
          name(                                                                                \
              n, m, k,                                                                         \
              thrust::raw_pointer_cast(dev_b.data()), n,                                       \
              thrust::raw_pointer_cast(dev_a.data()), k,                                       \
              thrust::raw_pointer_cast(dev_c.data()), n                                        \
          );                                                                                   \
        }                                                                                      \
        CUDA_CHECK(cudaEventRecord(end_event, default_stream));                                \
        CUDA_CHECK(cudaEventSynchronize(end_event));                                           \
        float duration_ms;                                                                     \
        CUDA_CHECK(cudaEventElapsedTime(&duration_ms, start_event, end_event));                \
        CUDA_CHECK(cudaMemcpy(                                                                 \
            c.mutable_data(),                                                                  \
            thrust::raw_pointer_cast(dev_c.data()),                                            \
            dev_c.size() * sizeof(float), cudaMemcpyDeviceToHost                               \
        ));                                                                                    \
        return duration_ms / repeats;                                                          \
      },                                                                                       \
      py::arg("a"),                                                                            \
      py::arg("b"),                                                                            \
      py::arg("c"),                                                                            \
      py::kw_only(),                                                                           \
      py::arg("repeats")                                                                       \
  );

namespace column_major {
// MSVC is not happy with function local forward decl
MATMUL_SIGNATURE(matmul_reference);
MATMUL_SIGNATURE(launch_matmul_kernel_naive_cta16x16);
MATMUL_SIGNATURE(launch_matmul_kernel_naive_cta16x32);
MATMUL_SIGNATURE(launch_matmul_kernel_naive_cta32x16);
MATMUL_SIGNATURE(launch_matmul_kernel_naive_cta32x32);

PYBIND11_MODULE(matmul, m) {
  REGISTER(matmul_reference);
  REGISTER(launch_matmul_kernel_naive_cta16x16);
  REGISTER(launch_matmul_kernel_naive_cta16x32);
  REGISTER(launch_matmul_kernel_naive_cta32x16);
  REGISTER(launch_matmul_kernel_naive_cta32x32);
}

}  // namespace column_major

#undef REGISTER
