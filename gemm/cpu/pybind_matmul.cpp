#include "matmul.hpp"

#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"

#include <iostream>
#include <chrono>

namespace py = pybind11;

#define REGISTER(name)                                                                                                 \
  MATMUL_SIGNATURE(name);                                                                                              \
  m.def(                                                                                                               \
      #name,                                                                                                           \
      [&](py::array_t<float> a, py::array_t<float> b, py::array_t<float> c, int repeats = 1) {                         \
        ENFORCE(a.ndim() == 2, "a must be a matrix");                                                                  \
        ENFORCE(b.ndim() == 2, "b must be a matrix");                                                                  \
        ENFORCE(c.ndim() == 2, "c must be a matrix");                                                                  \
        int64_t m = c.shape(0);                                                                                        \
        int64_t n = c.shape(1);                                                                                        \
        int64_t k = a.shape(1);                                                                                        \
        ENFORCE(m == a.shape(0), "a should be of " + std::to_string(m) + "rows");                                      \
        ENFORCE(k == b.shape(0), "b should be of " + std::to_string(k) + "rows");                                      \
        ENFORCE(n == b.shape(1), "b should be of " + std::to_string(n) + "columns");                                   \
                                                                                                                       \
        const auto start = std::chrono::steady_clock::now();                                                           \
        for (int i = 0; i < repeats; i++) {                                                                            \
          name(n, m, k, b.data(), n, a.data(), k, c.mutable_data(), n);                                                \
        }                                                                                                              \
        const auto end = std::chrono::steady_clock::now();                                                             \
        return std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(end - start).count() / repeats;    \
      },                                                                                                               \
      py::arg("a"),                                                                                                    \
      py::arg("b"),                                                                                                    \
      py::arg("c"),                                                                                                    \
      py::kw_only(),                                                                                                   \
      py::arg("repeats")                                                                                               \
  );

namespace column_major {
PYBIND11_MODULE(matmul, m) {
  REGISTER(matmul_reference);

  REGISTER(matmul_IJP);
  REGISTER(matmul_JIP);

  REGISTER(matmul_IPJ);
  REGISTER(matmul_JPI);

  REGISTER(matmul_PJI);
  REGISTER(matmul_PIJ);

  REGISTER(matmul_tile_IJ_naive_4x4);
  REGISTER(matmul_tile_JI_naive_4x4);

  REGISTER(matmul_tile_IJ_avx_4x4);
  REGISTER(matmul_tile_JI_avx_4x4);


  REGISTER(matmul_tile_IJ_avx2_8x4);
  REGISTER(matmul_tile_JI_avx2_8x4);

  REGISTER(matmul_simple_pipeline_8x4);
  REGISTER(matmul_simple_pipeline_16x4);
  REGISTER(matmul_simple_pipeline_16x6);
  REGISTER(matmul_simple_pipeline_16x8);

  REGISTER(matmul_fiveloop_pipeline_JPI_96_96_96_16x6);
  REGISTER(matmul_fiveloop_pipeline_JPI_144_144_144_16x6);

  REGISTER(matmul_fiveloop_pipeline_copied_JPI_320_2016_60_16x6);
  REGISTER(matmul_fiveloop_pipeline_copied_JPI_144_96_144_16x6);

  REGISTER(matmul_fiveloop_pipeline_packed_JPI_144_96_144_16x6);
  REGISTER(matmul_fiveloop_pipeline_packed_JPI_320_2016_60_16x6);
}

}  // namespace column_major

#undef REGISTER
