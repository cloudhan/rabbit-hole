#include "cuda/pybind_matmul.hpp"
#include "cuda/matmul.cuh"

#include <cuda_runtime_api.h>

namespace py = pybind11;

static py::module::module_def matmul_module_def;

py::module get_matmul_module() {
  static pybind11::module_ m = []() {
    PYBIND11_ENSURE_INTERNALS_READY;
    auto tmp = pybind11::module_::create_extension_module(
        "matmul", "", &matmul_module_def
    );
    tmp.dec_ref();
    return tmp;
  }();
  return m;
}

PYBIND11_PLUGIN_IMPL(matmul) {
  PYBIND11_CHECK_PYTHON_VERSION;
  return get_matmul_module().ptr();
}
