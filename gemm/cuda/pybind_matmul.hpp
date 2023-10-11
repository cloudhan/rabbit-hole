#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>

pybind11::module get_matmul_module();

// Decentralized/Distributed matmul module for pybind11
class matmulDModuleInit {
public:
  explicit matmulDModuleInit(void (*init_func)(pybind11::module module)) {
    pybind11::gil_scoped_acquire gil{};
    init_func(get_matmul_module());
  }
};

#define MATMUL_DMODULE_IMPL(unique_id, variable)                                                 \
  static void matmulDModuleInitFunc##unique_id(pybind11::module variable);                       \
  static const matmulDModuleInit matmulDModuleInit##unique_id{matmulDModuleInitFunc##unique_id}; \
  void matmulDModuleInitFunc##unique_id(pybind11::module variable)

#define MATMUL_DMODULE_IMPL_(unique_id, variable) MATMUL_DMODULE_IMPL(unique_id, variable)  // indirection to expand __COUNTER__
#define MATMUL_DMODULE(variable) MATMUL_DMODULE_IMPL_(__COUNTER__, variable)                // same as  PYBIND11_MODULE(matmul, variable)
