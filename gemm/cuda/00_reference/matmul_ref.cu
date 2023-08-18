#include "cpu/matmul.hpp"

#include <memory>
#include <type_traits>
#include "cublas_v2.h"

#define CUBLAS_CHECK(expr)                                                     \
  do {                                                                         \
    cublasStatus_t err = (expr);                                               \
    if (err != CUBLAS_STATUS_SUCCESS) {                                        \
      fprintf(stderr, "CUBLAS Error: %d at %s:%d\n", err, __FILE__, __LINE__); \
      exit(err);                                                               \
    }                                                                          \
  } while (0)

cublasHandle_t get_cublas_handle() {
  using handle_deref_t = std::remove_pointer_t<cublasHandle_t>;
  static std::shared_ptr<handle_deref_t> handle = []() {
    cublasHandle_t tmp;
    CUBLAS_CHECK(cublasCreate(&tmp));
    return std::shared_ptr<handle_deref_t>(tmp, [](cublasHandle_t p) { CUBLAS_CHECK(cublasDestroy(p)); });
  }();

  return handle.get();
}

namespace column_major {

MATMUL_SIGNATURE(matmul_reference) {
  float one = 1.0f;
  float zero = 0.0f;
  CUBLAS_CHECK(cublasSgemm(
      get_cublas_handle(),
      CUBLAS_OP_N, CUBLAS_OP_N,
      m, n, k,
      &one,
      a, lda,
      b, ldb,
      &zero,
      c, ldc
  ));
}

}  // namespace column_major
