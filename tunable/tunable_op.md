# TunableOp in ONNXRuntime (Work In Progress)

---
## Glossary

- Op: Kernel wrapped in unified calling interface
  - Kernel: An implementation of a specific algorithm. In CUDA/ROCm, a `__global__` function.
- TunableOp: A collection of Ops

---
## Motivation

Select a good implementation at runtime.

---
### You can implement MatMul as follows (extremely slow, don't do it in real world, CPU will be crying)
```cpp
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j)
      for (int p = 0; p < k; ++p)
        gamma(i, j) = gamma(i, j) + alpha(i, p) * beta(p, j);
```

---
### Or you can make a much faster MatMul (but still not fast enough)

```cpp
  for ( int j=0; j<n; j+=NB ){
    int jb = min( n-j, NB );    /* Size for "finge" block */
    for ( int i=0; i<m; i+=MB ){
      int ib = min( m-i, MB );    /* Size for "finge" block */
      for ( int p=0; p<k; p+=KB ){
        int pb = min( k-p, KB );    /* Size for "finge" block */
        Gemm_PJI(ib, jb, pb, &alpha(i, p), ldA, &beta(p, j), ldB,
		             &gamma(i, j), ldC );
      }
    }
  }

void Gemm_PJI(int m, int n, int k, float* A, int ldA, float* B, int ldB, float* C, int ldC) {
  for ( int p=0; p<k; p++ )
    for ( int j=0; j<n; j++ )
      for ( int i=0; i<m; i++ )
        gamma(i, j) += alpha(i, p) * beta(p, j);
}
```

## TunableOp Interanl as in Pseudocode

```py
def GemmImpl1(a: ndarray, b: ndarray, c: ndarray): ...
def GemmImpl2(a: ndarray, b: ndarray, c: ndarray): ...
...

class TunableGemm:
  def __init__():
    candidate[to_id(GemmImpl1)] = GemmImpl1
    candidate[to_id(GemmImpl2)] = GemmImpl2
    ...
  def __call__(a: ndarray, b: ndarray, c: ndarray):
    if not_done_find_fastest([a, b, c]):
      id = find_fastest(a, b, c)
      store_cache([a, b, c], id) # keep a mapping from [a, b, c] -> id
    impl = read_cache([a, b, c])
    return impl(a, b, c)
  def find_fastest(a: ndarray, b: ndarray, c: ndarray) -> Id: ...
```

---
## Observation - We need a unified calling interface

We are unable (automatically) to call it if the different `GemmImpl`s have different signature.

---
### Observation - We need a unified calling interface (cont.)

The `rocblas_sgemm` is

```c
rocblas_status rocblas_sgemm(rocblas_handle handle, rocblas_operation transA, rocblas_operation transB,
                             int m, int n, int k, const float *alpha, const float *A,
                             int lda, const float *B, int ldb, const float *beta, float *C int ldc)
```

---
### Observation - We need a unified calling interface (cont.)

The composable_kernel GEMM (or matmul)
```cpp
using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
using Nop = ck::tensor_operation::element_wise::PassThrough;

using CKDataType = // ... // The dtype of the blob
using DeviceGemm = ck::tensor_operation::device::DeviceGemm<
  ALayout, BLayout, Row, CKDataType, CKDataType, CKDataType, Nop, Nop, Nop>;
using InstanceFactory = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<DeviceGemm>;
for (auto&& impl : InstanceFactory::GetInstances()) {
  auto invoker = impl->MakeInvokerPointer();
  auto nop = Nop{};
  auto arg = impl->MakeArgumentPointer(a, b, c, m, n, k, lda, ldb, ldc, nop, nop, nop);
  invoker->Run(arg.get(), StreamConfig{stream});
}
```

---
### Observation - We need a unified calling interface (cont.)

We need an interface adaptor (a layer of indirection).
In C++, we do it with [type erasure](https://www.modernescpp.com/index.php/type-erasure).

```cpp
template <typename ParamsT>
class Op {
 public:
  template <typename T>
  explicit Op(T&& c) : callable_{std::make_unique<CallableImpl<T>>(std::forward<T>(c))} {}
  Status operator()(const ParamsT* param) { return (*callable_)(param); }

 private:
  struct ICallable {
    virtual ~ICallable() = default;
    virtual Status operator()(const ParamsT*) = 0;
  };

  template <typename T>
  struct CallableImpl : ICallable {
    explicit CallableImpl(T&& c) : c_{std::move(c)} {}
    Status operator()(const ParamsT* param) override { return c_(param); }

   private:
    T c_;
  };

  std::unique_ptr<ICallable> callable_;
};
```

---
### Observation - We need a unified calling interface (cont.)

Then we **say** an Op is something with interface `(params: ParamsT) -> Status`, a.k.a, a **callable** that takes as input an `ParamsT` object and returns `Status` object.

Then the adaptor can work with [function, functor and lambda](https://github.com/microsoft/onnxruntime/blob/c43ce64795df8d0284c8b612693ff237439f25d3/onnxruntime/test/framework/tunable_op_test.cc#L90-L207).

And since the wrapped object have same type signature, we can put them into a container to form a candidate set.


---
## Observation - We need a way to map params to an identifier

We want to keep a mapping from passed in `params` to be best kernel.

The impose a requirement on `ParamsT` that it must be hashable.

---
### Observation - We need a way to map params to an identifier (cont.)

But there is no way to automatically do it. We do it manually:

```cpp
template <typename T>
struct GemmParams : tunable::OpParams {
  std::string Signature() const override {
    return MakeString(BlasOpToString(opa), BlasOpToString(opb), "_", m, "_", n, "_", k);
  }
  // ...
}
```

That is, use `Signature()` to produce a string which is used to identify the performance characteristics.

---
## Pitfall - What if kernel use a buffer a both input and output?

Say GEMM, it supports $$C = \alpha A B + \beta C$$ C is used as both input and output when $\beta \ne 0$

$$
\begin{aligned}
C^{(1)} &= \alpha A B + \beta C^{(0)}  \\
C^{(2)} &= \alpha A B + \beta C^{(1)}  \\
&\quad\vdots \\
C^{(n)} &= \alpha A B + \beta C^{(n-1)}\\
\end{aligned}
$$

And we basically garbage filled the buffer during the tuning.

---
## Pitfall - What if kernel use a buffer a both input and output?

```cpp
template </*...*/>
class TunableOp {
  virtual const ParamsT* PreTuning(const ParamsT* params) {
    return params;
  }
  virtual void PostTuning(const ParamsT* /*params*/) {
    // Do nothing if we are not playing around with params
  }
}
```

User override this two methods to construct and destruct a proxy params from actual params.

Then the accumulated update will only affect the proxy params. And we will not garbage fill the actual buffer.

---
## What we get?

A runtime TunableOp,

We do not have very broad Op coverage at the moment.

---
## Programming for Correctness

> 面试官：“你简历上写着说你心算速度很快，那我问问你，13乘以19是多少？”
>
> 我脱口而出：“45！”
>
> 面试官：“这也差太远了吧。”
>
> 我：“你就说快不快吧！”

---
### Programming for Correctness (cont.)

- ONNXRuntime's OpTester only tests for call path, not for numerical correctness.
  - At least for OpKernel that wraps vendor libraries.

- We test for kernel implementation.
  - Cross validated with reference implementation.
  - Account for FP roundoff error and use `rtol` bound derives from papers.
  - Test cover real world shape.
  - Already found bugs in rocblas and composable_kernel.

---
### Programming for Correctness (cont.) - Binding to Python and Profiling and Testing from Python

Not much engineering here. We just manually binds all previously implemented `Op`s and `TunableOp`s to verify performance and correctness.

- Run the script with `python` to get microbenchmark result for the kernels.
```console
KERNEL_EXPLORER_BUILD_DIR=<...> python onnxruntime/python/tools/kernel_explorer/kernels/gemm_test.py
```

- Run the script with `pytest` to test correctness of all kernels.
```console
KERNEL_EXPLORER_BUILD_DIR=<...> pytest onnxruntime/python/tools/kernel_explorer/kernels/gemm_test.py
```

---
### Programming for Correctness (cont.) - OpTester

[OpTester](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/test/providers/provider_test_utils.cc),
the test driver is extremely convoluted. Basically:

> Combine all ingredients in a bowl, and stir until thoroughly mixed.
>
> <div style="text-align: right"> — a common instruction in recipes </div>

- It constructs EP in the innermost of the code and run (test) the `OpKernel` with the newly constructed EP
- We need the `OpKernel` of ROCm EP to run twice with tuning disabled and enabled
   - Added a way to easily achieve it in [PR 13378](https://github.com/microsoft/onnxruntime/pull/13378)

### Programming for Correctness (cont.) - Fearlessly Change Code

We will not fear to adding a new Op to candidate.
  - We will be correct if tests passed.
  - We won't be slower.
    - When tuning is disabled, we always fallback to the default one.
    - When tuning is enabled, we select the faster one.


## Open Question

- Offline Tuning
  -
  ```py
  raise NotImplementedError
  ```
- Tuning Speed
  - Run for fixed iterations, much slower due to long tail slow kernels.
- Production Runtime Environment
  - It might not be clean enough to obtain a stable tuning result due to background noisy.
