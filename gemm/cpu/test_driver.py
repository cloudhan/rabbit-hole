import os
import sys
if not os.path.exists(os.path.join(os.path.dirname(__file__), "../bazel-bin/cpu/matmul.so")):
  raise EnvironmentError("bazel build -c opt --config=linux '//cpu:matmul.so'")
sys.path.append(os.path.join(os.path.dirname(__file__), "../bazel-bin/cpu"))

import numpy as np
import pytest
import matmul


def get_bound(dtype: str, a: np.ndarray, b: np.ndarray, c: np.ndarray, transa: bool, transb: bool):
    k = b.shape[1] if transb else b.shape[0]
    # The machine epsilon, unit roundoff, the smallest positive floating point number n such that the floating point
    # number that represents 1 + n is greater than 1.
    machine_eps = 2.0 ** -(24 if dtype == "float32" else 11)

    # The following implements error bound 5.7 in paper I. C. Ipsen and H. Zhou, “Probabilistic error analysis for
    # Inner Products,” SIAM Journal on Matrix Analysis and Applications, vol. 41, no. 4, pp. 1726–1741, 2020.
    # NOTE: the bound is not tight for float16 when k is large
    absa_mul_absb = np.abs(a.T if transa else a) @ np.abs(b.T if transb else b)
    coeff = np.max(absa_mul_absb / np.abs(c))
    gamma_2k = (1.0 + machine_eps) ** (2 * k) - 1.0
    bound_5_7 = coeff * np.sqrt(np.log(2 / 1e-10) * machine_eps * gamma_2k / 2)
    bound = bound_5_7

    return bound

def create_testcase(func, dtype="float32", sizes=range(32, 2048, 512)):
  @pytest.mark.parametrize("size", sizes)
  def test(size):
    a = np.random.rand(size, size).astype(np.float64)
    b = np.random.rand(size, size).astype(np.float64)
    ref_c = a @ b
    bound = get_bound(dtype, a, b, ref_c, False, False)

    a = a.astype(dtype)
    b = b.astype(dtype)
    my_c = np.zeros_like(ref_c).astype(dtype)
    func(a, b, my_c, repeats=1)


    print("max abs diff:", np.max(np.abs(ref_c) - np.abs(my_c)), "bound:", bound)
    np.testing.assert_allclose(my_c, ref_c, rtol=bound)

  return test

for name in filter(lambda name: name.startswith("matmul_"),  dir(matmul)):
  f = getattr(matmul, name)
  if name.endswith("16x6"):
    vars()["test_" + name] = create_testcase(f, sizes=range(48, 2048, 480))
  else:
    vars()["test_" + name] = create_testcase(f)
