import os
import sys
if os.name == "posix":
  if not os.path.exists(os.path.join(os.path.dirname(__file__), "../linux/bazel-bin/cuda/matmul.so")):
    raise EnvironmentError("bazel build -c opt --config=linux '//cuda:matmul.so'")
  sys.path.append(os.path.join(os.path.dirname(__file__), "../linux/bazel-bin/cuda"))
if os.name == "nt":
  if not os.path.exists(os.path.join(os.path.dirname(__file__), "../windows/bazel-bin/cuda/matmul.pyd")):
    raise EnvironmentError("bazel build -c opt --config=windows '//cuda:matmul.pyd'")
  sys.path.append(os.path.join(os.path.dirname(__file__), "../windows/bazel-bin/cuda"))
  if tuple(sys.version_info) > (3, 8):
    # fuck this shit, see https://stackoverflow.com/a/64472088/2091555
    # always use winmode=0 and preload the library. So that I don't suffer from the add_dll_directory chaos
    import ctypes
    matmul_lib = ctypes.CDLL(
        os.path.join(os.path.dirname(__file__), "../windows/bazel-bin/cuda/matmul.pyd"), winmode=0)

import numpy as np
import pytest
import matmul

print(matmul.__file__)


def get_bound(dtype: str, a: np.ndarray, b: np.ndarray, c: np.ndarray, transa: bool, transb: bool):
  k = b.shape[1] if transb else b.shape[0]
  # The machine epsilon, unit roundoff, the smallest positive floating point number n such that the floating point
  # number that represents 1 + n is greater than 1.
  machine_eps = 2.0**-(24 if dtype == "float32" else 11)

  # The following implements error bound 5.7 in paper I. C. Ipsen and H. Zhou, “Probabilistic error analysis for
  # Inner Products,” SIAM Journal on Matrix Analysis and Applications, vol. 41, no. 4, pp. 1726–1741, 2020.
  # NOTE: the bound is not tight for float16 when k is large
  absa_mul_absb = np.abs(a.T if transa else a) @ np.abs(b.T if transb else b)
  coeff = np.max(absa_mul_absb / np.abs(c))
  gamma_2k = (1.0 + machine_eps)**(2 * k) - 1.0
  bound_5_7 = coeff * np.sqrt(np.log(2 / 1e-10) * machine_eps * gamma_2k / 2)
  bound = bound_5_7

  return bound


def create_testcase(func, dtype="float32", sizes=(1, 11, 111, 1111) + tuple(range(512, 4096, 256))):

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


for name in filter(lambda name: name.startswith("matmul_") or name.startswith("launch_"), dir(matmul)):
  f = getattr(matmul, name)
  vars()["test_" + name] = create_testcase(f)

if __name__ == "__main__":
  available = list(filter(lambda name: name.startswith("matmul_") or name.startswith("launch_"), dir(matmul)))

  print("Available tests:")
  for name in available:
    print("   ", name)

  selected = sys.argv[1] if len(sys.argv) > 1 else available[0]
  f = getattr(matmul, selected)
  size = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
  print(f"Running {selected} for [{size}x{size}] * [{size}x{size}] ...")
  create_testcase(f)(size)
