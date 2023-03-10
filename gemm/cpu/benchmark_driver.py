import os
import sys
if not os.path.exists(os.path.join(os.path.dirname(__file__), "../bazel-bin/cpu/matmul.so")):
  raise EnvironmentError("bazel build -c opt --config=linux '//cpu:matmul.so'")
sys.path.append(os.path.join(os.path.dirname(__file__), "../bazel-bin/cpu"))

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matmul import *


@dataclass
class stat:

  size: int
  milliseconds: float
  repeats: int

  num_ops: float = field(init=False)
  flops: float = field(init=False)
  gflops: float = field(init=False)

  # nanoseconds: float = field(init=False)
  # microseconds: float = field(init=False)
  # seconds: float = field(init=False)

  def __post_init__(self):
    self.nanoseconds = self.milliseconds * 1e6
    self.microseconds = self.milliseconds * 1e3
    self.seconds = self.milliseconds * 1e-3

    self.num_ops = 2 * (self.size**3)
    self.flops = self.num_ops / self.seconds
    self.gflops = self.flops * 1e-9

  def __str__(self):
    return "size: {0:<5d} time per iter(ms): {1:<8.2f} GFLOPS: {2:<8.2f} repeats: {3:<3d}".format(
        self.size, self.milliseconds, self.gflops, self.repeats)


def run_benchmark(f, size, *, benchmark_seconds=0.5, max_iterations=1000) -> stat:
  a = np.zeros((size, size), dtype=np.float32)
  b = np.zeros((size, size), dtype=np.float32)
  c = np.zeros((size, size), dtype=np.float32)
  probe_stat = stat(size, f(a, b, c, repeats=1), None)  # this is also warmup

  expected_iterations = int(np.ceil(benchmark_seconds / probe_stat.seconds))
  repeats = min(expected_iterations, max_iterations)
  return stat(size, f(a, b, c, repeats=repeats), repeats)


def benchmark(f, sizes, cached=False, cached_name=None):
  if cached_name is None:
    cached_name = "result." + f.__name__ + ".pkl"
  if cached and os.path.exists(cached_name):
    return pd.read_pickle(cached_name)

  df = pd.DataFrame([run_benchmark(f, size) for size in sizes]).set_index("size")
  df = pd.concat([df], keys=[f.__name__], axis=1)
  if cached:
    df.to_pickle(cached_name)
  return df


def benchmark_aggregate(*dataframes):
  df = pd.concat(dataframes, axis=1)
  df = df.swaplevel(0, 1, axis=1)
  df = df.sort_index()
  return df


def benchmark_plot(*dataframes, **kwargs):
  df = benchmark_aggregate(*dataframes)

  newfig = kwargs.get("newfig", True)
  if newfig:
    fig = plt.figure(figsize=(15, 7))
  plt.xlabel("Matrix Size")
  plt.ylabel("GFLOPS")

  gflops = df["gflops"]
  for col in gflops:
    s = gflops[col]
    line, = plt.plot(s.dropna(), marker="o", linestyle='-')
    line.set_label(col)

  plt.legend()
