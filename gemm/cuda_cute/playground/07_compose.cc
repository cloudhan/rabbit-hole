#include <vector>

#include <cute/tensor.hpp>
#include <cute/layout.hpp>

using namespace cute;

int main() {
  constexpr const auto MemShapeN = 8;
  constexpr const auto MemShapeK = 3;
  constexpr const auto VecSize = 2;
  constexpr const auto NumThreads = MemShapeN * MemShapeK / VecSize;

  {  // ex1
    auto mem_layout = make_layout(make_shape(MemShapeK, MemShapeN));
    print_layout(mem_layout);
    // (3,8):(_1,3)
    //        0    1    2    3    4    5    6    7
    //     +----+----+----+----+----+----+----+----+
    //  0  |  0 |  3 |  6 |  9 | 12 | 15 | 18 | 21 |
    //     +----+----+----+----+----+----+----+----+
    //  1  |  1 |  4 |  7 | 10 | 13 | 16 | 19 | 22 |
    //     +----+----+----+----+----+----+----+----+
    //  2  |  2 |  5 |  8 | 11 | 14 | 17 | 20 | 23 |
    //     +----+----+----+----+----+----+----+----+

    std::vector<int> mem_buffer(size(mem_layout));
    auto tensor = make_tensor(mem_buffer.data(), mem_layout);

    // We want to assign each threads to process values at positions as follows (the tid), how to achive that?
    //     0    0    3    3    6    6    9    9
    //     1    1    4    4    7    7   10   10
    //     2    2    5    5    8    8   11   11

    // Lets first review the Layout semantics:
    //
    // - Shape:  unmap the input 1d logical id as logical coordinate
    // - Stride: map the logical coordinate as 1d id as output
    //
    // And this is why layout can be composed: input and output are both 1d id.
    //
    // But nothing is explicitly stated for how the "unmap" is achived and is often the source of confusion.
    // To "unmap", we need to define the speed of each mode, and it is implicitly defined as the left to be the fastest and the right to be the slowest.
    // That is, "unmap" is achived in column-majored style.
    //
    //      -----------------> N ----------------->
    //        0    1    2    3    4    5    6    7
    //     +====+====+----+----+----+----+----+----+    |
    //  0  || 0    3||  6 |  9 | 12 | 15 | 18 | 21 |    |
    //     +====+====+----+----+----+----+----+----+    v
    //  1  |  1 |  4 |  7 | 10 | 13 | 16 | 19 | 22 |    K
    //     +----+----+----+----+----+----+----+----+    |
    //  2  |  2 |  5 |  8 | 11 | 14 | 17 | 20 | 23 |    |
    //     +----+----+----+----+----+----+----+----+    v
    //
    // How to obtain the shape of the Thr part?
    // 1. The highlighted part is the first thread's value. The second move along K, that is the fastest mode is K. The shape will be in form of (K,N)
    // 2. Along K, no fancy, thus the value is just K
    // 3. Along N, each threads access 2 values, thus the value will be N/2
    // Thus the shape of the Thr is (K,N/2)
    //
    // For strides for the Thr, just read the value out, we get (1, 6)
    //
    // Val can also be just read out as (1,2):(0,3), and be simplified as 2:3
    // The simplification is rule 2 of the `coalesce` operation
    //
    auto ThrVal = make_layout(
        make_layout(make_shape(Int<MemShapeK>{}, Int<MemShapeN / VecSize>{}), make_stride(Int<1>{}, Int<MemShapeK * VecSize>{})),
        make_layout(make_shape(Int<VecSize>{}), make_stride(Int<MemShapeK>{}))
    );
    for (int i = 0; i < NumThreads; i++) {
      // fill the tensor each thread's slice with the thread id
      fill(tensor.compose(ThrVal)(i, _), i);
    }
    print_tensor(tensor);
    // raw_ptr_32b(0x5613ca3be2c0) o (3,8):(_1,3):
    //     0    0    3    3    6    6    9    9
    //     1    1    4    4    7    7   10   10
    //     2    2    5    5    8    8   11   11
  }

  print("======================================================================================================================\n");

  {  // ex2, lets transpose ex1
    auto mem_layout = make_layout(make_shape(MemShapeN, MemShapeK));
    print_layout(mem_layout);
    // (8,3):(_1,8)
    //     ----->  K ----->
    //        0    1    2
    //     +====+----+----+   |
    //  0  || 0||  8 | 16 |   |
    //     ++  ++----+----+   |
    //  1  || 1||  9 | 17 |   |
    //     +====+----+----+   |
    //  2  |  2 | 10 | 18 |   |
    //     +----+----+----+   v
    //  3  |  3 | 11 | 19 |
    //     +----+----+----+   N
    //  4  |  4 | 12 | 20 |
    //     +----+----+----+   |
    //  5  |  5 | 13 | 21 |   |
    //     +----+----+----+   |
    //  6  |  6 | 14 | 22 |   |
    //     +----+----+----+   |
    //  7  |  7 | 15 | 23 |   |
    //     +----+----+----+   v
    //
    // And we want to get access order as follows:
    //     0    1    2
    //     0    1    2
    //     3    4    5
    //     3    4    5
    //     6    7    8
    //     6    7    8
    //     9   10   11
    //     9   10   11
    //
    // Notice the fastest mode of remapped thread (output) is still K and the slowest is still N, so it remain in (K,N/2).
    // The strides can be again read out follow order of the speed as (8, 2)
    //
    // For the Val, (2,1):(1,0), and can be simplified as 2:1
    //
    // We can also make a conclusion that the Thr part is not affected by transposition the tensor that it is composed to.
    // This sometime is also very confusion, but totally make sense.
    //   1. Thr is only affected by the thread ordering, that is, it only describe the (re)mapping of the threads (tid -> new id).
    //   2. After being remapped, the tensor's layout is responsible for unmapping from new id to coord then map out as memory offset.
    // The confusion is arised when we think Thr is also responsible for the second part.
    std::vector<int> mem_buffer(size(mem_layout));
    auto tensor = make_tensor(mem_buffer.data(), mem_layout);

    auto ThrVal = make_layout(
        make_layout(make_shape(Int<MemShapeK>{}, Int<MemShapeN / VecSize>{}), make_stride(Int<MemShapeN>{}, Int<VecSize>{})),
        make_layout(make_shape(Int<VecSize>{}), make_stride(Int<1>{}))
    );
    for (int i = 0; i < NumThreads; i++) {
      fill(tensor.compose(ThrVal)(i, _), i);
    }
    print_tensor(tensor);
    // raw_ptr_32b(0x5613ca3be2c0) o (8,3):(_1,8):
    //     0    1    2
    //     0    1    2
    //     3    4    5
    //     3    4    5
    //     6    7    8
    //     6    7    8
    //     9   10   11
    //     9   10   11
  }

  return 0;
}
