#include <vector>

#include <cute/tensor.hpp>
#include <cute/layout.hpp>

using namespace cute;

int main() {
  constexpr const auto MemShapeN = 8;
  constexpr const auto MemShapeK = 3;
  constexpr const auto TileN = 3;
  constexpr const auto TileK = 2;

  {  // ex3, pad the memory to round up the shape
    constexpr const auto MemShapeNRoundUp = ceil_div(MemShapeN, TileN) * TileN;
    constexpr const auto MemShapeKRoundUp = ceil_div(MemShapeK, TileK) * TileK;
    constexpr const auto NumThreads = MemShapeNRoundUp * MemShapeKRoundUp / (TileN * TileK);

    auto layout = make_layout(make_shape(MemShapeKRoundUp, MemShapeNRoundUp));
    print_layout(layout);
    // (4,9):(_1,4)
    //        0    1    2    3    4    5    6    7    8
    //     +----+----+----+----+----+----+----+----+----+
    //  0  |  0 |  4 |  8 | 12 | 16 | 20 | 24 | 28 | 32 |
    //     +----+----+----+----+----+----+----+----+----+
    //  1  |  1 |  5 |  9 | 13 | 17 | 21 | 25 | 29 | 33 |
    //     +----+----+----+----+----+----+----+----+----+
    //  2  |  2 |  6 | 10 | 14 | 18 | 22 | 26 | 30 | 34 |
    //     +----+----+----+----+----+----+----+----+----+
    //  3  |  3 |  7 | 11 | 15 | 19 | 23 | 27 | 31 | 35 |
    //     +----+----+----+----+----+----+----+----+----+
    // We want to assign each threads to process values at positions as follows (the tid):
    // 0    0    0    1    1    1    2    2    2
    // 0    0    0    1    1    1    2    2    2
    // 3    3    3    4    4    4    5    5    5
    // 3    3    3    4    4    4    5    5    5
    auto buffer = std::vector<int>(cosize(layout));
    auto tensor = make_tensor(buffer.data(), layout);

    auto ThrVal = make_layout(
        make_layout(make_shape(MemShapeNRoundUp / TileN, MemShapeKRoundUp / TileK), make_stride(Int<TileN * MemShapeKRoundUp>{}, Int<TileK>{})),
        make_layout(make_shape(Int<TileK>{}, Int<TileN>{}), make_stride(_1{}, Int<MemShapeKRoundUp>{}))
    );

    for (int t = 0; t < NumThreads; t++) {
      for (int v = 0; v < TileN * TileK; v++) {
        tensor.compose(ThrVal)(t, v) = t;
      }
    }
    print_tensor(tensor);
    // results are as expected
    // raw_ptr_32b(0x560c24bce2c0) o (4,9):(_1,4):
    //     0    0    0    1    1    1    2    2    2
    //     0    0    0    1    1    1    2    2    2
    //     3    3    3    4    4    4    5    5    5
    //     3    3    3    4    4    4    5    5    5
  }

  {  // ex4, memory non-padded, clobbered
    auto layout = make_layout(make_shape(MemShapeK, MemShapeN));
    print_layout(layout);
    // (3,8):(_1,3)
    //        0    1    2    3    4    5    6    7
    //     +----+----+----+----+----+----+----+----+
    //  0  |  0 |  3 |  6 |  9 | 12 | 15 | 18 | 21 |
    //     +----+----+----+----+----+----+----+----+
    //  1  |  1 |  4 |  7 | 10 | 13 | 16 | 19 | 22 |
    //     +----+----+----+----+----+----+----+----+
    //  2  |  2 |  5 |  8 | 11 | 14 | 17 | 20 | 23 |
    //     +----+----+----+----+----+----+----+----+
    // We want to assign each threads to process values at positions as follows (the tid):
    // 0    0    0    1    1    1    2    2
    // 0    0    0    1    1    1    2    2
    // 3    3    3    4    4    4    5    5
    auto buffer = std::vector<int>(cosize(layout));
    auto tensor = make_tensor(buffer.data(), layout);
    auto tcoord = make_identity_tensor(layout.shape());

    auto ThrVal = make_layout(
        make_layout(make_shape(ceil_div(MemShapeN, TileN), ceil_div(MemShapeK, TileK)), make_stride(Int<TileN * MemShapeK>{}, Int<TileK>{})),
        make_layout(make_shape(Int<TileK>{}, Int<TileN>{}), make_stride(_1{}, Int<MemShapeK>{}))
    );

    auto tensor_composed = tensor.compose(ThrVal);

    // Only fill **half** of the threads mapped positions
    for (int t = 0; t < size<0>(tensor_composed) / 2; t++) {
      for (int v = 0; v < TileN * TileK; v++) {
        if (elem_less(tcoord.compose(ThrVal)(t, v), make_coord(MemShapeK, MemShapeN))) {
          tensor_composed(t, v) = t;
        }
      }
    }
    print_tensor(tensor);
    // results are as expected
    // raw_ptr_32b(0x55afa4701360) o (3,8):(_1,3):
    //     0    0    0    1    1    1    2    2
    //     0    0    0    1    1    1    2    2
    //     0    0    0    0    0    0    0    0

    // Fill all positions
    clear(tensor);
    for (int t = 0; t < size<0>(tensor_composed); t++) {
      for (int v = 0; v < TileN * TileK; v++) {
        if (elem_less(tcoord.compose(ThrVal)(t, v), make_coord(MemShapeK, MemShapeN))) {
          tensor_composed(t, v) = t;
        }
      }
    }
    print_tensor(tensor);
    // results are clobbered
    // raw_ptr_32b(0x55afa4701360) o (3,8):(_1,3):
    //     0    3    3    3    4    4    4    5
    //     0    3    3    3    4    4    4    5
    //     0    0    0    0    0    0    0    0

    // Fill with copy_if
    clear(tensor);
    for (int t = 0; t < size<0>(tensor_composed); t++) {
      auto value = make_tensor_like<int>(get<1>(ThrVal));
      auto pred = make_tensor_like<bool>(get<1>(ThrVal));
      fill(value, t);
      clear(pred);
      for (int v = 0; v < TileN * TileK; v++) {
        pred(v) = elem_less(tcoord.compose(ThrVal)(t, v), make_coord(MemShapeK, MemShapeN));
      }
      copy_if(pred, value, tensor_composed(t, _));
    }
    print_tensor(tensor);
    // results are clobbered as previous
    // raw_ptr_32b(0x55afa4701360) o (3,8):(_1,3):
    //     0    3    3    3    4    4    4    5
    //     0    3    3    3    4    4    4    5
    //     0    0    0    0    0    0    0    0
  }

  {  // ex5, Tiler o TV_Layout
    constexpr const auto MemShapeNRoundUp = ceil_div(MemShapeN, TileN) * TileN;
    constexpr const auto MemShapeKRoundUp = ceil_div(MemShapeK, TileK) * TileK;
    constexpr const auto NumThreads = MemShapeNRoundUp * MemShapeKRoundUp / (TileN * TileK);

    auto layout = make_layout(make_shape(MemShapeK, MemShapeN));
    print_layout(layout);
    // (3,8):(_1,3)
    //        0    1    2    3    4    5    6    7
    //     +----+----+----+----+----+----+----+----+
    //  0  |  0 |  3 |  6 |  9 | 12 | 15 | 18 | 21 |
    //     +----+----+----+----+----+----+----+----+
    //  1  |  1 |  4 |  7 | 10 | 13 | 16 | 19 | 22 |
    //     +----+----+----+----+----+----+----+----+
    //  2  |  2 |  5 |  8 | 11 | 14 | 17 | 20 | 23 |
    //     +----+----+----+----+----+----+----+----+
    // We want to assign each threads to process values at positions as follows (the tid):
    // 0    0    0    1    1    1    2    2
    // 0    0    0    1    1    1    2    2
    // 3    3    3    4    4    4    5    5
    auto buffer = std::vector<int>(cosize(layout));
    auto tensor = make_tensor(buffer.data(), layout);
    auto tcoord = make_identity_tensor(tensor.shape());

    // NOTE: this is copied from ex3, not ex4!
    auto ThrVal = make_layout(
        make_layout(make_shape(MemShapeNRoundUp / TileN, MemShapeKRoundUp / TileK), make_stride(Int<TileN * MemShapeKRoundUp>{}, Int<TileK>{})),
        make_layout(make_shape(Int<TileK>{}, Int<TileN>{}), make_stride(_1{}, Int<MemShapeKRoundUp>{}))
    );

    // The division is for "virtually" padding the tensor, that is, we are manually over-approximate (pad with roundup)
    // the memory's layout. This is because compose intrinsically does not support over-approximate like local_tile or
    // *_divide. This is a clever trick for handling it. NOTE: we have cleaner way to achieve this, see ex6.
    auto tiler = make_tile(Int<MemShapeKRoundUp>{}, Int<MemShapeNRoundUp>{});
    auto tensor_composed = zipped_divide(tensor, tiler).compose(ThrVal, _);
    auto tcoord_composed = zipped_divide(tcoord, tiler).compose(ThrVal, _);

    print_layout(flatten(zipped_divide(tensor.layout(), tiler)(_, 0)));
    // (_4,_9):(_1,3)
    //        0    1    2    3    4    5    6    7    8
    //     +----+----+----+----+----+----+----+----+----+
    //  0  |  0 |  3 |  6 |  9 | 12 | 15 | 18 | 21 | 24 |
    //     +----+----+----+----+----+----+----+----+----+
    //  1  |  1 |  4 |  7 | 10 | 13 | 16 | 19 | 22 | 25 |
    //     +----+----+----+----+----+----+----+----+----+
    //  2  |  2 |  5 |  8 | 11 | 14 | 17 | 20 | 23 | 26 |
    //     +----+----+----+----+----+----+----+----+----+
    //  3  |  3 |  6 |  9 | 12 | 15 | 18 | 21 | 24 | 27 |
    //     +----+----+----+----+----+----+----+----+----+

    // Fill all positions
    clear(tensor);
    for (int i = 0; i < size<1>(tensor_composed); i++) {
      for (int t = 0; t < size<0, 0>(tensor_composed); t++) {
        for (int v = 0; v < size<0, 1>(tensor_composed); v++) {
          auto c = make_coord(make_coord(t, v), i);
          if (elem_less(tcoord_composed(c), shape(tcoord))) {
            tensor_composed(c) = t;
          }
        }
      }
    }
    print_tensor(tensor);
    // results are as expected this time
    // ptr[32b](0x55a985da8360) o (3,8):(_1,3):
    //     0    0    0    1    1    1    2    2
    //     0    0    0    1    1    1    2    2
    //     3    3    3    4    4    4    5    5
  }

  return 0;
}
