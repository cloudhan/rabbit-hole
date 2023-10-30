#include <vector>

#include <cute/tensor.hpp>
#include <cute/layout.hpp>

using namespace cute;

int main() {
  constexpr const auto MemShapeN = 8;
  constexpr const auto MemShapeK = 3;
  constexpr const auto TileN = 3;
  constexpr const auto TileK = 2;

  // See https://github.com/NVIDIA/cutlass/issues/1230

  {  // ex6
    auto layout = make_layout(make_shape(Int<MemShapeK>{}, Int<MemShapeN>{}));
    print_layout(layout);
    // (_3,_8):(_1,_3)
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
    //
    // In ex5, there is still a problem. We are designing the Thr part for the padded layout, and this additional
    // level of indirection makes it a little bit less intuitive.
    auto buffer = std::vector<int>(cosize(layout));
    auto tensor = make_tensor(buffer.data(), layout);
    auto tcoord = make_identity_tensor(tensor.shape());

    // https://github.com/NVIDIA/cutlass/issues/1230#issuecomment-1839809282
    constexpr const auto Thr = make_layout(
        make_shape(Int<ceil_div(MemShapeK, TileK)>{}, Int<ceil_div(MemShapeN, TileN)>{}),
        make_stride(Int<ceil_div(MemShapeN, TileN)>{}, _1{})
    );                                                                                                                // thread (k,n) -> thr_idx
    constexpr const auto Val = make_layout(make_shape(Int<TileK>{}, Int<TileN>{}), make_stride(_1{}, Int<TileK>{}));  // value (k,n) -> val_idx
    constexpr const auto KN_ThrVal = raked_product(Thr, Val);                                                         // (K,N) -> (thr_idx,val_idx), not in cute the output is 1d logical id, not hierarchical coord.
    constexpr const auto ThrVal_flat = right_inverse(KN_ThrVal);                                                      // (thr_idx,val_idx) -> (K,N), see note on the print_layout
    constexpr const auto ThrVal = ThrVal_flat.with_shape(size(Thr), size(Val));
    print_layout(Thr);
    // (_2,_3):(_3,_1)
    //       0   1   2
    //     +---+---+---+
    //  0  | 0 | 1 | 2 |
    //     +---+---+---+
    //  1  | 3 | 4 | 5 |
    //     +---+---+---+
    print_layout(Val);
    // (_2,_3):(_1,_2)
    //       0   1   2
    //     +---+---+---+
    //  0  | 0 | 2 | 4 |
    //     +---+---+---+
    //  1  | 1 | 3 | 5 |
    //     +---+---+---+
    print_layout(KN_ThrVal);
    //
    // right_inverse of layout L is defined as: L(right_inverse(L)(id)) == id,
    // KN_ThrVal maps from (K,N) -> (thr_idx,val_idx). right_inverse(KN_ThrVal) maps from (thr_idx,val_idx) -> (K,N)
    // due to its definition, the precondition domain(right_inverse(L)) == codomain(L)
    print_layout(ThrVal);
    // ((_3,_2),(_2,_3)):((_12,_2),(_1,_4))
    //        0    1    2    3    4    5
    //     +----+----+----+----+----+----+
    //  0  |  0 |  1 |  4 |  5 |  8 |  9 |
    //     +----+----+----+----+----+----+
    //  1  | 12 | 13 | 16 | 17 | 20 | 21 |
    //     +----+----+----+----+----+----+
    //  2  | 24 | 25 | 28 | 29 | 32 | 33 |
    //     +----+----+----+----+----+----+
    //  3  |  2 |  3 |  6 |  7 | 10 | 11 |
    //     +----+----+----+----+----+----+
    //  4  | 14 | 15 | 18 | 19 | 22 | 23 |
    //     +----+----+----+----+----+----+
    //  5  | 26 | 27 | 30 | 31 | 34 | 35 |
    //     +----+----+----+----+----+----+

    {
      // left_inverse might also work, as it is defined as left_inverse(L)(L(id)) == id, the mapping should be somehow
      // (thr_idx,val_idx) -> (K,N), but due to the definition, the precondition is not the same as previous,
      // it is codomain(left_inverse(L)) == domain(L)
      constexpr const auto left_inv = left_inverse(KN_ThrVal);  // (_3,_2,_2,_3):(_12,_2,_1,_4)
      // Result is flattened, but can be recovered as follows.
      // This is equivalent to previous ThrVal_flat.with_shape(size(Thr), size(Val))
      print_layout(make_layout(
          make_layout(get<0>(left_inv), get<1>(left_inv)),
          make_layout(get<2>(left_inv), get<3>(left_inv))
      ));
    }

    // NOTE for ex5, previously we use zipped_divide, but it creates an annoying Iter mode. compose will not.
    //
    // The division is for "virtually" padding the tensor, that is, we are manually over-approximate (pad with roundup)
    // the memory's layout. This is because compose intrinsically does not support over-approximate like local_tile or
    // *_divide. This is a clever trick for handling it.
    auto tiler = make_tile(Int<ceil_div(MemShapeK, TileK) * TileK>{}, Int<ceil_div(MemShapeN, TileN) * TileN>{});
    auto tensor_composed = tensor.compose(tiler).compose(ThrVal);
    auto tcoord_composed = tcoord.compose(tiler).compose(ThrVal);

    print_layout(tensor.compose(tiler).layout());
    // (_4,_9):(_1,_3)
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
    for (int t = 0; t < size<0>(tensor_composed); t++) {
      for (int v = 0; v < size<1>(tensor_composed); v++) {
        auto c = make_coord(t, v);
        if (elem_less(tcoord_composed(c), shape(tcoord))) {
          tensor_composed(c) = t;
        }
      }
    }
    print_tensor(tensor);
    // results are expected this time
    // ptr[32b](0x55a985da8360) o (3,8):(_1,3):
    //     0    0    0    1    1    1    2    2
    //     0    0    0    1    1    1    2    2
    //     3    3    3    4    4    4    5    5
  }

  {  // TiledCopy
    // This is an abstraction over our previous version, but zipped_divide is used instread of an intermediate compose.
    auto layout = make_layout(make_shape(Int<MemShapeK>{}, Int<MemShapeN>{}));
    print_layout(layout);
    // (_3,_8):(_1,_3)
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

    Layout thr_layout = Layout<Shape<_2, _3>, Stride<_3, _1>>{};  // ThrLayout: (m,n) -> thr_idx
    Layout val_layout = Layout<Shape<_2, _3>, Stride<_1, _2>>{};  // ValLayout: (m,n) -> val_idx

    auto tiled_copy = make_tiled_copy(Copy_Atom<DefaultCopy, int>{}, thr_layout, val_layout);

    clear(tensor);
    Tensor tensor_composed = tiled_copy.tidfrg_S(tensor);  // (Thr,Val,Iter)
    Tensor tcoord_composed = tiled_copy.tidfrg_S(tcoord);  // (Thr,Val,Iter)

    // Fill all valid positions
    for (int i = 0; i < size<2>(tensor_composed); ++i) {      // For each Iter (Tile)
      for (int t = 0; t < size<0>(tensor_composed); ++t) {    // For each Thread
        for (int v = 0; v < size<1>(tensor_composed); ++v) {  // For each Value
          auto c = make_coord(t, v, i);
          if (elem_less(tcoord_composed(c), shape(tcoord))) {
            tensor_composed(c) = t;
          }
        }
      }
    }
    print_tensor(tensor);
    // ptr[32b](0x5563eeea36c0) o (3,8):(_1,3):
    //     0    0    0    1    1    1    2    2
    //     0    0    0    1    1    1    2    2
    //     3    3    3    4    4    4    5    5
  }
  return 0;
}
