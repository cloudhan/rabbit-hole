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
    // Notice that in Thr part, we describe the threads' layout more directly, the shape is now transposed to be aligned
    // with the tensor's (memory) layout (compare to ex3, ex4, ex5). The tensor is K-by-N shape, the Thr is also in
    // K-by-N form now.
    //
    // And why is that?
    //   - In previous examples, ThrVal is a concatenated layout that maps from (thr_idx,val_idx) -> (K,N)
    //   - In current setting, the Thr is a layout that maps from (tid) -> (tid), to be more precise, it is actually
    //     (k,n) -> (tid), where size((k,n)) == num_threads. (k,n) does not necessarily cover every elements of the
    //     tensor, tho.
    //
    // Thr, Val and ThrVal are related as follows, and the computing is wrapped in TiledCopy, TiledCopy(Thr, Val) produces ThrVal.
    constexpr const auto Thr = make_layout(
        make_shape(Int<ceil_div(MemShapeK, TileK)>{}, Int<ceil_div(MemShapeN, TileN)>{}),
        make_stride(Int<ceil_div(MemShapeN, TileN)>{}, _1{})
    );                                                                                                                // thread (k1,n1) -> thr_idx
    constexpr const auto Val = make_layout(make_shape(Int<TileK>{}, Int<TileN>{}), make_stride(_1{}, Int<TileK>{}));  // value  (k2,n2) -> val_idx
    constexpr const auto KN_ThrVal = raked_product(Thr, Val);                                                         // (K,N) -> (thr_idx,val_idx), note in cute the output is 1d logical id, not hierarchical coord.
    constexpr const auto ThrVal_flat = right_inverse(KN_ThrVal);                                                      // (thr_idx,val_idx) -> (K,N), see note on the print_layout and left_inverse
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
    // The compose(tiler) is for "virtually" padding the tensor.
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
    // results are as expected this time
    // ptr[32b](0x55a985da8360) o (3,8):(_1,3):
    //     0    0    0    1    1    1    2    2
    //     0    0    0    1    1    1    2    2
    //     3    3    3    4    4    4    5    5
  }

  {  // TiledCopy
    // This is an abstraction over our previous version, but zipped_divide is used instead of an intermediate compose.
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

    // To make the TiledCopy algorithm work correctly, there is an implicit contract between the implementer and the users:
    //
    //   We must follow the same shape order as the tensor we want to work on to describe Thr and Val.
    //
    // Here we use the (K,N) order.
    Layout thr_layout = Layout<Shape<_2, _3>, Stride<_3, _1>>{};  // ThrLayout: (k1,n1) -> thr_idx
    Layout val_layout = Layout<Shape<_2, _3>, Stride<_1, _2>>{};  // ValLayout: (k2,n2) -> val_idx

    auto tiled_copy = make_tiled_copy(Copy_Atom<DefaultCopy, int>{}, thr_layout, val_layout);
    print(tiled_copy);
    // TiledCopy
    //   Tiler_MN:       (_4,_9)
    //   TiledLayout_TV: ((_3,_2),(_2,_3)):((_12,_2),(_1,_4))
    // Copy_Atom
    //   ThrID:        _1:_0
    //   ValLayoutSrc: (_1,_1):(_0,_0)
    //   ValLayoutDst: (_1,_1):(_0,_0)
    //   ValLayoutRef: (_1,_1):(_0,_0)
    //   ValueType:    32b

    clear(tensor);
    Tensor tensor_composed = tiled_copy.tidfrg_D(tensor);  // (Thr,Val,Iter)
    Tensor tcoord_composed = tiled_copy.tidfrg_D(tcoord);  // (Thr,Val,Iter)

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

    clear(tensor);
    for (int threadIdx_x = 0; threadIdx_x < size<0>(tensor_composed); ++threadIdx_x) {  // For each Thread
      // Think of 👆 as kernel launch, then 👇 is in the kernel.
      // In previous tiled_copy, the copy view work on tile level (thr,val,iter), we need to further slice the view to
      // get the view that a thread will work on. the tiled_copy.get_thread_slice(thr_idx) will create the view for us.
      auto thr_copy = tiled_copy.get_thread_slice(threadIdx_x);
      auto tv = thr_copy.partition_D(tensor);
      auto tc = thr_copy.partition_D(tcoord);
      for (int i = 0; i < size(tv); i++) {  // For values in the view of the thread's
        if (elem_less(tc(i), shape(tcoord))) {
          tv(i) = threadIdx_x;
        }
      }
    }
    print_tensor(tensor);
    // ptr[32b](0x55d9cae0b2c0) o (_3,_8):(_1,_3):
    //     0    0    0    1    1    1    2    2
    //     0    0    0    1    1    1    2    2
    //     3    3    3    4    4    4    5    5

    // Final note of the intermediate compose vs intermediate zipped_divide, which one is better?
    //
    // In the example, compose is clearly a winner.
    //
    // In this example, our (2,3)-shaped thread block (the Thr) fully covered our tensor in 1 go, each thread operates
    // on (2,3)-shaped tile (the Val).
    //
    // In practice, zipped_divede is more general.
    //
    // What if we have a little thread block but a huge tensor? Enlarge Val might work but may not be good for
    // performance. In this case, zipped_divide allows us to design a ThrVal for best performance and do the whole
    // work in multiple batches, by batch, you just iterate on the Iter mode.
  }

  return 0;
}
