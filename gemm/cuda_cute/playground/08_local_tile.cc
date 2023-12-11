#include <vector>

#include <cute/tensor.hpp>
#include <cute/layout.hpp>

using namespace cute;

int main() {
  auto layout = make_layout(make_shape(15, 15));
  std::vector<int> buffer(size(layout));
  auto tensor = make_tensor(buffer.data(), layout);
  for (int i = 0; i < size(tensor); i++) {
    tensor(i) = i;
  }
  print_tensor(tensor);
  std::cout << "\n";

  // This is what we have done in in 03_local_tile.cc, but make_tile itself can be must powerful!
  // make_tile(5, 5) is actually read as make_tile(make_layout(make_shape(5), make_stride(1)), make_layout(make_shape(5), make_stride(1)))
  print_tensor(local_tile(tensor, make_tile(5, 5), make_coord(0, 0)));
  print_tensor(local_tile(tensor, make_tile(make_layout(make_shape(_5{}), make_stride(_1{})), make_layout(make_shape(_5{}), make_stride(_1{}))), make_coord(0, 0)));
  // The second argument for local_tile is a Tiler.
  //
  // A Tiler can be many things:
  // A Layout. (X,(Y,Z)):(dx,(dy,dz))
  // A Tile<Layout...>. A tuple of Layouts. <L0, <L1, L2>>
  // A Shape. But this is just shorthand for a tuple of Layouts with stride-1.
  // (1) logical_product(Layout, Layout) and logical_divide(Layout, Layout) are the core operations. Once you understand what these do (and think of them in 1D), you understand all of it.
  // (2) Recursively apply (1) by-mode. For each corresponding mode of the Layout and the Tiler, apply (1). This is often the simplest way from 1D to ND.
  // (3) Essentially the same as (2), but call make_layout on the shape extent first to generate a default Layout.
  //
  // See https://github.com/NVIDIA/cutlass/issues/1148#issuecomment-1767641027
  //
  // So here we are actually apply the second rule

  // To be more powerful and expressive, for example, if in previous example, if 5x5 is the data to be processed by its corresponding thread.
  // Then we can easily extend to 2x2 of 5x5 (sub-)tiles that each thread will process.
  print_tensor(local_tile(
      tensor,
      make_tile(
          make_layout(make_shape(_5{}, _2{}), make_stride(_1{}, _10{})),
          make_layout(make_shape(_5{}, _2{}), make_stride(_1{}, _10{}))
      ),
      make_coord(0, 0)
  ));

  return 0;
}
