#include <cute/layout.hpp>

using namespace cute;

int main() {
  print_layout(make_layout(make_shape(3, 3), make_stride(1, 3)));  // col-majored
  print_layout(make_layout(make_shape(3, 3), make_stride(3, 1)));  // row-majored

  print_layout(make_layout(make_shape(3, 3), make_stride(6, 2)));  // non-contiguous

  // hierarchical
  print_layout(make_layout(make_shape(make_tuple(2, 3), make_tuple(3, 2)), make_stride(make_tuple(1, 6), make_tuple(2, 18))));

  return 0;
}
