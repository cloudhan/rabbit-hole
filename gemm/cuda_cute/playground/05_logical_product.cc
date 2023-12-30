#include <vector>

#include <cute/tensor.hpp>
#include <cute/layout.hpp>

using namespace cute;

int main() {
  auto layout_inner = make_layout(make_shape(Int<2>{}, Int<2>{}));
  auto layout_outer = make_layout(make_shape(Int<3>{}, Int<3>{}));
  auto layout = logical_product(layout_inner, layout_outer);

  std::vector<int> buffer(size(layout));
  auto tensor = make_tensor(buffer.data(), layout);
  for (size_t i = 0; i < size(tensor); i++) {
    tensor(i) = i;
  }

  print_tensor(tensor);
  // ptr[32b](0x56391a931eb0) o ((_2,_2),(_3,_3)):((_1,_2),(_4,_12)):
  //     0    4    8   12   16   20   24   28   32
  //     1    5    9   13   17   21   25   29   33
  //     2    6   10   14   18   22   26   30   34
  //     3    7   11   15   19   23   27   31   35

  print_tensor(tensor(make_coord(make_coord(_, _), make_coord(0,0))));
  // ptr[32b](0x56391a931eb0) o (_2,_2):(_1,_2):
  //     0    2
  //     1    3

  print_tensor(tensor(make_coord(make_coord(_, _), make_coord(1, 0))));
  // ptr[32b](0x56391a931ec0) o (_2,_2):(_1,_2):
  //     4    6
  //     5    7

  print_tensor(tensor(make_coord(make_coord(_, _), make_coord(2, 1))));
  // ptr[32b](0x56391a931f00) o (_2,_2):(_1,_2):
  //    20   22
  //    21   23

  print_tensor(tensor(make_coord(make_coord(1, 0), make_coord(_, _))));
  // ptr[32b](0x56391a931eb4) o (_3,_3):(_4,_12):
  //     1   13   25
  //     5   17   29
  //     9   21   33

  return 0;
}
