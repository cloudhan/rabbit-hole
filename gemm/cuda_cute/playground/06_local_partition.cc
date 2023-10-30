#include <vector>

#include <cute/tensor.hpp>
#include <cute/layout.hpp>

using namespace cute;

int main() {
  auto layout = make_layout(make_shape(15, 15));
  std::vector<int> buffer(size(layout));
  auto tensor = make_tensor(buffer.data(), layout);
  for (size_t i = 0; i < size(tensor); i++) {
    tensor(i) = i;
  }
  print_tensor(tensor);
  std::cout << "\n";

  {
    print_tensor(local_partition(tensor, make_tile(5, 5), make_coord(0, 0)));
    std::cout << "\n";
  }

  {
    print_tensor(local_partition(tensor, make_tile(5, 5), make_coord(2, 3)));
    std::cout << "\n";
  }

  {
    auto div = tiled_divide(tensor.layout(), make_tile(5,5));
    print(div);
    print_layout(div(0, _, _));
    print_layout(flatten(div(_, 0, 0)));
  }

  {
    auto div = logical_divide(tensor.layout(), make_tile(5,5));
    print(div);
    print_layout(div(make_coord(0, _), make_coord(0, _)));
    print_layout(div(make_coord(_, 0), make_coord(_, 0)));
  }

  return 0;
}
