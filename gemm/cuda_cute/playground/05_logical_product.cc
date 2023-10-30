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
  print_tensor(tensor(make_coord(make_coord(_, _), make_coord(0,0))));
  std::cout << "\n";

  print_tensor(tensor(make_coord(make_coord(_, _), make_coord(1, 0))));
  std::cout << "\n";

  print_tensor(tensor(make_coord(make_coord(_, _), make_coord(2, 1))));
  std::cout << "\n";

  print_tensor(tensor(make_coord(make_coord(1, 0), make_coord(_, _))));
  std::cout << "\n";

  return 0;
}
