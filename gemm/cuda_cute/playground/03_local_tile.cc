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

  {
    print_tensor(local_tile(tensor, make_tile(5, 5), make_coord(0, 0)));
    std::cout << "\n";
  }

  {
    // NOTE: coordinate is for indexing the tile. It is NOT the coordinate for original tensor!
    print_tensor(local_tile(tensor, make_tile(5, 5), make_coord(1, 2)));
    std::cout << "\n";
  }

  {
    // assuming blockIdx.x == 0 and blockIdx.y == 0
    auto block_shape = make_shape(5, 5, 15);
    auto block_coord = make_coord(0, 0, _);
    std::cout << "local tile for C:";
    print_tensor(local_tile(tensor, block_shape, block_coord, make_step(1, 1, _)));  // diced logical coordinate is in form of (m, n)
    std::cout << "\n";

    std::cout << "local tile for A:";
    print_tensor(local_tile(tensor, block_shape, block_coord, make_step(1, _, 1)));  // diced logical coordinate is in form of (m, k)
    std::cout << "\n";

    std::cout << "local tile for B:";
    // print_tensor(local_tile(tensor, block_shape, block_coord, make_step(_, 1, 1))); // diced logical coordinate is in form of (n, k), which is not OK, expect (k, n)
    // fixing the logical coordinate order we feed into the tensor
    print_tensor(local_tile(tensor, make_shape(15, 5), make_coord(_, 0)));

    // fixing the logical coordinate order the tensor expect:
    auto layout_nk = make_layout(tensor.shape(), make_stride(15, 1));
    // change the stride of layout to make the index speed match with the physical array.
    // print_layout(layout_nk);
    auto tensor_nk = make_tensor(buffer.data(), layout_nk);
    print_tensor(local_tile(tensor_nk, block_shape, block_coord, make_step(_, 1, 1)));
    std::cout << "\n";
  }
}
