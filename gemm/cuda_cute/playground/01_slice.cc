#include <cute/layout.hpp>

using namespace cute;

int main() {
  auto layout = make_layout(make_shape(make_tuple(2, 3), 5), make_stride(make_tuple(3, 6), 18));
  print_layout(layout);

  {
    auto coord = make_coord(1, _);
    std::cout << "slice at " << coord << " without offset\n\t";
    auto sliced = slice(coord, layout);
    for (int i = 0; i < size(sliced); i++) {
      std::cout << sliced(i) << " ";
    }
    std::cout << "\n\n";
  }

  {
    auto coord = make_coord(1, _);
    std::cout << "slice at " << coord << "\n\t";
    auto [sliced, offset] = slice_and_offset(coord, layout);
    for (int i = 0; i < size(sliced); i++) {
      std::cout << sliced(i) + offset << " ";
    }
    std::cout << "\n\n";
  }

  {
    auto coord = make_coord(_, 3);
    std::cout << "slice at " << coord << "\n\t";
    auto [sliced, offset] = slice_and_offset(coord, layout);
    for (int i = 0; i < size(sliced); i++) {
      std::cout << sliced(i) + offset << " ";
    }
    std::cout << "\n\n";
  }

  {
    auto coord = make_coord(make_coord(1, _), _);
    std::cout << "slice at " << coord << "\n\t";
    auto [sliced, offset] = slice_and_offset(coord, layout);
    for (int i = 0; i < size(sliced); i++) {
      std::cout << sliced(i) + offset << " ";
    }
    std::cout << "\n\n";
  }

  return 0;
}
