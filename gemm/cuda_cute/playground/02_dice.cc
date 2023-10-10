#include <cute/layout.hpp>

using namespace cute;

int main() {
  auto layout = make_layout(make_shape(make_tuple(2, 3), 5), make_stride(make_tuple(3, 6), 18));
  print_layout(layout);

  {
    auto coord = make_coord(make_coord(2,_), 0);
    std::cout << "dice at " << coord << "\n\t";
    auto diced = dice(coord, layout);
    print(diced);
    std::cout << "\n\n";
  }

  {
    auto coord = make_coord(1, _);
    std::cout << "dice at " << coord << "\n\t";
    auto diced = dice(coord, layout);
    print(diced);
    std::cout << "\n\n";
  }

  return 0;
}
