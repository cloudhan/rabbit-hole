#include <vector>

#include <cute/tensor.hpp>
#include <cute/layout.hpp>

using namespace cute;

template <typename T>
void print_identity_tensor(const T& t) {
  static_assert(rank_v<typename T::layout_type> == 2);
  print("identity tensor shape: "); print(t.shape()); print("\n");
  for (int i = 0; i < size<0>(t); i++) {
    for (int j = 0; j < size<1>(t); j++) {
      print(t(i,j)); print(" ");
    }
    print("\n");
  }
}

int main() {
  auto layout = make_layout(make_shape(6, 7));
  const auto pred = make_identity_tensor(layout.shape());
  print_identity_tensor(pred);

  std::cout << "\nlocal_tile 4x4 at (0,0):\n";
  print_identity_tensor(local_tile(pred, make_shape(4,4), make_coord(0,0)));

  std::cout << "\nlocal_tile 4x4 at (0,1):\n";
  print_identity_tensor(local_tile(pred, make_shape(4,4), make_coord(0,1)));

  std::cout << "\nlocal_tile 4x4 at (1,0):\n";
  print_identity_tensor(local_tile(pred, make_shape(4,4), make_coord(1,0)));

  std::cout << "\nlocal_tile 4x4 at (1,1):\n";
  auto local = local_tile(pred, make_shape(4,4), make_coord(1,1));
  print_identity_tensor(local);
  std::cout << "local_tile 4x4 at (1,1) and convert to bool:\n";
  for (int i = 0; i < size<0>(local); i++) {
    for (int j = 0; j < size<1>(local); j++) {
      print(elem_less(local(i, j), make_coord(6, 7))); print(" ");
    }
    print("\n");
  }
}
