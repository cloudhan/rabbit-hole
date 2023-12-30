#include <cute/tensor.hpp>

using namespace cute;

int main() {
  auto inner = make_layout(make_shape(_2{}, _3{}));
  auto tiler = make_layout(make_shape(3, 4));
  auto tiled = blocked_product(inner, tiler);   // (_x, _y) -> linear_idx
  auto naive = make_layout(make_shape(6, 12));  // (.x, .y) -> linear_idx, naively linearize blockIdx

  print_layout(naive);
  // c'mon, you don't need a print for this...

  print_layout(tiled);
  // ((_2,3),(_3,4)):((_1,_6),(_2,18))
  //        0    1    2    3    4    5    6    7    8    9   10   11
  //     +----+----+----+----+----+----+----+----+----+----+----+----+
  //  0  |  0 |  2 |  4 | 18 | 20 | 22 | 36 | 38 | 40 | 54 | 56 | 58 |
  //     +----+----+----+----+----+----+----+----+----+----+----+----+
  //  1  |  1 |  3 |  5 | 19 | 21 | 23 | 37 | 39 | 41 | 55 | 57 | 59 |
  //     +----+----+----+----+----+----+----+----+----+----+----+----+
  //  2  |  6 |  8 | 10 | 24 | 26 | 28 | 42 | 44 | 46 | 60 | 62 | 64 |
  //     +----+----+----+----+----+----+----+----+----+----+----+----+
  //  3  |  7 |  9 | 11 | 25 | 27 | 29 | 43 | 45 | 47 | 61 | 63 | 65 |
  //     +----+----+----+----+----+----+----+----+----+----+----+----+
  //  4  | 12 | 14 | 16 | 30 | 32 | 34 | 48 | 50 | 52 | 66 | 68 | 70 |
  //     +----+----+----+----+----+----+----+----+----+----+----+----+
  //  5  | 13 | 15 | 17 | 31 | 33 | 35 | 49 | 51 | 53 | 67 | 69 | 71 |
  //     +----+----+----+----+----+----+----+----+----+----+----+----+

  dim3 blockIdx;
  for (blockIdx.y = 0; blockIdx.y < size<1>(naive); blockIdx.y++) {
    for (blockIdx.x = 0; blockIdx.x < size<0>(naive); blockIdx.x++) {
      auto linear_idx = naive(blockIdx.x, blockIdx.y);
      // auto [blockIdx_x_tuple, blockIdx_y_tuple] = tiled[linear_idx];
      auto [blockIdx_x, blockIdx_xx, blockIdx_y, blockIdx_yy] = flatten(tiled)[linear_idx];
      blockIdx_x += blockIdx_xx * inner.shape<0>();
      blockIdx_y += blockIdx_yy * inner.shape<1>();
      std::cout << linear_idx << "\t(.x,.y)=(" << blockIdx.x << "," << blockIdx.y << ")\t(_x,_y)=(" << blockIdx_x << "," << blockIdx_y << ")\n";
      // 0       (.x,.y)=(0,0)   (_x,_y)=(0,0)
      // 1       (.x,.y)=(1,0)   (_x,_y)=(1,0)
      // 2       (.x,.y)=(2,0)   (_x,_y)=(0,1)
      // 3       (.x,.y)=(3,0)   (_x,_y)=(1,1)
      // 4       (.x,.y)=(4,0)   (_x,_y)=(0,2)
      // 5       (.x,.y)=(5,0)   (_x,_y)=(1,2)
      // 6       (.x,.y)=(0,1)   (_x,_y)=(2,0)
      // 7       (.x,.y)=(1,1)   (_x,_y)=(3,0)
      // 8       (.x,.y)=(2,1)   (_x,_y)=(2,1)
      // 9       (.x,.y)=(3,1)   (_x,_y)=(3,1)
      // 10      (.x,.y)=(4,1)   (_x,_y)=(2,2)
      // 11      (.x,.y)=(5,1)   (_x,_y)=(3,2)
      // 12      (.x,.y)=(0,2)   (_x,_y)=(4,0)
      // 13      (.x,.y)=(1,2)   (_x,_y)=(5,0)
      // ...
    }
  }

  auto coord = make_identity_tensor(shape(naive));
  for (blockIdx.y = 0; blockIdx.y < size<1>(naive); blockIdx.y++) {
    for (blockIdx.x = 0; blockIdx.x < size<0>(naive); blockIdx.x++) {
      auto linear_idx = naive(blockIdx.x, blockIdx.y);
      auto [blockIdx_x, blockIdx_y] = coord(tiled(linear_idx));
      std::cout << linear_idx << "\t(.x,.y)=(" << blockIdx.x << "," << blockIdx.y << ")\t(_x,_y)=(" << blockIdx_x << "," << blockIdx_y << ")\n";
    }
  }

  return 0;
}
