#pragma once

#include <algorithm>
#include <vector>
#include <cstdint>

namespace column_major {
namespace packers {

template <int H, int W, typename DT = float>
struct direct_copy {
  direct_copy() {
    this->buffer.resize(H * W);
  }

  void load(const DT* data, int64_t rs, int64_t cs, int64_t rows, int64_t cols) {
    auto ptr = this->data();
    int j = 0;
    for (; j < cols; j++) {
      auto src = data;
      int i = 0;
      for (; i < rows; i++) {
        *ptr++ = *src;
        src += rs;
      }
      for (; i < H; i++) {
        *ptr++ = 0;
      }
      data += cs;
    }
    for (; j < W; j++) {
      for (int i = 0; i < H; i++) {
        *ptr++ = 0;
      }
    }
  }

  DT* data() {
    return buffer.data();
  }

  int64_t stride() const {
    return H;
  }

  std::vector<DT> buffer;
};

}  // namespace packers
}  // namespace column_major
