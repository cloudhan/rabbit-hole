#pragma once

#include <algorithm>
#include <vector>
#include <cstdint>

namespace column_major {
namespace packers {

template <int MC, int KC, int MR, typename DT = float>
struct a_packer {
  a_packer() {
    this->buffer.resize(MC * KC);
  }

  void load(const DT* data, int64_t rs, int64_t cs, int64_t rows, int64_t cols) {
    pack_row_panels(data, rs, cs, rows, cols);
  }

  void pack_row_panels(const DT* data, int64_t rs, int64_t cs, int64_t rows, int64_t cols) {
    auto out = buffer.data();
    for (int i = 0; i < rows; i += MR) {
      out = pack_single_row_panel(&data[rs * i + cs * 0], rs, cs, std::min<int64_t>(MR, rows - i), cols, out);
    }
  }

  inline DT* pack_single_row_panel(const DT* data, int64_t rs, int64_t cs, int64_t rows, int64_t cols, DT* out) {
    if (rows == MR) {                   // pack a full panel
      for (int p = 0; p < cols; p++) {  // panel_width
        for (int i = 0; i < MR; i++) {  // panel_height
          *(out++) = data[rs * i + cs * p];
        }
      }
    } else {
      // similar to the if branch, but fill pack buffer where i \in [rows, MR) with 0
      for (int p = 0; p < cols; p++) {
        int i = 0;
        for (; i < rows; i++) {
          *(out++) = data[rs * i + cs * p];
        }
        for (; i < MR; i++) {
          *(out++) = 0;
        }
      }
    }
    return out;
  }

  DT* data() {
    return buffer.data();
  }

  std::vector<DT> buffer;
};

template <int KC, int NC, int NR, typename DT = float>
struct b_packer {
  b_packer() {
    this->buffer.resize(KC * NC);
  }

  void load(const DT* data, int64_t rs, int64_t cs, int64_t rows, int64_t cols) {
    pack_col_panels(data, rs, cs, rows, cols);
  }

  void pack_col_panels(const DT* data, int64_t rs, int64_t cs, int64_t rows, int64_t cols) {
    auto out = buffer.data();
    for (int j = 0; j < cols; j += NR) {
      out = pack_single_col_panel(&data[rs * 0 + cs * j], rs, cs, rows, std::min<int64_t>(NR, cols - j), out);
    }
  }

  DT* pack_single_col_panel(const DT* data, int64_t rs, int64_t cs, int64_t rows, int64_t cols, DT* out) {
    if (cols == NR) {
      for (int p = 0; p < rows; p++) {
        for (int j = 0; j < NR; j++) {
          *(out++) = data[rs * p + cs * j];
        }
      }
    } else {
      for (int p = 0; p < rows; p++) {
        int j = 0;
        for (; j < cols; j++) {
          *(out++) = data[rs * p + cs * j];
        }
        for (; j < NR; j++) {
          *(out++) = 0;
        }
      }
    }
    return out;
  }

  DT* data() {
    return buffer.data();
  }

  std::vector<DT> buffer;
};

}  // namespace packers
}  // namespace column_major
