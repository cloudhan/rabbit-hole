#include <cute/tensor.hpp>
#include <cute/layout.hpp>

using namespace cute;

enum class Order {
  Z,
  W
};
std::ostream& operator<<(std::ostream& oss, const Order& ord) {
  if (ord == Order::Z) {
    oss << "Z";
  } else {
    oss << "ᴎ";
  }
  return oss;
}

template <int _X = 0, int _Y = 0>
struct Z {
  constexpr const static auto Ord = Order::Z;
  constexpr const static auto X = Int<_X>{};
  constexpr const static auto Y = Int<_Y>{};
  constexpr const static auto Stride = Int<_Y>{};
};

template <int _X = 0, int _Y = 0>
struct W {  // convenience for ᴎ
  constexpr const static auto Ord = Order::W;
  constexpr const static auto X = Int<_X>{};
  constexpr const static auto Y = Int<_Y>{};
  constexpr const static auto Stride = Int<X>{};
};


struct Summary {
  explicit Summary(std::string_view s) {
    std::cout << "<details><summary>" << s << "</summary>\n\n```\n";
  }

  template <typename T, typename... Ts>
  Summary(T a, Ts... args) {
  }

  ~Summary() {
    std::cout << "\n```\n</details>\n\n";
  }

  template<typename T1, typename T2>
  std::string MakeString() const {

  }
};

int main() {
  constexpr const int WarpSize = 32;
  constexpr const int NumThread = 128;
  constexpr const int NumWarp = NumThread / WarpSize;

#if !defined(CONFIG) || CONFIG == 0
  using ConfigMemory = Z<8, 64>;
  using ConfigThread = Z<1, 1>;
  using ConfigWarp = Z<2, WarpSize / 2>;
  using ConfigCta = Z<1, NumWarp>;
  using ConfigBatch = Z<>;
#elif CONFIG == 1
  using ConfigMemory = Z<8, 64>;
  using ConfigThread = Z<1, 2>;
  using ConfigWarp = Z<4, WarpSize / 4>;
  using ConfigCta = Z<1, NumWarp>;
  using ConfigBatch = Z<>;
#elif CONFIG == 2
  using ConfigMemory = Z<8, 64>;
  using ConfigThread = Z<1, 2>;
  using ConfigWarp = Z<4, WarpSize / 4>;
  using ConfigCta = Z<2, NumWarp / 2>;
  using ConfigBatch = Z<>;
#elif CONFIG == 3
  using ConfigMemory = Z<8, 64>;
  using ConfigThread = Z<1, 2>;
  using ConfigWarp = Z<4, WarpSize / 4>;
  using ConfigCta = W<2, NumWarp / 2>;
  using ConfigBatch = Z<>;
#elif CONFIG == 4
  using ConfigMemory = Z<8, 64>;
  using ConfigThread = Z<1, 1>;
  using ConfigWarp = Z<2, WarpSize / 2>;
  using ConfigCta = Z<2, NumWarp / 2>;
  using ConfigBatch = W<>;
#elif CONFIG == 5
  using ConfigMemory = W<64, 8>;
  using ConfigThread = W<4, 1>;
  using ConfigWarp = W<WarpSize / 2, 2>;
  using ConfigCta = W<1, NumWarp>;
  using ConfigBatch = Z<>;
#elif CONFIG == 6
  using ConfigMemory = W<64, 8>;
  using ConfigThread = W<4, 1>;
  using ConfigWarp = W<WarpSize / 4, 4>;
  using ConfigCta = W<2, NumWarp / 2>;
  using ConfigBatch = Z<>;
#elif CONFIG == 7
  using ConfigMemory = W<64, 8>;
  using ConfigThread = W<2, 2>;
  using ConfigWarp = W<WarpSize / 2, 2>;
  using ConfigCta = W<2, NumWarp / 2>;
  using ConfigBatch = Z<>;
#endif

  std::cout << "Each warp has " << WarpSize << " threads.\n"
            << "We launched a grid with " << NumThread << " of total threads\n\n";

  // Load A
  std::cout << "# Load with " << ConfigMemory::Ord << " ordered " << int(ConfigMemory::X) << "x" << int(ConfigMemory::Y) << "\n\n";

  constexpr const auto MemoryLayout = [&]() {
    if constexpr (ConfigMemory::Ord == Order::W) {
      return make_layout(make_shape(ConfigMemory::X, ConfigMemory::Y), make_stride(_1{}, ConfigMemory::Stride));
    } else {
      return make_layout(make_shape(ConfigMemory::X, ConfigMemory::Y), make_stride(ConfigMemory::Stride, _1{}));
    }
  }();
  std::cout << "\n\n<details><summary>Each Memory Block</summary>\n\n```\n";
  print_layout(MemoryLayout);
  std::cout << "```\n</details>\n\n";

  constexpr const auto ThreadLoad = make_layout(make_shape(ConfigThread::X, ConfigThread::Y), make_stride(Int<0>{}, Int<0>{}));
  std::cout << "\n\n<details><summary>Each Thread Load</summary>\n\n```\n";
  print_layout(ThreadLoad);
  std::cout << "```\n</details>\n\n";

  constexpr const auto WarpLoad = [&]() {
    if constexpr (ConfigWarp::Ord == Order::W) {
      return blocked_product(ThreadLoad, make_layout(make_shape(ConfigWarp::X, ConfigWarp::Y), make_stride(_1{}, ConfigWarp::Stride)));

    } else {
      return blocked_product(ThreadLoad, make_layout(make_shape(ConfigWarp::X, ConfigWarp::Y), make_stride(ConfigWarp::Stride, _1{})));
    }
  }();
  std::cout << "\n\n<details><summary>Each Warp Load</summary>\n\n```\n";
  print_layout(WarpLoad);
  std::cout << "```\n</details>\n\n";

  constexpr const auto CtaSingleBatchLoad = [&]() {
    if constexpr (ConfigCta::Ord == Order::W) {
      return blocked_product(WarpLoad, make_layout(make_shape(ConfigCta::X, ConfigCta::Y), make_stride(_1{}, ConfigCta::Stride)));

    } else {
      return blocked_product(WarpLoad, make_layout(make_shape(ConfigCta::X, ConfigCta::Y), make_stride(ConfigCta::Stride, _1{})));
    }
  }();
  std::cout << "\n\n<details><summary>Each CTA Load In Single Batch (need " << int(NumWarp) << " batches)</summary>\n\n```\n";
  print_layout(CtaSingleBatchLoad);
  std::cout << "```\n</details>\n\n";

  constexpr const auto NumBatch = size<1>(logical_divide(MemoryLayout, CtaSingleBatchLoad)) / size(ThreadLoad);
  constexpr auto NumBatchX = ConfigMemory::X / size<0>(CtaSingleBatchLoad);
  constexpr auto NumBatchY = ConfigMemory::Y / size<1>(CtaSingleBatchLoad);

  const auto CtaFullLoad = [&]() {
    if constexpr (ConfigBatch::Ord == Order::W) {
      return blocked_product(CtaSingleBatchLoad, make_layout(make_shape(Int<NumBatchX>{}, Int<NumBatchY>{}), make_stride(_1{}, Int<NumBatchX>{})));
    } else {
      return blocked_product(CtaSingleBatchLoad, make_layout(make_shape(Int<NumBatchX>{}, Int<NumBatchY>{}), make_stride(Int<NumBatchY>{}, _1{})));
    }
  }();

  static_assert(ConfigWarp::X * ConfigWarp::Y == WarpSize, "invalid config");
  static_assert(ConfigCta::X * ConfigCta::Y == NumWarp, "invalid config");
  static_assert(NumBatch >= 1, "invalid config");
  static_assert(NumBatchX >= 1, "invalid config");
  static_assert(NumBatchY >= 1, "invalid config");
  static_assert(size<0>(CtaFullLoad) == ConfigMemory::X && size<1>(CtaFullLoad) == ConfigMemory::Y, "invalid config");

  std::cout << "\n\n<details><summary>Each CTA Load In " << int(NumBatch) << " Batches</summary>\n\n```\n";
  if (NumBatch > 1) {
    print_layout(CtaFullLoad);
  } else {
    std::cout << "only 1 batch, skip!\n";
  }
  std::cout << "```\n</details>\n\n";

  std::cout << "```\n"
            << "ConfigMemory " << ConfigMemory::Ord << " " << int(ConfigMemory::X) << "x" << int(ConfigMemory::Y) << "\n"
            << "ConfigThread " << ConfigThread::Ord << " " << int(ConfigThread::X) << "x" << int(ConfigThread::Y) << "\n"
            << "ConfigWarp   " << ConfigWarp::Ord << " " << int(ConfigWarp::X) << "x" << int(ConfigWarp::Y) << "\n"
            << "ConfigCta    " << ConfigCta::Ord << " " << int(ConfigCta::X) << "x" << int(ConfigCta::Y) << "\n"
            << "ConfigBatch  " << ConfigBatch::Ord << " " << int(NumBatchX) << "x" << int(NumBatchY) << "\n"
            << "```\n";

  return 0;
}
