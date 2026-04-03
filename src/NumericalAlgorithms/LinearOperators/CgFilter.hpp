// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <pup.h>
#include <string>
#include <unordered_set>

#include "Options/Auto.hpp"
#include "Options/String.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class Matrix;
template <size_t Dim>
class Mesh;

/// \endcond

namespace Filters {
template <size_t FilterIndex>
class CgFilter {
 public:
  struct Alpha {
    using type = double;
    static constexpr Options::String help =
        "exp(-alpha) is rescaling of highest coefficient";
    static type lower_bound() { return 0.0; }
  };

  struct HalfPower {
    using type = unsigned;
    static constexpr Options::String help =
        "Half of the exponent in the generalized Gaussian";
    static type lower_bound() { return 1; }
  };

  struct Enable {
    using type = bool;
    static constexpr Options::String help = {"Enable the filter"};
  };

  struct BlocksToFilter {
    using type =
        Options::Auto<std::vector<std::string>, Options::AutoLabel::All>;
    static constexpr Options::String help = {
        "List of blocks or block groups to apply filtering to. All other "
        "blocks will have no filtering. You can also specify 'All' to do "
        "filtering in all blocks of the domain."};
  };

  using options = tmpl::list<Alpha, HalfPower, Enable, BlocksToFilter>;
  static constexpr Options::String help = {"A CG compatible filter."};
  static std::string name() { return "CGFilter" + std::to_string(FilterIndex); }

  CgFilter() = default;

  CgFilter(double alpha, unsigned half_power, bool enable,
           const std::optional<std::vector<std::string>>& blocks_to_filter,
           const Options::Context& context = {});

  /// A cached matrix used to apply the filter to the given mesh
  const Matrix& filter_matrix(const Mesh<1>& mesh) const;

  bool enable() const { return enable_; }

  const std::optional<std::unordered_set<std::string>>& blocks_to_filter()
      const {
    return blocks_to_filter_;
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

 private:
  template <size_t LocalFilterIndex>
  // NOLINTNEXTLINE(readability-redundant-declaration)
  friend bool operator==(const CgFilter<LocalFilterIndex>& lhs,
                         const CgFilter<LocalFilterIndex>& rhs);

  double alpha_{36.0};
  unsigned half_power_{16};
  bool enable_{true};
  std::optional<std::unordered_set<std::string>> blocks_to_filter_{};
};

template <size_t LocalFilterIndex>
bool operator==(const CgFilter<LocalFilterIndex>& lhs,
                const CgFilter<LocalFilterIndex>& rhs);

template <size_t FilterIndex>
bool operator!=(const CgFilter<FilterIndex>& lhs,
                const CgFilter<FilterIndex>& rhs);
}  // namespace Filters
