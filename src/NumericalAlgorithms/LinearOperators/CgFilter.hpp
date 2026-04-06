// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <functional>
#include <optional>
#include <pup.h>
#include <string>
#include <tuple>
#include <unordered_set>

#include "DataStructures/ApplyMatrices.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Filter.hpp"
#include "Options/Auto.hpp"
#include "Options/String.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

/// \cond
class Matrix;
template <size_t Dim>
class Mesh;

/// \endcond

namespace Filters {
template <size_t Dim>
class CgFilter : public Filter {
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

  struct BlocksToFilter {
    using type =
        Options::Auto<std::vector<std::string>, Options::AutoLabel::All>;
    static constexpr Options::String help = {
        "List of blocks or block groups to apply filtering to. All other "
        "blocks will have no filtering. You can also specify 'All' to do "
        "filtering in all blocks of the domain."};
  };

  using options = tmpl::list<Alpha, HalfPower, BlocksToFilter>;
  static constexpr Options::String help = {"A CG compatible filter."};
  static std::string name() { return "CGFilter"; }

  CgFilter() = default;

  CgFilter(double alpha, unsigned half_power,
           const std::optional<std::vector<std::string>>& blocks_to_filter,
           const Options::Context& context = {});

  WRAPPED_PUPable_decl_template(CgFilter);  // NOLINT
  explicit CgFilter(CkMigrateMessage* msg) : Filter(msg) {}

  /// A cached matrix used to apply the filter to the given mesh
  const Matrix& filter_matrix(const Mesh<1>& mesh) const;

  std::optional<std::unordered_set<std::string>> blocks_to_filter()
      const override {
    return blocks_to_filter_;
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

 public:
  using argument_tags = tmpl::list<domain::Tags::Mesh<Dim>>;

  template <typename TagsList>
  void operator()(const gsl::not_null<Variables<TagsList>*> vars,
                  const Mesh<Dim>& mesh) const {
    *vars = apply_matrices(filter_matrices(mesh), *vars, mesh.extents());
  }

  template <typename... TensorTypes>
    requires((not tt::is_a_v<Variables, std::decay_t<TensorTypes>>) and ...)
  void operator()(const std::tuple<gsl::not_null<TensorTypes*>...>& tensors,
                  const Mesh<Dim>& mesh) const {
    const auto filter = filter_matrices(mesh);
    std::apply(
        [&filter, extents = mesh.extents()](const auto... tensor_ptrs) {
          (
              [&filter, &extents](const auto tensor_ptr) {
                for (auto& component : *tensor_ptr) {
                  component = apply_matrices(filter, component, extents);
                }
              }(tensor_ptrs),
              ...);
        },
        tensors);
  }

 private:
  std::array<std::reference_wrapper<const Matrix>, Dim> filter_matrices(
      const Mesh<Dim>& mesh) const;

  template <size_t LocalDim>
  // NOLINTNEXTLINE(readability-redundant-declaration)
  friend bool operator==(const CgFilter<LocalDim>& lhs,
                         const CgFilter<LocalDim>& rhs);

  double alpha_{36.0};
  unsigned half_power_{16};
  std::optional<std::unordered_set<std::string>> blocks_to_filter_{};
};

template <size_t LocalDim>
bool operator==(const CgFilter<LocalDim>& lhs, const CgFilter<LocalDim>& rhs);

template <size_t LocalDim>
bool operator!=(const CgFilter<LocalDim>& lhs, const CgFilter<LocalDim>& rhs);

/// \cond
template <size_t Dim>
PUP::able::PUP_ID CgFilter<Dim>::my_PUP_ID = 0;  // NOLINT
/// \endcond

}  // namespace Filters
