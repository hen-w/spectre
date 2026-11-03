// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "Domain/ExplicitElementPlacement.hpp"
#include "Options/Auto.hpp"
#include "Options/Options.hpp"
#include "Options/ParseError.hpp"
#include "Utilities/TypeTraits/CreateGetStaticMemberVariableOrDefault.hpp"

/// \cond
template <size_t Dim>
class Block;

template <size_t Dim>
class ElementId;

namespace PUP {
class er;
}  // namespace PUP

namespace Spectral {
enum class Basis : uint8_t;
enum class Quadrature : uint8_t;
}  // namespace Spectral
/// \endcond

namespace domain {
/// The weighting scheme for assigning computational costs to `Element`s for
/// distributing balanced compuational costs per processor (see
/// `BlockZCurveProcDistribution`)
enum class ElementWeight {
  /// A weighting scheme where each `Element` is assigned the same computational
  /// cost
  Uniform,
  /// A weighting scheme where each `Element`'s computational cost is equal to
  /// the number of grid points in that `Element`. In a filled-sphere domain
  /// (a block named `InnerCube`), the elements of the static DG-subcell (FD)
  /// region — the InnerCube block and the innermost radial layer of the
  /// innermost shell's wedges — are weighted by their FD point count
  /// \f$(2n-1)^3\f$ instead of the declared \f$n^3\f$
  NumGridPoints,
  /// A weighting scheme where each `Element`'s computational cost is weighted
  /// by both the number of grid points and minimum spacing between grid points
  /// in that `Element` (see `get_num_points_and_grid_spacing_cost()` for
  /// details)
  NumGridPointsAndGridSpacing,
  /// A test weighting scheme specialized to a single static DG-subcell layout:
  /// a 3D filled sphere in which every block except the one named `InnerCube`
  /// is DG-only (listed in the subcell option `OnlyDgBlocksAndGroups`). The
  /// subcell initialization then puts on FD exactly the `InnerCube` elements
  /// that do not touch the cube's boundary, i.e. those whose segment index
  /// \f$i\f$ satisfies \f$0 < i < 2^L - 1\f$ in every dimension for cube
  /// refinement level \f$L\f$. The one-element-thick collar of the cube
  /// borders the DG-only blocks and stays on DG, as do all other blocks. FD
  /// weights are \f$\prod_i (2N_i - 1)\f$; all other weights are
  /// \f$\prod_i N_i\f$.
  ///
  /// \warning This is not a general subcell-aware weighting.
  /// `get_element_costs()` has no access to `OnlyDgBlocksAndGroups` or to the
  /// TCI configuration, so it cannot detect other FD layouts and weights them
  /// incorrectly. It checks only the domain structure: the domain must be 3D
  /// with exactly one block named `InnerCube`, connected to other blocks on all
  /// six faces, with isotropic initial refinement \f$L \geq 2\f$ and isotropic
  /// initial extents. Any other structure is an error.
  TestInteriorCubeGridPoints,
  /// Temporary CCZ4 work estimate for the same static FD layout and with the
  /// same restrictions as TestInteriorCubeGridPoints. An FD element has weight
  /// equal to its active point count. A DG element with P points has weight
  /// P * (1.824 + 0.000519 * P). The coefficients are an empirical fit to the
  /// gradient-projection executable's FD11/DG6 puncture-domain profiles, not a
  /// general DG cost model. They may need recalibration after changing the
  /// numerical algorithms or substantially changing the meshes.
  ///
  /// This changes only the weights passed to BlockZCurveProcDistribution.
  /// It does not change the placement algorithm or detect the TCI options.
  TestCcz4InteriorCubeWork
};

std::ostream& operator<<(std::ostream& os, ElementWeight weight);

/*!
 * \brief How to distribute the initial elements on processors
 *
 * \details Holds one of
 * - `std::nullopt`: place each element on the next processor (round robin),
 * - an `ElementWeight`: place the elements with the
 *   `BlockZCurveProcDistribution` using costs from `get_element_costs()`,
 * - an `ExplicitElementPlacement`: place each initial element on the
 *   processor given for it.
 *
 * In an input file the first two are given as `RoundRobin` or the name of
 * the `ElementWeight`, and the third as a map under the key `Explicit` (see
 * `ExplicitElementPlacement`).
 */
class ElementDistribution {
 public:
  /// Option label for round robin
  struct RoundRobin {};

  /// Round robin
  ElementDistribution() = default;
  /// Z-curve distribution with the given weight, or round robin for
  /// `std::nullopt`
  explicit ElementDistribution(std::optional<ElementWeight> element_weight);
  explicit ElementDistribution(ExplicitElementPlacement explicit_placement);

  /// Whether the elements are placed by an `ExplicitElementPlacement`
  bool is_explicit() const {
    return std::holds_alternative<ExplicitElementPlacement>(distribution_);
  }

  /// The weight of the Z-curve distribution, or `std::nullopt` for round
  /// robin. Throws `std::bad_variant_access` if `is_explicit()`.
  const std::optional<ElementWeight>& element_weight() const {
    return std::get<std::optional<ElementWeight>>(distribution_);
  }

  /// Throws `std::bad_variant_access` unless `is_explicit()`.
  const ExplicitElementPlacement& explicit_placement() const {
    return std::get<ExplicitElementPlacement>(distribution_);
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

 private:
  friend bool operator==(const ElementDistribution& lhs,
                         const ElementDistribution& rhs);

  std::variant<std::optional<ElementWeight>, ExplicitElementPlacement>
      distribution_;
};

bool operator!=(const ElementDistribution& lhs, const ElementDistribution& rhs);

/// \brief Get the cost of each `Element` in a list of `Block`s where
/// `element_weight` specifies which weight distribution scheme to use
///
/// \details It is only necessary to pass in a value for `i1_basis` and
/// `i1_quadrature` if the value for `element_weight` is
/// `ElementWeight::NumGridPointsAndGridSpacing`. Otherwise, the argument isn't
/// needed and will have no effect if it does have a value.
template <size_t Dim>
std::unordered_map<ElementId<Dim>, double> get_element_costs(
    const std::vector<Block<Dim>>& blocks,
    const std::vector<std::array<size_t, Dim>>& initial_refinement_levels,
    const std::vector<std::array<size_t, Dim>>& initial_extents,
    ElementWeight element_weight,
    const std::optional<Spectral::Basis>& i1_basis,
    const std::optional<Spectral::Quadrature>& i1_quadrature);

/*!
 * \brief Distribution strategy for assigning elements to CPUs using a
 * Morton ('Z-order') space-filling curve to determine placement within each
 * block, where `Element`s are distributed across CPUs
 *
 * \details The element distribution attempts to assign a balanced total
 * computational cost to each processor that is allowed to have `Element`s.
 * First, each `Block`'s `Element`s are ordered by their Z-curve index (see more
 * below). `Element`s are traversed in this order and assigned to CPUs in order,
 * moving onto the next CPU once the target cost per CPU is met. The target cost
 * per CPU is defined as the remaining cost to distribute divided by the
 * remaining number of CPUs to distribute to. This is an important distinction
 * from simply having one constant target cost per CPU defined as the total cost
 * divided by the total number of CPUs with elements. Since the total cost of
 * `Element`s on a processor will nearly never add up to be exactly the average
 * cost per CPU, this means that we would either have to decide to overshoot or
 * undershoot the average as we iterate over the CPUs and assign `Element`s. If
 * we overshoot the average on each processor, the final processor could have a
 * much lower cost than the rest of the processors and we run the risk of
 * overshooting so much that one or more of the requested processors don't get
 * assigned any `Element`s at all. If we undershoot the average on each
 * processor, the final processor could have a much higher cost than the others
 * due to remainder cost piling up. This algorithm avoids these risks by instead
 * adjusting the target cost per CPU as we finish assigning cost to previous
 * CPUs.
 *
 * Morton curves are a simple and easily-computed space-filling curve that
 * (unlike Hilbert curves) permit diagonal traversal. See, for instance,
 * \cite Borrell2018 for a discussion of mesh partitioning using space-filling
 * curves.
 * A concrete example of the use of a Morton curve in 2d is given below.
 *
 * A sketch of a 2D block with 4x2 elements, with each element labeled according
 * to the order on the Morton curve:
 * ```
 *          x-->
 *          0   1   2   3
 *        ----------------
 *  y  0 |  0   2   4   6
 *  |    |  | / | / | / |
 *  v  1 |  1   3   5   7
 * ```
 * (forming a zig-zag path, that under some rotation/reflection has a 'Z'
 * shape).
 *
 * The Morton curve method is a quick way of getting acceptable spatial locality
 * -- usually, for approximately even distributions, it will ensure that
 * elements are assigned in large volume chunks, and the structure of the Morton
 * curve ensures that for a given processor and block, the elements will be
 * assigned in no more than two orthogonally connected clusters. In principle, a
 * Hilbert curve could potentially improve upon the gains obtained by this class
 * by guaranteeing that all elements within each block form a single
 * orthogonally connected cluster.
 *
 * The assignment of portions of blocks to processors may use partial blocks,
 * and/or multiple blocks to ensure an even distribution of elements to
 * processors.
 * We currently make no distinction between dividing elements between processors
 * within a node and dividing elements between processors across nodes. The
 * current technique aims to have a simple method of reducing communication
 * globally, though it would likely be more efficient to prioritize minimization
 * of inter-node communication, because communication across interconnects is
 * the primary cost of communication in charm++ runs.
 *
 * \warning The use of the Morton curve to generate a well-clustered element
 * distribution currently assumes that the refinement is uniform over each
 * block, with no internal structure that would be generated by, for instance
 * AMR.
 * This distribution method will need alteration to perform well for blocks with
 * internal structure from h-refinement. Morton curves can be defined
 * recursively, so a generalization of the present method is possible for blocks
 * with internal refinement
 *
 * \tparam Dim the number of spatial dimensions of the `Block`s
 */
template <size_t Dim>
struct BlockZCurveProcDistribution {
  BlockZCurveProcDistribution() = default;

  /// The `number_of_procs_with_elements` argument represents how many procs
  /// will have elements. This is not necessarily equal to the total number of
  /// procs because some global procs may be ignored by the sixth argument
  /// `global_procs_to_ignore`.
  BlockZCurveProcDistribution(
      const std::unordered_map<ElementId<Dim>, double>& element_costs,
      size_t number_of_procs_with_elements,
      const std::vector<Block<Dim>>& blocks,
      const std::vector<std::array<size_t, Dim>>& initial_refinement_levels,
      const std::vector<std::array<size_t, Dim>>& initial_extents,
      const std::unordered_set<size_t>& global_procs_to_ignore = {});

  /// Gets the suggested processor number for a particular `ElementId`,
  /// determined by the Morton curve weighted element assignment described in
  /// detail in the parent class documentation.
  size_t get_proc_for_element(const ElementId<Dim>& element_id) const;

  const std::vector<std::vector<std::pair<size_t, size_t>>>&
  block_element_distribution() const {
    return block_element_distribution_;
  }

 private:
  // in this nested data structure:
  // - The block id is the first index
  // - There is an arbitrary number of CPUs per block, each with an element
  //   allowance
  // - Each element allowance is represented by a pair of proc number, number of
  //   elements in the allowance
  std::vector<std::vector<std::pair<size_t, size_t>>>
      block_element_distribution_;
};
}  // namespace domain

namespace element_weight_detail {
CREATE_GET_STATIC_MEMBER_VARIABLE_OR_DEFAULT(local_time_stepping)
}  // namespace element_weight_detail

template <>
struct Options::create_from_yaml<domain::ElementWeight> {
  template <typename Metavariables>
  static domain::ElementWeight create(const Options::Option& options) {
    const auto ordering = options.parse_as<std::string>();
    if (ordering == "Uniform") {
      return domain::ElementWeight::Uniform;
    } else if (ordering == "NumGridPoints") {
      return domain::ElementWeight::NumGridPoints;
    } else if (ordering == "NumGridPointsAndGridSpacing") {
      if constexpr (not element_weight_detail::
                        get_local_time_stepping_or_default_v<Metavariables,
                                                             false>) {
        PARSE_ERROR(
            options.context(),
            "When not using local time stepping, you cannot use "
            "NumGridPointsAndGridSpacing for the element distribution. Please "
            "choose another element distribution.");
      }
      return domain::ElementWeight::NumGridPointsAndGridSpacing;
    } else if (ordering == "TestInteriorCubeGridPoints") {
      return domain::ElementWeight::TestInteriorCubeGridPoints;
    } else if (ordering == "TestCcz4InteriorCubeWork") {
      return domain::ElementWeight::TestCcz4InteriorCubeWork;
    }
    PARSE_ERROR(options.context(),
                "ElementWeight must be 'Uniform', 'NumGridPoints', "
                "'NumGridPointsAndGridSpacing', 'TestInteriorCubeGridPoints', "
                "or 'TestCcz4InteriorCubeWork'");
  }
};

namespace domain::ElementDistribution_detail {
// The placement given under the key `Explicit` if the option is a map, else
// `std::nullopt`
std::optional<ExplicitElementPlacement> parse_explicit_placement(
    const Options::Option& options);
}  // namespace domain::ElementDistribution_detail

template <>
struct Options::create_from_yaml<domain::ElementDistribution> {
  template <typename Metavariables>
  static domain::ElementDistribution create(const Options::Option& options) {
    std::optional<domain::ExplicitElementPlacement> explicit_placement =
        domain::ElementDistribution_detail::parse_explicit_placement(options);
    if (explicit_placement.has_value()) {
      return domain::ElementDistribution{std::move(*explicit_placement)};
    }
    const auto element_weight =
        options.parse_as<Options::Auto<domain::ElementWeight,
                                       domain::ElementDistribution::RoundRobin>,
                         Metavariables>();
    return domain::ElementDistribution{
        static_cast<const std::optional<domain::ElementWeight>&>(
            element_weight)};
  }
};
