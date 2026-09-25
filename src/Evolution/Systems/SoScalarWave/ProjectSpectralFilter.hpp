// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <unordered_map>
#include <utility>

#include "DataStructures/TaggedTuple.hpp"
#include "Evolution/Systems/SoScalarWave/System.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Tag.hpp"
#include "ParallelAlgorithms/Amr/Protocols/Projector.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <size_t Dim>
class Element;
template <size_t Dim>
class ElementId;
template <size_t Dim>
class Mesh;
/// \endcond

namespace SoScalarWave {
/*!
 * \ingroup DiscontinuousGalerkinGroup
 * \brief Placeholder AMR projector for the per-element runtime spectral
 * filter selected by `evolution::dg::Initialization::SpectralFilters`.
 *
 * AMR is not supported for SoScalarWave. This projector exists only because
 * `amr::Actions::AdjustDomain` statically requires every mutable DataBox item
 * to be covered by a projector; every overload ERRORs if it is ever invoked,
 * rather than pretending the filter can be projected.
 */
template <size_t Dim>
struct ProjectSpectralFilter : tt::ConformsTo<amr::protocols::Projector> {
 private:
  using filter_tag = Filters::runtime::Tags::SpectralFilter<
      Dim, typename System<Dim>::variables_tag::tags_list>;
  using filter_type = typename filter_tag::type;

 public:
  using return_tags = tmpl::list<filter_tag>;
  using argument_tags = tmpl::list<>;

  static void apply(
      const gsl::not_null<filter_type*> /*filter*/,
      const std::pair<Mesh<Dim>, Element<Dim>>& /*old_mesh_and_element*/) {
    ERROR(
        "AMR is not supported for SoScalarWave, so the runtime spectral "
        "filter cannot be projected (p-refinement).");
  }

  template <typename... ParentTags>
  static void apply(
      const gsl::not_null<filter_type*> /*filter*/,
      const tuples::TaggedTuple<ParentTags...>& /*parent_items*/) {
    ERROR(
        "AMR is not supported for SoScalarWave, so the runtime spectral "
        "filter cannot be projected (h-refinement split).");
  }

  template <typename... ChildrenTags>
  static void apply(const gsl::not_null<filter_type*> /*filter*/,
                    const std::unordered_map<
                        ElementId<Dim>, tuples::TaggedTuple<ChildrenTags...>>&
                    /*children_items*/) {
    ERROR(
        "AMR is not supported for SoScalarWave, so the runtime spectral "
        "filter cannot be projected (h-refinement join).");
  }
};
}  // namespace SoScalarWave
