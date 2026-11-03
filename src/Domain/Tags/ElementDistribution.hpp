// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>

#include "DataStructures/DataBox/Tag.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementDistribution.hpp"
#include "Options/String.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Parallel::OptionTags {
struct Parallelization;
}  // namespace Parallel::OptionTags
/// \endcond

namespace domain {
namespace OptionTags {
/// \ingroup OptionTagsGroup
/// \ingroup ComputationalDomainGroup
struct ElementDistribution {
  using type = ::domain::ElementDistribution;
  static constexpr Options::String help = {
      "Weighting pattern to use for ZCurve element distribution. Specify "
      "RoundRobin to just place each element on the next core. "
      "TestInteriorCubeGridPoints is a test weighting valid only for a 3D "
      "filled sphere in which every block except InnerCube is DG-only: it "
      "weights the InnerCube elements not touching the cube's boundary by "
      "their FD point count and all other elements by their DG point count. "
      "TestCcz4InteriorCubeWork uses the same layout restrictions, with "
      "temporary CCZ4 work weights: FD weight P, DG weight "
      "P*(1.824+0.000519*P), where P is the active point count. "
      "Alternatively, specify 'Explicit:' followed by a map from the quoted "
      "name of every initial element, e.g. \"[B0,(L1I0,L1I0,L0I0)]\", to the "
      "global processor (Charm++ PE) number on which it is created. This "
      "chooses only the initial placement; binding processors to cores "
      "remains the launcher's task, and the map must be regenerated when the "
      "domain or the processors change. Explicit placement is supported only "
      "by the evolution element array and element collection."};
  using group = Parallel::OptionTags::Parallelization;
};
}  // namespace OptionTags

namespace Tags {
/// \ingroup DataBoxTagsGroup
/// \ingroup ComputationalDomainGroup
/// Tag that holds method for how to distribute the elements on the given
/// resources.
///
/// \note When not using local time stepping (LTS), a user cannot choose the
/// NumGridPointsAndGridSpacing element distribution because grid spacing does
/// not affect the computational cost at all. Therefore, if a user does choose
/// NumGridPointsAndGridSpacing when not using LTS, an error will occur.
struct ElementDistribution : db::SimpleTag {
  using type = ::domain::ElementDistribution;
  using option_tags = tmpl::list<OptionTags::ElementDistribution>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& element_distribution) {
    return element_distribution;
  }
};
}  // namespace Tags
}  // namespace domain
