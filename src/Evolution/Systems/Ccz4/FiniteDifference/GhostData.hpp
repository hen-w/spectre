// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/VariablesTag.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
template <typename T>
class Variables;
/// \endcond

namespace Ccz4::fd {
/*!
 * \brief Get the Ccz4 ghost data for subcell communication.
 *
 * - `GhostVariablesImpl<true>` (first pass, and all DG sends): copies the 9
 *   physical fields; the gradients `Tags::PotentialGradients` are zero.
 * - `GhostVariablesImpl<false>` (second finite-difference pass): copies the 9
 *   physical fields and the gradients computed for the DG neighbors.
 *
 * In both cases the buffer is sized for the full
 * `Variables<variables_tag_list>` (physical fields and gradients, 55
 * components) so that the receiver can interpret it uniformly.
 */
template <bool IsAuxiliary>
class GhostVariablesImpl {
 public:
  using return_tags = tmpl::list<>;
  using argument_tags =
      tmpl::list<::Tags::Variables<Ccz4::fd::System::variables_tag_list>>;

  static DataVector apply(
      const Variables<Ccz4::fd::System::variables_tag_list>& evolved_vars,
      size_t rdmp_size);
};

/// The physical fields with zero gradients.
using GhostVariables = GhostVariablesImpl<true>;
/// The physical fields and the gradients.
using GhostVariablesPhysical = GhostVariablesImpl<false>;
}  // namespace Ccz4::fd
