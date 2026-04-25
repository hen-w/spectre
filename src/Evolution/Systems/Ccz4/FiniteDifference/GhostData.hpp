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
 * - `GhostVariablesImpl<true>` (round 1, auxiliary): copies the 9 original
 *   evolved variables only.
 * - `GhostVariablesImpl<false>` (round 2, physical): copies the 9 original
 *   evolved variables AND the 4 auxiliary fields (FieldA/B/D/P).
 *
 * In both cases the buffer is sized for a full
 * `Variables<variables_tag_list>` (17 tags) so that the receiver can
 * interpret it uniformly.  Tags that are not explicitly copied are zero.
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

/// Round 1 (auxiliary pass): 9 evolved variables only.
using GhostVariables = GhostVariablesImpl<true>;
/// Round 2 (physical pass): 9 evolved + 4 auxiliary fields.
using GhostVariablesPhysical = GhostVariablesImpl<false>;
}  // namespace Ccz4::fd
