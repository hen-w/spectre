// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Protocols/Mutator.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Time/Actions/SelfStartActions.hpp"
#include "Time/Tags/TimeStep.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace Ccz4::fd {
/*!
 * \brief Add sixth-order Kreiss-Oliger dissipation to the physical RHS.
 *
 * The filter increment is evaluated from volume and ghost values at the
 * current RHS time, before the time-stepper history is recorded. Dividing by
 * the signed full step preserves the nominal per-step strength epsilon and
 * damps in either evolution direction. During self-start the saved evolution
 * step is used, so the generated history solves the same semidiscrete system
 * as the first evolution step. Stored auxiliary-gradient derivatives are
 * untouched. The explicit time stepper's stability limit also applies to
 * this dissipation; this is not a post-step filter.
 */
struct AddKreissOligerDissipation : tt::ConformsTo<db::protocols::Mutator> {
  using dt_variables_tag =
      db::add_tag_prefix<::Tags::dt, System::variables_tag>;
  using return_tags = tmpl::list<dt_variables_tag>;
  using argument_tags =
      tmpl::list<System::variables_tag, evolution::dg::subcell::Tags::Mesh<3>,
                 Tags::EvolveLapseAndShift, Tags::KreissOligerEpsilon,
                 evolution::dg::subcell::Tags::GhostDataForReconstruction<3>,
                 ::Tags::TimeStep, ::Tags::TimeStepId,
                 SelfStart::Tags::InitialValue<::Tags::TimeStep>>;

  static void apply(
      gsl::not_null<typename dt_variables_tag::type*> dt_vars,
      const typename System::variables_tag::type& evolved_vars,
      const Mesh<3>& subcell_mesh, bool evolve_lapse_and_shift,
      double kreiss_oliger_epsilon,
      const DirectionalIdMap<3, evolution::dg::subcell::GhostData>& ghost_data,
      const TimeDelta& time_step, const TimeStepId& time_step_id,
      const std::tuple<TimeDelta>& initial_time_step);
};
}  // namespace Ccz4::fd
