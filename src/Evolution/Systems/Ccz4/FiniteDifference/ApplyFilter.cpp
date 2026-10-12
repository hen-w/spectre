// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Ccz4/FiniteDifference/ApplyFilter.hpp"

#include <cmath>
#include <tuple>

#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Filter.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"

namespace Ccz4::fd {

void AddKreissOligerDissipation::apply(
    const gsl::not_null<typename dt_variables_tag::type*> dt_vars,
    const typename System::variables_tag::type& evolved_vars,
    const Mesh<3>& subcell_mesh, const bool evolve_lapse_and_shift,
    const double kreiss_oliger_epsilon,
    const DirectionalIdMap<3, evolution::dg::subcell::GhostData>& ghost_data,
    const TimeDelta& time_step, const TimeStepId& time_step_id,
    const std::tuple<TimeDelta>& initial_time_step) {
  constexpr size_t fd_order = 4;

  if (kreiss_oliger_epsilon == 0.0) {
    return;
  } else if (kreiss_oliger_epsilon > 0.0 and kreiss_oliger_epsilon <= 1.0) {
    const double full_step = time_step_id.slab_number() < 0
                                 ? std::get<0>(initial_time_step).value()
                                 : time_step.value();
    ASSERT(std::isfinite(full_step) and full_step != 0.0,
           "KO dissipation requires a finite nonzero full time step.");
    typename System::variables_tag::type filtered_vars = evolved_vars;

    Ccz4::fd::ccz4_kreiss_oliger_filter(make_not_null(&filtered_vars),
                                        evolved_vars, ghost_data,
                                        evolve_lapse_and_shift, subcell_mesh,
                                        fd_order + 2, kreiss_oliger_epsilon);

    tmpl::for_each<System::original_evolved_variables_tags>(
        [&dt_vars, &filtered_vars, &evolved_vars,
         full_step]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
          auto& dt_tensor = get<::Tags::dt<Tag>>(*dt_vars);
          for (size_t c = 0; c < dt_tensor.size(); ++c) {
            dt_tensor[c] +=
                (get<Tag>(filtered_vars)[c] - get<Tag>(evolved_vars)[c]) /
                full_step;
          }
        });
  } else {
    ERROR("Kreiss-Oliger epsilon should be in the interval [0, 1]. Got "
          << kreiss_oliger_epsilon << " instead.");
  }
}

}  // namespace Ccz4::fd
