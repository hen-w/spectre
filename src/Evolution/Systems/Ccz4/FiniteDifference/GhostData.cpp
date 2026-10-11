// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Ccz4/FiniteDifference/GhostData.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Utilities/TMPL.hpp"

namespace Ccz4::fd {
template <bool IsAuxiliary>
DataVector GhostVariablesImpl<IsAuxiliary>::apply(
    const Variables<Ccz4::fd::System::variables_tag_list>& evolved_vars,
    const size_t rdmp_size) {
  DataVector buffer{evolved_vars.number_of_grid_points() *
                        Variables<Ccz4::fd::System::variables_tag_list>::
                            number_of_independent_components +
                    rdmp_size};
  buffer = 0.0;
  Variables<Ccz4::fd::System::variables_tag_list> vars_to_reconstruct(
      buffer.data(), buffer.size() - rdmp_size);

  // The physical fields (always), then the gradients only when they are
  // valid (the second finite-difference pass); otherwise they stay zero.
  tmpl::for_each<System::original_evolved_variables_tags>(
      [&evolved_vars,
       &vars_to_reconstruct]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
        get<Tag>(vars_to_reconstruct) = get<Tag>(evolved_vars);
      });
  if constexpr (not IsAuxiliary) {
    get<Tags::PotentialGradients>(vars_to_reconstruct) =
        get<Tags::PotentialGradients>(evolved_vars);
  }

  return buffer;
}

template class GhostVariablesImpl<true>;
template class GhostVariablesImpl<false>;
}  // namespace Ccz4::fd
