// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>

#include "Evolution/Systems/Ccz4/ApplyTensorYlmFilter.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Factory.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Filter.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/SphericalShell.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/TensorYlmFilter.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Ccz4 {
/*!
 * \ingroup DiscontinuousGalerkinGroup
 * \brief A `tmpl::list` of all concrete `Filters::runtime` filter types for
 * the CCZ4 system.
 *
 * Extends `Filters::runtime::all_filters` with
 * `Filters::runtime::SphericalShell`. The name distinguishes these
 * runtime-filtering framework classes from the legacy `Filters::Filter`
 * volume filters (`Filters::Exponential`, `Ccz4::TensorYlmFilter`) that are
 * registered alongside them in the CCZ4 executable.
 */
using all_runtime_filters = tmpl::append<
    Filters::runtime::all_filters<3, fd::System::variables_tag_list>,
    tmpl::list<
        Filters::runtime::SphericalShell<fd::System::variables_tag_list>>>;
}  // namespace Ccz4

namespace Filters::runtime {
// No filter may modify the LDG auxiliary variables FieldA/B/D/P (rebuilt
// from the filtered variables every step with lifted boundary
// corrections): restrict all filters to the 9 original CCZ4 evolved
// variables, matching the legacy Ccz4::TensorYlmFilter and the tag list
// of the legacy filter action in EvolveCcz4.hpp.
template <>
struct FilteredTags<Ccz4::fd::System::variables_tag_list> {
  using type = Ccz4::filter_detail::ccz4_vars_list<Frame::Inertial>;
};
}  // namespace Filters::runtime

namespace ylm::TensorYlm {
// Declarations of the CCZ4 explicit specializations (defined in
// Evolution/Systems/Ccz4/ApplyTensorYlmFilter.cpp) so that translation
// units instantiating
// Filters::runtime::SphericalShell<Ccz4::fd::System::variables_tag_list>
// see them before use.
template <>
void fill_tensor_ylm_filters<Ccz4::fd::System::variables_tag_list>(
    gsl::not_null<FilterMatrixHolder*> matrix, size_t ell_max,
    size_t number_of_ell_modes_to_kill, std::optional<size_t> half_power,
    CoefficientNormalization coefficient_normalization);
template <>
void apply_tensor_ylm_filter(
    gsl::not_null<Variables<Ccz4::fd::System::variables_tag_list>*> vars,
    gsl::not_null<Variables<Ccz4::fd::System::variables_tag_list>*>
        temp_storage,
    const InverseJacobian<DataVector, 3, Frame::Inertial, Frame::Grid>&
        jac_inertial_to_grid,
    const InverseJacobian<DataVector, 3, Frame::Grid, Frame::Inertial>&
        jac_grid_to_inertial,
    const FilterMatrixHolder& filter_matrices, size_t ell_max,
    size_t radial_extents);
}  // namespace ylm::TensorYlm
