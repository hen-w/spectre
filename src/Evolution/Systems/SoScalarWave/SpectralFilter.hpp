// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/SoScalarWave/System.hpp"
#include "Evolution/Systems/SoScalarWave/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Factory.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Filter.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/SphericalShell.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/TensorYlmFilter.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace SoScalarWave {
/*!
 * \ingroup DiscontinuousGalerkinGroup
 * \brief A `tmpl::list` of all concrete `Filters::runtime` filter types for
 * the second-order scalar wave system in `Dim` spatial dimensions.
 *
 * In 3D this extends `Filters::runtime::all_filters` with
 * `Filters::runtime::SphericalShell`, which is available only in 3D and
 * requires the system-specific `ylm::TensorYlm` filter specializations
 * declared below. In 1D and 2D only `Filters::runtime::None` and
 * `Filters::runtime::Hypercube` are registered.
 */
template <size_t Dim>
using all_runtime_filters = tmpl::conditional_t<
    Dim == 3,
    tmpl::append<Filters::runtime::all_filters<
                     3, typename System<3>::variables_tag::tags_list>,
                 tmpl::list<Filters::runtime::SphericalShell<
                     typename System<3>::variables_tag::tags_list>>>,
    Filters::runtime::all_filters<
        Dim, typename System<Dim>::variables_tag::tags_list>>;
}  // namespace SoScalarWave

namespace Filters::runtime {
// No filter may modify the LDG auxiliary variable Phi (rebuilt from Psi
// every step with lifted boundary corrections) or the boundary-only
// evolved variable BoundaryPsi: restrict all filters to the leading
// scalars Psi and Pi.
template <>
struct FilteredTags<
    typename SoScalarWave::System<1>::variables_tag::tags_list> {
  using type = tmpl::list<SoScalarWave::Tags::Psi, SoScalarWave::Tags::Pi>;
};
template <>
struct FilteredTags<
    typename SoScalarWave::System<2>::variables_tag::tags_list> {
  using type = tmpl::list<SoScalarWave::Tags::Psi, SoScalarWave::Tags::Pi>;
};
template <>
struct FilteredTags<
    typename SoScalarWave::System<3>::variables_tag::tags_list> {
  using type = tmpl::list<SoScalarWave::Tags::Psi, SoScalarWave::Tags::Pi>;
};
}  // namespace Filters::runtime

namespace ylm::TensorYlm {
// Declarations of the SoScalarWave explicit specializations (defined in
// Evolution/Systems/SoScalarWave/SpectralFilter.cpp) so that translation
// units instantiating
// Filters::runtime::SphericalShell<
//     SoScalarWave::System<3>::variables_tag::tags_list>
// see them before use. Only Psi and Pi (the leading two scalar tags) are
// filtered; Phi and BoundaryPsi are passed through untouched.
template <>
void fill_tensor_ylm_filters<
    typename SoScalarWave::System<3>::variables_tag::tags_list>(
    gsl::not_null<FilterMatrixHolder*> matrix, size_t ell_max,
    size_t number_of_ell_modes_to_kill, std::optional<size_t> half_power,
    CoefficientNormalization coefficient_normalization);
template <>
void apply_tensor_ylm_filter(
    gsl::not_null<
        Variables<typename SoScalarWave::System<3>::variables_tag::tags_list>*>
        vars,
    gsl::not_null<
        Variables<typename SoScalarWave::System<3>::variables_tag::tags_list>*>
        temp_storage,
    const InverseJacobian<DataVector, 3, Frame::Inertial, Frame::Grid>&
        jac_inertial_to_grid,
    const InverseJacobian<DataVector, 3, Frame::Grid, Frame::Inertial>&
        jac_grid_to_inertial,
    const FilterMatrixHolder& filter_matrices, size_t ell_max,
    size_t radial_extents);
}  // namespace ylm::TensorYlm
