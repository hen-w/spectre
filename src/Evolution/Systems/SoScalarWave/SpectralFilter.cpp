// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/SoScalarWave/SpectralFilter.hpp"

#include <cstddef>
#include <optional>
#include <type_traits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/SimpleSparseMatrix.hpp"
#include "DataStructures/Tensor/Structure.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/SpectralFilters.tpp"
#include "Evolution/Systems/SoScalarWave/System.hpp"
#include "Evolution/Systems/SoScalarWave/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Hypercube.tpp"
#include "NumericalAlgorithms/LinearOperators/Filters/None.tpp"
#include "NumericalAlgorithms/LinearOperators/Filters/SphericalShell.tpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackCache.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/TensorYlmFilter.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

#include "NumericalAlgorithms/SphericalHarmonics/ApplyTensorYlmFilter.tpp"

namespace {
using tags_1d = typename SoScalarWave::System<1>::variables_tag::tags_list;
using tags_2d = typename SoScalarWave::System<2>::variables_tag::tags_list;
using tags_3d = typename SoScalarWave::System<3>::variables_tag::tags_list;
using filtered_tags =
    tmpl::list<SoScalarWave::Tags::Psi, SoScalarWave::Tags::Pi>;

// The filter operates in place through a non-owning view over the leading
// block of the full evolved Variables, so the two filtered scalars (Psi, Pi)
// must be the first two tags of the evolved list, in the same order. The
// trailing Phi and BoundaryPsi variables are never touched.
template <typename FullList>
using leading_two_tags =
    tmpl::list<tmpl::at_c<FullList, 0>, tmpl::at_c<FullList, 1>>;
static_assert(std::is_same_v<filtered_tags, leading_two_tags<tags_3d>>,
              "The two filtered SoScalarWave variables (Psi, Pi) must be the "
              "leading tags of the full evolved variables list, in the order "
              "assumed by the filter.");
}  // namespace

// Specializations of the generic TensorYlm filter entry points
// (declared in NumericalAlgorithms/SphericalHarmonics/TensorYlmFilter.hpp)
// for the full SoScalarWave evolved-variable list. These are what
// Filters::runtime::SphericalShell<
//     SoScalarWave::System<3>::variables_tag::tags_list>
// calls for volume and boundary filtering. Scalars are frame-invariant, so
// no Jacobian frame transforms are performed and only the scalar filter
// matrix is needed.
namespace ylm::TensorYlm {

template <>
void fill_tensor_ylm_filters<tags_3d>(
    const gsl::not_null<FilterMatrixHolder*> matrix, const size_t ell_max,
    const size_t number_of_ell_modes_to_kill,
    const std::optional<size_t> half_power,
    const CoefficientNormalization coefficient_normalization) {
  const bool parameters_match =
      matrix->number_of_ell_modes_to_kill == number_of_ell_modes_to_kill and
      matrix->half_power == half_power and
      matrix->coefficient_normalization == coefficient_normalization;
  if (not parameters_match or not matrix->scalar.has_value()) {
    matrix->scalar = decltype(matrix->scalar)::value_type{};
    ylm::TensorYlm::fill_filter<Scalar<DataVector>::structure>(
        make_not_null(&matrix->scalar.value()), ell_max,
        number_of_ell_modes_to_kill, half_power, coefficient_normalization);
  }

  matrix->number_of_ell_modes_to_kill = number_of_ell_modes_to_kill;
  matrix->half_power = half_power;
  matrix->coefficient_normalization = coefficient_normalization;
}

template <>
void apply_tensor_ylm_filter(
    const gsl::not_null<Variables<tags_3d>*> vars,
    const gsl::not_null<Variables<tags_3d>*> temp_storage,
    const InverseJacobian<DataVector, 3, Frame::Inertial, Frame::Grid>&
    /*jac_inertial_to_grid*/,
    const InverseJacobian<DataVector, 3, Frame::Grid, Frame::Inertial>&
    /*jac_grid_to_inertial*/,
    const FilterMatrixHolder& filter_matrices, const size_t ell_max,
    const size_t radial_extents) {
  const auto& ylm = ::ylm::get_spherepack_cache(ell_max);
  ASSERT(radial_extents * ylm.physical_size() == vars->number_of_grid_points(),
         "Mismatch " << radial_extents * ylm.physical_size() << " must equal "
                     << vars->number_of_grid_points());
  ASSERT(filter_matrices.scalar.has_value(),
         "The scalar filter matrix must be filled by fill_tensor_ylm_filters "
         "before calling apply_tensor_ylm_filter for the SoScalarWave "
         "variables. Filled: scalar = "
             << filter_matrices.scalar.has_value());

  constexpr size_t number_of_filtered_components =
      Variables<filtered_tags>::number_of_independent_components;

  // Non-owning view over the leading two-scalar block (Psi, Pi) of vars: the
  // trailing Phi and BoundaryPsi variables are passed through untouched by
  // construction.
  Variables<filtered_tags> filtered_vars(
      vars->data(),
      number_of_filtered_components * vars->number_of_grid_points());
  // Non-owning view over temp_storage with EXACTLY radial_extents *
  // spectral_size grid points: the spectral scratch derives component strides
  // from the view's grid-point count, and the sparse filter matrices index a
  // component-major layout with that exact stride. temp_storage is sized for
  // all evolved components at >= this grid-point count (see the size contract
  // in TensorYlmFilter.hpp), so capacity always suffices.
  const size_t spectral_grid_points = radial_extents * ylm.spectral_size();
  ASSERT(temp_storage->size() >=
             number_of_filtered_components * spectral_grid_points,
         "The temp_storage buffer with "
             << temp_storage->size() << " doubles cannot hold the "
             << number_of_filtered_components * spectral_grid_points
             << " doubles of filtered-variable spectral scratch.");
  Variables<filtered_tags> spectral_vars(
      temp_storage->data(),
      number_of_filtered_components * spectral_grid_points);

  // 1. Nodal to modal transformation.
  // src: filtered_vars
  // dest: spectral_vars
  ylm::TensorYlm::filter_detail::nodal_to_modal_ylm(
      make_not_null(&spectral_vars), filtered_vars, ylm, radial_extents);

  // 2. Filter
  // src: spectral_vars
  // dest: spectral_vars
  // using filtered_vars as temp storage for each scalar
  tmpl::for_each<filtered_tags>([&spectral_vars, &filtered_vars, radial_extents,
                                 &filter_matrices]<class Tag>(
                                    const tmpl::type_<Tag> /*meta*/) {
    constexpr size_t num_independent_components = Tag::type::structure::size();
    ASSERT(spectral_vars.number_of_grid_points() * num_independent_components <=
               filtered_vars.size(),
           "Insufficient size: must have "
               << spectral_vars.number_of_grid_points() *
                      num_independent_components
               << " <= " << filtered_vars.size());

    Variables<tmpl::list<Tag>> dest_tensor(
        filtered_vars.data(),
        spectral_vars.number_of_grid_points() * num_independent_components);

    // Delta term
    get<Tag>(dest_tensor) = get<Tag>(spectral_vars);

    const gsl::span<double> src(
        get<Tag>(spectral_vars)[0].data(),
        num_independent_components * spectral_vars.number_of_grid_points());
    gsl::span<double> dest(
        get<Tag>(dest_tensor)[0].data(),
        num_independent_components * dest_tensor.number_of_grid_points());
    const size_t stride = radial_extents;
    for (size_t offset = 0; offset < stride; ++offset) {
      filter_matrices.scalar->increment_multiply_on_right(
          make_not_null(&dest), offset, stride, src, offset, stride);
    }
    // Copy the result for this scalar back into spectral_vars.
    get<Tag>(spectral_vars) = get<Tag>(dest_tensor);
  });

  // 3. Modal to nodal transformation, written back in place into vars.
  // src: spectral_vars
  // dest: filtered_vars
  ylm::TensorYlm::filter_detail::modal_to_nodal_ylm(
      make_not_null(&filtered_vars), spectral_vars, ylm, radial_extents);
}
}  // namespace ylm::TensorYlm

namespace ylm::TensorYlm::filter_detail {
YLM_TENSORYLM_INSTANTIATE_MODAL_NODAL_TRANSFORMS(filtered_tags);
}  // namespace ylm::TensorYlm::filter_detail

// Explicit instantiations of the runtime filter framework for each dimension.
template class Filters::runtime::Hypercube<1, tags_1d>;
template class Filters::runtime::None<1, tags_1d>;
template struct evolution::dg::Initialization::SpectralFilters<1, tags_1d>;
template class Filters::runtime::Hypercube<2, tags_2d>;
template class Filters::runtime::None<2, tags_2d>;
template struct evolution::dg::Initialization::SpectralFilters<2, tags_2d>;
template class Filters::runtime::Hypercube<3, tags_3d>;
template class Filters::runtime::None<3, tags_3d>;
template class Filters::runtime::SphericalShell<tags_3d>;
template struct evolution::dg::Initialization::SpectralFilters<3, tags_3d>;
