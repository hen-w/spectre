// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cstddef>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ComputeTimeDerivativeHelpers.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/VolumeTermsImpl.tpp"
#include "Evolution/Systems/SoScalarWave/System.hpp"
#include "Evolution/Systems/SoScalarWave/Tags.hpp"
#include "Evolution/Systems/SoScalarWave/TimeDerivative.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/TMPL.hpp"

// Instantiate volume_terms and the partial_derivatives call used by the
// infrastructure. These must be instantiated with exactly the differentiation
// source `ComputeTimeDerivative` builds, so we reuse its shared definition
// (`vars_to_differentiate_tags`) rather than re-deriving it here.

namespace {
// Local shorthand keyed on `Dim`. Named distinctly from the shared
// `detail::vars_to_differentiate_tags` it forwards to: in an explicit
// instantiation of `detail::volume_terms`, an unqualified name matching one in
// `detail` would resolve to that (type-parameter) template and reject the
// integer `Dim`.
template <size_t Dim>
using differentiation_source_tags =
    evolution::dg::Actions::detail::vars_to_differentiate_tags<
        ::SoScalarWave::System<Dim>>;
}  // namespace

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(_, data)                                                \
  template void evolution::dg::Actions::detail::volume_terms<                 \
      ::SoScalarWave::TimeDerivative<DIM(data)>>(                             \
      const gsl::not_null<Variables<db::wrap_tags_in<                         \
          ::Tags::dt, typename ::SoScalarWave::System<DIM(                    \
                          data)>::variables_tag::tags_list>>*>                \
          dt_vars_ptr,                                                        \
      const gsl::not_null<Variables<db::wrap_tags_in<                         \
          ::Tags::Flux,                                                       \
          typename ::SoScalarWave::System<DIM(data)>::flux_variables,         \
          tmpl::size_t<DIM(data)>, Frame::Inertial>>*>                        \
          volume_fluxes,                                                      \
      const gsl::not_null<Variables<db::wrap_tags_in<                         \
          ::Tags::deriv,                                                      \
          typename ::SoScalarWave::System<DIM(data)>::gradient_variables,     \
          tmpl::size_t<DIM(data)>, Frame::Inertial>>*>                        \
          partial_derivs,                                                     \
      const gsl::not_null<Variables<typename ::SoScalarWave::System<DIM(      \
          data)>::compute_volume_time_derivative_terms::temporary_tags>*>     \
          temporaries,                                                        \
      const gsl::not_null<Variables<db::wrap_tags_in<                         \
          ::Tags::div,                                                        \
          db::wrap_tags_in<                                                   \
              ::Tags::Flux,                                                   \
              typename ::SoScalarWave::System<DIM(data)>::flux_variables,     \
              tmpl::size_t<DIM(data)>, Frame::Inertial>>>*>                   \
          div_fluxes,                                                         \
      const Variables<differentiation_source_tags<DIM(data)>>& evolved_vars,  \
      const ::dg::Formulation dg_formulation, const Mesh<DIM(data)>& mesh,    \
      [[maybe_unused]] const tnsr::I<DataVector, DIM(data), Frame::Inertial>& \
          inertial_coordinates,                                               \
      const InverseJacobian<DataVector, DIM(data), Frame::ElementLogical,     \
                            Frame::Inertial>&                                 \
          logical_to_inertial_inverse_jacobian,                               \
      [[maybe_unused]] const Scalar<DataVector>* const det_inverse_jacobian,  \
      const std::optional<tnsr::I<DataVector, DIM(data), Frame::Inertial>>&   \
          mesh_velocity,                                                      \
      const std::optional<Scalar<DataVector>>& div_mesh_velocity,             \
      const Scalar<DataVector>& pi,                                           \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>& phi,             \
      const Mesh<DIM(data)>& mesh_args,                                       \
      const InverseJacobian<DataVector, DIM(data), Frame::ElementLogical,     \
                            Frame::Inertial>&                                 \
          logical_to_inertial_inverse_jacobian_args,                          \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>&                  \
          inertial_coordinates_args,                                          \
      const double& time_args);                                               \
  template void partial_derivatives<                                          \
      db::wrap_tags_in<                                                       \
          ::Tags::deriv,                                                      \
          typename ::SoScalarWave::System<DIM(data)>::gradient_variables,     \
          tmpl::size_t<DIM(data)>, Frame::Inertial>,                          \
      differentiation_source_tags<DIM(data)>, DIM(data), Frame::Inertial>(    \
      gsl::not_null<Variables<db::wrap_tags_in<                               \
          ::Tags::deriv,                                                      \
          typename ::SoScalarWave::System<DIM(data)>::gradient_variables,     \
          tmpl::size_t<DIM(data)>, Frame::Inertial>>*>                        \
          du,                                                                 \
      const Variables<differentiation_source_tags<DIM(data)>>& u,             \
      const Mesh<DIM(data)>& mesh,                                            \
      const InverseJacobian<DataVector, DIM(data), Frame::ElementLogical,     \
                            Frame::Inertial>& inverse_jacobian,               \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>& inertial_coords);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
