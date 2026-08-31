// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cstddef>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ComputeTimeDerivativeHelpers.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/VolumeTermsImpl.tpp"
#include "Evolution/Systems/SecondOrderScalarWave/System.hpp"
#include "Evolution/Systems/SecondOrderScalarWave/Tags.hpp"
#include "Evolution/Systems/SecondOrderScalarWave/TimeDerivative.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/TMPL.hpp"

// Instantiate volume_terms and the partial_derivatives call used by the
// infrastructure. These must be instantiated with exactly the differentiation
// source `ComputeTimeDerivative` builds, so we reuse its shared definition
// (`evolved_and_auxiliary_vars_tags`) rather than re-deriving it here. The
// system's variables_tag is a tmpl::list (split volume/boundary variables);
// the DG actions operate on the volume entry, so it is what these
// instantiations are keyed on.

namespace {
template <size_t Dim>
using volume_variables_tag = ::Tags::Variables<
    typename ::SecondOrderScalarWave::System<Dim>::volume_vars>;

// Local shorthand keyed on `Dim`. Named distinctly from the shared
// `detail::evolved_and_auxiliary_vars_tags` it forwards to: in an explicit
// instantiation of `detail::volume_terms`, an unqualified name matching one in
// `detail` would resolve to that (type-parameter) template and reject the
// integer `Dim`.
template <size_t Dim>
using differentiation_source_tags =
    evolution::dg::Actions::detail::evolved_and_auxiliary_vars_tags<
        ::SecondOrderScalarWave::System<Dim>, volume_variables_tag<Dim>>;
}  // namespace

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(_, data)                                                 \
  template void evolution::dg::Actions::detail::volume_terms<                  \
      ::SecondOrderScalarWave::TimeDerivative<DIM(data)>>(                     \
      const gsl::not_null<Variables<db::wrap_tags_in<                          \
          ::Tags::dt,                                                          \
          typename ::SecondOrderScalarWave::System<DIM(data)>::volume_vars>>*> \
          dt_vars_ptr,                                                         \
      const gsl::not_null<Variables<db::wrap_tags_in<                          \
          ::Tags::Flux,                                                        \
          typename ::SecondOrderScalarWave::System<DIM(data)>::flux_variables, \
          tmpl::size_t<DIM(data)>, Frame::Inertial>>*>                         \
          volume_fluxes,                                                       \
      const gsl::not_null<Variables<                                           \
          db::wrap_tags_in<::Tags::deriv,                                      \
                           typename ::SecondOrderScalarWave::System<DIM(       \
                               data)>::gradient_variables,                     \
                           tmpl::size_t<DIM(data)>, Frame::Inertial>>*>        \
          partial_derivs,                                                      \
      const gsl::not_null<Variables<typename ::SecondOrderScalarWave::System<  \
          DIM(data)>::compute_volume_time_derivative_terms::temporary_tags>*>  \
          temporaries,                                                         \
      const gsl::not_null<Variables<db::wrap_tags_in<                          \
          ::Tags::div,                                                         \
          db::wrap_tags_in<::Tags::Flux,                                       \
                           typename ::SecondOrderScalarWave::System<DIM(       \
                               data)>::flux_variables,                         \
                           tmpl::size_t<DIM(data)>, Frame::Inertial>>>*>       \
          div_fluxes,                                                          \
      const Variables<differentiation_source_tags<DIM(data)>>& evolved_vars,   \
      const ::dg::Formulation dg_formulation, const Mesh<DIM(data)>& mesh,     \
      [[maybe_unused]] const tnsr::I<DataVector, DIM(data), Frame::Inertial>&  \
          inertial_coordinates,                                                \
      const InverseJacobian<DataVector, DIM(data), Frame::ElementLogical,      \
                            Frame::Inertial>&                                  \
          logical_to_inertial_inverse_jacobian,                                \
      [[maybe_unused]] const Scalar<DataVector>* const det_inverse_jacobian,   \
      const std::optional<tnsr::I<DataVector, DIM(data), Frame::Inertial>>&    \
          mesh_velocity,                                                       \
      const std::optional<Scalar<DataVector>>& div_mesh_velocity,              \
      const Scalar<DataVector>& pi);                                           \
  template void partial_derivatives<                                           \
      db::wrap_tags_in<::Tags::deriv,                                          \
                       typename ::SecondOrderScalarWave::System<DIM(           \
                           data)>::gradient_variables,                         \
                       tmpl::size_t<DIM(data)>, Frame::Inertial>,              \
      differentiation_source_tags<DIM(data)>, DIM(data), Frame::Inertial>(     \
      gsl::not_null<Variables<                                                 \
          db::wrap_tags_in<::Tags::deriv,                                      \
                           typename ::SecondOrderScalarWave::System<DIM(       \
                               data)>::gradient_variables,                     \
                           tmpl::size_t<DIM(data)>, Frame::Inertial>>*>        \
          du,                                                                  \
      const Variables<differentiation_source_tags<DIM(data)>>& u,              \
      const Mesh<DIM(data)>& mesh,                                             \
      const InverseJacobian<DataVector, DIM(data), Frame::ElementLogical,      \
                            Frame::Inertial>& inverse_jacobian,                \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>& inertial_coords);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
