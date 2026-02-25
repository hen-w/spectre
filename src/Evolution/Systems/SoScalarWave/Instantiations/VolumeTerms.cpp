// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/VolumeTermsImpl.tpp"
#include "Evolution/Systems/SoScalarWave/System.hpp"
#include "Evolution/Systems/SoScalarWave/Tags.hpp"
#include "Evolution/Systems/SoScalarWave/TimeDerivative.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "Utilities/GenerateInstantiations.hpp"

// SoScalarWave computes derivatives internally, so we manually instantiate
// the specific partial_derivatives calls used in TimeDerivative.cpp

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(_, data)                                                \
  template void evolution::dg::Actions::detail::volume_terms<                 \
      ::SoScalarWave::TimeDerivative<DIM(data)>, true>(                       \
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
      const Variables<typename ::SoScalarWave::System<DIM(                    \
          data)>::variables_tag::tags_list>& evolved_vars,                    \
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
      const Variables<typename ::SoScalarWave::System<DIM(                    \
          data)>::variables_tag::tags_list>& evolved_vars_args,               \
      const Mesh<DIM(data)>& mesh_args,                                       \
      const InverseJacobian<DataVector, DIM(data), Frame::ElementLogical,     \
                            Frame::Inertial>&                                 \
          logical_to_inertial_inverse_jacobian_args);                         \
  template void partial_derivatives<                                          \
      tmpl::list<::Tags::deriv<SoScalarWave::Tags::Psi,                       \
                               tmpl::size_t<DIM(data)>, Frame::Inertial>,     \
                 ::Tags::deriv<SoScalarWave::Tags::Pi,                        \
                               tmpl::size_t<DIM(data)>, Frame::Inertial>>,    \
      tmpl::list<SoScalarWave::Tags::Psi, SoScalarWave::Tags::Pi>, DIM(data), \
      Frame::Inertial>(                                                       \
      gsl::not_null<Variables<tmpl::list<                                     \
          ::Tags::deriv<SoScalarWave::Tags::Psi, tmpl::size_t<DIM(data)>,     \
                        Frame::Inertial>,                                     \
          ::Tags::deriv<SoScalarWave::Tags::Pi, tmpl::size_t<DIM(data)>,      \
                        Frame::Inertial>>>*>                                  \
          du,                                                                 \
      const Variables<                                                        \
          tmpl::list<SoScalarWave::Tags::Psi, SoScalarWave::Tags::Pi>>& u,    \
      const Mesh<DIM(data)>& mesh,                                            \
      const InverseJacobian<DataVector, DIM(data), Frame::ElementLogical,     \
                            Frame::Inertial>& inverse_jacobian);              \
  template void partial_derivatives<                                          \
      tmpl::list<::Tags::deriv<                                               \
                     ::Tags::deriv<SoScalarWave::Tags::Psi,                   \
                                   tmpl::size_t<DIM(data)>, Frame::Inertial>, \
                     tmpl::size_t<DIM(data)>, Frame::Inertial>,               \
                 ::Tags::deriv<                                               \
                     ::Tags::deriv<SoScalarWave::Tags::Pi,                    \
                                   tmpl::size_t<DIM(data)>, Frame::Inertial>, \
                     tmpl::size_t<DIM(data)>, Frame::Inertial>>,              \
      tmpl::list<::Tags::deriv<SoScalarWave::Tags::Psi,                       \
                               tmpl::size_t<DIM(data)>, Frame::Inertial>,     \
                 ::Tags::deriv<SoScalarWave::Tags::Pi,                        \
                               tmpl::size_t<DIM(data)>, Frame::Inertial>>,    \
      DIM(data), Frame::Inertial>(                                            \
      gsl::not_null<Variables<tmpl::list<                                     \
          ::Tags::deriv<                                                      \
              ::Tags::deriv<SoScalarWave::Tags::Psi, tmpl::size_t<DIM(data)>, \
                            Frame::Inertial>,                                 \
              tmpl::size_t<DIM(data)>, Frame::Inertial>,                      \
          ::Tags::deriv<                                                      \
              ::Tags::deriv<SoScalarWave::Tags::Pi, tmpl::size_t<DIM(data)>,  \
                            Frame::Inertial>,                                 \
              tmpl::size_t<DIM(data)>, Frame::Inertial>>>*>                   \
          du,                                                                 \
      const Variables<tmpl::list<                                             \
          ::Tags::deriv<SoScalarWave::Tags::Psi, tmpl::size_t<DIM(data)>,     \
                        Frame::Inertial>,                                     \
          ::Tags::deriv<SoScalarWave::Tags::Pi, tmpl::size_t<DIM(data)>,      \
                        Frame::Inertial>>>& u,                                \
      const Mesh<DIM(data)>& mesh,                                            \
      const InverseJacobian<DataVector, DIM(data), Frame::ElementLogical,     \
                            Frame::Inertial>& inverse_jacobian);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
