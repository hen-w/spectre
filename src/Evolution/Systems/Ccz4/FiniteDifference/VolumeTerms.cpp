// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/Creators/Tags/ExternalBoundaryConditions.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/VolumeTermsImpl.tpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/LdgTimeDerivative.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace {
using system = Ccz4::fd::System;
}  // namespace

template void
evolution::dg::Actions::detail::volume_terms<Ccz4::fd::LdgTimeDerivative>(
    const gsl::not_null<Variables<db::wrap_tags_in<
        ::Tags::dt, typename system::variables_tag::tags_list>>*>
        dt_vars_ptr,
    const gsl::not_null<Variables<
        db::wrap_tags_in<::Tags::Flux, typename system::flux_variables,
                         tmpl::size_t<3>, Frame::Inertial>>*>
        volume_fluxes,
    const gsl::not_null<Variables<
        db::wrap_tags_in<::Tags::deriv, typename system::gradient_variables,
                         tmpl::size_t<3>, Frame::Inertial>>*>
        partial_derivs,
    const gsl::not_null<Variables<
        typename system::compute_volume_time_derivative_terms::temporary_tags>*>
        temporaries,
    const gsl::not_null<Variables<db::wrap_tags_in<
        ::Tags::div,
        db::wrap_tags_in<::Tags::Flux, typename system::flux_variables,
                         tmpl::size_t<3>, Frame::Inertial>>>*>
        div_fluxes,
    const Variables<typename system::variables_tag::tags_list>& evolved_vars,
    const ::dg::Formulation dg_formulation, const Mesh<3>& mesh,
    const tnsr::I<DataVector, 3, Frame::Inertial>& inertial_coordinates,
    const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                          Frame::Inertial>&
        logical_to_inertial_inverse_jacobian,
    const Scalar<DataVector>* det_inverse_jacobian,
    const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>& mesh_velocity,
    const std::optional<Scalar<DataVector>>& div_mesh_velocity,
    // argument_tags expanded:
    const tnsr::ii<DataVector, 3, Frame::Inertial>& conformal_metric,
    const Scalar<DataVector>& conformal_factor,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& a_tilde,
    const Scalar<DataVector>& trace_extrinsic_curvature,
    const Scalar<DataVector>& theta,
    const tnsr::I<DataVector, 3, Frame::Inertial>& gamma_hat,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
    const tnsr::I<DataVector, 3, Frame::Inertial>& auxiliary_shift_b,
    const tnsr::i<DataVector, 3, Frame::Inertial>& field_a,
    const tnsr::iJ<DataVector, 3, Frame::Inertial>& field_b,
    const tnsr::ijj<DataVector, 3, Frame::Inertial>& field_d,
    const tnsr::i<DataVector, 3, Frame::Inertial>& field_p,
    const double& kappa_1, const double& kappa_2, const double& kappa_3,
    const Scalar<DataVector>& eta, const Scalar<DataVector>& k_0,
    const bool& evolve_lapse_and_shift, const Element<3>& element,
    const Mesh<3>& mesh_arg,
    const std::vector<DirectionMap<
        3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>&
        all_boundary_conditions,
    const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                          Frame::Inertial>& inv_jacobian);

template void partial_derivatives<
    db::wrap_tags_in<::Tags::deriv, typename system::gradient_variables,
                     tmpl::size_t<3>, Frame::Inertial>,
    typename system::variables_tag::tags_list, 3, Frame::Inertial>(
    gsl::not_null<Variables<
        db::wrap_tags_in<::Tags::deriv, typename system::gradient_variables,
                         tmpl::size_t<3>, Frame::Inertial>>*>
        du,
    const Variables<typename system::variables_tag::tags_list>& u,
    const Mesh<3>& mesh,
    const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                          Frame::Inertial>& inverse_jacobian,
    const tnsr::I<DataVector, 3, Frame::Inertial>& inertial_coords);

// Instantiation for the 3-arg overload used by
// OverwriteExternalBoundaryDtDirichlet
template auto partial_derivatives<typename system::variables_tag::tags_list,
                                  typename system::variables_tag::tags_list, 3,
                                  Frame::Inertial>(
    const Variables<typename system::variables_tag::tags_list>& u,
    const Mesh<3>& mesh,
    const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                          Frame::Inertial>& inverse_jacobian)
    -> Variables<db::wrap_tags_in<::Tags::deriv,
                                  typename system::variables_tag::tags_list,
                                  tmpl::size_t<3>, Frame::Inertial>>;
