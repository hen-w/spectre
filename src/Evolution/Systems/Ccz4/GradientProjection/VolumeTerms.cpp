// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <optional>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/VolumeTermsImpl.tpp"
#include "Evolution/Systems/Ccz4/GradientProjection/System.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Tags.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/TimeDerivative.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
using system = Ccz4::GradientProjection::System;
}  // namespace

template void evolution::dg::Actions::detail::volume_terms<
    Ccz4::GradientProjection::TimeDerivative>(
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
    const Ccz4::GradientProjection::Tags::ChartU::type& u,
    const Ccz4::GradientProjection::Tags::ChartW::type& w,
    const Ccz4::GradientProjection::Tags::ChartQ::type& q,
    const Scalar<DataVector>& kappa_1, const Scalar<DataVector>& kappa_2,
    const double& kappa_3, const Scalar<DataVector>& eta,
    const Scalar<DataVector>& k_0);
