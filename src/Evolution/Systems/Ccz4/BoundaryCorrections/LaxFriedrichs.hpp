// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>
#include <memory>
#include <optional>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/BoundaryCorrection.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "Options/String.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
template <size_t Dim>
class Direction;
template <size_t Dim>
class Mesh;
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace Ccz4::BoundaryCorrections {
/*!
 * \brief A boundary correction class used for LDG collocation method.
 *
 * This boundary correction uses Lax-Friedrichs flux with a fixed coefficient
 * \tau1 and \tau2.
 *
 * \note We assume gamma-driver condition is used. If lapse and shift are
 * freezed the SoCcz4 system is not strongly hyperbolic and it does not make
 * sense to use LDG method.
 */
template <size_t Dim>
class LaxFriedrichs final : public evolution::BoundaryCorrection {
 public:
  struct Tau1 {
    using type = double;
    static constexpr Options::String help = {
        "The penalty parameter tau1 for the Lax-Friedrichs numerical flux."};
  };
  struct Tau2 {
    using type = double;
    static constexpr Options::String help = {
        "The penalty parameter tau2 for the auxiliary numerical flux"};
  };

  using options = tmpl::list<Tau1, Tau2>;
  static constexpr Options::String help = {
      "A boundary correction that enables the LDG method using DG "
      "infrastructure. "};

  LaxFriedrichs() = default;
  explicit LaxFriedrichs(double tau1, double tau2);
  LaxFriedrichs(const LaxFriedrichs&) = default;
  LaxFriedrichs& operator=(const LaxFriedrichs&) = default;
  LaxFriedrichs(LaxFriedrichs&&) = default;
  LaxFriedrichs& operator=(LaxFriedrichs&&) = default;
  ~LaxFriedrichs() override = default;

  /// \cond
  explicit LaxFriedrichs(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(LaxFriedrichs);  // NOLINT
  /// \endcond
  void pup(PUP::er& p) override;  // NOLINT

  std::unique_ptr<BoundaryCorrection> get_clone() const override;

  using dg_package_field_tags = tmpl::list<
      // evolved variables
      ::Ccz4::Tags::ConformalMetric<DataVector, 3>,
      ::Ccz4::Tags::ConformalFactor<DataVector>,
      ::Ccz4::Tags::ATilde<DataVector, 3>,
      gr::Tags::TraceExtrinsicCurvature<DataVector>,
      ::Ccz4::Tags::Theta<DataVector>, ::Ccz4::Tags::GammaHat<DataVector, 3>,
      gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>,
      ::Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>,
      // auxiliary reduction variables
      ::Ccz4::Tags::FieldA<DataVector, 3>, ::Ccz4::Tags::FieldB<DataVector, 3>,
      ::Ccz4::Tags::FieldD<DataVector, 3>, ::Ccz4::Tags::FieldP<DataVector, 3>,
      // boundary mode variables
      ::Ccz4::fd::Tags::UTensorMinus<DataVector, 3, Frame::Inertial>,
      // boundary second-order fields
      ::Ccz4::Tags::BoundaryConformalMetric<DataVector, 3>,
      ::Ccz4::Tags::BoundaryConformalFactor<DataVector>,
      ::Ccz4::Tags::BoundaryLapse<DataVector>,
      ::Ccz4::Tags::BoundaryShift<DataVector, 3>,
      ::Ccz4::Tags::BoundaryTheta<DataVector>,
      ::Ccz4::Tags::BoundaryZ<DataVector, 3, Frame::Inertial>,
      // normal covector and inverse grid spacing
      ::Ccz4::Tags::NormalCovector<DataVector, 3>,
      ::Ccz4::Tags::InverseGridSpacing<DataVector>>;
  using dg_package_data_temporary_tags = tmpl::list<>;
  using dg_package_data_volume_tags =
      tmpl::list<domain::Tags::Mesh<Dim>,
                 domain::Tags::Coordinates<Dim, Frame::Inertial>>;
  using dg_boundary_terms_volume_tags = tmpl::list<>;

  double dg_package_data(
      gsl::not_null<tnsr::ii<DataVector, Dim, Frame::Inertial>*>
          packaged_conformal_metric,
      gsl::not_null<Scalar<DataVector>*> packaged_conformal_factor,
      gsl::not_null<tnsr::ii<DataVector, Dim, Frame::Inertial>*>
          packaged_a_tilde,
      gsl::not_null<Scalar<DataVector>*> packaged_trace_extrinsic_curvature,
      gsl::not_null<Scalar<DataVector>*> packaged_theta,
      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          packaged_gamma_hat,
      gsl::not_null<Scalar<DataVector>*> packaged_lapse,
      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> packaged_shift,
      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          packaged_auxiliary_shift_b,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
          packaged_field_a,
      gsl::not_null<tnsr::iJ<DataVector, Dim, Frame::Inertial>*>
          packaged_field_b,
      gsl::not_null<tnsr::ijj<DataVector, Dim, Frame::Inertial>*>
          packaged_field_d,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
          packaged_field_p,
      gsl::not_null<tnsr::ii<DataVector, Dim, Frame::Inertial>*>
          packaged_u_tensor_minus,
      gsl::not_null<tnsr::ii<DataVector, Dim, Frame::Inertial>*>
          packaged_boundary_conformal_metric,
      gsl::not_null<Scalar<DataVector>*> packaged_boundary_conformal_factor,
      gsl::not_null<Scalar<DataVector>*> packaged_boundary_lapse,
      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          packaged_boundary_shift,
      gsl::not_null<Scalar<DataVector>*> packaged_boundary_theta,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
          packaged_boundary_z,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
          packaged_normal_covector,
      gsl::not_null<Scalar<DataVector>*> packaged_inverse_grid_spacing,

      const tnsr::ii<DataVector, Dim, Frame::Inertial>& conformal_metric,
      const Scalar<DataVector>& conformal_factor,
      const tnsr::ii<DataVector, Dim, Frame::Inertial>& a_tilde,
      const Scalar<DataVector>& trace_extrinsic_curvature,
      const Scalar<DataVector>& theta,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& gamma_hat,
      const Scalar<DataVector>& lapse,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& shift,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& auxiliary_shift_b,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& field_a,
      const tnsr::iJ<DataVector, Dim, Frame::Inertial>& field_b,
      const tnsr::ijj<DataVector, Dim, Frame::Inertial>& field_d,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& field_p,

      const tnsr::ii<DataVector, Dim, Frame::Inertial>& /*u_tensor_minus*/,

      const tnsr::ii<DataVector, Dim, Frame::Inertial>&
      /*boundary_conformal_metric*/,
      const Scalar<DataVector>& /*boundary_conformal_factor*/,
      const Scalar<DataVector>& /*boundary_lapse*/,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& /*boundary_shift*/,
      const Scalar<DataVector>& /*boundary_theta*/,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& /*boundary_z*/,

      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
      /*mesh_velocity*/,
      const std::optional<Scalar<DataVector>>& /*normal_dot_mesh_velocity*/,
      const Direction<Dim>& face_direction, const Mesh<Dim>& volume_mesh,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& volume_coords) const;

  void dg_boundary_terms(
      gsl::not_null<tnsr::ii<DataVector, Dim, Frame::Inertial>*>
          conformal_metric_boundary_correction,
      gsl::not_null<Scalar<DataVector>*> conformal_factor_boundary_correction,
      gsl::not_null<tnsr::ii<DataVector, Dim, Frame::Inertial>*>
          a_tilde_boundary_correction,
      gsl::not_null<Scalar<DataVector>*>
          trace_extrinsic_curvature_boundary_correction,
      gsl::not_null<Scalar<DataVector>*> theta_boundary_correction,
      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          gamma_hat_boundary_correction,
      gsl::not_null<Scalar<DataVector>*> lapse_boundary_correction,
      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          shift_boundary_correction,
      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          auxiliary_shift_b_boundary_correction,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
          field_a_boundary_correction,
      gsl::not_null<tnsr::iJ<DataVector, Dim, Frame::Inertial>*>
          field_b_boundary_correction,
      gsl::not_null<tnsr::ijj<DataVector, Dim, Frame::Inertial>*>
          field_d_boundary_correction,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
          field_p_boundary_correction,
      gsl::not_null<tnsr::ii<DataVector, Dim, Frame::Inertial>*>
          u_tensor_minus_boundary_correction,
      gsl::not_null<tnsr::ii<DataVector, Dim, Frame::Inertial>*>
          boundary_conformal_metric_boundary_correction,
      gsl::not_null<Scalar<DataVector>*>
          boundary_conformal_factor_boundary_correction,
      gsl::not_null<Scalar<DataVector>*> boundary_lapse_boundary_correction,
      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          boundary_shift_boundary_correction,
      gsl::not_null<Scalar<DataVector>*> boundary_theta_boundary_correction,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
          boundary_z_boundary_correction,

      const tnsr::ii<DataVector, Dim, Frame::Inertial>& conformal_metric_int,
      const Scalar<DataVector>& conformal_factor_int,
      const tnsr::ii<DataVector, Dim, Frame::Inertial>& a_tilde_int,
      const Scalar<DataVector>& trace_extrinsic_curvature_int,
      const Scalar<DataVector>& theta_int,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& gamma_hat_int,
      const Scalar<DataVector>& lapse_int,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& shift_int,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& auxiliary_shift_b_int,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& field_a_int,
      const tnsr::iJ<DataVector, Dim, Frame::Inertial>& field_b_int,
      const tnsr::ijj<DataVector, Dim, Frame::Inertial>& field_d_int,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& field_p_int,
      const tnsr::ii<DataVector, Dim, Frame::Inertial>& /*u_tensor_minus_int*/,
      const tnsr::ii<DataVector, Dim, Frame::Inertial>&
      /*boundary_conformal_metric_int*/,
      const Scalar<DataVector>& /*boundary_conformal_factor_int*/,
      const Scalar<DataVector>& /*boundary_lapse_int*/,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& /*boundary_shift_int*/,
      const Scalar<DataVector>& /*boundary_theta_int*/,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& /*boundary_z_int*/,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector_int,
      const Scalar<DataVector>& inverse_grid_spacing_int,

      const tnsr::ii<DataVector, Dim, Frame::Inertial>& conformal_metric_ext,
      const Scalar<DataVector>& conformal_factor_ext,
      const tnsr::ii<DataVector, Dim, Frame::Inertial>& a_tilde_ext,
      const Scalar<DataVector>& trace_extrinsic_curvature_ext,
      const Scalar<DataVector>& theta_ext,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& gamma_hat_ext,
      const Scalar<DataVector>& lapse_ext,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& shift_ext,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& auxiliary_shift_b_ext,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& field_a_ext,
      const tnsr::iJ<DataVector, Dim, Frame::Inertial>& field_b_ext,
      const tnsr::ijj<DataVector, Dim, Frame::Inertial>& field_d_ext,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& field_p_ext,
      const tnsr::ii<DataVector, Dim, Frame::Inertial>& /*u_tensor_minus_ext*/,
      const tnsr::ii<DataVector, Dim, Frame::Inertial>&
      /*boundary_conformal_metric_ext*/,
      const Scalar<DataVector>& /*boundary_conformal_factor_ext*/,
      const Scalar<DataVector>& /*boundary_lapse_ext*/,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& /*boundary_shift_ext*/,
      const Scalar<DataVector>& /*boundary_theta_ext*/,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& /*boundary_z_ext*/,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector_ext,
      const Scalar<DataVector>& inverse_grid_spacing_ext,

      dg::Formulation /*dg_formulation*/) const;

  using dg_auxiliary_package_field_tags = tmpl::list<
      Ccz4::Tags::ConformalMetric<DataVector, 3>,
      Ccz4::Tags::ConformalFactor<DataVector>, gr::Tags::Lapse<DataVector>,
      gr::Tags::Shift<DataVector, 3>, Ccz4::Tags::NormalCovector<DataVector, 3>,
      Ccz4::Tags::InverseGridSpacing<DataVector>,
      Ccz4::Tags::FieldA<DataVector, 3>, Ccz4::Tags::FieldB<DataVector, 3>,
      Ccz4::Tags::FieldD<DataVector, 3>, Ccz4::Tags::FieldP<DataVector, 3>>;
  using dg_auxiliary_package_data_temporary_tags = tmpl::list<>;
  using dg_auxiliary_package_data_volume_tags =
      tmpl::list<domain::Tags::Mesh<Dim>,
                 domain::Tags::Coordinates<Dim, Frame::Inertial>>;
  using dg_auxiliary_boundary_terms_volume_tags = tmpl::list<>;

  double dg_auxiliary_package_data(
      gsl::not_null<tnsr::ii<DataVector, Dim, Frame::Inertial>*>
          packaged_conformal_metric,
      gsl::not_null<Scalar<DataVector>*> packaged_conformal_factor,
      gsl::not_null<Scalar<DataVector>*> packaged_lapse,
      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> packaged_shift,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
          packaged_normal_covector,
      gsl::not_null<Scalar<DataVector>*> packaged_inverse_grid_spacing,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
          packaged_field_a,
      gsl::not_null<tnsr::iJ<DataVector, Dim, Frame::Inertial>*>
          packaged_field_b,
      gsl::not_null<tnsr::ijj<DataVector, Dim, Frame::Inertial>*>
          packaged_field_d,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
          packaged_field_p,

      const tnsr::ii<DataVector, Dim, Frame::Inertial>& conformal_metric,
      const Scalar<DataVector>& conformal_factor,
      const tnsr::ii<DataVector, Dim, Frame::Inertial>& /*a_tilde*/,
      const Scalar<DataVector>& /*trace_extrinsic_curvature*/,
      const Scalar<DataVector>& /*theta*/,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& /*gamma_hat*/,
      const Scalar<DataVector>& lapse,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& shift,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& /*auxiliary_shift_b*/,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& field_a,
      const tnsr::iJ<DataVector, Dim, Frame::Inertial>& field_b,
      const tnsr::ijj<DataVector, Dim, Frame::Inertial>& field_d,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& field_p,

      const tnsr::ii<DataVector, Dim, Frame::Inertial>& /*u_tensor_minus*/,

      const tnsr::ii<DataVector, Dim, Frame::Inertial>&
      /*boundary_conformal_metric*/,
      const Scalar<DataVector>& /*boundary_conformal_factor*/,
      const Scalar<DataVector>& /*boundary_lapse*/,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& /*boundary_shift*/,
      const Scalar<DataVector>& /*boundary_theta*/,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& /*boundary_z*/,

      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
      /*mesh_velocity*/,
      const std::optional<Scalar<DataVector>>& /*normal_dot_mesh_velocity*/,
      const Direction<Dim>& face_direction, const Mesh<Dim>& volume_mesh,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& volume_coords) const;

  void dg_auxiliary_boundary_terms(
      gsl::not_null<tnsr::ii<DataVector, Dim, Frame::Inertial>*>
          conformal_metric_boundary_correction,
      gsl::not_null<Scalar<DataVector>*> conformal_factor_boundary_correction,
      gsl::not_null<tnsr::ii<DataVector, Dim, Frame::Inertial>*>
          a_tilde_boundary_correction,
      gsl::not_null<Scalar<DataVector>*>
          trace_extrinsic_curvature_boundary_correction,
      gsl::not_null<Scalar<DataVector>*> theta_boundary_correction,
      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          gamma_hat_boundary_correction,
      gsl::not_null<Scalar<DataVector>*> lapse_boundary_correction,
      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          shift_boundary_correction,
      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          auxiliary_shift_b_boundary_correction,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
          field_a_boundary_correction,
      gsl::not_null<tnsr::iJ<DataVector, Dim, Frame::Inertial>*>
          field_b_boundary_correction,
      gsl::not_null<tnsr::ijj<DataVector, Dim, Frame::Inertial>*>
          field_d_boundary_correction,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
          field_p_boundary_correction,
      gsl::not_null<tnsr::ii<DataVector, Dim, Frame::Inertial>*>
          u_tensor_minus_boundary_correction,
      gsl::not_null<tnsr::ii<DataVector, Dim, Frame::Inertial>*>
          boundary_conformal_metric_boundary_correction,
      gsl::not_null<Scalar<DataVector>*>
          boundary_conformal_factor_boundary_correction,
      gsl::not_null<Scalar<DataVector>*> boundary_lapse_boundary_correction,
      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          boundary_shift_boundary_correction,
      gsl::not_null<Scalar<DataVector>*> boundary_theta_boundary_correction,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
          boundary_z_boundary_correction,

      const tnsr::ii<DataVector, Dim, Frame::Inertial>& conformal_metric_int,
      const Scalar<DataVector>& conformal_factor_int,
      const Scalar<DataVector>& lapse_int,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& shift_int,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector_int,
      const Scalar<DataVector>& inverse_grid_spacing_int,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& field_a_int,
      const tnsr::iJ<DataVector, Dim, Frame::Inertial>& field_b_int,
      const tnsr::ijj<DataVector, Dim, Frame::Inertial>& field_d_int,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& field_p_int,

      const tnsr::ii<DataVector, Dim, Frame::Inertial>& conformal_metric_ext,
      const Scalar<DataVector>& conformal_factor_ext,
      const Scalar<DataVector>& lapse_ext,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& shift_ext,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector_ext,
      const Scalar<DataVector>& inverse_grid_spacing_ext,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& field_a_ext,
      const tnsr::iJ<DataVector, Dim, Frame::Inertial>& field_b_ext,
      const tnsr::ijj<DataVector, Dim, Frame::Inertial>& field_d_ext,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& field_p_ext,

      dg::Formulation /*dg_formulation*/) const;

 private:
  double tau1_ = std::numeric_limits<double>::signaling_NaN();
  double tau2_ = std::numeric_limits<double>::signaling_NaN();
};
}  // namespace Ccz4::BoundaryCorrections
