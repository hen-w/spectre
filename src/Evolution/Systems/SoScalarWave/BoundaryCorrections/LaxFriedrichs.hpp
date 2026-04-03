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
#include "Evolution/Systems/SoScalarWave/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "Options/String.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
template <size_t Dim>
class Direction;
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace SoScalarWave::BoundaryCorrections {
/*!
 * \brief A boundary correction class used for LDG collocation method.
 *
 * This boundary correction uses Lax-Friedrichs flux with a fixed coefficient
 * \tau1 and \tau2.
 *
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

  using dg_package_field_tags = tmpl::list<Tags::Pi, Tags::NormalDotPhi>;
  using dg_package_data_temporary_tags = tmpl::list<>;
  using dg_package_data_volume_tags = tmpl::list<>;
  using dg_boundary_terms_volume_tags = tmpl::list<>;

  double dg_package_data(
      gsl::not_null<Scalar<DataVector>*> packaged_pi,
      gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_phi,

      const Scalar<DataVector>& /*psi*/, const Scalar<DataVector>& pi,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& phi,

      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
      /*mesh_velocity*/,
      const std::optional<Scalar<DataVector>>& /*normal_dot_mesh_velocity*/,
      const Direction<Dim>& /*face_direction*/) const;

  void dg_boundary_terms(
      gsl::not_null<Scalar<DataVector>*> psi_boundary_correction,
      gsl::not_null<Scalar<DataVector>*> pi_boundary_correction,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
          phi_boundary_correction,

      const Scalar<DataVector>& pi_int,
      const Scalar<DataVector>& normal_dot_phi_int,

      const Scalar<DataVector>& pi_ext,
      const Scalar<DataVector>& normal_dot_phi_ext,

      dg::Formulation /*dg_formulation*/) const;

  using dg_auxiliary_package_field_tags =
      tmpl::list<Tags::Psi, Tags::PsiTimesNormal<Dim>>;
  using dg_auxiliary_package_data_temporary_tags = tmpl::list<>;
  using dg_auxiliary_package_data_volume_tags = tmpl::list<>;
  using dg_auxiliary_boundary_terms_volume_tags = tmpl::list<>;

  double dg_auxiliary_package_data(
      gsl::not_null<Scalar<DataVector>*> packaged_psi,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
          psi_times_normal,

      const Scalar<DataVector>& psi, const Scalar<DataVector>& /*pi*/,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& /*phi*/,

      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
      /*mesh_velocity*/,
      const std::optional<Scalar<DataVector>>& /*normal_dot_mesh_velocity*/,
      const Direction<Dim>& /*face_direction*/) const;

  void dg_auxiliary_boundary_terms(
      gsl::not_null<Scalar<DataVector>*> psi_boundary_correction,
      gsl::not_null<Scalar<DataVector>*> pi_boundary_correction,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
          phi_boundary_correction,

      const Scalar<DataVector>& psi_int,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& psi_times_normal_int,

      const Scalar<DataVector>& psi_ext,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& psi_times_normal_ext,

      dg::Formulation /*dg_formulation*/) const;

 private:
  double tau1_ = std::numeric_limits<double>::signaling_NaN();
  double tau2_ = std::numeric_limits<double>::signaling_NaN();
};
}  // namespace SoScalarWave::BoundaryCorrections
