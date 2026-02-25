// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <optional>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/BoundaryCorrection.hpp"
#include "Evolution/Systems/SoScalarWave/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "Options/String.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
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
 * \brief A boundary correction class used for CG collocation method.
 *
 * This boundary correction exchanges time derivatives between neighboring
 * elements and modifies the time derivatives on each side of the interface
 * by an weighted average.
 *
 */
template <size_t Dim>
class CgCollocation final : public evolution::BoundaryCorrection {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help = {
      "A boundary correction that enables the CG collocation method using DG "
      "infrastructure. "
      "Exchanges time derivatives between neighboring elements and modifies "
      "them by a weighted average."};

  CgCollocation() = default;
  CgCollocation(const CgCollocation&) = default;
  CgCollocation& operator=(const CgCollocation&) = default;
  CgCollocation(CgCollocation&&) = default;
  CgCollocation& operator=(CgCollocation&&) = default;
  ~CgCollocation() override = default;

  /// \cond
  explicit CgCollocation(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(CgCollocation);  // NOLINT
  /// \endcond
  void pup(PUP::er& p) override;  // NOLINT

  std::unique_ptr<BoundaryCorrection> get_clone() const override;

  using dg_package_field_tags =
      tmpl::list<::Tags::dt<Tags::Psi>, ::Tags::dt<Tags::Pi>>;

  using dg_package_data_temporary_tags =
      tmpl::list<::Tags::dt<Tags::Psi>, ::Tags::dt<Tags::Pi>>;

  using dg_package_data_volume_tags = tmpl::list<>;
  using dg_boundary_terms_volume_tags = tmpl::list<>;

  double dg_package_data(
      gsl::not_null<Scalar<DataVector>*> packaged_dt_psi,
      gsl::not_null<Scalar<DataVector>*> packaged_dt_pi,

      const Scalar<DataVector>& /*psi*/, const Scalar<DataVector>& /*pi*/,

      const Scalar<DataVector>& dt_psi, const Scalar<DataVector>& dt_pi,

      const tnsr::i<DataVector, Dim, Frame::Inertial>& /*normal_covector*/,
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
      /*mesh_velocity*/,
      const std::optional<Scalar<DataVector>>& /*normal_dot_mesh_velocity*/)
      const;

  void dg_boundary_terms(
      gsl::not_null<Scalar<DataVector>*> psi_boundary_correction,
      gsl::not_null<Scalar<DataVector>*> pi_boundary_correction,

      const Scalar<DataVector>& dt_psi_int, const Scalar<DataVector>& dt_pi_int,

      const Scalar<DataVector>& dt_psi_ext, const Scalar<DataVector>& dt_pi_ext,

      dg::Formulation /*dg_formulation*/,
      bool used_for_external_bc = false) const;
};
}  // namespace SoScalarWave::BoundaryCorrections
