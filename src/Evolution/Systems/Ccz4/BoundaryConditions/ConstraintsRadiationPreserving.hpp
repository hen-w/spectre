// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <pup.h>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/BoundaryConditions/Type.hpp"
#include "Evolution/DgSubcell/GhostZoneLogicalCoordinates.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Systems/Ccz4/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Evolution/TypeTraits.hpp"
#include "Options/String.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Tags {
struct Time;
}  // namespace Tags
/// \endcond

namespace Ccz4::BoundaryConditions {
/*!
 * \brief Set constraints and radiation preserving
 * boundary conditions (CRPBC) for SoCcz4.
 *
 * Unlike time-independent subcell external boundary conditions,
 * the CRPBC is applied at the outermost
 * interior points in the volume instead of the ghost zone.
 * This is because the CRPBC requires spatial derivatives
 * of the fields and modify their time derivatives, which then need to
 * be time integrated. Under current infrastructure, it is therefore
 * the simplest to apply this boundary condition in the interior
 * rather than the ghost zone. This file extrapolates the interior
 * evolved variables into the ghost zone. Their spatial derivatives and
 * time derivatives are then computed and applied in SoTimeDerivative.hpp for
 * the outermost interior points.
 *
 * The CRPBC is designed to preserve the incoming constraint and radiation
 * characteristics at the boundary; see \ref constraint_characteristic_fields()
 * and \ref radiation_characteristic_fields(). We control the
 * constraint characteristic fields $C_i^{-\beta^n}$ via the main
 * characteristic field $U_i^{-\alpha-\beta^n}$
 * and similarly $C^{\pm\alpha-\beta^n}$ via $U_{(1)}^{\pm\alpha-\beta^n}$
 * or $U_{(2)}^{\pm\alpha-\beta^n}$. See \ref characteristic_fields()
 * for the definition of the main characteristic fields. We also control
 * the gravitatonal radiation characteristics $C_{ij}^{\pm\alpha-\beta^n}$,
 * which are proportional to Newman-Penrose quantity $\Psi_4$ and $\Psi_0$
 * respectively, via the normal derivative
 * $\partial_n U_{ij}^{\pm\alpha-\beta^n}$. For the rest of the incoming
 * main characteristic fields (gauge), we simply apply Sommerfeld
 * boundary conditions.
 *
 * \warning This boundary condition assumes a complete sphere domain
 * (all wedges), as we only apply it on the \ref upper_zeta
 * direction in blocks with external boundaries.
 *
 * \note This file has exactly the same implementation as Sommerfeld.hpp
 * since they only serve to extrapolate into the ghost zone. The actual
 * difference in the boundary condition is applied in SoTimeDerivative.hpp,
 * which needs this spearate class from Sommerfeld to identify the correct
 * boundary condition to impose.
 */
class ConstraintsRadiationPreserving final : public BoundaryCondition {
 public:
  /// \brief What extrapolation order to use to extrapolate
  /// into the ghost zone.
  struct ExtrapolationOrder {
    static constexpr Options::String help =
        "What extrapolation order to use to extrapolate into the ghost zone.";
    using type = size_t;
    static type lower_bound() { return 1; }
    static type upper_bound() { return 3; }
  };
  using options = tmpl::list<ExtrapolationOrder>;
  static constexpr Options::String help{
      "Constraints and radiation preserving boundary conditions."};

  ConstraintsRadiationPreserving() = default;
  ConstraintsRadiationPreserving(ConstraintsRadiationPreserving&&) = default;
  ConstraintsRadiationPreserving& operator=(ConstraintsRadiationPreserving&&) =
      default;
  ConstraintsRadiationPreserving(const ConstraintsRadiationPreserving&);
  ConstraintsRadiationPreserving& operator=(
      const ConstraintsRadiationPreserving&);
  ~ConstraintsRadiationPreserving() override = default;

  explicit ConstraintsRadiationPreserving(CkMigrateMessage* msg);

  explicit ConstraintsRadiationPreserving(size_t extrapolation_order);

  WRAPPED_PUPable_decl_base_template(
      domain::BoundaryConditions::BoundaryCondition,
      ConstraintsRadiationPreserving);

  auto get_clone() const -> std::unique_ptr<
      domain::BoundaryConditions::BoundaryCondition> override;

  static constexpr evolution::BoundaryConditions::Type bc_type =
      evolution::BoundaryConditions::Type::Ghost;

  void pup(PUP::er& p) override;

  using fd_interior_evolved_variables_tags =
      ::Ccz4::fd::System::variables_tag_list;
  using fd_interior_temporary_tags =
      tmpl::list<evolution::dg::subcell::Tags::Mesh<3>>;
  using fd_interior_primitive_variables_tags = tmpl::list<>;
  using fd_gridless_tags = tmpl::list<::Ccz4::fd::Tags::Reconstructor, ::Ccz4::fd::Tags::EvolveLapseAndShift>;
  void fd_ghost(
      gsl::not_null<tnsr::ii<DataVector, 3, Frame::Inertial>*> conformal_metric,
      gsl::not_null<Scalar<DataVector>*> lapse,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> shift,
      gsl::not_null<Scalar<DataVector>*> conformal_factor,
      gsl::not_null<tnsr::ii<DataVector, 3, Frame::Inertial>*> a_tilde,
      gsl::not_null<Scalar<DataVector>*> trace_extrinsic_curvature,
      gsl::not_null<Scalar<DataVector>*> theta,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> gamma_hat,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> auxiliary_shift_b,
      const Direction<3>& direction,

      // fd_interior_evolved_variables_tags (variables_tag_list order)
      const tnsr::ii<DataVector, 3, Frame::Inertial>& interior_conformal_metric,
      const Scalar<DataVector>& interior_conformal_factor,
      const tnsr::ii<DataVector, 3, Frame::Inertial>& interior_a_tilde,
      const Scalar<DataVector>& interior_trace_extrinsic_curvature,
      const Scalar<DataVector>& interior_theta,
      const tnsr::I<DataVector, 3, Frame::Inertial>& interior_gamma_hat,
      const Scalar<DataVector>& interior_lapse,
      const tnsr::I<DataVector, 3, Frame::Inertial>& interior_shift,
      const tnsr::I<DataVector, 3, Frame::Inertial>& interior_auxiliary_shift_b,

      // fd_interior_temporary_tags
      const Mesh<3>& subcell_mesh,

      // fd_gridless_tags
      const fd::Reconstructor& reconstructor,
      const bool evolve_lapse_and_shift) const;

 private:
  size_t extrapolation_order_;
};
}  // namespace Ccz4::BoundaryConditions
