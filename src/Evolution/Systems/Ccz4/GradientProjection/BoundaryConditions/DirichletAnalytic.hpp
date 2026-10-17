// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <optional>
#include <pup.h>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/BoundaryConditions/Type.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Systems/Ccz4/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Tags {
struct Time;
}  // namespace Tags
/// \endcond

/// Boundary conditions of the CCZ4 gradient-projection scheme.
namespace Ccz4::GradientProjection::BoundaryConditions {
/*!
 * \brief Exterior data from an analytic solution.
 *
 * \details The analytic physical fields and their analytic first derivatives
 * (`FieldA`, `FieldB`, `FieldD`, `FieldP` of `Ccz4WrappedGr`) are evaluated at
 * the face. The exterior physical fields are the analytic fields, and the
 * exterior gradients are the exact chain-rule gradients
 * \f$q_{H,i}=\mathcal T_H^{-1}(\partial_i\tilde\gamma)
 * =\mathcal T_H^{-1}(2\,\mathrm{FieldD}_i)\f$,
 * \f$q_{\phi,i}=\phi\,\mathrm{FieldP}_i\f$,
 * \f$q_{\alpha,i}=\alpha\,\mathrm{FieldA}_i\f$, \f$q_{\beta,i}{}^j=
 * \mathrm{FieldB}_i{}^j\f$. The boundary correction packages the exterior
 * chart fields from these and applies its external-face upwind target
 * correction, which has no scalar jump penalty (see
 * `Ccz4::GradientProjection::BoundaryCorrections::UpwindGradientProjection`).
 *
 * On elements using finite differences, `fd_ghost` sets the physical fields
 * of the exterior ghost cells to the analytic solution at the inertial
 * coordinates of the ghost cells at the given time. Normal to the face, the
 * ghost cells continue the uniform cell-centered subcell grid outside the
 * element; tangentially, they are at the subcell points. The gradients of the
 * ghost cells are not set (see `Ccz4::fd::BoundaryConditionGhostData`).
 */
class DirichletAnalytic final
    : public ::Ccz4::BoundaryConditions::BoundaryCondition {
 public:
  struct AnalyticPrescription {
    static constexpr Options::String help =
        "What analytic solution to prescribe.";
    using type = std::unique_ptr<evolution::initial_data::InitialData>;
  };
  using options = tmpl::list<AnalyticPrescription>;
  static constexpr Options::String help{
      "Exterior fields of the CCZ4 gradient-projection scheme from an "
      "analytic solution."};

  DirichletAnalytic() = default;
  DirichletAnalytic(DirichletAnalytic&&) = default;
  DirichletAnalytic& operator=(DirichletAnalytic&&) = default;
  DirichletAnalytic(const DirichletAnalytic&);
  DirichletAnalytic& operator=(const DirichletAnalytic&);
  ~DirichletAnalytic() override = default;

  explicit DirichletAnalytic(CkMigrateMessage* msg);

  explicit DirichletAnalytic(
      std::unique_ptr<evolution::initial_data::InitialData>
          analytic_prescription);

  WRAPPED_PUPable_decl_base_template(
      domain::BoundaryConditions::BoundaryCondition, DirichletAnalytic);

  auto get_clone() const -> std::unique_ptr<
      domain::BoundaryConditions::BoundaryCondition> override;

  static constexpr evolution::BoundaryConditions::Type bc_type =
      evolution::BoundaryConditions::Type::Ghost;

  void pup(PUP::er& p) override;

  using dg_interior_evolved_variables_tags = tmpl::list<>;
  using dg_interior_dt_vars_tags = tmpl::list<>;
  using dg_interior_temporary_tags =
      tmpl::list<domain::Tags::Coordinates<3, Frame::Inertial>>;
  using dg_gridless_tags = tmpl::list<::Tags::Time>;

  std::optional<std::string> dg_ghost(
      gsl::not_null<tnsr::ii<DataVector, 3>*> conformal_metric,
      gsl::not_null<Scalar<DataVector>*> conformal_factor,
      gsl::not_null<tnsr::ii<DataVector, 3>*> a_tilde,
      gsl::not_null<Scalar<DataVector>*> trace_extrinsic_curvature,
      gsl::not_null<Scalar<DataVector>*> theta,
      gsl::not_null<tnsr::I<DataVector, 3>*> gamma_hat,
      gsl::not_null<Scalar<DataVector>*> lapse,
      gsl::not_null<tnsr::I<DataVector, 3>*> shift,
      gsl::not_null<tnsr::I<DataVector, 3>*> b,
      gsl::not_null<Tags::PotentialGradients::type*> q,
      const std::optional<tnsr::I<DataVector, 3>>& face_mesh_velocity,
      const tnsr::i<DataVector, 3>& normal_covector,
      const tnsr::I<DataVector, 3>& coords, double time) const;

  using fd_interior_evolved_variables_tags = tmpl::list<>;
  using fd_interior_temporary_tags =
      tmpl::list<evolution::dg::subcell::Tags::Mesh<3>>;
  using fd_gridless_tags =
      tmpl::list<::Tags::Time, domain::Tags::FunctionsOfTime,
                 domain::Tags::ElementMap<3, Frame::Grid>,
                 domain::CoordinateMaps::Tags::CoordinateMap<3, Frame::Grid,
                                                             Frame::Inertial>,
                 ::Ccz4::fd::Tags::Reconstructor>;

  void fd_ghost(gsl::not_null<tnsr::ii<DataVector, 3>*> conformal_metric,
                gsl::not_null<Scalar<DataVector>*> conformal_factor,
                gsl::not_null<tnsr::ii<DataVector, 3>*> a_tilde,
                gsl::not_null<Scalar<DataVector>*> trace_extrinsic_curvature,
                gsl::not_null<Scalar<DataVector>*> theta,
                gsl::not_null<tnsr::I<DataVector, 3>*> gamma_hat,
                gsl::not_null<Scalar<DataVector>*> lapse,
                gsl::not_null<tnsr::I<DataVector, 3>*> shift,
                gsl::not_null<tnsr::I<DataVector, 3>*> b,
                const Direction<3>& direction, const Mesh<3>& subcell_mesh,
                double time,
                const domain::FunctionsOfTimeMap& functions_of_time,
                const ElementMap<3, Frame::Grid>& logical_to_grid_map,
                const domain::CoordinateMapBase<Frame::Grid, Frame::Inertial,
                                                3>& grid_to_inertial_map,
                const ::Ccz4::fd::Reconstructor& reconstructor) const;

 private:
  std::unique_ptr<evolution::initial_data::InitialData> analytic_prescription_;
};

/*!
 * \brief The gradients \f$q\f$ at the given points from the physical fields
 * and their analytic first derivatives, see `DirichletAnalytic`.
 */
void gradients_from_analytic(gsl::not_null<Tags::PotentialGradients::type*> q,
                             const tnsr::ii<DataVector, 3>& conformal_metric,
                             const Scalar<DataVector>& conformal_factor,
                             const Scalar<DataVector>& lapse,
                             const tnsr::i<DataVector, 3>& field_a,
                             const tnsr::iJ<DataVector, 3>& field_b,
                             const tnsr::ijj<DataVector, 3>& field_d,
                             const tnsr::i<DataVector, 3>& field_p);
}  // namespace Ccz4::GradientProjection::BoundaryConditions
