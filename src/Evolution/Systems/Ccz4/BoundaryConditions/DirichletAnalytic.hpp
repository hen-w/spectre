// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>
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
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Tags {
struct Time;
}  // namespace Tags
namespace domain::Tags {
template <size_t Dim, typename Frame>
struct Coordinates;
}  // namespace domain::Tags
/// \endcond

namespace Ccz4::BoundaryConditions {
/*!
 * \brief Sets Dirichlet boundary conditions using the analytic solution or
 * analytic data.
 */
class DirichletAnalytic final : public BoundaryCondition {
 public:
  /// \brief What analytic solution/data to prescribe.
  struct AnalyticPrescription {
    static constexpr Options::String help =
        "What analytic solution/data to prescribe.";
    using type = std::unique_ptr<evolution::initial_data::InitialData>;
  };
  using options = tmpl::list<AnalyticPrescription>;
  static constexpr Options::String help{
      "DirichletAnalytic boundary conditions using either analytic solution or "
      "analytic data."};

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

  auto get_clone() const
      -> std::unique_ptr<
          domain::BoundaryConditions::BoundaryCondition> override;

  static constexpr evolution::BoundaryConditions::Type bc_type =
      evolution::BoundaryConditions::Type::Ghost;

  void pup(PUP::er& p) override;

  // DG interface
  using dg_interior_evolved_variables_tags = tmpl::list<>;
  using dg_interior_dt_vars_tags = tmpl::list<>;
  using dg_interior_temporary_tags =
      tmpl::list<domain::Tags::Coordinates<3, Frame::Inertial>>;
  using dg_gridless_tags = tmpl::list<::Tags::Time>;

  std::optional<std::string> dg_ghost(
      gsl::not_null<tnsr::ii<DataVector, 3, Frame::Inertial>*> conformal_metric,
      gsl::not_null<Scalar<DataVector>*> conformal_factor,
      gsl::not_null<tnsr::ii<DataVector, 3, Frame::Inertial>*> a_tilde,
      gsl::not_null<Scalar<DataVector>*> trace_extrinsic_curvature,
      gsl::not_null<Scalar<DataVector>*> theta,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> gamma_hat,
      gsl::not_null<Scalar<DataVector>*> lapse,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> shift,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> auxiliary_shift_b,
      gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*> field_a,
      gsl::not_null<tnsr::iJ<DataVector, 3, Frame::Inertial>*> field_b,
      gsl::not_null<tnsr::ijj<DataVector, 3, Frame::Inertial>*> field_d,
      gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*> field_p,
      gsl::not_null<Scalar<DataVector>*> u_scalar3_minus,
      gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*> u_vector2_minus,
      gsl::not_null<Scalar<DataVector>*> u_scalar2_minus,
      gsl::not_null<tnsr::ii<DataVector, 3, Frame::Inertial>*> u_tensor_minus,
      const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
          face_mesh_velocity,
      const tnsr::i<DataVector, 3, Frame::Inertial>& normal_covector,
      const tnsr::I<DataVector, 3, Frame::Inertial>& coords, double time) const;

  // FD interface
  using fd_interior_evolved_variables_tags = tmpl::list<>;
  using fd_interior_temporary_tags =
      tmpl::list<evolution::dg::subcell::Tags::Mesh<3>>;
  using fd_interior_primitive_variables_tags = tmpl::list<>;
  using fd_gridless_tags =
      tmpl::list<::Tags::Time, ::domain::Tags::FunctionsOfTime,
                 domain::Tags::ElementMap<3, Frame::Grid>,
                 domain::CoordinateMaps::Tags::CoordinateMap<3, Frame::Grid,
                                                             Frame::Inertial>,
                 ::Ccz4::fd::Tags::Reconstructor>;
  void fd_ghost(
      gsl::not_null<tnsr::ii<DataVector, 3, Frame::Inertial>*>
          conformal_metric,
      gsl::not_null<Scalar<DataVector>*> lapse,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> shift,
      gsl::not_null<Scalar<DataVector>*> conformal_factor,
      gsl::not_null<tnsr::ii<DataVector, 3, Frame::Inertial>*> a_tilde,
      gsl::not_null<Scalar<DataVector>*> trace_extrinsic_curvature,
      gsl::not_null<Scalar<DataVector>*> theta,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> gamma_hat,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          auxiliary_shift_b,
      const Direction<3>& direction,

      // fd_interior_temporary_tags
      const Mesh<3>& subcell_mesh,

      // fd_gridless_tags
      double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<::domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time,
      const ElementMap<3, Frame::Grid>& logical_to_grid_map,
      const domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>&
          grid_to_inertial_map,
      const fd::Reconstructor& reconstructor) const;

 private:
  std::unique_ptr<evolution::initial_data::InitialData> analytic_prescription_;
};
}  // namespace Ccz4::BoundaryConditions
