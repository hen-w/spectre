// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <optional>
#include <pup.h>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/BoundaryConditions/Type.hpp"
#include "Evolution/Systems/Ccz4/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Tags.hpp"
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
 * \brief Exterior data from an analytic solution, converted to the chart.
 *
 * \details The analytic physical fields and their analytic first derivatives
 * (`FieldA`, `FieldB`, `FieldD`, `FieldP` of `Ccz4WrappedGr`) are evaluated at
 * the face. The exterior chart fields are \f$u^+=(H,\log\phi,\alpha,\beta)\f$,
 * \f$w^+=(2\mathcal T_H^{-1}\tilde A,K,\Theta,\hat\Gamma,b)\f$ and the exact
 * chain-rule gradients \f$q_{H,i}=\mathcal T_H^{-1}(\partial_i\tilde\gamma)
 * =\mathcal T_H^{-1}(2\,\mathrm{FieldD}_i)\f$,
 * \f$q_{\ell,i}=\mathrm{FieldP}_i\f$,
 * \f$q_{\alpha,i}=\alpha\,\mathrm{FieldA}_i\f$, \f$q_{\beta,i}{}^j=
 * \mathrm{FieldB}_i{}^j\f$. The exterior physical fields are the analytic
 * fields. The boundary correction then applies the same upwind and scalar
 * penalty target correction as on internal faces.
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
      "Exterior chart fields of the CCZ4 gradient-projection scheme from an "
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
      gsl::not_null<Tags::ChartW::type*> w,
      gsl::not_null<Tags::ChartQ::type*> q,
      gsl::not_null<Tags::ChartU::type*> u,
      gsl::not_null<tnsr::ii<DataVector, 3>*> conformal_metric,
      gsl::not_null<Scalar<DataVector>*> conformal_factor,
      gsl::not_null<tnsr::ii<DataVector, 3>*> a_tilde,
      gsl::not_null<Scalar<DataVector>*> trace_extrinsic_curvature,
      gsl::not_null<Scalar<DataVector>*> theta,
      gsl::not_null<tnsr::I<DataVector, 3>*> gamma_hat,
      gsl::not_null<Scalar<DataVector>*> lapse,
      gsl::not_null<tnsr::I<DataVector, 3>*> shift,
      gsl::not_null<tnsr::I<DataVector, 3>*> b,
      const std::optional<tnsr::I<DataVector, 3>>& face_mesh_velocity,
      const tnsr::i<DataVector, 3>& normal_covector,
      const tnsr::I<DataVector, 3>& coords, double time) const;

 private:
  std::unique_ptr<evolution::initial_data::InitialData> analytic_prescription_;
};

/*!
 * \brief Fill the chart fields \f$(u,w,q)\f$ at the given points from physical
 * fields and their analytic first derivatives, see `DirichletAnalytic`.
 */
void chart_fields_from_analytic(
    gsl::not_null<Tags::ChartW::type*> w, gsl::not_null<Tags::ChartQ::type*> q,
    gsl::not_null<Tags::ChartU::type*> u,
    const tnsr::ii<DataVector, 3>& conformal_metric,
    const Scalar<DataVector>& conformal_factor,
    const tnsr::ii<DataVector, 3>& a_tilde,
    const Scalar<DataVector>& trace_extrinsic_curvature,
    const Scalar<DataVector>& theta, const tnsr::I<DataVector, 3>& gamma_hat,
    const Scalar<DataVector>& lapse, const tnsr::I<DataVector, 3>& shift,
    const tnsr::I<DataVector, 3>& b, const tnsr::i<DataVector, 3>& field_a,
    const tnsr::iJ<DataVector, 3>& field_b,
    const tnsr::ijj<DataVector, 3>& field_d,
    const tnsr::i<DataVector, 3>& field_p);
}  // namespace Ccz4::GradientProjection::BoundaryConditions
