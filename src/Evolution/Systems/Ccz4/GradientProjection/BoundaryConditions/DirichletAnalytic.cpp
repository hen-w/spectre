// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Ccz4/GradientProjection/BoundaryConditions/DirichletAnalytic.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <pup.h>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Chart.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/PointwiseFields.hpp"
#include "Evolution/Systems/Ccz4/Solutions/Factory.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Evolution/TypeTraits.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"

namespace Ccz4::GradientProjection::BoundaryConditions {
DirichletAnalytic::DirichletAnalytic(CkMigrateMessage* const msg)
    : BoundaryCondition(msg) {}

DirichletAnalytic::DirichletAnalytic(const DirichletAnalytic& rhs)
    : BoundaryCondition{dynamic_cast<const BoundaryCondition&>(rhs)},
      analytic_prescription_(rhs.analytic_prescription_->get_clone()) {}

DirichletAnalytic& DirichletAnalytic::operator=(const DirichletAnalytic& rhs) {
  if (&rhs == this) {
    return *this;
  }
  analytic_prescription_ = rhs.analytic_prescription_->get_clone();
  return *this;
}

DirichletAnalytic::DirichletAnalytic(
    std::unique_ptr<evolution::initial_data::InitialData> analytic_prescription)
    : analytic_prescription_(std::move(analytic_prescription)) {}

std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
DirichletAnalytic::get_clone() const {
  return std::make_unique<DirichletAnalytic>(*this);
}

void DirichletAnalytic::pup(PUP::er& p) {
  BoundaryCondition::pup(p);
  p | analytic_prescription_;
}

// NOLINTNEXTLINE
PUP::able::PUP_ID DirichletAnalytic::my_PUP_ID = 0;

void chart_fields_from_analytic(
    const gsl::not_null<Tags::ChartW::type*> w,
    const gsl::not_null<Tags::ChartQ::type*> q,
    const gsl::not_null<Tags::ChartU::type*> u,
    const tnsr::ii<DataVector, 3>& conformal_metric,
    const Scalar<DataVector>& conformal_factor,
    const tnsr::ii<DataVector, 3>& a_tilde,
    const Scalar<DataVector>& trace_extrinsic_curvature,
    const Scalar<DataVector>& theta, const tnsr::I<DataVector, 3>& gamma_hat,
    const Scalar<DataVector>& lapse, const tnsr::I<DataVector, 3>& shift,
    const tnsr::I<DataVector, 3>& b, const tnsr::i<DataVector, 3>& field_a,
    const tnsr::iJ<DataVector, 3>& field_b,
    const tnsr::ijj<DataVector, 3>& field_d,
    const tnsr::i<DataVector, 3>& field_p) {
  const size_t number_of_points = get(lapse).size();
  for (size_t s = 0; s < number_of_points; ++s) {
    PhysicalPointState state{};
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        state.conformal_metric(i, j) = conformal_metric.get(i, j)[s];
        state.a_tilde(i, j) = a_tilde.get(i, j)[s];
      }
      gsl::at(state.gamma_hat, i) = gamma_hat.get(i)[s];
      gsl::at(state.shift, i) = shift.get(i)[s];
      gsl::at(state.b, i) = b.get(i)[s];
    }
    state.conformal_factor = get(conformal_factor)[s];
    state.trace_extrinsic_curvature = get(trace_extrinsic_curvature)[s];
    state.theta = get(theta)[s];
    state.lapse = get(lapse)[s];
    PotentialVector u_s{};
    VelocityVector w_s{};
    chart_from_physical(make_not_null(&u_s), make_not_null(&w_s), state);
    const LogMetricChart chart =
        LogMetricChart::from_metric(state.conformal_metric);
    GradientVector q_s{};
    for (size_t i = 0; i < 3; ++i) {
      Matrix3 metric_derivative{};
      for (size_t j = 0; j < 3; ++j) {
        for (size_t k = 0; k < 3; ++k) {
          metric_derivative(j, k) = 2.0 * field_d.get(i, j, k)[s];
        }
      }
      const std::array<double, 5> q_h =
          stf_encode(chart.inverse_tangent(metric_derivative));
      PotentialVector& q_i = gsl::at(q_s, i);
      for (size_t c = 0; c < 5; ++c) {
        gsl::at(q_i, PotentialIndex::log_metric + c) = gsl::at(q_h, c);
      }
      q_i[PotentialIndex::log_conformal_factor] = field_p.get(i)[s];
      q_i[PotentialIndex::lapse] = state.lapse * field_a.get(i)[s];
      for (size_t k = 0; k < 3; ++k) {
        gsl::at(q_i, PotentialIndex::shift + k) = field_b.get(i, k)[s];
      }
    }
    set_potentials_at(u, s, u_s);
    set_velocities_at(w, s, w_s);
    set_gradients_at(q, s, q_s);
  }
}

std::optional<std::string> DirichletAnalytic::dg_ghost(
    const gsl::not_null<Tags::ChartW::type*> w,
    const gsl::not_null<Tags::ChartQ::type*> q,
    const gsl::not_null<Tags::ChartU::type*> u,
    const gsl::not_null<tnsr::ii<DataVector, 3>*> conformal_metric,
    const gsl::not_null<Scalar<DataVector>*> conformal_factor,
    const gsl::not_null<tnsr::ii<DataVector, 3>*> a_tilde,
    const gsl::not_null<Scalar<DataVector>*> trace_extrinsic_curvature,
    const gsl::not_null<Scalar<DataVector>*> theta,
    const gsl::not_null<tnsr::I<DataVector, 3>*> gamma_hat,
    const gsl::not_null<Scalar<DataVector>*> lapse,
    const gsl::not_null<tnsr::I<DataVector, 3>*> shift,
    const gsl::not_null<tnsr::I<DataVector, 3>*> b,
    const std::optional<tnsr::I<DataVector, 3>>& /*face_mesh_velocity*/,
    const tnsr::i<DataVector, 3>& /*normal_covector*/,
    const tnsr::I<DataVector, 3>& coords, const double time) const {
  using analytic_tags = tmpl::list<
      ::Ccz4::Tags::ConformalMetric<DataVector, 3>,
      ::Ccz4::Tags::ConformalFactor<DataVector>,
      ::Ccz4::Tags::ATilde<DataVector, 3>,
      gr::Tags::TraceExtrinsicCurvature<DataVector>,
      ::Ccz4::Tags::Theta<DataVector>, ::Ccz4::Tags::GammaHat<DataVector, 3>,
      gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>,
      ::Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>,
      ::Ccz4::Tags::FieldA<DataVector, 3>, ::Ccz4::Tags::FieldB<DataVector, 3>,
      ::Ccz4::Tags::FieldD<DataVector, 3>, ::Ccz4::Tags::FieldP<DataVector, 3>>;
  const auto values =
      call_with_dynamic_type<tuples::tagged_tuple_from_typelist<analytic_tags>,
                             ::Ccz4::Solutions::all_solutions>(
          analytic_prescription_.get(),
          [&coords, &time](const auto* const initial_data) {
            if constexpr (is_analytic_solution_v<
                              std::decay_t<decltype(*initial_data)>>) {
              return initial_data->variables(coords, time, analytic_tags{});
            } else {
              (void)time;
              ERROR(
                  "The CCZ4 gradient-projection DirichletAnalytic boundary "
                  "condition requires an analytic solution.");
            }
          });
  *conformal_metric = get<::Ccz4::Tags::ConformalMetric<DataVector, 3>>(values);
  *conformal_factor = get<::Ccz4::Tags::ConformalFactor<DataVector>>(values);
  *a_tilde = get<::Ccz4::Tags::ATilde<DataVector, 3>>(values);
  *trace_extrinsic_curvature =
      get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(values);
  *theta = get<::Ccz4::Tags::Theta<DataVector>>(values);
  *gamma_hat = get<::Ccz4::Tags::GammaHat<DataVector, 3>>(values);
  *lapse = get<gr::Tags::Lapse<DataVector>>(values);
  *shift = get<gr::Tags::Shift<DataVector, 3>>(values);
  *b = get<::Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>>(values);
  chart_fields_from_analytic(w, q, u, *conformal_metric, *conformal_factor,
                             *a_tilde, *trace_extrinsic_curvature, *theta,
                             *gamma_hat, *lapse, *shift, *b,
                             get<::Ccz4::Tags::FieldA<DataVector, 3>>(values),
                             get<::Ccz4::Tags::FieldB<DataVector, 3>>(values),
                             get<::Ccz4::Tags::FieldD<DataVector, 3>>(values),
                             get<::Ccz4::Tags::FieldP<DataVector, 3>>(values));
  return std::nullopt;
}
}  // namespace Ccz4::GradientProjection::BoundaryConditions
