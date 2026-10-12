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
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Evolution/DgSubcell/GhostZoneLogicalCoordinates.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Chart.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/PointwiseFields.hpp"
#include "Evolution/Systems/Ccz4/Solutions/Factory.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Evolution/TypeTraits.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Ccz4::GradientProjection::BoundaryConditions {
namespace {
// The fields `AnalyticTags` of the analytic prescription at the given points.
template <typename... AnalyticTags>
tuples::TaggedTuple<AnalyticTags...> analytic_variables(
    const evolution::initial_data::InitialData& analytic_prescription,
    const tnsr::I<DataVector, 3>& coords, const double time,
    tmpl::list<AnalyticTags...> /*meta*/) {
  return call_with_dynamic_type<tuples::TaggedTuple<AnalyticTags...>,
                                ::Ccz4::Solutions::all_solutions>(
      &analytic_prescription, [&coords, &time](const auto* const initial_data) {
        if constexpr (is_analytic_solution_v<
                          std::decay_t<decltype(*initial_data)>>) {
          return initial_data->variables(coords, time,
                                         tmpl::list<AnalyticTags...>{});
        } else {
          (void)time;
          ERROR(
              "The CCZ4 gradient-projection DirichletAnalytic boundary "
              "condition requires an analytic solution.");
        }
      });
}
}  // namespace

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

void gradients_from_analytic(
    const gsl::not_null<Tags::PotentialGradients::type*> q,
    const tnsr::ii<DataVector, 3>& conformal_metric,
    const Scalar<DataVector>& conformal_factor, const Scalar<DataVector>& lapse,
    const tnsr::i<DataVector, 3>& field_a,
    const tnsr::iJ<DataVector, 3>& field_b,
    const tnsr::ijj<DataVector, 3>& field_d,
    const tnsr::i<DataVector, 3>& field_p) {
  const size_t number_of_points = get(lapse).size();
  for (size_t s = 0; s < number_of_points; ++s) {
    Matrix3 metric{};
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        metric(i, j) = conformal_metric.get(i, j)[s];
      }
    }
    const LogMetricChart chart = LogMetricChart::from_metric(metric);
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
      q_i[PotentialIndex::conformal_factor] =
          get(conformal_factor)[s] * field_p.get(i)[s];
      q_i[PotentialIndex::lapse] = get(lapse)[s] * field_a.get(i)[s];
      for (size_t k = 0; k < 3; ++k) {
        gsl::at(q_i, PotentialIndex::shift + k) = field_b.get(i, k)[s];
      }
    }
    set_gradients_at(q, s, q_s);
  }
}

std::optional<std::string> DirichletAnalytic::dg_ghost(
    const gsl::not_null<tnsr::ii<DataVector, 3>*> conformal_metric,
    const gsl::not_null<Scalar<DataVector>*> conformal_factor,
    const gsl::not_null<tnsr::ii<DataVector, 3>*> a_tilde,
    const gsl::not_null<Scalar<DataVector>*> trace_extrinsic_curvature,
    const gsl::not_null<Scalar<DataVector>*> theta,
    const gsl::not_null<tnsr::I<DataVector, 3>*> gamma_hat,
    const gsl::not_null<Scalar<DataVector>*> lapse,
    const gsl::not_null<tnsr::I<DataVector, 3>*> shift,
    const gsl::not_null<tnsr::I<DataVector, 3>*> b,
    const gsl::not_null<Tags::PotentialGradients::type*> q,
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
  const auto values = analytic_variables(*analytic_prescription_, coords, time,
                                         analytic_tags{});
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
  gradients_from_analytic(q, *conformal_metric, *conformal_factor, *lapse,
                          get<::Ccz4::Tags::FieldA<DataVector, 3>>(values),
                          get<::Ccz4::Tags::FieldB<DataVector, 3>>(values),
                          get<::Ccz4::Tags::FieldD<DataVector, 3>>(values),
                          get<::Ccz4::Tags::FieldP<DataVector, 3>>(values));
  return std::nullopt;
}

void DirichletAnalytic::fd_ghost(
    const gsl::not_null<tnsr::ii<DataVector, 3>*> conformal_metric,
    const gsl::not_null<Scalar<DataVector>*> conformal_factor,
    const gsl::not_null<tnsr::ii<DataVector, 3>*> a_tilde,
    const gsl::not_null<Scalar<DataVector>*> trace_extrinsic_curvature,
    const gsl::not_null<Scalar<DataVector>*> theta,
    const gsl::not_null<tnsr::I<DataVector, 3>*> gamma_hat,
    const gsl::not_null<Scalar<DataVector>*> lapse,
    const gsl::not_null<tnsr::I<DataVector, 3>*> shift,
    const gsl::not_null<tnsr::I<DataVector, 3>*> b,
    const Direction<3>& direction, const Mesh<3>& subcell_mesh,
    const double time, const domain::FunctionsOfTimeMap& functions_of_time,
    const ElementMap<3, Frame::Grid>& logical_to_grid_map,
    const domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>&
        grid_to_inertial_map,
    const ::Ccz4::fd::Reconstructor& reconstructor) const {
  const auto ghost_inertial_coords = grid_to_inertial_map(
      logical_to_grid_map(
          evolution::dg::subcell::fd::ghost_zone_logical_coordinates(
              subcell_mesh, reconstructor.ghost_zone_size(), direction)),
      time, functions_of_time);
  const auto values =
      analytic_variables(*analytic_prescription_, ghost_inertial_coords, time,
                         ::Ccz4::fd::System::original_evolved_variables_tags{});
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
}
}  // namespace Ccz4::GradientProjection::BoundaryConditions
