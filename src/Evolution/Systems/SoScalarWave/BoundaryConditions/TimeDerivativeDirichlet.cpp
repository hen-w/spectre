// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/SoScalarWave/BoundaryConditions/TimeDerivativeDirichlet.hpp"

#include <cstddef>
#include <memory>
#include <pup.h>
#include <type_traits>

#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/Factory.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace SoScalarWave::BoundaryConditions {
template <size_t Dim>
TimeDerivativeDirichlet<Dim>::TimeDerivativeDirichlet(
    const TimeDerivativeDirichlet& rhs)
    : BoundaryCondition<Dim>{dynamic_cast<const BoundaryCondition<Dim>&>(rhs)},
      analytic_prescription_(rhs.analytic_prescription_->get_clone()) {}

template <size_t Dim>
TimeDerivativeDirichlet<Dim>& TimeDerivativeDirichlet<Dim>::operator=(
    const TimeDerivativeDirichlet& rhs) {
  if (&rhs == this) {
    return *this;
  }
  analytic_prescription_ = rhs.analytic_prescription_->get_clone();
  return *this;
}

template <size_t Dim>
TimeDerivativeDirichlet<Dim>::TimeDerivativeDirichlet(
    CkMigrateMessage* const msg)
    : BoundaryCondition<Dim>(msg) {}

template <size_t Dim>
TimeDerivativeDirichlet<Dim>::TimeDerivativeDirichlet(
    std::unique_ptr<evolution::initial_data::InitialData> analytic_prescription)
    : analytic_prescription_(std::move(analytic_prescription)) {}

template <size_t Dim>
std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
TimeDerivativeDirichlet<Dim>::get_clone() const {
  return std::make_unique<TimeDerivativeDirichlet>(*this);
}

template <size_t Dim>
void TimeDerivativeDirichlet<Dim>::pup(PUP::er& p) {
  BoundaryCondition<Dim>::pup(p);
  p | analytic_prescription_;
}

template <size_t Dim>
std::optional<std::string> TimeDerivativeDirichlet<Dim>::dg_time_derivative(
    const gsl::not_null<Scalar<DataVector>*> dt_psi_correction,
    const gsl::not_null<Scalar<DataVector>*> dt_pi_correction,
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
        dt_phi_correction,
    const gsl::not_null<Scalar<DataVector>*> dt_boundary_psi_correction,

    const std::optional<
        tnsr::I<DataVector, Dim, Frame::Inertial>>& /*face_mesh_velocity*/,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& /*normal_covector*/,

    const tnsr::I<DataVector, Dim, Frame::Inertial>& coords,
    const Scalar<DataVector>& volume_dt_psi,
    const Scalar<DataVector>& volume_dt_pi,
    [[maybe_unused]] const double time) const {
  // Evaluate analytic time derivatives at the face coordinates.
  // The analytic solution returns dt<Psi>, dt<Pi>, and dt<Phi<Dim>> together;
  // we only use dt<Psi> and dt<Pi>.
  auto analytic_dt_values = call_with_dynamic_type<
      tuples::TaggedTuple<::Tags::dt<SoScalarWave::Tags::Psi>,
                          ::Tags::dt<SoScalarWave::Tags::Pi>,
                          ::Tags::dt<SoScalarWave::Tags::Phi<Dim>>>,
      tmpl::append<SoScalarWave::Solutions::all_solutions<Dim>>>(
      analytic_prescription_.get(),
      [&coords, &time](const auto* const analytic_solution_or_data) {
        if constexpr (is_analytic_solution_v<
                          std::decay_t<decltype(*analytic_solution_or_data)>>) {
          return analytic_solution_or_data->variables(
              coords, time,
              tmpl::list<::Tags::dt<SoScalarWave::Tags::Psi>,
                         ::Tags::dt<SoScalarWave::Tags::Pi>,
                         ::Tags::dt<SoScalarWave::Tags::Phi<Dim>>>{});
        } else {
          (void)time;
          (void)coords;
          ERROR(
              "TimeDerivativeDirichlet boundary condition requires an analytic "
              "solution that provides time derivatives, not analytic data.");
        }
      });

  // Correction = -volume_dt + analytic_dt
  // After the infrastructure adds this, the result is analytic_dt.
  get(*dt_psi_correction) =
      -get(volume_dt_psi) +
      get(get<::Tags::dt<SoScalarWave::Tags::Psi>>(analytic_dt_values));
  get(*dt_pi_correction) =
      -get(volume_dt_pi) +
      get(get<::Tags::dt<SoScalarWave::Tags::Pi>>(analytic_dt_values));

  // No correction to Phi or BoundaryPsi
  for (size_t d = 0; d < Dim; ++d) {
    dt_phi_correction->get(d) = 0.0;
  }
  get(*dt_boundary_psi_correction) = 0.0;

  return std::nullopt;
}

template <size_t Dim>
// NOLINTNEXTLINE
PUP::able::PUP_ID TimeDerivativeDirichlet<Dim>::my_PUP_ID = 0;

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data) \
  template class TimeDerivativeDirichlet<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace SoScalarWave::BoundaryConditions
