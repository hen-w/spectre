// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/SoScalarWave/BoundaryConditions/DirichletAnalytic.hpp"

#include <cstddef>
#include <memory>
#include <pup.h>
#include <type_traits>

#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/Factory.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace SoScalarWave::BoundaryConditions {
template <size_t Dim>
DirichletAnalytic<Dim>::DirichletAnalytic(const DirichletAnalytic& rhs)
    : BoundaryCondition<Dim>{dynamic_cast<const BoundaryCondition<Dim>&>(rhs)},
      analytic_prescription_(rhs.analytic_prescription_->get_clone()) {}

template <size_t Dim>
DirichletAnalytic<Dim>& DirichletAnalytic<Dim>::operator=(
    const DirichletAnalytic& rhs) {
  if (&rhs == this) {
    return *this;
  }
  analytic_prescription_ = rhs.analytic_prescription_->get_clone();
  return *this;
}

template <size_t Dim>
DirichletAnalytic<Dim>::DirichletAnalytic(CkMigrateMessage* const msg)
    : BoundaryCondition<Dim>(msg) {}

template <size_t Dim>
DirichletAnalytic<Dim>::DirichletAnalytic(
    std::unique_ptr<evolution::initial_data::InitialData> analytic_prescription)
    : analytic_prescription_(std::move(analytic_prescription)) {}

template <size_t Dim>
std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
DirichletAnalytic<Dim>::get_clone() const {
  return std::make_unique<DirichletAnalytic>(*this);
}

template <size_t Dim>
void DirichletAnalytic<Dim>::pup(PUP::er& p) {
  BoundaryCondition<Dim>::pup(p);
  p | analytic_prescription_;
}

template <size_t Dim>
std::optional<std::string> DirichletAnalytic<Dim>::dg_ghost(
    const gsl::not_null<Scalar<DataVector>*> psi,
    const gsl::not_null<Scalar<DataVector>*> pi,
    const gsl::not_null<Scalar<DataVector>*> dt_psi,
    const gsl::not_null<Scalar<DataVector>*> dt_pi,
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*> d_psi,
    const gsl::not_null<Scalar<DataVector>*> det_jacobian,
    const std::optional<
        tnsr::I<DataVector, Dim, Frame::Inertial>>& /*face_mesh_velocity*/,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& /*normal_covector*/,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& coords,
    [[maybe_unused]] const double time) const {
  // Set Psi and Pi to zero since they are not used by CgCollocation.
  // Only dt<Psi> and dt<Pi> are actually used by dg_package_data.
  // Setting to zero helps catch bugs if something unexpectedly uses these
  // values.
  get(*psi) = std::numeric_limits<double>::signaling_NaN();
  get(*pi) = std::numeric_limits<double>::signaling_NaN();

  // Set the time derivatives from the analytic solution.
  auto boundary_values = call_with_dynamic_type<
      tuples::TaggedTuple<::Tags::dt<SoScalarWave::Tags::Psi>,
                          ::Tags::dt<SoScalarWave::Tags::Pi>>,
      tmpl::append<SoScalarWave::Solutions::all_solutions<Dim>>>(
      analytic_prescription_.get(),
      [&coords, &time](const auto* const analytic_solution_or_data) {
        if constexpr (is_analytic_solution_v<
                          std::decay_t<decltype(*analytic_solution_or_data)>>) {
          return analytic_solution_or_data->variables(
              coords, time,
              tmpl::list<::Tags::dt<SoScalarWave::Tags::Psi>,
                         ::Tags::dt<SoScalarWave::Tags::Pi>>{});

        } else {
          (void)time;
          return analytic_solution_or_data->variables(
              coords, tmpl::list<::Tags::dt<SoScalarWave::Tags::Psi>,
                                 ::Tags::dt<SoScalarWave::Tags::Pi>>{});
        }
      });
  *dt_psi = get<::Tags::dt<SoScalarWave::Tags::Psi>>(boundary_values);
  *dt_pi = get<::Tags::dt<SoScalarWave::Tags::Pi>>(boundary_values);

  // d_psi is unused at external boundary
  for (size_t d = 0; d < Dim; ++d) {
    d_psi->get(d) = 0.0;
  }
  // det_jacobian is unused at external boundary
  get(*det_jacobian) = std::numeric_limits<double>::signaling_NaN();

  return std::nullopt;
}

template <size_t Dim>
// NOLINTNEXTLINE
PUP::able::PUP_ID DirichletAnalytic<Dim>::my_PUP_ID = 0;

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data) template class DirichletAnalytic<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace SoScalarWave::BoundaryConditions
