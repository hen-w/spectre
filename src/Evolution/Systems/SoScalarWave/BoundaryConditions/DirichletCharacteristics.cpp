// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/SoScalarWave/BoundaryConditions/DirichletCharacteristics.hpp"

#include <cstddef>
#include <memory>
#include <pup.h>
#include <type_traits>

#include "Evolution/Systems/SoScalarWave/Characteristics.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/Factory.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace SoScalarWave::BoundaryConditions {
template <size_t Dim>
DirichletCharacteristics<Dim>::DirichletCharacteristics(
    const DirichletCharacteristics& rhs)
    : BoundaryCondition<Dim>{dynamic_cast<const BoundaryCondition<Dim>&>(rhs)},
      analytic_prescription_(rhs.analytic_prescription_->get_clone()),
      prescribe_zero_speed_modes_(rhs.prescribe_zero_speed_modes_) {}

template <size_t Dim>
DirichletCharacteristics<Dim>& DirichletCharacteristics<Dim>::operator=(
    const DirichletCharacteristics& rhs) {
  if (&rhs == this) {
    return *this;
  }
  analytic_prescription_ = rhs.analytic_prescription_->get_clone();
  prescribe_zero_speed_modes_ = rhs.prescribe_zero_speed_modes_;
  return *this;
}

template <size_t Dim>
DirichletCharacteristics<Dim>::DirichletCharacteristics(
    CkMigrateMessage* const msg)
    : BoundaryCondition<Dim>(msg) {}

template <size_t Dim>
DirichletCharacteristics<Dim>::DirichletCharacteristics(
    std::unique_ptr<evolution::initial_data::InitialData> analytic_prescription,
    const bool prescribe_zero_speed_modes)
    : analytic_prescription_(std::move(analytic_prescription)),
      prescribe_zero_speed_modes_(prescribe_zero_speed_modes) {}

template <size_t Dim>
std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
DirichletCharacteristics<Dim>::get_clone() const {
  return std::make_unique<DirichletCharacteristics>(*this);
}

template <size_t Dim>
void DirichletCharacteristics<Dim>::pup(PUP::er& p) {
  BoundaryCondition<Dim>::pup(p);
  p | analytic_prescription_;
  p | prescribe_zero_speed_modes_;
}

namespace {
// Pointwise selection: outgoing (speed > 0) from interior,
// incoming (speed < 0) from analytic, zero from prescribe option.
void mix_scalar_mode(const gsl::not_null<DataVector*> result,
                     const DataVector& speed,
                     const DataVector& interior_val,
                     const DataVector& analytic_val,
                     const bool prescribe_zero) {
  for (size_t s = 0; s < result->size(); ++s) {
    if (speed[s] > 0.0) {
      (*result)[s] = interior_val[s];
    } else if (speed[s] < 0.0) {
      (*result)[s] = analytic_val[s];
    } else {
      (*result)[s] = prescribe_zero ? analytic_val[s] : interior_val[s];
    }
  }
}
}  // namespace

template <size_t Dim>
std::optional<std::string> DirichletCharacteristics<Dim>::dg_ghost(
    const gsl::not_null<Scalar<DataVector>*> psi,
    const gsl::not_null<Scalar<DataVector>*> pi,
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*> phi,
    const std::optional<
        tnsr::I<DataVector, Dim, Frame::Inertial>>& face_mesh_velocity,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
    const Scalar<DataVector>& interior_psi,
    const Scalar<DataVector>& interior_pi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& interior_phi,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& coords,
    [[maybe_unused]] const double time) const {
  // 1. Evaluate analytic solution at face
  auto boundary_values = call_with_dynamic_type<
      tuples::TaggedTuple<SoScalarWave::Tags::Psi, SoScalarWave::Tags::Pi,
                          SoScalarWave::Tags::Phi<Dim>>,
      tmpl::append<SoScalarWave::Solutions::all_solutions<Dim>>>(
      analytic_prescription_.get(),
      [&coords, &time](const auto* const analytic_solution_or_data) {
        if constexpr (is_analytic_solution_v<
                          std::decay_t<decltype(*analytic_solution_or_data)>>) {
          return analytic_solution_or_data->variables(
              coords, time,
              tmpl::list<SoScalarWave::Tags::Psi, SoScalarWave::Tags::Pi,
                         SoScalarWave::Tags::Phi<Dim>>{});
        } else {
          (void)time;
          return analytic_solution_or_data->variables(
              coords,
              tmpl::list<SoScalarWave::Tags::Psi, SoScalarWave::Tags::Pi,
                         SoScalarWave::Tags::Phi<Dim>>{});
        }
      });

  const auto& analytic_psi =
      get<SoScalarWave::Tags::Psi>(boundary_values);
  const auto& analytic_pi =
      get<SoScalarWave::Tags::Pi>(boundary_values);
  const auto& analytic_phi =
      get<SoScalarWave::Tags::Phi<Dim>>(boundary_values);

  // 2. Compute characteristic fields for interior and analytic
  const auto interior_char_fields = characteristic_fields(
      interior_psi, interior_pi, interior_phi, normal_covector);
  const auto analytic_char_fields = characteristic_fields(
      analytic_psi, analytic_pi, analytic_phi, normal_covector);

  // 3. Compute characteristic speeds (accounting for mesh velocity)
  const auto char_speeds =
      characteristic_speeds(normal_covector, face_mesh_velocity);
  const auto& speed_vpsi = char_speeds[0];
  const auto& speed_vzero = char_speeds[1];
  const auto& speed_vplus = char_speeds[2];
  const auto& speed_vminus = char_speeds[3];

  // 4. Mix modes: outgoing from interior, incoming from analytic
  const size_t num_points = get(interior_psi).size();

  Scalar<DataVector> v_psi_ext{num_points};
  mix_scalar_mode(make_not_null(&get(v_psi_ext)), speed_vpsi,
                  get(get<Tags::VPsi>(interior_char_fields)),
                  get(get<Tags::VPsi>(analytic_char_fields)),
                  prescribe_zero_speed_modes_);

  Scalar<DataVector> v_plus_ext{num_points};
  mix_scalar_mode(make_not_null(&get(v_plus_ext)), speed_vplus,
                  get(get<Tags::VPlus>(interior_char_fields)),
                  get(get<Tags::VPlus>(analytic_char_fields)),
                  prescribe_zero_speed_modes_);

  Scalar<DataVector> v_minus_ext{num_points};
  mix_scalar_mode(make_not_null(&get(v_minus_ext)), speed_vminus,
                  get(get<Tags::VMinus>(interior_char_fields)),
                  get(get<Tags::VMinus>(analytic_char_fields)),
                  prescribe_zero_speed_modes_);

  tnsr::i<DataVector, Dim, Frame::Inertial> v_zero_ext{num_points};
  for (size_t d = 0; d < Dim; ++d) {
    mix_scalar_mode(make_not_null(&v_zero_ext.get(d)), speed_vzero,
                    get<Tags::VZero<Dim>>(interior_char_fields).get(d),
                    get<Tags::VZero<Dim>>(analytic_char_fields).get(d),
                    prescribe_zero_speed_modes_);
  }

  // 5. Inverse transform to evolved fields
  auto evolved = evolved_fields_from_characteristic_fields(
      v_psi_ext, v_zero_ext, v_plus_ext, v_minus_ext, normal_covector);

  *psi = get<Tags::Psi>(evolved);
  *pi = get<Tags::Pi>(evolved);
  *phi = get<Tags::Phi<Dim>>(evolved);

  return std::nullopt;
}

template <size_t Dim>
// NOLINTNEXTLINE
PUP::able::PUP_ID DirichletCharacteristics<Dim>::my_PUP_ID = 0;

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data) \
  template class DirichletCharacteristics<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace SoScalarWave::BoundaryConditions
