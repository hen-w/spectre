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
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace SoScalarWave::BoundaryConditions {
template <size_t Dim>
DirichletCharacteristics<Dim>::DirichletCharacteristics(
    const DirichletCharacteristics& rhs)
    : BoundaryCondition<Dim>{dynamic_cast<const BoundaryCondition<Dim>&>(rhs)},
      analytic_prescription_(rhs.analytic_prescription_->get_clone()),
      prescribe_zero_speed_modes_(rhs.prescribe_zero_speed_modes_),
      copy_psi_from_interior_(rhs.copy_psi_from_interior_),
      zero_incoming_mode_(rhs.zero_incoming_mode_) {}

template <size_t Dim>
DirichletCharacteristics<Dim>& DirichletCharacteristics<Dim>::operator=(
    const DirichletCharacteristics& rhs) {
  if (&rhs == this) {
    return *this;
  }
  analytic_prescription_ = rhs.analytic_prescription_->get_clone();
  prescribe_zero_speed_modes_ = rhs.prescribe_zero_speed_modes_;
  copy_psi_from_interior_ = rhs.copy_psi_from_interior_;
  zero_incoming_mode_ = rhs.zero_incoming_mode_;
  return *this;
}

template <size_t Dim>
DirichletCharacteristics<Dim>::DirichletCharacteristics(
    CkMigrateMessage* const msg)
    : BoundaryCondition<Dim>(msg) {}

template <size_t Dim>
DirichletCharacteristics<Dim>::DirichletCharacteristics(
    std::unique_ptr<evolution::initial_data::InitialData> analytic_prescription,
    const bool prescribe_zero_speed_modes, const bool copy_psi_from_interior,
    const bool zero_incoming_mode)
    : analytic_prescription_(std::move(analytic_prescription)),
      prescribe_zero_speed_modes_(prescribe_zero_speed_modes),
      copy_psi_from_interior_(copy_psi_from_interior),
      zero_incoming_mode_(zero_incoming_mode) {
  if (prescribe_zero_speed_modes_ and copy_psi_from_interior_) {
    ERROR(
        "DirichletCharacteristics: CopyPsiFromInterior and "
        "PrescribeZeroSpeedModes cannot both be true. "
        "CopyPsiFromInterior copies Psi from the interior evolved value, "
        "while PrescribeZeroSpeedModes sets VPsi from the analytic solution.");
  }
}

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
  p | copy_psi_from_interior_;
  p | zero_incoming_mode_;
}

namespace {
// Pointwise selection: outgoing (speed > 0) from interior,
// incoming (speed < 0) from analytic, zero from prescribe option.
void mix_scalar_mode(const gsl::not_null<DataVector*> result,
                     const DataVector& speed, const DataVector& interior_val,
                     const DataVector& analytic_val, const bool prescribe_zero,
                     const bool zero_incoming) {
  for (size_t s = 0; s < result->size(); ++s) {
    if (speed[s] > 0.0) {
      (*result)[s] = interior_val[s];
    } else if (speed[s] < 0.0) {
      (*result)[s] = zero_incoming ? 0.0 : analytic_val[s];
    } else {
      (*result)[s] = prescribe_zero ? analytic_val[s] : interior_val[s];
    }
  }
}

// Evaluate the analytic solution at the face coordinates
template <size_t Dim>
auto evaluate_analytic(
    const evolution::initial_data::InitialData& analytic_prescription,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& coords,
    const double time) {
  return call_with_dynamic_type<
      tuples::TaggedTuple<SoScalarWave::Tags::Psi, SoScalarWave::Tags::Pi,
                          SoScalarWave::Tags::Phi<Dim>>,
      tmpl::append<SoScalarWave::Solutions::all_solutions<Dim>>>(
      &analytic_prescription,
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
}

// Mix characteristic modes and return (v_plus, v_minus) after mixing
template <size_t Dim>
struct MixedCharData {
  Scalar<DataVector> v_psi_ext;
  Scalar<DataVector> v_plus_ext;
  Scalar<DataVector> v_minus_ext;
  tnsr::i<DataVector, Dim, Frame::Inertial> v_zero_ext;
};

template <size_t Dim>
MixedCharData<Dim> mix_char_modes(
    const Scalar<DataVector>& interior_psi,
    const Scalar<DataVector>& interior_pi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& interior_phi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
    const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
        face_mesh_velocity,
    const Scalar<DataVector>& analytic_psi,
    const Scalar<DataVector>& analytic_pi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& analytic_phi,
    const bool prescribe_zero, const bool zero_incoming) {
  const auto interior_char_fields = SoScalarWave::characteristic_fields(
      interior_psi, interior_pi, interior_phi, normal_covector);
  const auto analytic_char_fields = SoScalarWave::characteristic_fields(
      analytic_psi, analytic_pi, analytic_phi, normal_covector);
  const auto char_speeds =
      SoScalarWave::characteristic_speeds(normal_covector, face_mesh_velocity);

  const size_t num_points = get(interior_psi).size();

  MixedCharData<Dim> result;
  result.v_psi_ext = Scalar<DataVector>{num_points};
  result.v_plus_ext = Scalar<DataVector>{num_points};
  result.v_minus_ext = Scalar<DataVector>{num_points};
  result.v_zero_ext = tnsr::i<DataVector, Dim, Frame::Inertial>{num_points};

  mix_scalar_mode(make_not_null(&get(result.v_psi_ext)), char_speeds[0],
                  get(get<SoScalarWave::Tags::VPsi>(interior_char_fields)),
                  get(get<SoScalarWave::Tags::VPsi>(analytic_char_fields)),
                  prescribe_zero, zero_incoming);
  mix_scalar_mode(make_not_null(&get(result.v_plus_ext)), char_speeds[2],
                  get(get<SoScalarWave::Tags::VPlus>(interior_char_fields)),
                  get(get<SoScalarWave::Tags::VPlus>(analytic_char_fields)),
                  prescribe_zero, zero_incoming);
  mix_scalar_mode(make_not_null(&get(result.v_minus_ext)), char_speeds[3],
                  get(get<SoScalarWave::Tags::VMinus>(interior_char_fields)),
                  get(get<SoScalarWave::Tags::VMinus>(analytic_char_fields)),
                  prescribe_zero, zero_incoming);
  for (size_t d = 0; d < Dim; ++d) {
    mix_scalar_mode(
        make_not_null(&result.v_zero_ext.get(d)), char_speeds[1],
        get<SoScalarWave::Tags::VZero<Dim>>(interior_char_fields).get(d),
        get<SoScalarWave::Tags::VZero<Dim>>(analytic_char_fields).get(d),
        prescribe_zero, zero_incoming);
  }
  return result;
}
}  // namespace

template <size_t Dim>
std::optional<std::string> DirichletCharacteristics<Dim>::dg_ghost(
    const gsl::not_null<Scalar<DataVector>*> psi,
    const gsl::not_null<Scalar<DataVector>*> pi,
    const gsl::not_null<Scalar<DataVector>*> boundary_psi,
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*> phi,
    const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
        face_mesh_velocity,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
    const Scalar<DataVector>& interior_psi,
    const Scalar<DataVector>& interior_pi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& interior_phi,
    const Scalar<DataVector>& interior_boundary_psi,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& coords,
    [[maybe_unused]] const double time) const {
  const auto boundary_values =
      evaluate_analytic<Dim>(*analytic_prescription_, coords, time);
  const auto& analytic_psi = get<SoScalarWave::Tags::Psi>(boundary_values);
  const auto& analytic_pi = get<SoScalarWave::Tags::Pi>(boundary_values);
  const auto& analytic_phi = get<SoScalarWave::Tags::Phi<Dim>>(boundary_values);

  const auto mixed = mix_char_modes<Dim>(
      interior_psi, interior_pi, interior_phi, normal_covector,
      face_mesh_velocity, analytic_psi, analytic_pi, analytic_phi,
      prescribe_zero_speed_modes_, zero_incoming_mode_);

  auto evolved = evolved_fields_from_characteristic_fields(
      mixed.v_psi_ext, mixed.v_zero_ext, mixed.v_plus_ext, mixed.v_minus_ext,
      normal_covector);

  // Ghost Psi selection:
  //   CopyPsiFromInterior: use interior evolved Psi directly
  //   PrescribeZeroSpeedModes: use char-decomposed Psi (VPsi from analytic)
  //   Otherwise: use the time-integrated BoundaryPsi
  if (copy_psi_from_interior_) {
    *psi = interior_psi;
  } else if (prescribe_zero_speed_modes_) {
    *psi = get<Tags::Psi>(evolved);
  } else {
    *psi = interior_boundary_psi;
  }
  *pi = get<Tags::Pi>(evolved);
  *phi = get<Tags::Phi<Dim>>(evolved);
  // Ghost BoundaryPsi = interior value (zero jump -> zero flux correction)
  *boundary_psi = interior_boundary_psi;

  return std::nullopt;
}

template <size_t Dim>
std::optional<std::string> DirichletCharacteristics<Dim>::dg_time_derivative(
    const gsl::not_null<Scalar<DataVector>*> dt_psi_correction,
    const gsl::not_null<Scalar<DataVector>*> dt_pi_correction,
    const gsl::not_null<Scalar<DataVector>*> dt_boundary_psi_correction,
    const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
        face_mesh_velocity,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
    const Scalar<DataVector>& interior_psi,
    const Scalar<DataVector>& interior_pi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& interior_phi,
    const Scalar<DataVector>& /*interior_boundary_psi*/,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& coords,
    [[maybe_unused]] const double time) const {
  // No corrections to Psi or Pi from the time derivative path
  get(*dt_psi_correction) = 0.0;
  get(*dt_pi_correction) = 0.0;

  if (copy_psi_from_interior_) {
    // BoundaryPsi is not used; no time derivative correction needed
    get(*dt_boundary_psi_correction) = 0.0;
  } else {
    // Compute Pi_boundary from characteristic decomposition and set
    // dt_boundary_psi_correction = -Pi_boundary
    const auto boundary_values =
        evaluate_analytic<Dim>(*analytic_prescription_, coords, time);
    const auto& analytic_psi = get<SoScalarWave::Tags::Psi>(boundary_values);
    const auto& analytic_pi = get<SoScalarWave::Tags::Pi>(boundary_values);
    const auto& analytic_phi =
        get<SoScalarWave::Tags::Phi<Dim>>(boundary_values);

    const auto mixed = mix_char_modes<Dim>(
        interior_psi, interior_pi, interior_phi, normal_covector,
        face_mesh_velocity, analytic_psi, analytic_pi, analytic_phi,
        prescribe_zero_speed_modes_, zero_incoming_mode_);

    // Pi_boundary = (v_plus + v_minus) / 2
    const DataVector pi_boundary =
        0.5 * (get(mixed.v_plus_ext) + get(mixed.v_minus_ext));

    // dt BoundaryPsi = -Pi_boundary
    get(*dt_boundary_psi_correction) = -pi_boundary;
  }

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
