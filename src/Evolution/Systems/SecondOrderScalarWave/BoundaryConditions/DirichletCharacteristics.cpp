// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/SecondOrderScalarWave/BoundaryConditions/DirichletCharacteristics.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <pup.h>
#include <type_traits>

#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "Evolution/Systems/SecondOrderScalarWave/Characteristics.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/Factory.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Math.hpp"

namespace SecondOrderScalarWave::BoundaryConditions {
template <size_t Dim>
DirichletCharacteristics<Dim>::DirichletCharacteristics(
    const DirichletCharacteristics& rhs)
    : BoundaryCondition<Dim>{dynamic_cast<const BoundaryCondition<Dim>&>(rhs)},
      analytic_prescription_(rhs.analytic_prescription_->get_clone()),
      zero_incoming_mode_(rhs.zero_incoming_mode_) {}

template <size_t Dim>
DirichletCharacteristics<Dim>& DirichletCharacteristics<Dim>::operator=(
    const DirichletCharacteristics& rhs) {
  if (&rhs == this) {
    return *this;
  }
  analytic_prescription_ = rhs.analytic_prescription_->get_clone();
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
    const bool zero_incoming_mode)
    : analytic_prescription_(std::move(analytic_prescription)),
      zero_incoming_mode_(zero_incoming_mode) {}

template <size_t Dim>
std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
DirichletCharacteristics<Dim>::get_clone() const {
  return std::make_unique<DirichletCharacteristics>(*this);
}

template <size_t Dim>
void DirichletCharacteristics<Dim>::pup(PUP::er& p) {
  BoundaryCondition<Dim>::pup(p);
  p | analytic_prescription_;
  p | zero_incoming_mode_;
}

namespace {
// Evaluate the analytic solution at the face coordinates
template <size_t Dim>
auto evaluate_analytic(
    const evolution::initial_data::InitialData& analytic_prescription,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& coords,
    const double time) {
  using tags = tmpl::list<SecondOrderScalarWave::Tags::Psi,
                          SecondOrderScalarWave::Tags::Pi,
                          SecondOrderScalarWave::Tags::Phi<Dim>>;
  return call_with_dynamic_type<
      tuples::tagged_tuple_from_typelist<tags>,
      SecondOrderScalarWave::Solutions::all_solutions<Dim>>(
      &analytic_prescription,
      [&coords, &time](const auto* const analytic_solution_or_data) {
        if constexpr (is_analytic_solution_v<
                          std::decay_t<decltype(*analytic_solution_or_data)>>) {
          return analytic_solution_or_data->variables(coords, time, tags{});
        } else {
          (void)time;
          return analytic_solution_or_data->variables(coords, tags{});
        }
      });
}

// Compute the per-point selected characteristic modes for the ghost state.
//
// The grid-frame speeds [VZero, VPlus, VMinus] are evaluated from
// `characteristic_speeds(normal, mesh_velocity)`. Pointwise, a mode with
// non-negative speed is taken from the interior, a mode with negative speed
// is incoming and taken from the data (the analytic prescription, or zero
// with `zero_incoming`). Selection uses Heaviside masks, with the convention
// step_function(0) == 1 so that a mode at speed exactly zero is interior.
template <size_t Dim>
Variables<tmpl::list<SecondOrderScalarWave::Tags::VZero<Dim>,
                     SecondOrderScalarWave::Tags::VPlus,
                     SecondOrderScalarWave::Tags::VMinus>>
selected_characteristic_modes(
    const Scalar<DataVector>& interior_pi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& interior_phi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& coords, const double time,
    const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
        face_mesh_velocity,
    const bool zero_incoming,
    const evolution::initial_data::InitialData& analytic_prescription) {
  const std::array<DataVector, 3> speeds =
      characteristic_speeds(normal_covector, face_mesh_velocity);
  // Interior char fields (non-negative-speed modes are kept from these).
  const auto interior_char_fields =
      characteristic_fields(interior_pi, interior_phi, normal_covector);

  // The step-function masks: 1 where the mode is interior (speed >= 0), 0
  // where it is incoming (speed < 0).
  const DataVector interior_mask_zero = step_function(speeds[0]);
  const DataVector interior_mask_plus = step_function(speeds[1]);
  const DataVector interior_mask_minus = step_function(speeds[2]);

  auto selected = interior_char_fields;
  auto& v_zero_sel = get<SecondOrderScalarWave::Tags::VZero<Dim>>(selected);
  auto& v_plus_sel = get(get<SecondOrderScalarWave::Tags::VPlus>(selected));
  auto& v_minus_sel = get(get<SecondOrderScalarWave::Tags::VMinus>(selected));

  if (zero_incoming) {
    // data == 0, so selected_X = interior_mask_X * interior_X.
    for (size_t i = 0; i < Dim; ++i) {
      v_zero_sel.get(i) *= interior_mask_zero;
    }
    v_plus_sel *= interior_mask_plus;
    v_minus_sel *= interior_mask_minus;
  } else {
    const auto data_values =
        evaluate_analytic<Dim>(analytic_prescription, coords, time);
    const auto data_char_fields = characteristic_fields(
        get<SecondOrderScalarWave::Tags::Pi>(data_values),
        get<SecondOrderScalarWave::Tags::Phi<Dim>>(data_values),
        normal_covector);
    const auto& v_zero_data =
        get<SecondOrderScalarWave::Tags::VZero<Dim>>(data_char_fields);
    const auto& v_plus_data =
        get(get<SecondOrderScalarWave::Tags::VPlus>(data_char_fields));
    const auto& v_minus_data =
        get(get<SecondOrderScalarWave::Tags::VMinus>(data_char_fields));
    for (size_t i = 0; i < Dim; ++i) {
      v_zero_sel.get(i) = interior_mask_zero * v_zero_sel.get(i) +
                          (1.0 - interior_mask_zero) * v_zero_data.get(i);
    }
    v_plus_sel = interior_mask_plus * v_plus_sel +
                 (1.0 - interior_mask_plus) * v_plus_data;
    v_minus_sel = interior_mask_minus * v_minus_sel +
                  (1.0 - interior_mask_minus) * v_minus_data;
  }
  return selected;
}
}  // namespace

template <size_t Dim>
std::optional<std::string> DirichletCharacteristics<Dim>::dg_ghost(
    const gsl::not_null<Scalar<DataVector>*> psi,
    const gsl::not_null<Scalar<DataVector>*> pi,
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*> phi,
    const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
        face_mesh_velocity,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
    const Scalar<DataVector>& boundary_psi_value,
    const Scalar<DataVector>& interior_pi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& interior_phi,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& coords,
    const double time) const {
  // Per-point mode selection from the grid-frame characteristic speeds:
  // non-negative-speed modes come from the interior, negative-speed
  // (incoming) modes from the analytic data (or zero).
  const auto selected = selected_characteristic_modes<Dim>(
      interior_pi, interior_phi, normal_covector, coords, time,
      face_mesh_velocity, zero_incoming_mode_, *analytic_prescription_);

  const auto evolved = fields_from_inverse_characteristic_transform(
      get<Tags::VZero<Dim>>(selected), get<Tags::VPlus>(selected),
      get<Tags::VMinus>(selected), normal_covector);

  // The ghost Psi is the time-integrated boundary-evolved value.
  *psi = boundary_psi_value;
  *pi = get<Tags::Pi>(evolved);
  *phi = get<Tags::Phi<Dim>>(evolved);

  return std::nullopt;
}

template <size_t Dim>
std::optional<std::string>
DirichletCharacteristics<Dim>::boundary_field_time_derivatives(
    const gsl::not_null<Scalar<DataVector>*> dt_boundary_psi,
    const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
        face_mesh_velocity,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
    const Scalar<DataVector>& /*boundary_psi_value*/,
    const Scalar<DataVector>& interior_pi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& interior_phi,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& coords,
    const double time) const {
  // From the same mixed-mode ghost state that dg_ghost reconstructs, the
  // boundary (Pi, Phi) follow from the inverse characteristic transform:
  //   Pi_b     = 0.5 (v^+_sel + v^-_sel),
  //   (Phi_b)_i = 0.5 (v^+_sel - v^-_sel) n_i + (v^0_sel)_i,
  // and dt BoundaryPsi = -Pi_b + v^i (Phi_b)_i.
  const auto selected = selected_characteristic_modes<Dim>(
      interior_pi, interior_phi, normal_covector, coords, time,
      face_mesh_velocity, zero_incoming_mode_, *analytic_prescription_);

  const auto ghost = fields_from_inverse_characteristic_transform(
      get<Tags::VZero<Dim>>(selected), get<Tags::VPlus>(selected),
      get<Tags::VMinus>(selected), normal_covector);

  get(*dt_boundary_psi) = -get(get<Tags::Pi>(ghost));
  if (face_mesh_velocity.has_value()) {
    get(*dt_boundary_psi) +=
        get(dot_product(*face_mesh_velocity, get<Tags::Phi<Dim>>(ghost)));
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
}  // namespace SecondOrderScalarWave::BoundaryConditions
