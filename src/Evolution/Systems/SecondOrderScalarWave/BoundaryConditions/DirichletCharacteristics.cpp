// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/SecondOrderScalarWave/BoundaryConditions/DirichletCharacteristics.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <pup.h>
#include <type_traits>

#include "Evolution/Systems/SecondOrderScalarWave/Characteristics.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/Factory.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"

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

constexpr char moving_mesh_error[] =
    "DirichletCharacteristics does not support moving meshes: the "
    "characteristic speeds are defined without a mesh velocity.";
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
  if (face_mesh_velocity.has_value()) {
    return moving_mesh_error;
  }
  const auto interior_char_fields =
      characteristic_fields(interior_pi, interior_phi, normal_covector);

  // The characteristic speeds are constant (+1, -1, 0 for v^+, v^-, v^0), so
  // the mode selection has no per-point branching: v^+ and the zero-speed
  // v^0 are always taken from the interior, v^- is always incoming (analytic
  // or zero).
  const auto& v_plus_ext = get<Tags::VPlus>(interior_char_fields);
  const auto& v_zero_ext = get<Tags::VZero<Dim>>(interior_char_fields);
  Scalar<DataVector> v_minus_ext{};
  if (zero_incoming_mode_) {
    get(v_minus_ext) = DataVector{get(interior_pi).size(), 0.0};
  } else {
    const auto boundary_values =
        evaluate_analytic<Dim>(*analytic_prescription_, coords, time);
    const auto analytic_char_fields = characteristic_fields(
        get<Tags::Pi>(boundary_values), get<Tags::Phi<Dim>>(boundary_values),
        normal_covector);
    v_minus_ext = get<Tags::VMinus>(analytic_char_fields);
  }

  const auto evolved = fields_from_inverse_characteristic_transform(
      v_zero_ext, v_plus_ext, v_minus_ext, normal_covector);

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
  if (face_mesh_velocity.has_value()) {
    return moving_mesh_error;
  }
  // Pi on the boundary from the mixed characteristic modes:
  //   Pi_boundary = 0.5 (v^+ + v^-),
  // with the outgoing v^+ from the interior and the incoming v^- from the
  // analytic data (or zero). dt BoundaryPsi = -Pi_boundary.
  const auto interior_char_fields =
      characteristic_fields(interior_pi, interior_phi, normal_covector);
  const auto& v_plus_ext = get(get<Tags::VPlus>(interior_char_fields));
  if (zero_incoming_mode_) {
    get(*dt_boundary_psi) = -0.5 * v_plus_ext;
  } else {
    const auto boundary_values =
        evaluate_analytic<Dim>(*analytic_prescription_, coords, time);
    const auto analytic_char_fields = characteristic_fields(
        get<Tags::Pi>(boundary_values), get<Tags::Phi<Dim>>(boundary_values),
        normal_covector);
    get(*dt_boundary_psi) =
        -0.5 * (v_plus_ext + get(get<Tags::VMinus>(analytic_char_fields)));
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
