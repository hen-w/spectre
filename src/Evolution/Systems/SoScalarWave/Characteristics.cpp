// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/SoScalarWave/Characteristics.hpp"

#include <array>

#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/SetNumberOfGridPoints.hpp"
#include "Utilities/TMPL.hpp"

namespace SoScalarWave {
template <size_t Dim>
void characteristic_speeds(
    const gsl::not_null<std::array<DataVector, 4>*> char_speeds,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& unit_normal_one_form,
    const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
        mesh_velocity) {
  set_number_of_grid_points(char_speeds, unit_normal_one_form);
  if (mesh_velocity.has_value()) {
    const auto vg_dot_n = dot_product(*mesh_velocity, unit_normal_one_form);
    (*char_speeds)[0] = -get(vg_dot_n);        // v(VPsi)
    (*char_speeds)[1] = -get(vg_dot_n);        // v(VZero)
    (*char_speeds)[2] = 1. - get(vg_dot_n);    // v(VPlus)
    (*char_speeds)[3] = -1. - get(vg_dot_n);   // v(VMinus)
  } else {
    (*char_speeds)[0] = 0.;   // v(VPsi)
    (*char_speeds)[1] = 0.;   // v(VZero)
    (*char_speeds)[2] = 1.;   // v(VPlus)
    (*char_speeds)[3] = -1.;  // v(VMinus)
  }
}

template <size_t Dim>
std::array<DataVector, 4> characteristic_speeds(
    const tnsr::i<DataVector, Dim, Frame::Inertial>& unit_normal_one_form,
    const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
        mesh_velocity) {
  auto char_speeds = make_with_value<std::array<DataVector, 4>>(
      get<0>(unit_normal_one_form), 0.);
  characteristic_speeds(make_not_null(&char_speeds), unit_normal_one_form,
                        mesh_velocity);
  return char_speeds;
}

template <size_t Dim>
void characteristic_speeds(
    const gsl::not_null<std::array<DataVector, 4>*> char_speeds,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& unit_normal_one_form) {
  characteristic_speeds(
      char_speeds, unit_normal_one_form,
      std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>{std::nullopt});
}

template <size_t Dim>
std::array<DataVector, 4> characteristic_speeds(
    const tnsr::i<DataVector, Dim, Frame::Inertial>& unit_normal_one_form) {
  return characteristic_speeds(
      unit_normal_one_form,
      std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>{std::nullopt});
}

template <size_t Dim>
void characteristic_fields(
    const gsl::not_null<Variables<
        tmpl::list<Tags::VPsi, Tags::VZero<Dim>, Tags::VPlus, Tags::VMinus>>*>
        char_fields,
    const Scalar<DataVector>& psi, const Scalar<DataVector>& pi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& phi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& unit_normal_one_form) {
  if (UNLIKELY(char_fields->number_of_grid_points() != get(psi).size())) {
    char_fields->initialize(get(psi).size());
  }
  const auto phi_dot_normal = dot_product(unit_normal_one_form, phi);

  for (size_t i = 0; i < Dim; ++i) {
    get<Tags::VZero<Dim>>(*char_fields).get(i) =
        phi.get(i) - unit_normal_one_form.get(i) * get(phi_dot_normal);
  }

  get<Tags::VPsi>(*char_fields) = psi;

  get(get<Tags::VPlus>(*char_fields)) = get(pi) + get(phi_dot_normal);
  get(get<Tags::VMinus>(*char_fields)) = get(pi) - get(phi_dot_normal);
}

template <size_t Dim>
Variables<tmpl::list<Tags::VPsi, Tags::VZero<Dim>, Tags::VPlus, Tags::VMinus>>
characteristic_fields(
    const Scalar<DataVector>& psi, const Scalar<DataVector>& pi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& phi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& unit_normal_one_form) {
  Variables<tmpl::list<Tags::VPsi, Tags::VZero<Dim>, Tags::VPlus, Tags::VMinus>>
      char_fields(get_size(get(psi)));
  characteristic_fields(make_not_null(&char_fields), psi, pi, phi,
                        unit_normal_one_form);
  return char_fields;
}

template <size_t Dim>
void evolved_fields_from_characteristic_fields(
    const gsl::not_null<
        Variables<tmpl::list<Tags::Psi, Tags::Pi, Tags::Phi<Dim>>>*>
        evolved_fields,
    const Scalar<DataVector>& v_psi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& v_zero,
    const Scalar<DataVector>& v_plus, const Scalar<DataVector>& v_minus,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& unit_normal_one_form) {
  if (UNLIKELY(evolved_fields->number_of_grid_points() != get(v_psi).size())) {
    evolved_fields->initialize(get(v_psi).size());
  }
  get<Tags::Psi>(*evolved_fields) = v_psi;

  get<Tags::Pi>(*evolved_fields).get() = 0.5 * (get(v_plus) + get(v_minus));
  for (size_t i = 0; i < Dim; ++i) {
    get<Tags::Phi<Dim>>(*evolved_fields).get(i) =
        0.5 * (get(v_plus) - get(v_minus)) * unit_normal_one_form.get(i) +
        v_zero.get(i);
  }
}

template <size_t Dim>
Variables<tmpl::list<Tags::Psi, Tags::Pi, Tags::Phi<Dim>>>
evolved_fields_from_characteristic_fields(
    const Scalar<DataVector>& v_psi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& v_zero,
    const Scalar<DataVector>& v_plus, const Scalar<DataVector>& v_minus,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& unit_normal_one_form) {
  Variables<tmpl::list<Tags::Psi, Tags::Pi, Tags::Phi<Dim>>> evolved_fields(
      get_size(get(v_psi)));
  evolved_fields_from_characteristic_fields(make_not_null(&evolved_fields),
                                            v_psi, v_zero, v_plus, v_minus,
                                            unit_normal_one_form);
  return evolved_fields;
}
}  // namespace SoScalarWave

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                   \
  template void SoScalarWave::characteristic_speeds(                           \
      const gsl::not_null<std::array<DataVector, 4>*> char_speeds,             \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&                   \
          unit_normal_one_form,                                                \
      const std::optional<tnsr::I<DataVector, DIM(data), Frame::Inertial>>&    \
          mesh_velocity);                                                      \
  template std::array<DataVector, 4> SoScalarWave::characteristic_speeds(      \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&                   \
          unit_normal_one_form,                                                \
      const std::optional<tnsr::I<DataVector, DIM(data), Frame::Inertial>>&    \
          mesh_velocity);                                                      \
  template void SoScalarWave::characteristic_speeds(                           \
      const gsl::not_null<std::array<DataVector, 4>*> char_speeds,             \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&                   \
          unit_normal_one_form);                                               \
  template std::array<DataVector, 4> SoScalarWave::characteristic_speeds(      \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&                   \
          unit_normal_one_form);                                               \
  template struct SoScalarWave::Tags::CharacteristicSpeedsCompute<DIM(data)>;  \
  template void SoScalarWave::characteristic_fields(                           \
      const gsl::not_null<Variables<tmpl::list<                                \
          SoScalarWave::Tags::VPsi, SoScalarWave::Tags::VZero<DIM(data)>,      \
          SoScalarWave::Tags::VPlus, SoScalarWave::Tags::VMinus>>*>            \
          char_fields,                                                         \
      const Scalar<DataVector>& psi, const Scalar<DataVector>& pi,             \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>& phi,              \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&                   \
          unit_normal_one_form);                                               \
  template Variables<tmpl::list<                                               \
      SoScalarWave::Tags::VPsi, SoScalarWave::Tags::VZero<DIM(data)>,          \
      SoScalarWave::Tags::VPlus, SoScalarWave::Tags::VMinus>>                  \
  SoScalarWave::characteristic_fields(                                         \
      const Scalar<DataVector>& psi, const Scalar<DataVector>& pi,             \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>& phi,              \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&                   \
          unit_normal_one_form);                                               \
  template struct SoScalarWave::Tags::CharacteristicFieldsCompute<DIM(data)>;  \
  template void SoScalarWave::evolved_fields_from_characteristic_fields(       \
      const gsl::not_null<Variables<                                           \
          tmpl::list<SoScalarWave::Tags::Psi, SoScalarWave::Tags::Pi,          \
                     SoScalarWave::Tags::Phi<DIM(data)>>>*>                    \
          evolved_fields,                                                      \
      const Scalar<DataVector>& v_psi,                                         \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>& v_zero,           \
      const Scalar<DataVector>& v_plus, const Scalar<DataVector>& v_minus,     \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&                   \
          unit_normal_one_form);                                               \
  template Variables<                                                          \
      tmpl::list<SoScalarWave::Tags::Psi, SoScalarWave::Tags::Pi,              \
                 SoScalarWave::Tags::Phi<DIM(data)>>>                          \
  SoScalarWave::evolved_fields_from_characteristic_fields(                     \
      const Scalar<DataVector>& v_psi,                                         \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>& v_zero,           \
      const Scalar<DataVector>& v_plus, const Scalar<DataVector>& v_minus,     \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&                   \
          unit_normal_one_form);                                               \
  template struct SoScalarWave::Tags::                                         \
      EvolvedFieldsFromCharacteristicFieldsCompute<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM
