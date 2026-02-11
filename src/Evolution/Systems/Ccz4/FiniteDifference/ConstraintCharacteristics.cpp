// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Ccz4/FiniteDifference/ConstraintCharacteristics.hpp"

#include "DataStructures/Tensor/ContractFirstNIndices.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "PointwiseFunctions/GeneralRelativity/ProjectionOperators.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace Ccz4::fd {

template <typename Frame>
std::array<DataVector, 3> constraint_characteristic_speeds(
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, ::Ccz4::fd::System::volume_dim, Frame>& shift,
    const tnsr::i<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
        unit_normal_one_form) {
  auto char_speeds = make_with_value<
      typename Tags::ConstraintCharacteristicSpeeds<DataVector>::type>(
      get(lapse), 0.0);
  constraint_characteristic_speeds(make_not_null(&char_speeds), lapse, shift,
                                   unit_normal_one_form);
  return char_speeds;
}

template <typename Frame>
void constraint_characteristic_speeds(
    const gsl::not_null<std::array<DataVector, 3>*> char_speeds,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, ::Ccz4::fd::System::volume_dim, Frame>& shift,
    const tnsr::i<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
        unit_normal_one_form) {
  const DataVector shift_n = get(dot_product(shift, unit_normal_one_form));
  (*char_speeds)[0] = -shift_n;
  (*char_speeds)[1] = -shift_n + get(lapse);
  (*char_speeds)[2] = -shift_n - get(lapse);
}

template <typename Frame>
typename Tags::ConstraintCharacteristicFields<
    DataVector, ::Ccz4::fd::System::volume_dim, Frame>::type
constraint_characteristic_fields(
    const Scalar<DataVector>& theta,
    const tnsr::I<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
        upper_spatial_z4_constraint,
    const tnsr::ii<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
        spatial_metric,
    const tnsr::i<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
        unit_normal_one_form) {
  typename Tags::ConstraintCharacteristicFields<
      DataVector, ::Ccz4::fd::System::volume_dim, Frame>::type
      constraint_char_fields{};
  constraint_characteristic_fields(make_not_null(&constraint_char_fields),
                                   theta, upper_spatial_z4_constraint,
                                   spatial_metric, unit_normal_one_form);
  return constraint_char_fields;
}

template <typename Frame>
void constraint_characteristic_fields(
    const gsl::not_null<typename Tags::ConstraintCharacteristicFields<
        DataVector, ::Ccz4::fd::System::volume_dim, Frame>::type*>
        constraint_char_fields,
    const Scalar<DataVector>& theta,
    const tnsr::I<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
        upper_spatial_z4_constraint,
    const tnsr::ii<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
        spatial_metric,
    const tnsr::i<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
        unit_normal_one_form) {
  const auto number_of_grid_points = get(theta).size();
  if (UNLIKELY(number_of_grid_points !=
               constraint_char_fields->number_of_grid_points())) {
    constraint_char_fields->initialize(number_of_grid_points);
  }

  const auto q_dd =
      gr::transverse_projection_operator(spatial_metric, unit_normal_one_form);

  // Piece together c_vector_zero
  auto& c_vector_zero =
      get<::Ccz4::fd::Tags::CVectorZero<DataVector,
                                        ::Ccz4::fd::System::volume_dim, Frame>>(
          *constraint_char_fields);
  ::tenex::evaluate<ti::i>(
      make_not_null(&c_vector_zero),
      q_dd(ti::i, ti::j) * upper_spatial_z4_constraint(ti::J));

  // Piece together c_scalar_plus and c_scalar_minus
  auto& c_scalar_plus =
      get<::Ccz4::fd::Tags::CScalarPlus<DataVector>>(*constraint_char_fields);
  ::tenex::evaluate(make_not_null(&c_scalar_plus),
                    -1.0 * theta() + unit_normal_one_form(ti::i) *
                                         upper_spatial_z4_constraint(ti::I));
  auto& c_scalar_minus =
      get<::Ccz4::fd::Tags::CScalarMinus<DataVector>>(*constraint_char_fields);
  ::tenex::evaluate(make_not_null(&c_scalar_minus),
                    theta() + unit_normal_one_form(ti::i) *
                                  upper_spatial_z4_constraint(ti::I));
}
}  // namespace Ccz4::fd

#define FRAME(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(_, data)                                                 \
  template std::array<DataVector, 3>                                           \
  Ccz4::fd::constraint_characteristic_speeds<FRAME(data)>(                     \
      const Scalar<DataVector>& lapse,                                         \
      const tnsr::I<DataVector, ::Ccz4::fd::System::volume_dim, FRAME(data)>&  \
          shift,                                                               \
      const tnsr::i<DataVector, ::Ccz4::fd::System::volume_dim, FRAME(data)>&  \
          unit_normal_one_form);                                               \
  template void Ccz4::fd::constraint_characteristic_speeds<FRAME(data)>(       \
      const gsl::not_null<std::array<DataVector, 3>*> char_speeds,             \
      const Scalar<DataVector>& lapse,                                         \
      const tnsr::I<DataVector, ::Ccz4::fd::System::volume_dim, FRAME(data)>&  \
          shift,                                                               \
      const tnsr::i<DataVector, ::Ccz4::fd::System::volume_dim, FRAME(data)>&  \
          unit_normal_one_form);                                               \
  template struct Ccz4::fd::ConstraintCharacteristicSpeedsCompute<FRAME(       \
      data)>;                                                                  \
  template typename Ccz4::fd::Tags::ConstraintCharacteristicFields<            \
      DataVector, ::Ccz4::fd::System::volume_dim, FRAME(data)>::type           \
  Ccz4::fd::constraint_characteristic_fields<FRAME(data)>(                     \
      const Scalar<DataVector>& theta,                                         \
      const tnsr::I<DataVector, ::Ccz4::fd::System::volume_dim, FRAME(data)>&  \
          upper_spatial_z4_constraint,                                         \
      const tnsr::ii<DataVector, ::Ccz4::fd::System::volume_dim, FRAME(data)>& \
          spatial_metric,                                                      \
      const tnsr::i<DataVector, ::Ccz4::fd::System::volume_dim, FRAME(data)>&  \
          unit_normal_one_form);                                               \
  template void Ccz4::fd::constraint_characteristic_fields<FRAME(data)>(       \
      const gsl::not_null<                                                     \
          typename Ccz4::fd::Tags::ConstraintCharacteristicFields<             \
              DataVector, ::Ccz4::fd::System::volume_dim, FRAME(data)>::type*> \
          constraint_char_fields,                                              \
      const Scalar<DataVector>& theta,                                         \
      const tnsr::I<DataVector, ::Ccz4::fd::System::volume_dim, FRAME(data)>&  \
          upper_spatial_z4_constraint,                                         \
      const tnsr::ii<DataVector, ::Ccz4::fd::System::volume_dim, FRAME(data)>& \
          spatial_metric,                                                      \
      const tnsr::i<DataVector, ::Ccz4::fd::System::volume_dim, FRAME(data)>&  \
          unit_normal_one_form);                                               \
  template struct Ccz4::fd::ConstraintCharacteristicFieldsCompute<FRAME(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (Frame::Inertial, Frame::Grid))

#undef INSTANTIATION
#undef FRAME
