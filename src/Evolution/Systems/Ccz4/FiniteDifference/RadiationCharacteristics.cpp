// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Ccz4/FiniteDifference/RadiationCharacteristics.hpp"

#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Characteristics.hpp"
#include "PointwiseFunctions/GeneralRelativity/ProjectionOperators.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace Ccz4::fd {

template <typename Frame>
std::array<DataVector, 2> radiation_characteristic_speeds(
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, ::Ccz4::fd::System::volume_dim, Frame>& shift,
    const tnsr::i<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
        unit_normal_one_form) {
  auto char_speeds = make_with_value<
      typename Tags::RadiationCharacteristicSpeeds<DataVector>::type>(
      get(lapse), 0.0);
  radiation_characteristic_speeds(make_not_null(&char_speeds), lapse, shift,
                                  unit_normal_one_form);
  return char_speeds;
}

template <typename Frame>
void radiation_characteristic_speeds(
    const gsl::not_null<std::array<DataVector, 2>*> char_speeds,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, ::Ccz4::fd::System::volume_dim, Frame>& shift,
    const tnsr::i<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
        unit_normal_one_form) {
  const DataVector shift_n = get(dot_product(shift, unit_normal_one_form));
  (*char_speeds)[0] = -shift_n + get(lapse);
  (*char_speeds)[1] = -shift_n - get(lapse);
}

template <typename Frame>
typename Tags::RadiationCharacteristicFields<
    DataVector, ::Ccz4::fd::System::volume_dim, Frame>::type
radiation_characteristic_fields(
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& conformal_factor_squared,
    const tnsr::ii<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
        conformal_spatial_metric,
    const tnsr::ii<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
        spatial_metric,
    const tnsr::II<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
        inverse_spatial_metric,
    const Scalar<DataVector>& trace_extrinsic_curvature,
    const tnsr::ii<DataVector, ::Ccz4::fd::System::volume_dim, Frame>& a_tilde,
    const tnsr::i<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
        d_conformal_factor,
    const tnsr::i<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
        d_trace_extrinsic_curvature,
    const tnsr::ijj<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
        d_conformal_spatial_metric,
    const tnsr::ijj<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
        d_a_tilde,
    const tnsr::ii<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
        spatial_ricci_tensor,
    const tnsr::Ijj<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
        christoffel,
    const tnsr::i<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
        unit_normal_one_form) {
  typename Tags::RadiationCharacteristicFields<
      DataVector, ::Ccz4::fd::System::volume_dim, Frame>::type
      radiation_char_fields{};
  radiation_characteristic_fields(
      make_not_null(&radiation_char_fields), conformal_factor,
      conformal_factor_squared, conformal_spatial_metric, spatial_metric,
      inverse_spatial_metric, trace_extrinsic_curvature, a_tilde,
      d_conformal_factor, d_trace_extrinsic_curvature,
      d_conformal_spatial_metric, d_a_tilde, spatial_ricci_tensor, christoffel,
      unit_normal_one_form);
  return radiation_char_fields;
}

template <typename Frame>
void radiation_characteristic_fields(
    const gsl::not_null<typename Tags::RadiationCharacteristicFields<
        DataVector, ::Ccz4::fd::System::volume_dim, Frame>::type*>
        radiation_char_fields,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& conformal_factor_squared,
    const tnsr::ii<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
        conformal_spatial_metric,
    const tnsr::ii<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
        spatial_metric,
    const tnsr::II<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
        inverse_spatial_metric,
    const Scalar<DataVector>& trace_extrinsic_curvature,
    const tnsr::ii<DataVector, ::Ccz4::fd::System::volume_dim, Frame>& a_tilde,
    const tnsr::i<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
        d_conformal_factor,
    const tnsr::i<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
        d_trace_extrinsic_curvature,
    const tnsr::ijj<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
        d_conformal_spatial_metric,
    const tnsr::ijj<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
        d_a_tilde,
    const tnsr::ii<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
        spatial_ricci_tensor,
    const tnsr::Ijj<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
        christoffel,
    const tnsr::i<DataVector, ::Ccz4::fd::System::volume_dim, Frame>&
        unit_normal_one_form) {
  const auto number_of_grid_points = get(conformal_factor).size();
  if (UNLIKELY(number_of_grid_points !=
               radiation_char_fields->number_of_grid_points())) {
    radiation_char_fields->initialize(number_of_grid_points);
  }

  // Piece together the extrinsic curvature tensor
  tnsr::ii<DataVector, ::Ccz4::fd::System::volume_dim, Frame>
      extrinsic_curvature{};
  ::tenex::evaluate<ti::i, ti::j>(
      make_not_null(&extrinsic_curvature),
      a_tilde(ti::i, ti::j) / conformal_factor_squared() +
          trace_extrinsic_curvature() * spatial_metric(ti::i, ti::j) / 3.0);

  // Piece together the covariant derivative of the extrinsic curvature
  tnsr::ijj<DataVector, ::Ccz4::fd::System::volume_dim, Frame>
      covd_extrinsic_curvature{};
  ::tenex::evaluate<ti::m, ti::k, ti::l>(
      make_not_null(&covd_extrinsic_curvature),
      -2.0 / (conformal_factor_squared() * conformal_factor()) *
              a_tilde(ti::k, ti::l) * d_conformal_factor(ti::m) +
          1.0 / conformal_factor_squared() * d_a_tilde(ti::m, ti::k, ti::l) +
          1.0 / (3.0 * conformal_factor_squared()) *
              d_trace_extrinsic_curvature(ti::m) *
              conformal_spatial_metric(ti::k, ti::l) +
          1.0 / (3.0 * conformal_factor_squared()) *
              trace_extrinsic_curvature() *
              d_conformal_spatial_metric(ti::m, ti::k, ti::l) -
          2.0 / (3.0 * conformal_factor_squared() * conformal_factor()) *
              trace_extrinsic_curvature() *
              conformal_spatial_metric(ti::k, ti::l) *
              d_conformal_factor(ti::m) -
          christoffel(ti::J, ti::m, ti::k) * extrinsic_curvature(ti::j, ti::l) -
          christoffel(ti::J, ti::m, ti::l) * extrinsic_curvature(ti::j, ti::k));

  // Piece together the unprojected radiation characteristic fields
  auto& c_tensor_plus =
      get<::Ccz4::fd::Tags::CTensorPlus<DataVector,
                                        ::Ccz4::fd::System::volume_dim, Frame>>(
          *radiation_char_fields);
  ::tenex::evaluate<ti::k, ti::l>(
      make_not_null(&c_tensor_plus),
      spatial_ricci_tensor(ti::k, ti::l) +
          trace_extrinsic_curvature() * extrinsic_curvature(ti::k, ti::l) -
          inverse_spatial_metric(ti::M, ti::J) *
              extrinsic_curvature(ti::j, ti::k) *
              extrinsic_curvature(ti::m, ti::l) -
          inverse_spatial_metric(ti::M, ti::J) * unit_normal_one_form(ti::j) *
              covd_extrinsic_curvature(ti::m, ti::k, ti::l) +
          0.5 * inverse_spatial_metric(ti::M, ti::J) *
              unit_normal_one_form(ti::j) *
              (covd_extrinsic_curvature(ti::k, ti::l, ti::m) +
               covd_extrinsic_curvature(ti::l, ti::k, ti::m)));
  auto& c_tensor_minus = get<::Ccz4::fd::Tags::CTensorMinus<
      DataVector, ::Ccz4::fd::System::volume_dim, Frame>>(
      *radiation_char_fields);
  ::tenex::evaluate<ti::k, ti::l>(
      make_not_null(&c_tensor_minus),
      spatial_ricci_tensor(ti::k, ti::l) +
          trace_extrinsic_curvature() * extrinsic_curvature(ti::k, ti::l) -
          inverse_spatial_metric(ti::M, ti::J) *
              extrinsic_curvature(ti::j, ti::k) *
              extrinsic_curvature(ti::m, ti::l) +
          inverse_spatial_metric(ti::M, ti::J) * unit_normal_one_form(ti::j) *
              covd_extrinsic_curvature(ti::m, ti::k, ti::l) -
          0.5 * inverse_spatial_metric(ti::M, ti::J) *
              unit_normal_one_form(ti::j) *
              (covd_extrinsic_curvature(ti::k, ti::l, ti::m) +
               covd_extrinsic_curvature(ti::l, ti::k, ti::m)));

  // TT project to get the radiation characteristic fields
  c_tensor_plus =
      compute_tt_symmetric_tensor(c_tensor_plus, spatial_metric,
                                  inverse_spatial_metric, unit_normal_one_form);
  c_tensor_minus =
      compute_tt_symmetric_tensor(c_tensor_minus, spatial_metric,
                                  inverse_spatial_metric, unit_normal_one_form);
}
}  // namespace Ccz4::fd

#define FRAME(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(_, data)                                                 \
  template std::array<DataVector, 2>                                           \
  Ccz4::fd::radiation_characteristic_speeds<FRAME(data)>(                      \
      const Scalar<DataVector>& lapse,                                         \
      const tnsr::I<DataVector, ::Ccz4::fd::System::volume_dim, FRAME(data)>&  \
          shift,                                                               \
      const tnsr::i<DataVector, ::Ccz4::fd::System::volume_dim, FRAME(data)>&  \
          unit_normal_one_form);                                               \
  template void Ccz4::fd::radiation_characteristic_speeds<FRAME(data)>(        \
      const gsl::not_null<std::array<DataVector, 2>*> char_speeds,             \
      const Scalar<DataVector>& lapse,                                         \
      const tnsr::I<DataVector, ::Ccz4::fd::System::volume_dim, FRAME(data)>&  \
          shift,                                                               \
      const tnsr::i<DataVector, ::Ccz4::fd::System::volume_dim, FRAME(data)>&  \
          unit_normal_one_form);                                               \
  template struct Ccz4::fd::RadiationCharacteristicSpeedsCompute<FRAME(data)>; \
  template typename Ccz4::fd::Tags::RadiationCharacteristicFields<             \
      DataVector, ::Ccz4::fd::System::volume_dim, FRAME(data)>::type           \
  Ccz4::fd::radiation_characteristic_fields<FRAME(data)>(                      \
      const Scalar<DataVector>& conformal_factor,                              \
      const Scalar<DataVector>& conformal_factor_squared,                      \
      const tnsr::ii<DataVector, ::Ccz4::fd::System::volume_dim, FRAME(data)>& \
          conformal_spatial_metric,                                            \
      const tnsr::ii<DataVector, ::Ccz4::fd::System::volume_dim, FRAME(data)>& \
          spatial_metric,                                                      \
      const tnsr::II<DataVector, ::Ccz4::fd::System::volume_dim, FRAME(data)>& \
          inverse_spatial_metric,                                              \
      const Scalar<DataVector>& trace_extrinsic_curvature,                     \
      const tnsr::ii<DataVector, ::Ccz4::fd::System::volume_dim, FRAME(data)>& \
          a_tilde,                                                             \
      const tnsr::i<DataVector, ::Ccz4::fd::System::volume_dim, FRAME(data)>&  \
          d_conformal_factor,                                                  \
      const tnsr::i<DataVector, ::Ccz4::fd::System::volume_dim, FRAME(data)>&  \
          d_trace_extrinsic_curvature,                                         \
      const tnsr::ijj<DataVector, ::Ccz4::fd::System::volume_dim,              \
                      FRAME(data)>& d_conformal_spatial_metric,                \
      const tnsr::ijj<DataVector, ::Ccz4::fd::System::volume_dim,              \
                      FRAME(data)>& d_a_tilde,                                 \
      const tnsr::ii<DataVector, ::Ccz4::fd::System::volume_dim, FRAME(data)>& \
          spatial_ricci_tensor,                                                \
      const tnsr::Ijj<DataVector, ::Ccz4::fd::System::volume_dim,              \
                      FRAME(data)>& christoffel,                               \
      const tnsr::i<DataVector, ::Ccz4::fd::System::volume_dim, FRAME(data)>&  \
          unit_normal_one_form);                                               \
  template void Ccz4::fd::radiation_characteristic_fields<FRAME(data)>(        \
      const gsl::not_null<                                                     \
          typename Ccz4::fd::Tags::RadiationCharacteristicFields<              \
              DataVector, ::Ccz4::fd::System::volume_dim, FRAME(data)>::type*> \
          radiation_char_fields,                                               \
      const Scalar<DataVector>& conformal_factor,                              \
      const Scalar<DataVector>& conformal_factor_squared,                      \
      const tnsr::ii<DataVector, ::Ccz4::fd::System::volume_dim, FRAME(data)>& \
          conformal_spatial_metric,                                            \
      const tnsr::ii<DataVector, ::Ccz4::fd::System::volume_dim, FRAME(data)>& \
          spatial_metric,                                                      \
      const tnsr::II<DataVector, ::Ccz4::fd::System::volume_dim, FRAME(data)>& \
          inverse_spatial_metric,                                              \
      const Scalar<DataVector>& trace_extrinsic_curvature,                     \
      const tnsr::ii<DataVector, ::Ccz4::fd::System::volume_dim, FRAME(data)>& \
          a_tilde,                                                             \
      const tnsr::i<DataVector, ::Ccz4::fd::System::volume_dim, FRAME(data)>&  \
          d_conformal_factor,                                                  \
      const tnsr::i<DataVector, ::Ccz4::fd::System::volume_dim, FRAME(data)>&  \
          d_trace_extrinsic_curvature,                                         \
      const tnsr::ijj<DataVector, ::Ccz4::fd::System::volume_dim,              \
                      FRAME(data)>& d_conformal_spatial_metric,                \
      const tnsr::ijj<DataVector, ::Ccz4::fd::System::volume_dim,              \
                      FRAME(data)>& d_a_tilde,                                 \
      const tnsr::ii<DataVector, ::Ccz4::fd::System::volume_dim, FRAME(data)>& \
          spatial_ricci_tensor,                                                \
      const tnsr::Ijj<DataVector, ::Ccz4::fd::System::volume_dim,              \
                      FRAME(data)>& christoffel,                               \
      const tnsr::i<DataVector, ::Ccz4::fd::System::volume_dim, FRAME(data)>&  \
          unit_normal_one_form);                                               \
  template struct Ccz4::fd::RadiationCharacteristicFieldsCompute<FRAME(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (Frame::Inertial, Frame::Grid))

#undef INSTANTIATION
#undef FRAME
