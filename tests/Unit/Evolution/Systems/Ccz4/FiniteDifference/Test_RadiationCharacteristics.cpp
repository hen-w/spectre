// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Evolution/Systems/Ccz4/Christoffel.hpp"
#include "Evolution/Systems/Ccz4/DerivChristoffel.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Derivatives.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/RadiationCharacteristics.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/Ccz4/Ricci.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Helpers/Evolution/Systems/Ccz4/PrimReconstructor.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"

namespace {
void test_radiation_characteristics(const size_t points_per_dimension) {
  const size_t ghost_zone_size = 3;
  const size_t fd_deriv_order = 4;
  const Mesh<::Ccz4::fd::System::volume_dim> subcell_mesh{
      points_per_dimension, Spectral::Basis::FiniteDifference,
      Spectral::Quadrature::CellCentered};

  using Affine = domain::CoordinateMaps::Affine;
  using Affine3D =
      domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;

  const std::array<double, ::Ccz4::fd::System::volume_dim> lower_bound{1.0, 1.0,
                                                                       2.3};
  const std::array<double, ::Ccz4::fd::System::volume_dim> upper_bound{1.2, 1.2,
                                                                       2.4};
  const std::array<double, ::Ccz4::fd::System::volume_dim> coords_range =
      upper_bound - lower_bound;
  const auto coord_map =
      domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
          Affine3D{
              Affine{-1., 1., lower_bound[0], upper_bound[0]},
              Affine{-1., 1., lower_bound[1], upper_bound[1]},
              Affine{-1., 1., lower_bound[2], upper_bound[2]},
          });
  // set up displaced logical coords
  const auto logical_coords =
      TestHelpers::Ccz4::fd::detail::set_logical_coordinates(subcell_mesh);
  const auto x = coord_map(logical_coords);
  InverseJacobian<DataVector, ::Ccz4::fd::System::volume_dim,
                  Frame::ElementLogical, Frame::Inertial>
      cell_centered_logical_to_inertial_inv_jacobian{
          subcell_mesh.number_of_grid_points(), 0.0};
  for (size_t i = 0; i < ::Ccz4::fd::System::volume_dim; ++i) {
    cell_centered_logical_to_inertial_inv_jacobian.get(i, i) =
        2.0 / gsl::at(coords_range, i);
  }

  const Element<::Ccz4::fd::System::volume_dim> element =
      TestHelpers::Ccz4::fd::detail::set_element();

  // Setup solution
  const double mass = 2.0;
  // Use Schwarzschild spacetime so that Psi_0 and Psi_4 are zero
  const std::array<double, ::Ccz4::fd::System::volume_dim> spin{
      {0.0, 0.0, 0.0}};
  // Put BH at the center as Psi_0 and Psi_4 are only zero
  // for n^i pointing radially from the center
  const std::array<double, ::Ccz4::fd::System::volume_dim> center{
      {0.0, 0.0, 0.0}};
  const gr::Solutions::KerrSchild solution(mass, spin, center);

  // Arbitrary time for time-independent solution
  const double t = std::numeric_limits<double>::signaling_NaN();

  const double f = Ccz4::fd::System::f;
  const bool evolve_shift = true;
  const DirectionalIdMap<::Ccz4::fd::System::volume_dim,
                         evolution::dg::subcell::GhostData>
      all_ghost_data =
          TestHelpers::Ccz4::fd::detail::compute_ghost_data<Frame::Inertial>(
              subcell_mesh, x, element.neighbors(), ghost_zone_size,
              TestHelpers::Ccz4::fd::detail::KerrSchild::
                  compute_prim_solution_for_KerrSchild,
              coords_range, t, f, evolve_shift, solution);

  auto volume_evolved_vars = TestHelpers::Ccz4::fd::detail::KerrSchild::
      compute_prim_solution_for_KerrSchild(x, t, f, evolve_shift, solution);

  Variables<db::wrap_tags_in<
      Tags::deriv, typename Ccz4::fd::System::gradients_tags,
      tmpl::size_t<::Ccz4::fd::System::volume_dim>, Frame::Inertial>>
      deriv_of_Ccz4_vars{subcell_mesh.number_of_grid_points()};

  ::Ccz4::fd::spacetime_derivatives(
      make_not_null(&deriv_of_Ccz4_vars), volume_evolved_vars, all_ghost_data,
      fd_deriv_order, subcell_mesh,
      cell_centered_logical_to_inertial_inv_jacobian);

  Variables<db::wrap_tags_in<
      Tags::second_deriv, typename Ccz4::fd::System::gradients_tags,
      tmpl::size_t<::Ccz4::fd::System::volume_dim>, Frame::Inertial>>
      second_deriv_of_Ccz4_vars{subcell_mesh.number_of_grid_points()};

  ::Ccz4::fd::second_spacetime_derivatives(
      make_not_null(&second_deriv_of_Ccz4_vars), volume_evolved_vars,
      all_ghost_data, fd_deriv_order, subcell_mesh,
      cell_centered_logical_to_inertial_inv_jacobian);

  // Compute unit_normal_vector
  const auto& conformal_spatial_metric = get<::Ccz4::Tags::ConformalMetric<
      DataVector, ::Ccz4::fd::System::volume_dim>>(volume_evolved_vars);
  const auto& conformal_factor =
      get<::Ccz4::Tags::ConformalFactor<DataVector>>(volume_evolved_vars);
  tnsr::ii<DataVector, ::Ccz4::fd::System::volume_dim, Frame::Inertial>
      spatial_metric{};
  ::tenex::evaluate<ti::i, ti::j>(
      make_not_null(&spatial_metric),
      conformal_spatial_metric(ti::i, ti::j) /
          (conformal_factor() * conformal_factor()));
  const auto& inverse_spatial_metric =
      determinant_and_inverse(spatial_metric).second;
  Scalar<DataVector> radial_coords{subcell_mesh.number_of_grid_points(), 0.0};
  for (size_t i = 0; i < ::Ccz4::fd::System::volume_dim; ++i) {
    get(radial_coords) += pow<2>(x.get(i));
  }
  get(radial_coords) = sqrt(get(radial_coords));
  auto unit_normal_vector =
      ::tenex::evaluate<ti::I>(x(ti::I) / radial_coords());
  const auto norm = ::tenex::evaluate(
      sqrt((spatial_metric(ti::i, ti::j) * unit_normal_vector(ti::I) *
            unit_normal_vector(ti::J))));
  for (size_t i = 0; i < ::Ccz4::fd::System::volume_dim; ++i) {
    unit_normal_vector.get(i) /= get(norm);
  }
  const auto unit_normal_one_form =
      raise_or_lower_index(unit_normal_vector, spatial_metric);

  // Compute expected characteristic speeds
  const auto& lapse = get<gr::Tags::Lapse<DataVector>>(volume_evolved_vars);
  const auto& shift =
      get<gr::Tags::Shift<DataVector, ::Ccz4::fd::System::volume_dim,
                          Frame::Inertial>>(volume_evolved_vars);
  const auto shift_n = dot_product(shift, unit_normal_one_form);
  std::array<DataVector, 2> expected_char_speeds{};
  expected_char_speeds[0] = -get(shift_n) + get(lapse);
  expected_char_speeds[1] = -get(shift_n) - get(lapse);

  // Compute actual characteristic speeds
  std::array<DataVector, 2> char_speeds{};
  ::Ccz4::fd::RadiationCharacteristicSpeedsCompute<Frame::Inertial>::function(
      make_not_null(&char_speeds), lapse, shift, unit_normal_one_form);

  CHECK_ITERABLE_APPROX(expected_char_speeds, char_speeds);

  // Compute characteristic fields and check they are zero
  const auto conformal_factor_squared =
      ::tenex::evaluate(conformal_factor() * conformal_factor());
  const auto inverse_conformal_spatial_metric =
      determinant_and_inverse(conformal_spatial_metric).second;
  const auto& d_conformal_spatial_metric = get<Tags::deriv<
      Ccz4::Tags::ConformalMetric<DataVector, ::Ccz4::fd::System::volume_dim,
                                  Frame::Inertial>,
      tmpl::size_t<::Ccz4::fd::System::volume_dim>, Frame::Inertial>>(
      deriv_of_Ccz4_vars);
  tnsr::ijj<DataVector, ::Ccz4::fd::System::volume_dim, Frame::Inertial>
      field_d{};
  ::tenex::evaluate<ti::k, ti::i, ti::j>(
      make_not_null(&field_d),
      0.5 * d_conformal_spatial_metric(ti::k, ti::i, ti::j));
  const auto conformal_christoffel = ::Ccz4::conformal_christoffel_second_kind(
      inverse_conformal_spatial_metric, field_d);
  const auto& d_conformal_factor =
      get<Tags::deriv<Ccz4::Tags::ConformalFactor<DataVector>,
                      tmpl::size_t<::Ccz4::fd::System::volume_dim>,
                      Frame::Inertial>>(deriv_of_Ccz4_vars);
  const auto field_p =
      ::tenex::evaluate<ti::i>(d_conformal_factor(ti::i) / conformal_factor());
  const auto christoffel = ::Ccz4::christoffel_second_kind(
      conformal_spatial_metric, inverse_conformal_spatial_metric, field_p,
      conformal_christoffel);
  const auto contracted_christoffel =
      ::tenex::evaluate<ti::l>(christoffel(ti::M, ti::l, ti::m));
  tnsr::iJkk<DataVector, ::Ccz4::fd::System::volume_dim, Frame::Inertial>
      d_conformal_christoffel{};
  const auto d_d_conformal_metric = get<Tags::second_deriv<
      ::Ccz4::Tags::ConformalMetric<DataVector, ::Ccz4::fd::System::volume_dim,
                                    Frame::Inertial>,
      tmpl::size_t<::Ccz4::fd::System::volume_dim>, Frame::Inertial>>(
      second_deriv_of_Ccz4_vars);
  tnsr::iijj<DataVector, ::Ccz4::fd::System::volume_dim, Frame::Inertial>
      d_field_d{};
  ::tenex::evaluate<ti::i, ti::j, ti::k, ti::l>(
      make_not_null(&d_field_d),
      0.5 * d_d_conformal_metric(ti::i, ti::j, ti::k, ti::l));
  tnsr::iJJ<DataVector, ::Ccz4::fd::System::volume_dim, Frame::Inertial>
      field_d_up{};
  ::tenex::evaluate<ti::k, ti::I, ti::J>(
      make_not_null(&field_d_up),
      (inverse_conformal_spatial_metric)(ti::I, ti::N) *
          (inverse_conformal_spatial_metric)(ti::M, ti::J) *
          field_d(ti::k, ti::n, ti::m));
  ::Ccz4::deriv_conformal_christoffel_second_kind(
      make_not_null(&d_conformal_christoffel), inverse_conformal_spatial_metric,
      field_d, d_field_d, field_d_up);
  const auto contracted_d_conformal_christoffel_difference =
      ::tenex::evaluate<ti::i, ti::j>(
          (d_conformal_christoffel)(ti::m, ti::M, ti::i, ti::j) -
          (d_conformal_christoffel)(ti::j, ti::M, ti::i, ti::m));
  const auto contracted_field_d_up =
      ::tenex::evaluate<ti::L>((field_d_up)(ti::m, ti::M, ti::L));
  const auto& d_d_conformal_factor =
      get<Tags::second_deriv<Ccz4::Tags::ConformalFactor<DataVector>,
                             tmpl::size_t<::Ccz4::fd::System::volume_dim>,
                             Frame::Inertial>>(second_deriv_of_Ccz4_vars);
  tnsr::ii<DataVector, ::Ccz4::fd::System::volume_dim, Frame::Inertial>
      d_field_p{};
  ::tenex::evaluate<ti::i, ti::j>(
      make_not_null(&d_field_p),
      (d_d_conformal_factor)(ti::i, ti::j) / conformal_factor() -
          (d_conformal_factor(ti::i) * d_conformal_factor(ti::j)) /
              (conformal_factor() * conformal_factor()));
  tnsr::ii<DataVector, ::Ccz4::fd::System::volume_dim, Frame::Inertial>
      spatial_ricci_tensor{};
  ::Ccz4::spatial_ricci_tensor(
      make_not_null(&spatial_ricci_tensor), christoffel, contracted_christoffel,
      contracted_d_conformal_christoffel_difference, conformal_spatial_metric,
      inverse_conformal_spatial_metric, field_d, field_d_up,
      contracted_field_d_up, field_p, d_field_p);

  typename ::Ccz4::fd::Tags::RadiationCharacteristicFields<
      DataVector, ::Ccz4::fd::System::volume_dim, Frame::Inertial>::type
      radiation_char_fields{};
  ::Ccz4::fd::RadiationCharacteristicFieldsCompute<Frame::Inertial>::function(
      make_not_null(&radiation_char_fields), conformal_factor,
      conformal_factor_squared, conformal_spatial_metric, spatial_metric,
      inverse_spatial_metric,
      get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(volume_evolved_vars),
      get<::Ccz4::Tags::ATilde<DataVector, ::Ccz4::fd::System::volume_dim>>(
          volume_evolved_vars),
      get<Tags::deriv<Ccz4::Tags::ConformalFactor<DataVector>,
                      tmpl::size_t<::Ccz4::fd::System::volume_dim>,
                      Frame::Inertial>>(deriv_of_Ccz4_vars),
      get<Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DataVector>,
                      tmpl::size_t<::Ccz4::fd::System::volume_dim>,
                      Frame::Inertial>>(deriv_of_Ccz4_vars),
      get<Tags::deriv<
          Ccz4::Tags::ConformalMetric<
              DataVector, ::Ccz4::fd::System::volume_dim, Frame::Inertial>,
          tmpl::size_t<::Ccz4::fd::System::volume_dim>, Frame::Inertial>>(
          deriv_of_Ccz4_vars),
      get<Tags::deriv<
          ::Ccz4::Tags::ATilde<DataVector, ::Ccz4::fd::System::volume_dim,
                               Frame::Inertial>,
          tmpl::size_t<::Ccz4::fd::System::volume_dim>, Frame::Inertial>>(
          deriv_of_Ccz4_vars),
      spatial_ricci_tensor, christoffel, unit_normal_one_form);

  // Error dominated by FD truncation error, as convergence test gives 4th order
  // cleanly.
  const Approx custom_approx = Approx::custom().epsilon(1.0e-9).scale(1.0);
  double global_max_abs = 0.0;
  tmpl::for_each<typename ::Ccz4::fd::Tags::RadiationCharacteristicFields<
      DataVector, ::Ccz4::fd::System::volume_dim,
      Frame::Inertial>::type::tags_list>(
      [&radiation_char_fields, &custom_approx,
       &global_max_abs]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
        const std::string tag_name = db::tag_name<Tag>();
        CAPTURE(tag_name);
        for (const auto& component : get<Tag>(radiation_char_fields)) {
          for (size_t i = 0; i < component.size(); ++i) {
            global_max_abs = std::max(global_max_abs, std::abs(component[i]));
          }
          CHECK_ITERABLE_CUSTOM_APPROX(
              component, (DataVector{component.size(), 0.0}), custom_approx);
        }
      });
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Ccz4.RadiationCharacteristics",
                  "[Unit][Evolution]") {
  const size_t points_per_dimension = 20;
  test_radiation_characteristics(points_per_dimension);
}
