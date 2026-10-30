// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <pup.h>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/IndexIterator.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Block.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/Creators/Rectilinear.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/NumericalAlgorithms/Spectral/DiskTestFunctions.hpp"
#include "Helpers/NumericalAlgorithms/Spectral/FourierTestFunctions.hpp"
#include "Helpers/NumericalAlgorithms/SphericalHarmonics/YlmTestFunctions.hpp"
#include "NumericalAlgorithms/Interpolation/IrregularInterpolant.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/CollocationPoints.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/TovStar.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "PointwiseFunctions/MathFunctions/PowX.hpp"
#include "PointwiseFunctions/MathFunctions/TensorProduct.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace {
// Polynomial of given degree with leading coefficinet \f$a_0\f$, and with
// \f$a_{n+1} = a_n / falloff
class Polynomial {
 public:
  Polynomial(const size_t degree, const double a_0, const double falloff)
      : coefficients_(degree + 1) {
    double n = falloff * a_0;
    std::generate(coefficients_.begin(), coefficients_.end(), [&falloff, &n]() {
      n /= falloff;
      return n;
    });
  }
  Polynomial() = default;
  DataVector operator()(const DataVector& x) const {
    return evaluate_polynomial(coefficients_, x);
  }

 private:
  std::vector<double> coefficients_;
};

using Affine = domain::CoordinateMaps::Affine;
using Affine2D = domain::CoordinateMaps::ProductOf2Maps<Affine, Affine>;
using Affine3D = domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;

const double inertial_coord_min = -0.3;
const double inertial_coord_max = 0.7;

template <size_t VolumeDim>
auto make_affine_map();

template <>
auto make_affine_map<1>() {
  return domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
      Affine{-1.0, 1.0, inertial_coord_min, inertial_coord_max});
}

template <>
auto make_affine_map<2>() {
  return domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
      Affine2D{Affine{-1.0, 1.0, inertial_coord_min, inertial_coord_max},
               Affine{-1.0, 1.0, inertial_coord_min, inertial_coord_max}});
}

template <>
auto make_affine_map<3>() {
  return domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
      Affine3D{Affine{-1.0, 1.0, inertial_coord_min, inertial_coord_max},
               Affine{-1.0, 1.0, inertial_coord_min, inertial_coord_max},
               Affine{-1.0, 1.0, inertial_coord_min, inertial_coord_max}});
}

namespace TestTags {

template <typename DataType, size_t Dim>
struct Vector : db::SimpleTag {
  using type = tnsr::I<DataType, Dim>;
  static auto fill_values(const MathFunctions::TensorProduct<Dim>& f,
                          const tnsr::I<DataVector, Dim>& x) {
    auto result = make_with_value<tnsr::I<DataType, Dim>>(x, 0.);
    const auto f_of_x = f(x);
    for (size_t d = 0; d < Dim; ++d) {
      result.get(d) = (d + 0.5) * get(f_of_x);
    }
    return result;
  }
};

template <typename DataType, size_t Dim>
struct SymmetricTensor : db::SimpleTag {
  using type = tnsr::ii<DataType, Dim>;
  static auto fill_values(const MathFunctions::TensorProduct<Dim>& f,
                          const tnsr::I<DataVector, Dim>& x) {
    auto result = make_with_value<tnsr::ii<DataType, Dim>>(x, 0.);
    const auto f_of_x = f(x);
    for (size_t i = 0; i < Dim; ++i) {
      for (size_t j = i; j < Dim; ++j) {  // Symmetry
        result.get(i, j) = (static_cast<double>(i + j) + 0.33) * get(f_of_x);
      }
    }
    return result;
  }
};

}  // namespace TestTags

template <size_t Dim>
void test_interpolate_to_points(const Mesh<Dim>& mesh) {
  // Fill target interpolation coordinates with random values
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> dist(inertial_coord_min, inertial_coord_max);

  const auto nn_generator = make_not_null(&generator);
  const auto nn_dist = make_not_null(&dist);

  const size_t number_of_points = 6;
  const auto target_x_inertial =
      make_with_random_values<tnsr::I<DataVector, Dim>>(
          nn_generator, nn_dist, DataVector(number_of_points));

  const auto coordinate_map = make_affine_map<Dim>();
  const auto target_x = [&target_x_inertial, &coordinate_map,
                         &number_of_points]() {
    tnsr::I<DataVector, Dim, Frame::ElementLogical> result(number_of_points);
    for (size_t s = 0; s < number_of_points; ++s) {
      tnsr::I<double, Dim> x_inertial_local{};
      for (size_t d = 0; d < Dim; ++d) {
        x_inertial_local.get(d) = target_x_inertial.get(d)[s];
      }
      const auto x_local = coordinate_map.inverse(x_inertial_local).value();
      for (size_t d = 0; d < Dim; ++d) {
        result.get(d)[s] = x_local.get(d);
      }
    }
    return result;
  }();

  // Set up interpolator. Need do this only once.
  const intrp::Irregular<Dim> irregular_interpolant(mesh, target_x);
  test_serialization(irregular_interpolant);

  // ... but we construct another interpolator to test operator!=
  {
    auto target_x_new = target_x;
    target_x_new.get(0)[0] *= 0.98;  // Change one point slightly.
    const intrp::Irregular<Dim> irregular_interpolant_new(mesh, target_x_new);
    CHECK(irregular_interpolant_new != irregular_interpolant);
  }

  // ... and another to test the constructor from a single point
  {
    tnsr::I<double, Dim, Frame::ElementLogical> target_x_single{};
    tnsr::I<DataVector, Dim, Frame::ElementLogical> target_x_single_dv(1_st);
    for (size_t d = 0; d < Dim; ++d) {
      target_x_single.get(d) = target_x.get(d)[0];
      target_x_single_dv.get(d)[0] = target_x_single.get(d);
    }
    CHECK(intrp::Irregular<Dim>(mesh, target_x_single) ==
          intrp::Irregular<Dim>(mesh, target_x_single_dv));
  }

  // Coordinates on the grid
  const auto src_x = coordinate_map(logical_coordinates(mesh));

  // Set up variables
  using tags = tmpl::list<TestTags::Vector<DataVector, Dim>,
                          TestTags::SymmetricTensor<DataVector, Dim>>;
  Variables<tags> src_vars(mesh.number_of_grid_points());
  Variables<tags> expected_dest_vars(number_of_points);

  // We will make polynomials of the form x^a y^b z^c ...
  // for all a,b,c, that result in exact interpolation.
  // IndexIterator loops over "a,b,c"
  for (IndexIterator<Dim> iter(mesh.extents()); iter; ++iter) {
    // Set up analytic solution.  We fill a Variables with this solution,
    // interpolate to arbitrary points, and then check that the
    // values at arbitrary points match this solution.
    // We choose polynomials so that interpolation is exact on an LGL grid.
    std::array<std::unique_ptr<MathFunction<1, Frame::Inertial>>, Dim>
        functions;
    for (size_t d = 0; d < Dim; ++d) {
      gsl::at(functions, d) =
          std::make_unique<MathFunctions::PowX<1, Frame::Inertial>>(iter()[d]);
    }
    MathFunctions::TensorProduct<Dim> f(1.0, std::move(functions));

    // Fill source and expected destination Variables with analytic solution.
    tmpl::for_each<tags>([&f, &src_x, &target_x_inertial, &src_vars,
                          &expected_dest_vars](auto tag) {
      using Tag = tmpl::type_from<decltype(tag)>;
      get<Tag>(src_vars) = Tag::fill_values(f, src_x);
      get<Tag>(expected_dest_vars) = Tag::fill_values(f, target_x_inertial);
    });

    // Interpolate
    // (g++ 7.2.0 does not allow `const auto dest_vars` here)
    const Variables<tags> dest_vars =
        irregular_interpolant.interpolate(src_vars);

    CHECK_VARIABLES_APPROX(dest_vars, expected_dest_vars);

    const DataVector result_dv = irregular_interpolant.interpolate(
        get<0>(get<TestTags::Vector<DataVector, Dim>>(src_vars)));
    CHECK_ITERABLE_APPROX(
        result_dv,
        get<0>(get<TestTags::Vector<DataVector, Dim>>(expected_dest_vars)));

    {
      INFO("Complex data");
      // Copy the real data above into a complex Variables
      Variables<tmpl::list<TestTags::Vector<ComplexDataVector, Dim>>>
          src_vars_complex(mesh.number_of_grid_points());
      Variables<tmpl::list<TestTags::Vector<ComplexDataVector, Dim>>>
          expected_complex(number_of_points);
      const auto& src_vector = get<TestTags::Vector<DataVector, Dim>>(src_vars);
      const auto& expected_vector =
          get<TestTags::Vector<DataVector, Dim>>(expected_dest_vars);
      for (size_t d = 0; d < Dim; ++d) {
        for (size_t j = 0; j < src_vars.number_of_grid_points(); ++j) {
          get<TestTags::Vector<ComplexDataVector, Dim>>(src_vars_complex)
              .get(d)[j] = std::complex<double>(src_vector.get(d)[j],
                                                2. * src_vector.get(d)[j]);
        }
        for (size_t j = 0; j < number_of_points; ++j) {
          get<TestTags::Vector<ComplexDataVector, Dim>>(expected_complex)
              .get(d)[j] = std::complex<double>(expected_vector.get(d)[j],
                                                2. * expected_vector.get(d)[j]);
        }
      }
      // Interpolate the complex data
      const auto result_complex =
          irregular_interpolant.interpolate(src_vars_complex);
      CHECK_VARIABLES_APPROX(result_complex, expected_complex);
      const auto result_cdv = irregular_interpolant.interpolate(get<0>(
          get<TestTags::Vector<ComplexDataVector, Dim>>(src_vars_complex)));
      CHECK_ITERABLE_APPROX(
          result_cdv, get<0>(get<TestTags::Vector<ComplexDataVector, Dim>>(
                          expected_complex)));
      std::vector<std::complex<double>> result_data(expected_complex.size());
      auto result_span = gsl::make_span(result_data);
      irregular_interpolant.interpolate(
          make_not_null(&result_span),
          gsl::make_span(src_vars_complex.data(), src_vars_complex.size()));
    }
    {
      INFO("Single precision data");
      // Copy the data above into single precision
      std::vector<float> src_vars_single(src_vars.size());
      std::vector<float> expected_dest_single(expected_dest_vars.size());
      std::copy_n(src_vars.data(), src_vars.size(), src_vars_single.data());
      std::copy_n(expected_dest_vars.data(), expected_dest_vars.size(),
                  expected_dest_single.data());
      // Interpolate the single precision data
      std::vector<float> result_single(expected_dest_single.size());
      auto result_single_span = gsl::make_span(result_single);
      irregular_interpolant.interpolate(make_not_null(&result_single_span),
                                        gsl::make_span(src_vars_single));
      const Approx custom_approx =
          Approx::custom()
              .epsilon(std::numeric_limits<float>::epsilon() * 10.)
              .scale(1.);
      CHECK_ITERABLE_CUSTOM_APPROX(result_single, expected_dest_single,
                                   custom_approx);
    }
  }
}

template <Spectral::Basis Basis, Spectral::Quadrature Quadrature>
void test_irregular_interpolant() {
  const size_t start_points = 4;
  const size_t end_points = 6;
  for (size_t n0 = start_points; n0 < end_points; ++n0) {
    test_interpolate_to_points<1>(Mesh<1>{n0, Basis, Quadrature});
    for (size_t n1 = start_points; n1 < end_points; ++n1) {
      test_interpolate_to_points<2>(Mesh<2>{{{n0, n1}}, Basis, Quadrature});
      for (size_t n2 = start_points; n2 < end_points; ++n2) {
        test_interpolate_to_points<3>(
            Mesh<3>{{{n0, n1, n2}}, Basis, Quadrature});
      }
    }
  }
}

void test_irregular_interpolant_mixed_quadrature() {
  const size_t start_points = 4;
  const size_t end_points = 6;
  for (size_t n0 = start_points; n0 < end_points; ++n0) {
    for (size_t n1 = start_points; n1 < end_points; ++n1) {
      test_interpolate_to_points<2>(Mesh<2>{
          {{n0, n1}},
          {{Spectral::Basis::Legendre, Spectral::Basis::Legendre}},
          {{Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::Gauss}}});
      for (size_t n2 = start_points; n2 < end_points; ++n2) {
        test_interpolate_to_points<3>(Mesh<3>{
            {{n0, n1, n2}},
            {{Spectral::Basis::Legendre, Spectral::Basis::Legendre,
              Spectral::Basis::Legendre}},
            {{Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::Gauss,
              Spectral::Quadrature::GaussLobatto}}});
      }
    }
  }
}

template <size_t Dim>
Domain<Dim> create_domain(double length,
                          const std::array<size_t, Dim>& extents);

template <>
Domain<3> create_domain<3>(const double length,
                           const std::array<size_t, 3>& extents) {
  const domain::creators::Brick creator{{{0.0, 0.0, 0.0}},
                                        {{length, length, length}},
                                        {{0, 0, 0}},
                                        extents,
                                        {{false, false, false}}};
  return creator.create_domain();
}

template <>
Domain<2> create_domain<2>(const double length,
                           const std::array<size_t, 2>& extents) {
  const domain::creators::Rectangle creator{
      {{0.0, 0.0}}, {{length, length}}, {{0, 0}}, extents, {{false, false}}};
  return creator.create_domain();
}

template <>
Domain<1> create_domain<1>(const double length,
                           const std::array<size_t, 1>& extents) {
  const domain::creators::Interval creator{{{0.0}}, {{length}}, {{0}}, extents};
  return creator.create_domain();
}

template <size_t Dim>
tnsr::I<DataVector, Dim, Frame::ElementLogical> create_target_points(
    size_t n_random_target_points) {
  tnsr::I<DataVector, Dim, Frame::ElementLogical> xi_target{
      n_random_target_points + 2, -1.0};
  for (size_t d = 0; d < Dim; ++d) {
    xi_target.get(d)[1] = 1.0;
  }
  return xi_target;
}

namespace Tags {
struct Scalar : ::db::SimpleTag {
  using type = ::Scalar<DataVector>;
};
}  // namespace Tags

using var_tags = tmpl::list<Tags::Scalar>;

template <size_t Dim>
Variables<var_tags> polynomial(
    const tnsr::I<DataVector, Dim, Frame::Inertial>& x, const size_t degree) {
  tnsr::I<DataVector, Dim, Frame::Inertial> v = x;
  for (size_t d = 0; d < Dim; ++d) {
    for (size_t n = degree; n > 1; --n) {
      v.get(d) *= (x.get(d) + 1.0);
    }
  }
  Variables<var_tags> result{get<0>(x).size()};
  get(get<Tags::Scalar>(result)) = get<0>(v);
  for (size_t d = 1; d < Dim; ++d) {
    get(get<Tags::Scalar>(result)) *= v.get(d);
  }
  return result;
}

template <size_t Dim>
void test_polynomial_interpolant(const std::array<size_t, Dim>& extents,
                                 const size_t max_degree) {
  const size_t n_random_target_points = 10;
  // use a small domain to avoid huge polynomial values for large max_degree
  // which results in large absolute errors
  const auto domain = create_domain<Dim>(1.5, extents);
  const auto& block = domain.blocks()[0];
  const ElementMap<Dim, Frame::Inertial> element_map{
      ElementId<Dim>{0}, block.stationary_map().get_clone()};
  Mesh<Dim> mesh(extents, Spectral::Basis::FiniteDifference,
                 Spectral::Quadrature::CellCentered);

  const auto source_xi = logical_coordinates(mesh);
  const auto source_x = element_map(source_xi);
  const auto target_xi = create_target_points<Dim>(n_random_target_points);
  const auto target_x = element_map(target_xi);
  const intrp::Irregular irregular_interp{mesh, target_xi,
                                          std::optional<size_t>{max_degree}};

  // The target points lie half a grid spacing outside the outermost cell
  // centers, where the one-sided Lagrange stencils extrapolate and their
  // weights amplify roundoff (up to ~3e-14 for the 5-point order-4 stencil
  // in 3d). The tolerance below still asserts exactness to ~12 digits.
  Approx custom_approx = Approx::custom().epsilon(1.e-12).scale(1.);
  for (size_t degree = 0; degree <= max_degree; ++degree) {
    const auto source_vars = polynomial<Dim>(source_x, degree);
    const auto target_vars = irregular_interp.interpolate(source_vars);
    const auto expected_vars = polynomial<Dim>(target_x, degree);
    CAPTURE(Dim);
    CAPTURE(max_degree);
    CAPTURE(degree);
    CHECK_VARIABLES_CUSTOM_APPROX(target_vars, expected_vars, custom_approx);
  }
}

void test_tov(const size_t max_degree, const bool specified_interp_order) {
  const std::array<size_t, 3> isotropic_extents{{15, 15, 15}};
  constexpr size_t n_resolutions = 4;
  auto errors =
      make_array<n_resolutions>(std::numeric_limits<double>::signaling_NaN());
  const double central_density = 1.28e-3;
  for (size_t i = 0; i < n_resolutions; ++i) {
    const Domain<3> domain = create_domain<3>(
        6.6666666666666666666 / two_to_the(i), isotropic_extents);
    const Block<3>& cube = domain.blocks()[0];
    Mesh<3> mesh(isotropic_extents, Spectral::Basis::FiniteDifference,
                 Spectral::Quadrature::CellCentered);
    const auto xi = logical_coordinates(mesh);
    const ElementMap<3, Frame::Inertial> element_map{
        ElementId<3>{0}, cube.stationary_map().get_clone()};
    const auto x = element_map(xi);

    RelativisticEuler::Solutions::TovStar tov_star(
        central_density,
        std::make_unique<EquationsOfState::PolytropicFluid<true>>(100.0, 2.0));

    using rho_tag = hydro::Tags::RestMassDensity<DataVector>;
    auto vars = variables_from_tagged_tuple(
        tov_star.variables(x, 0.0, tmpl::list<rho_tag>{}));

    const tnsr::I<DataVector, 3, Frame::ElementLogical> xi_target{1_st, -1.0};
    intrp::Irregular irregular_interp{mesh, xi_target};
    if (specified_interp_order) {
      irregular_interp =
          intrp::Irregular{mesh, xi_target, std::optional<size_t>{max_degree}};
    }

    const auto target_vars = irregular_interp.interpolate(vars);
    gsl::at(errors, i) =
        fabs(central_density - get(get<rho_tag>(target_vars))[0]);
  }

  std::reverse(std::begin(errors), std::end(errors));
  auto ratio_of_errors =
      make_array<n_resolutions>(std::numeric_limits<double>::signaling_NaN());
  std::adjacent_difference(std::begin(errors), std::end(errors),
                           std::begin(ratio_of_errors), std::divides<>{});

  // since \rho is a symmetric function across the center, quadratic and
  // quartic extrapolation has one order higher convergence rate at the center
  const double expected_ratio =
      specified_interp_order
          ? (max_degree == 2
                 ? 16.0
                 : (max_degree == 4 ? 64.0 : two_to_the(max_degree + 1)))
          : 4.0;
  // A deviation of the convergence *order* by delta changes the ratio by
  // ~expected_ratio * 2^delta, so the allowed deviation must scale with the
  // expected ratio. The tolerance below corresponds to an order window of
  // about +-0.14 at every order.
  Approx custom_approx = Approx::custom().epsilon(5.e-2).scale(expected_ratio);
  for (size_t i = 1; i < n_resolutions; ++i) {
    CAPTURE(max_degree);
    CAPTURE(i);
    CHECK(expected_ratio == custom_approx(gsl::at(ratio_of_errors, i)));
  }
}

// Positions of FD target points along a dimension with
// `number_of_source_points` uniformly spaced source points, in units of the
// grid spacing from the first source point. They cover the source points, the
// points in between, and targets up to half a grid spacing outside the grid.
// Midpoints between source points are equally close to the centers of two
// windows of an odd number of source points, so they test how the stencil
// selection breaks ties. The computed midpoints are guaranteed to be exactly
// centered between the source points only if the grid spacing is a power of
// two, so `include_midpoints` must be false for other grids. (Source points
// are equally close to the centers of two windows of an even number of source
// points, but there the interpolated value does not depend on the window.)
std::vector<double> fd_target_positions(const size_t number_of_source_points,
                                        const bool include_midpoints) {
  std::vector<double> result{};
  for (size_t i = 0; i <= 4 * number_of_source_points; ++i) {
    const double position = 0.25 * static_cast<double>(i) - 0.5;
    const bool interior_midpoint =
        position > 0.0 and
        position < static_cast<double>(number_of_source_points - 1) and
        position - std::floor(position) == 0.5;
    if (include_midpoints or not interior_midpoint) {
      result.push_back(position);
    }
  }
  return result;
}

// The coordinate of the target point at `position` (see
// `fd_target_positions`) on the source grid `x`. Targets at integer positions
// are exactly on source points.
double fd_target_coordinate(const DataVector& x, const double position) {
  const size_t last = x.size() - 1;
  if (position < 0.0) {
    return x[0] + position * (x[1] - x[0]);
  }
  if (position >= static_cast<double>(last)) {
    return x[last] +
           (position - static_cast<double>(last)) * (x[last] - x[last - 1]);
  }
  const auto cell = static_cast<size_t>(position);
  return x[cell] +
         (position - static_cast<double>(cell)) * (x[cell + 1] - x[cell]);
}

// The first source point of the window of `order + 1` consecutive source
// points used by the FD interpolation of the given `order` for the target
// point at `position` (see `fd_target_positions`) on a uniform grid. This is
// the window whose center is closest to the target point. Ties are broken
// towards the upper window for orders 1 and 2 (so linear interpolation uses
// the two source points bracketing the target point) and towards the lower
// window for higher orders. Near and beyond the ends of the grid the window is
// shifted to stay inside the grid, i.e. the stencil becomes one-sided.
size_t expected_window_start(const double position,
                             const size_t number_of_source_points,
                             const size_t order) {
  // The window starting at source point `s` is centered on `s + order / 2`
  const double centered_start = position - 0.5 * static_cast<double>(order);
  const double start = order <= 2 ? std::floor(centered_start + 0.5)
                                  : std::ceil(centered_start - 0.5);
  return static_cast<size_t>(std::clamp(
      start, 0.0, static_cast<double>(number_of_source_points - order - 1)));
}

// Lagrange interpolation weights at `target` of the `width` consecutive points
// of `x` starting at `start`
std::vector<double> lagrange_weights(const DataVector& x, const size_t start,
                                     const size_t width, const double target) {
  std::vector<double> weights(width, 1.0);
  for (size_t i = 0; i < width; ++i) {
    for (size_t j = 0; j < width; ++j) {
      if (j != i) {
        weights[i] *= (target - x[start + j]) / (x[start + i] - x[start + j]);
      }
    }
  }
  return weights;
}

// For each target point, the source points (as indices into the source data)
// and weights of its tensor-product FD stencil
using ReferenceStencils = std::vector<std::vector<std::pair<size_t, double>>>;

template <size_t Dim>
ReferenceStencils reference_fd_stencils(
    const Mesh<Dim>& mesh,
    const std::vector<std::array<double, Dim>>& positions, const size_t order) {
  ReferenceStencils result(positions.size());
  for (size_t p = 0; p < positions.size(); ++p) {
    Index<Dim> window_start{};
    std::array<std::vector<double>, Dim> weights{};
    for (size_t d = 0; d < Dim; ++d) {
      const DataVector& x = Spectral::collocation_points(mesh.slice_through(d));
      const double position = gsl::at(positions[p], d);
      window_start[d] = expected_window_start(position, x.size(), order);
      gsl::at(weights, d) = lagrange_weights(x, window_start[d], order + 1,
                                             fd_target_coordinate(x, position));
    }
    for (IndexIterator<Dim> offset(Index<Dim>(order + 1)); offset; ++offset) {
      Index<Dim> source_index{};
      double weight = 1.0;
      for (size_t d = 0; d < Dim; ++d) {
        source_index[d] = window_start[d] + offset()[d];
        weight *= gsl::at(weights, d)[offset()[d]];
      }
      result[p].emplace_back(collapsed_index(source_index, mesh.extents()),
                             weight);
    }
  }
  return result;
}

DataVector reference_interpolate(const ReferenceStencils& stencils,
                                 const DataVector& source_data) {
  DataVector result(stencils.size(), 0.0);
  for (size_t p = 0; p < stencils.size(); ++p) {
    for (const auto& [source_index, weight] : stencils[p]) {
      result[p] += weight * source_data[source_index];
    }
  }
  return result;
}

// A smooth non-polynomial function, different for each `component`
template <size_t Dim>
DataVector nonpolynomial_data(
    const tnsr::I<DataVector, Dim, Frame::ElementLogical>& x,
    const size_t component) {
  const std::array<double, 3> wave_numbers{{2.1, 1.3, -1.7}};
  DataVector phase(get<0>(x).size(),
                   0.4 + 0.1 * static_cast<double>(component));
  for (size_t d = 0; d < Dim; ++d) {
    phase += gsl::at(wave_numbers, d) * x.get(d);
  }
  return DataVector{sin(phase) * exp(0.6 * get<0>(x))};
}

// A product of polynomials of degree `degree` in each dimension, which the FD
// interpolation of order `degree` reproduces exactly
template <size_t Dim>
DataVector polynomial_data(
    const tnsr::I<DataVector, Dim, Frame::ElementLogical>& x,
    const size_t degree) {
  DataVector result(get<0>(x).size(), 1.0);
  for (size_t d = 0; d < Dim; ++d) {
    result *=
        Polynomial{degree, 1.0 + 0.5 * static_cast<double>(d), 2.0}(x.get(d));
  }
  return result;
}

template <size_t Dim>
void check_fd_interpolation(const Mesh<Dim>& mesh, const size_t order,
                            const bool include_midpoints) {
  CAPTURE(mesh);
  CAPTURE(order);
  const size_t number_of_source_points = mesh.number_of_grid_points();
  // Each target point takes its position in each dimension from the list of
  // positions for that dimension, so that each position is used at least once
  // in each dimension, in varying combinations.
  std::array<std::vector<double>, Dim> positions_in_dim{};
  size_t number_of_targets = 0;
  for (size_t d = 0; d < Dim; ++d) {
    gsl::at(positions_in_dim, d) =
        fd_target_positions(mesh.extents(d), include_midpoints);
    number_of_targets =
        std::max(number_of_targets, gsl::at(positions_in_dim, d).size());
  }
  std::vector<std::array<double, Dim>> positions(number_of_targets);
  tnsr::I<DataVector, Dim, Frame::ElementLogical> target_points{
      number_of_targets};
  for (size_t p = 0; p < number_of_targets; ++p) {
    for (size_t d = 0; d < Dim; ++d) {
      const auto& positions_d = gsl::at(positions_in_dim, d);
      gsl::at(positions[p], d) =
          positions_d[(p + d * positions_d.size() / 3) % positions_d.size()];
      target_points.get(d)[p] = fd_target_coordinate(
          Spectral::collocation_points(mesh.slice_through(d)),
          gsl::at(positions[p], d));
    }
  }

  const intrp::Irregular<Dim> interpolant{mesh, target_points, order};
  const ReferenceStencils stencils =
      reference_fd_stencils(mesh, positions, order);
  const auto source_x = logical_coordinates(mesh);
  // Extrapolating stencils amplify roundoff errors. Shifting the stencil
  // window by one source point changes the interpolated non-polynomial data by
  // more than this tolerance.
  const Approx custom_approx = Approx::custom().epsilon(1.e-10).scale(1.);

  {
    INFO("DataVector");
    const DataVector source = nonpolynomial_data(source_x, 0);
    const DataVector expected = reference_interpolate(stencils, source);
    const DataVector result = interpolant.interpolate(source);
    CHECK_ITERABLE_CUSTOM_APPROX(result, expected, custom_approx);
    DataVector resized_result{2};
    interpolant.interpolate(make_not_null(&resized_result), source);
    CHECK_ITERABLE_CUSTOM_APPROX(resized_result, expected, custom_approx);

    const DataVector polynomial_result =
        interpolant.interpolate(polynomial_data(source_x, order));
    CHECK_ITERABLE_CUSTOM_APPROX(polynomial_result,
                                 polynomial_data(target_points, order),
                                 custom_approx);
  }
  {
    INFO("Variables");
    using vector_tag = TestTags::Vector<DataVector, Dim>;
    using tags = tmpl::list<Tags::Scalar, vector_tag>;
    Variables<tags> source{number_of_source_points};
    Variables<tags> expected{number_of_targets};
    get(get<Tags::Scalar>(source)) = nonpolynomial_data(source_x, 0);
    get(get<Tags::Scalar>(expected)) =
        reference_interpolate(stencils, get(get<Tags::Scalar>(source)));
    for (size_t d = 0; d < Dim; ++d) {
      get<vector_tag>(source).get(d) = nonpolynomial_data(source_x, d + 1);
      get<vector_tag>(expected).get(d) =
          reference_interpolate(stencils, get<vector_tag>(source).get(d));
    }
    const Variables<tags> result = interpolant.interpolate(source);
    CHECK_VARIABLES_CUSTOM_APPROX(result, expected, custom_approx);
  }
  {
    INFO("Many components");
    const size_t number_of_components = 55;
    std::vector<double> source(number_of_components * number_of_source_points);
    std::vector<double> expected(number_of_components * number_of_targets);
    for (size_t c = 0; c < number_of_components; ++c) {
      const DataVector component_source = nonpolynomial_data(source_x, c);
      const DataVector component_expected =
          reference_interpolate(stencils, component_source);
      for (size_t s = 0; s < number_of_source_points; ++s) {
        source[c * number_of_source_points + s] = component_source[s];
      }
      for (size_t p = 0; p < number_of_targets; ++p) {
        expected[c * number_of_targets + p] = component_expected[p];
      }
    }
    std::vector<double> result(expected.size());
    auto result_span = gsl::make_span(result);
    interpolant.interpolate(make_not_null(&result_span),
                            gsl::make_span(source));
    CHECK_ITERABLE_CUSTOM_APPROX(result, expected, custom_approx);
  }
  {
    INFO("Complex data");
    const size_t number_of_components = 2;
    std::vector<std::complex<double>> source(number_of_components *
                                             number_of_source_points);
    std::vector<std::complex<double>> expected(number_of_components *
                                               number_of_targets);
    for (size_t c = 0; c < number_of_components; ++c) {
      const DataVector real_source = nonpolynomial_data(source_x, 2 * c);
      const DataVector imag_source = nonpolynomial_data(source_x, 2 * c + 1);
      const DataVector real_expected =
          reference_interpolate(stencils, real_source);
      const DataVector imag_expected =
          reference_interpolate(stencils, imag_source);
      for (size_t s = 0; s < number_of_source_points; ++s) {
        source[c * number_of_source_points + s] = {real_source[s],
                                                   imag_source[s]};
      }
      for (size_t p = 0; p < number_of_targets; ++p) {
        expected[c * number_of_targets + p] = {real_expected[p],
                                               imag_expected[p]};
      }
    }
    std::vector<std::complex<double>> result(expected.size());
    auto result_span = gsl::make_span(result);
    interpolant.interpolate(make_not_null(&result_span),
                            gsl::make_span(source));
    CHECK_ITERABLE_CUSTOM_APPROX(result, expected, custom_approx);

    // The first component as a ComplexDataVector
    const ComplexDataVector source_dv(source.data(), number_of_source_points);
    const ComplexDataVector expected_dv(expected.data(), number_of_targets);
    const ComplexDataVector result_dv = interpolant.interpolate(source_dv);
    CHECK_ITERABLE_CUSTOM_APPROX(result_dv, expected_dv, custom_approx);
  }
  {
    INFO("Single precision data");
    const size_t number_of_components = 2;
    std::vector<float> source(number_of_components * number_of_source_points);
    for (size_t c = 0; c < number_of_components; ++c) {
      const DataVector component_source = nonpolynomial_data(source_x, c);
      for (size_t s = 0; s < number_of_source_points; ++s) {
        source[c * number_of_source_points + s] =
            static_cast<float>(component_source[s]);
      }
    }
    std::vector<float> result(number_of_components * number_of_targets);
    auto result_span = gsl::make_span(result);
    interpolant.interpolate(make_not_null(&result_span),
                            gsl::make_span(source));
    // The sum over the stencil is accumulated in single precision, so each
    // term can contribute a single-precision rounding error.
    for (size_t c = 0; c < number_of_components; ++c) {
      for (size_t p = 0; p < number_of_targets; ++p) {
        double expected = 0.0;
        double sum_of_magnitudes = 0.0;
        for (const auto& [source_index, weight] : stencils[p]) {
          const double term =
              weight * static_cast<double>(
                           source[c * number_of_source_points + source_index]);
          expected += term;
          sum_of_magnitudes += std::abs(term);
        }
        CAPTURE(p);
        CHECK(std::abs(result[c * number_of_targets + p] - expected) <=
              static_cast<double>(stencils[p].size()) *
                  std::numeric_limits<float>::epsilon() * sum_of_magnitudes);
      }
    }
  }
  {
    INFO("Single target point");
    const DataVector source = nonpolynomial_data(source_x, 0);
    const DataVector result = interpolant.interpolate(source);
    for (size_t p = 0; p < number_of_targets; p += 7) {
      CAPTURE(p);
      tnsr::I<double, Dim, Frame::ElementLogical> target_point{};
      tnsr::I<DataVector, Dim, Frame::ElementLogical> target_point_dv{1_st};
      for (size_t d = 0; d < Dim; ++d) {
        target_point.get(d) = target_points.get(d)[p];
        target_point_dv.get(d)[0] = target_point.get(d);
      }
      const intrp::Irregular<Dim> single_point_interpolant{mesh, target_point,
                                                           order};
      CHECK(single_point_interpolant ==
            intrp::Irregular<Dim>{mesh, target_point_dv, order});
      CHECK(single_point_interpolant.interpolate(source)[0] ==
            approx(result[p]));
    }
  }
  {
    INFO("No target points");
    const intrp::Irregular<Dim> no_targets_interpolant{
        mesh, tnsr::I<DataVector, Dim, Frame::ElementLogical>{0_st}, order};
    CHECK(no_targets_interpolant.interpolate(nonpolynomial_data(source_x, 0))
              .size() == 0);
    const Variables<tmpl::list<Tags::Scalar>> source{number_of_source_points,
                                                     1.0};
    CHECK(no_targets_interpolant.interpolate(source).number_of_grid_points() ==
          0);
  }
  {
    INFO("Default order");
    const intrp::Irregular<Dim> default_order_interpolant{mesh, target_points};
    if (order == 1) {
      CHECK(default_order_interpolant == interpolant);
    } else {
      CHECK(default_order_interpolant != interpolant);
    }
  }
  {
    INFO("Copies and serialization");
    const DataVector source = nonpolynomial_data(source_x, 0);
    const DataVector result = interpolant.interpolate(source);
    test_serialization(interpolant);
    const auto deserialized = serialize_and_deserialize(interpolant);
    CHECK(deserialized.interpolate(source) == result);
    intrp::Irregular<Dim> copy{};
    CHECK(copy != interpolant);
    test_serialization(copy);
    copy = interpolant;
    CHECK(copy == interpolant);
    const intrp::Irregular<Dim> moved{std::move(copy)};
    CHECK(moved.interpolate(source) == result);
  }
}

void test_fd_interpolation() {
  for (size_t order = 1; order <= 6; ++order) {
    for (const auto quadrature : {Spectral::Quadrature::CellCentered,
                                  Spectral::Quadrature::FaceCentered}) {
      // With these numbers of points the grid spacing is 1/4 (coarse) or 1/8
      // (fine), so all target positions are exact.
      const size_t n_coarse =
          quadrature == Spectral::Quadrature::CellCentered ? 8 : 9;
      const size_t n_fine =
          quadrature == Spectral::Quadrature::CellCentered ? 16 : 17;
      const auto basis = Spectral::Basis::FiniteDifference;
      check_fd_interpolation(Mesh<1>{n_coarse, basis, quadrature}, order, true);
      check_fd_interpolation(Mesh<1>{n_fine, basis, quadrature}, order, true);
      check_fd_interpolation(Mesh<1>{11, basis, quadrature}, order, false);
      check_fd_interpolation(Mesh<2>{{{n_fine, n_coarse}}, basis, quadrature},
                             order, true);
      check_fd_interpolation(Mesh<2>{{{7, 10}}, basis, quadrature}, order,
                             false);
      check_fd_interpolation(
          Mesh<3>{{{n_coarse, n_fine, n_coarse}}, basis, quadrature}, order,
          true);
      check_fd_interpolation(Mesh<3>{{{10, 7, 12}}, basis, quadrature}, order,
                             false);
    }
  }
}

void test_fd_coordinate_reuse() {
  const auto fd = Spectral::Basis::FiniteDifference;
  const auto cc = Spectral::Quadrature::CellCentered;
  const auto fc = Spectral::Quadrature::FaceCentered;
  // The existing FD interpolator reuses the first matching axis's coordinates
  // for equal extents, even if quadratures differ. Pin that behavior separately
  // from the interpolation-accuracy tests: changing it is not a storage change.
  const auto check_coordinate_reuse = [](const Mesh<3>& mesh,
                                         const Mesh<3>& effective_mesh) {
    CAPTURE(mesh);
    tnsr::I<DataVector, 3, Frame::ElementLogical> targets{};
    get<0>(targets) = DataVector{-0.93, 0.13, 1.02};
    get<1>(targets) = DataVector{-0.31, 0.91, -1.03};
    get<2>(targets) = DataVector{0.71, -0.43, 0.27};
    for (size_t order = 1; order <= 6; ++order) {
      CHECK(intrp::Irregular<3>{mesh, targets, order} ==
            intrp::Irregular<3>{effective_mesh, targets, order});
    }
  };
  check_coordinate_reuse(Mesh<3>{{{9, 9, 9}}, {fd, fd, fd}, {cc, fc, fc}},
                         Mesh<3>{{{9, 9, 9}}, fd, cc});
  check_coordinate_reuse(Mesh<3>{{{9, 11, 9}}, {fd, fd, fd}, {cc, fc, fc}},
                         Mesh<3>{{{9, 11, 9}}, {fd, fd, fd}, {cc, fc, cc}});
  check_coordinate_reuse(Mesh<3>{{{9, 11, 11}}, {fd, fd, fd}, {cc, fc, cc}},
                         Mesh<3>{{{9, 11, 11}}, {fd, fd, fd}, {cc, fc, fc}});
}

void test_fd_interpolation_errors() {
  const tnsr::I<DataVector, 1, Frame::ElementLogical> target_point{1_st, 0.1};
  const Mesh<1> mesh{8, Spectral::Basis::FiniteDifference,
                     Spectral::Quadrature::CellCentered};
  for (const size_t order : {0_st, 7_st}) {
    CHECK_THROWS_WITH((intrp::Irregular<1>{mesh, target_point, order}),
                      Catch::Matchers::ContainsSubstring(
                          "Unsupported fd_to_fd_interp_order: " +
                          std::to_string(order) + "; expected 1 to 6."));
  }
#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(
      (intrp::Irregular<1>{Mesh<1>{1, Spectral::Basis::FiniteDifference,
                                   Spectral::Quadrature::CellCentered},
                           target_point}),
      Catch::Matchers::ContainsSubstring(
          "Need at least 2 points for linear interpolation"));
  CHECK_THROWS_WITH(
      (intrp::Irregular<1>{Mesh<1>{4, Spectral::Basis::FiniteDifference,
                                   Spectral::Quadrature::CellCentered},
                           target_point, 4}),
      Catch::Matchers::ContainsSubstring(
          "Need at least 5 points for quartic interpolation"));
  const tnsr::I<DataVector, 3, Frame::ElementLogical> target_point_3d{1_st,
                                                                      0.1};
  CHECK_THROWS_WITH(
      (intrp::Irregular<3>{Mesh<3>{{{7, 7, 6}},
                                   Spectral::Basis::FiniteDifference,
                                   Spectral::Quadrature::FaceCentered},
                           target_point_3d, 6}),
      Catch::Matchers::ContainsSubstring(
          "Need at least 7 points for degree-6 interpolation"));
#endif
}

void test_2d_spherical(const gsl::not_null<std::mt19937*> generator) {
  std::uniform_real_distribution<> xi_distribution(-1.0, 1.0);
  std::uniform_real_distribution<> phi_distribution(0.0, 2.0 * M_PI);
  for (size_t n_target_points = 1; n_target_points < 13;
       n_target_points += 11) {
    tnsr::I<DataVector, 2, Frame::ElementLogical> xi_target{n_target_points};
    get<0>(xi_target) = acos(make_with_random_values<DataVector>(
        generator, make_not_null(&xi_distribution), xi_target));
    get<1>(xi_target) = make_with_random_values<DataVector>(
        generator, make_not_null(&phi_distribution), xi_target);
    for (size_t n_z = 0; n_z < 4; ++n_z) {
      for (size_t n_y = 0; n_y < 4; ++n_y) {
        for (size_t n_x = 0; n_x < 4; ++n_x) {
          const Mesh<2> source_mesh{
              std::array{n_x + n_y + n_z + 2, 2 * (n_x + n_y) + 3},
              std::array{Spectral::Basis::SphericalHarmonic,
                         Spectral::Basis::SphericalHarmonic},
              std::array{Spectral::Quadrature::Gauss,
                         Spectral::Quadrature::Equiangular}};
          const YlmTestFunctions::ProductOfPolynomials f(n_x, n_y, n_z);
          const auto xi_source = logical_coordinates(source_mesh);
          const DataVector f_source = f(xi_source);
          const DataVector f_expected = f(xi_target);
          const intrp::Irregular<2> interpolator(source_mesh, xi_target);
          const DataVector f_interpolated = interpolator.interpolate(f_source);
          CHECK_ITERABLE_APPROX(f_interpolated, f_expected);
        }
      }
    }
  }
}

void test_3d_spherical(const gsl::not_null<std::mt19937*> generator) {
  std::uniform_real_distribution<> xi_distribution(-1.0, 1.0);
  std::uniform_real_distribution<> phi_distribution(0.0, 2.0 * M_PI);
  for (size_t n_target_points = 1; n_target_points < 13;
       n_target_points += 11) {
    tnsr::I<DataVector, 3, Frame::ElementLogical> xi_target{n_target_points};
    get<0>(xi_target) = make_with_random_values<DataVector>(
        generator, make_not_null(&xi_distribution), xi_target);
    get<1>(xi_target) = acos(make_with_random_values<DataVector>(
        generator, make_not_null(&xi_distribution), xi_target));
    get<2>(xi_target) = make_with_random_values<DataVector>(
        generator, make_not_null(&phi_distribution), xi_target);
    for (size_t n_r = 2; n_r < 4; ++n_r) {
      for (size_t n_z = 0; n_z < 4; ++n_z) {
        for (size_t n_y = 0; n_y < 4; ++n_y) {
          for (size_t n_x = 0; n_x < 4; ++n_x) {
            const Mesh<3> source_mesh{
                std::array{n_r, n_x + n_y + n_z + 2, 2 * (n_x + n_y) + 3},
                std::array{Spectral::Basis::Legendre,
                           Spectral::Basis::SphericalHarmonic,
                           Spectral::Basis::SphericalHarmonic},
                std::array{Spectral::Quadrature::GaussLobatto,
                           Spectral::Quadrature::Gauss,
                           Spectral::Quadrature::Equiangular}};
            const Polynomial f_r{n_r - 1, 1.5, 2.0};
            const YlmTestFunctions::ProductOfPolynomials f_a(n_x, n_y, n_z);
            const auto xi_source = logical_coordinates(source_mesh);
            const DataVector f_source =
                f_r(get<0>(xi_source)) *
                f_a(get<1>(xi_source), get<2>(xi_source));
            const DataVector f_expected =
                f_r(get<0>(xi_target)) *
                f_a(get<1>(xi_target), get<2>(xi_target));
            const intrp::Irregular<3> interpolator(source_mesh, xi_target);
            const DataVector f_interpolated =
                interpolator.interpolate(f_source);
            CHECK_ITERABLE_APPROX(f_interpolated, f_expected);
          }
        }
      }
    }
  }
}

void test_2d_disk(const gsl::not_null<std::mt19937*> generator) {
  std::uniform_real_distribution<> xi_distribution(-1.0, 1.0);
  std::uniform_real_distribution<> phi_distribution(0.0, 2.0 * M_PI);
  for (size_t n_target_points = 1; n_target_points < 13;
       n_target_points += 11) {
    tnsr::I<DataVector, 2, Frame::ElementLogical> xi_target{n_target_points};
    get<0>(xi_target) = make_with_random_values<DataVector>(
        generator, make_not_null(&xi_distribution), xi_target);
    get<1>(xi_target) = make_with_random_values<DataVector>(
        generator, make_not_null(&phi_distribution), xi_target);
    for (size_t n_y = 0; n_y < 4; ++n_y) {
      CAPTURE(n_y);
      for (size_t n_x = 0; n_x < 4; ++n_x) {
        CAPTURE(n_x);
        const Mesh<2> source_mesh{
            n_x + n_y == 0 ? std::array{1_st, 1_st}
                           : std::array{n_x + n_y + 1, 2 * (n_x + n_y) + 1},
            std::array{Spectral::Basis::ZernikeB2, Spectral::Basis::ZernikeB2},
            std::array{Spectral::Quadrature::GaussRadauUpper,
                       Spectral::Quadrature::Equiangular}};
        const DiskTestFunctions::ProductOfPolynomials f{n_x, n_y};
        const auto xi_source = logical_coordinates(source_mesh);
        const DataVector f_source =
            f(0.5 * (get<0>(xi_source) + 1.0), get<1>(xi_source));
        const DataVector f_expected =
            f(0.5 * (get<0>(xi_target) + 1.0), get<1>(xi_target));
        const intrp::Irregular<2> interpolator(source_mesh, xi_target);
        const DataVector f_interpolated = interpolator.interpolate(f_source);
        CHECK_ITERABLE_APPROX(f_interpolated, f_expected);
      }
    }
  }
}

void test_3d_cylinder(const gsl::not_null<std::mt19937*> generator) {
  std::uniform_real_distribution<> xi_distribution(-1.0, 1.0);
  std::uniform_real_distribution<> phi_distribution(0.0, 2.0 * M_PI);
  for (size_t n_target_points = 1; n_target_points < 13;
       n_target_points += 11) {
    tnsr::I<DataVector, 3, Frame::ElementLogical> xi_target{n_target_points};
    get<0>(xi_target) = make_with_random_values<DataVector>(
        generator, make_not_null(&xi_distribution), xi_target);
    get<1>(xi_target) = make_with_random_values<DataVector>(
        generator, make_not_null(&phi_distribution), xi_target);
    get<2>(xi_target) = make_with_random_values<DataVector>(
        generator, make_not_null(&xi_distribution), xi_target);
    for (size_t n_z = 2; n_z < 4; ++n_z) {
      CAPTURE(n_z);
      for (size_t n_y = 0; n_y < 4; ++n_y) {
        CAPTURE(n_y);
        for (size_t n_x = 0; n_x < 4; ++n_x) {
          CAPTURE(n_x);
          const Mesh<3> source_mesh{
              n_x + n_y == 0
                  ? std::array{1_st, 1_st, n_z}
                  : std::array{n_x + n_y + 1, 2 * (n_x + n_y) + 1, n_z},
              std::array{Spectral::Basis::ZernikeB2, Spectral::Basis::ZernikeB2,
                         Spectral::Basis::Legendre},
              std::array{Spectral::Quadrature::GaussRadauUpper,
                         Spectral::Quadrature::Equiangular,
                         Spectral::Quadrature::GaussLobatto}};
          const DiskTestFunctions::ProductOfPolynomials f{n_x, n_y};
          const Polynomial f_z{n_z - 1, 1.5, 2.0};
          const auto xi_source = logical_coordinates(source_mesh);
          const DataVector f_source =
              f(0.5 * (get<0>(xi_source) + 1.0), get<1>(xi_source)) *
              f_z(get<2>(xi_source));
          const DataVector f_expected =
              f(0.5 * (get<0>(xi_target) + 1.0), get<1>(xi_target)) *
              f_z(get<2>(xi_target));
          const intrp::Irregular<3> interpolator(source_mesh, xi_target);
          const DataVector f_interpolated = interpolator.interpolate(f_source);
          CHECK_ITERABLE_APPROX(f_interpolated, f_expected);
        }
      }
    }
  }
}

void test_2d_hollow_disk(const gsl::not_null<std::mt19937*> generator) {
  std::uniform_real_distribution<> xi_distribution(-1.0, 1.0);
  std::uniform_real_distribution<> phi_distribution(0.0,
                                                    2.0 * std::numbers::pi);
  const Approx custom_approx = Approx::custom().epsilon(1e-12).scale(1.);
  for (size_t n_target_points = 1; n_target_points < 13;
       n_target_points += 11) {
    tnsr::I<DataVector, 2, Frame::ElementLogical> xi_target{n_target_points};
    get<0>(xi_target) = make_with_random_values<DataVector>(
        generator, make_not_null(&xi_distribution), xi_target);
    get<1>(xi_target) = make_with_random_values<DataVector>(
        generator, make_not_null(&phi_distribution), xi_target);
    for (size_t n_r = 2; n_r < 4; ++n_r) {
      CAPTURE(n_r);
      for (size_t n_y = 0; n_y < 4; ++n_y) {
        CAPTURE(n_y);
        for (size_t n_x = 0; n_x < 4; ++n_x) {
          CAPTURE(n_x);
          const Mesh<2> source_mesh{
              std::array{n_r, 2 * (n_x + n_y) + 1},
              std::array{Spectral::Basis::Legendre, Spectral::Basis::Fourier},
              std::array{Spectral::Quadrature::GaussLobatto,
                         Spectral::Quadrature::Equiangular}};
          const Polynomial f_r{n_r - 1, 1.5, 2.0};
          const FourierTestFunctions::ProductOfPolynomials f_m{n_x, n_y};
          const auto xi_source = logical_coordinates(source_mesh);
          const DataVector f_source =
              f_r(get<0>(xi_source)) * f_m(get<1>(xi_source));
          const DataVector f_expected =
              f_r(get<0>(xi_target)) * f_m(get<1>(xi_target));
          const intrp::Irregular<2> interpolator(source_mesh, xi_target);
          const DataVector f_interpolated = interpolator.interpolate(f_source);
          CHECK_ITERABLE_CUSTOM_APPROX(f_interpolated, f_expected,
                                       custom_approx);
        }
      }
    }
  }
}

void test_3d_hollow_cylinder(const gsl::not_null<std::mt19937*> generator) {
  std::uniform_real_distribution<> xi_distribution(-1.0, 1.0);
  std::uniform_real_distribution<> phi_distribution(0.0,
                                                    2.0 * std::numbers::pi);
  const Approx custom_approx = Approx::custom().epsilon(1e-12).scale(1.);
  for (size_t n_target_points = 1; n_target_points < 13;
       n_target_points += 11) {
    tnsr::I<DataVector, 3, Frame::ElementLogical> xi_target{n_target_points};
    get<0>(xi_target) = make_with_random_values<DataVector>(
        generator, make_not_null(&xi_distribution), xi_target);
    get<1>(xi_target) = make_with_random_values<DataVector>(
        generator, make_not_null(&phi_distribution), xi_target);
    get<2>(xi_target) = make_with_random_values<DataVector>(
        generator, make_not_null(&xi_distribution), xi_target);
    for (size_t n_r = 2; n_r < 4; ++n_r) {
      CAPTURE(n_r);
      for (size_t n_y = 0; n_y < 4; ++n_y) {
        CAPTURE(n_y);
        for (size_t n_x = 0; n_x < 4; ++n_x) {
          CAPTURE(n_x);
          for (size_t n_z = 2; n_z < 4; ++n_z) {
            CAPTURE(n_z);
            const Mesh<3> source_mesh{
                std::array{n_r, 2 * (n_x + n_y) + 1, n_z},
                std::array{Spectral::Basis::Legendre, Spectral::Basis::Fourier,
                           Spectral::Basis::Legendre},
                std::array{Spectral::Quadrature::GaussLobatto,
                           Spectral::Quadrature::Equiangular,
                           Spectral::Quadrature::GaussLobatto}};
            const Polynomial f_r{n_r - 1, 1.5, 2.0};
            const Polynomial f_z{n_z - 1, 1.6, 2.0};
            const FourierTestFunctions::ProductOfPolynomials f_m{n_x, n_y};
            const auto xi_source = logical_coordinates(source_mesh);
            const DataVector f_source = f_r(get<0>(xi_source)) *
                                        f_m(get<1>(xi_source)) *
                                        f_z(get<2>(xi_source));
            const DataVector f_expected = f_r(get<0>(xi_target)) *
                                          f_m(get<1>(xi_target)) *
                                          f_z(get<2>(xi_target));
            const intrp::Irregular<3> interpolator(source_mesh, xi_target);
            const DataVector f_interpolated =
                interpolator.interpolate(f_source);
            CHECK_ITERABLE_CUSTOM_APPROX(f_interpolated, f_expected,
                                         custom_approx);
          }
        }
      }
    }
  }
}

// Test that the Cartoon-basis interpolation in 3D delegates correctly to the
// lower-dimensional interpolator.  For the FD+Cartoon (axisymmetric) case the
// 3D result must match an Irregular<2> built from slice_through(0,1).  For the
// Cartoon+Cartoon (spherically-symmetric) case it must match an Irregular<1>
// built from slice_through(0).
void test_cartoon_fd(const gsl::not_null<std::mt19937*> generator) {
  std::uniform_real_distribution<> dist(-1.0, 1.0);
  const size_t n_target = 7;
  {
    INFO("FD+FD+Cartoon matches Irregular<2> on slice_through(0,1)");
    for (size_t n0 = 4; n0 <= 6; ++n0) {
      for (size_t n1 = 4; n1 <= 6; ++n1) {
        CAPTURE(n0);
        CAPTURE(n1);
        const Mesh<3> mesh3{
            {{n0, n1, 1}},
            {{Spectral::Basis::FiniteDifference,
              Spectral::Basis::FiniteDifference, Spectral::Basis::Cartoon}},
            {{Spectral::Quadrature::CellCentered,
              Spectral::Quadrature::CellCentered,
              Spectral::Quadrature::AxialSymmetry}}};
        const Mesh<2> mesh2 = mesh3.slice_through(0, 1);

        // Target points: xi2 is always zero (guaranteed by cartoon symmetry)
        tnsr::I<DataVector, 3, Frame::ElementLogical> xi3{n_target};
        get<0>(xi3) = make_with_random_values<DataVector>(
            generator, make_not_null(&dist), DataVector(n_target));
        get<1>(xi3) = make_with_random_values<DataVector>(
            generator, make_not_null(&dist), DataVector(n_target));
        get<2>(xi3) = DataVector(n_target, 0.0);

        tnsr::I<DataVector, 2, Frame::ElementLogical> xi2{n_target};
        get<0>(xi2) = get<0>(xi3);
        get<1>(xi2) = get<1>(xi3);

        // Source data: f(xi, eta) = (1 + xi) * (2 + eta) on the 2D slice,
        // tiled trivially into 3D (only one point in dim 2)
        const auto xi_src3 = logical_coordinates(mesh3);
        DataVector f3(mesh3.number_of_grid_points());
        for (size_t s = 0; s < mesh3.number_of_grid_points(); ++s) {
          f3[s] = (1.0 + get<0>(xi_src3)[s]) * (2.0 + get<1>(xi_src3)[s]);
        }
        // The 2D slice has the same data layout (dim2 has only one point)
        const DataVector f2(f3.data(), mesh2.number_of_grid_points());

        const intrp::Irregular<3> interp3(mesh3, xi3);
        const intrp::Irregular<2> interp2(mesh2, xi2);
        CHECK_ITERABLE_APPROX(interp3.interpolate(f3), interp2.interpolate(f2));

        tnsr::I<double, 3, Frame::ElementLogical> xi3_double{1};
        tnsr::I<double, 2, Frame::ElementLogical> xi2_double{1};
        get<0>(xi3_double) = get<0>(xi2_double) = get<0>(xi3)[0];
        get<1>(xi3_double) = get<1>(xi2_double) = get<1>(xi3)[0];
        get<2>(xi3_double) = get<2>(xi3)[0];
        const intrp::Irregular<3> interp3_double(mesh3, xi3_double);
        const intrp::Irregular<2> interp2_double(mesh2, xi2_double);
        CHECK_ITERABLE_APPROX(interp3_double.interpolate(f3),
                              interp2_double.interpolate(f2));
      }
    }
  }
  {
    INFO("FD+Cartoon+Cartoon matches Irregular<1> on slice_through(0)");
    for (size_t n0 = 4; n0 <= 6; ++n0) {
      CAPTURE(n0);
      const Mesh<3> mesh3{
          {{n0, 1, 1}},
          {{Spectral::Basis::FiniteDifference, Spectral::Basis::Cartoon,
            Spectral::Basis::Cartoon}},
          {{Spectral::Quadrature::CellCentered,
            Spectral::Quadrature::SphericalSymmetry,
            Spectral::Quadrature::SphericalSymmetry}}};
      const Mesh<1> mesh1 = mesh3.slice_through(0);

      tnsr::I<DataVector, 3, Frame::ElementLogical> xi3{n_target};
      get<0>(xi3) = make_with_random_values<DataVector>(
          generator, make_not_null(&dist), DataVector(n_target));
      get<1>(xi3) = DataVector(n_target, 0.0);
      get<2>(xi3) = DataVector(n_target, 0.0);

      tnsr::I<DataVector, 1, Frame::ElementLogical> xi1{n_target};
      get<0>(xi1) = get<0>(xi3);

      const auto xi_src3 = logical_coordinates(mesh3);
      DataVector f3(mesh3.number_of_grid_points());
      for (size_t s = 0; s < mesh3.number_of_grid_points(); ++s) {
        f3[s] = 1.0 + 2.0 * get<0>(xi_src3)[s];
      }
      const DataVector f1(f3.data(), mesh1.number_of_grid_points());

      const intrp::Irregular<3> interp3(mesh3, xi3);
      const intrp::Irregular<1> interp1(mesh1, xi1);
      CHECK_ITERABLE_APPROX(interp3.interpolate(f3), interp1.interpolate(f1));

      tnsr::I<double, 3, Frame::ElementLogical> xi3_double{1};
      tnsr::I<double, 1, Frame::ElementLogical> xi1_double{1};
      get<0>(xi3_double) = get<0>(xi1_double) = get<0>(xi3)[0];
      get<1>(xi3_double) = get<1>(xi3)[0];
      get<2>(xi3_double) = get<2>(xi3)[0];
      const intrp::Irregular<3> interp3_double(mesh3, xi3_double);
      const intrp::Irregular<1> interp1_double(mesh1, xi1_double);
      CHECK_ITERABLE_APPROX(interp3_double.interpolate(f3),
                            interp1_double.interpolate(f1));
    }
  }
}

void test_cartoon_spherical(const gsl::not_null<std::mt19937*> generator) {
  std::uniform_real_distribution<> xi_distribution(-1.0, 1.0);

  const size_t number_of_points = 6;
  tnsr::I<DataVector, 3, Frame::ElementLogical> xi_target{number_of_points};
  get<0>(xi_target) = make_with_random_values<DataVector>(
      generator, make_not_null(&xi_distribution), xi_target);
  get<1>(xi_target) = DataVector(number_of_points, 0.0);
  get<2>(xi_target) = DataVector(number_of_points, 0.0);
  for (size_t n_x = 4; n_x < 6; ++n_x) {
    const Mesh<3> source_mesh{
        {{n_x, 1, 1}},
        {{Spectral::Basis::Legendre, Spectral::Basis::Cartoon,
          Spectral::Basis::Cartoon}},
        {{Spectral::Quadrature::GaussLobatto,
          Spectral::Quadrature::SphericalSymmetry,
          Spectral::Quadrature::SphericalSymmetry}}};
    const Polynomial f_r{n_x - 1, 1.5, 2.0};
    const auto xi_source = logical_coordinates(source_mesh);
    const DataVector f_source = f_r(get<0>(xi_source));
    const DataVector f_expected = f_r(get<0>(xi_target));
    const intrp::Irregular<3> interpolator(source_mesh, xi_target);
    const DataVector f_interpolated = interpolator.interpolate(f_source);
    CHECK_ITERABLE_APPROX(f_interpolated, f_expected);
  }
}

void test_cartoon_axial(const gsl::not_null<std::mt19937*> generator) {
  std::uniform_real_distribution<> xi_distribution(-1.0, 1.0);

  const size_t number_of_points = 6;
  tnsr::I<DataVector, 3, Frame::ElementLogical> xi_target{number_of_points};
  get<0>(xi_target) = make_with_random_values<DataVector>(
      generator, make_not_null(&xi_distribution), xi_target);
  get<1>(xi_target) = make_with_random_values<DataVector>(
      generator, make_not_null(&xi_distribution), xi_target);
  get<2>(xi_target) = DataVector(number_of_points, 0.0);
  for (size_t n_x = 4; n_x < 6; ++n_x) {
    for (size_t n_y = 4; n_y < 6; ++n_y) {
      const Mesh<3> source_mesh{
          {{n_x, n_y, 1}},
          {{Spectral::Basis::Legendre, Spectral::Basis::Legendre,
            Spectral::Basis::Cartoon}},
          {{Spectral::Quadrature::GaussLobatto,
            Spectral::Quadrature::GaussLobatto,
            Spectral::Quadrature::AxialSymmetry}}};
      const Polynomial f_x{n_x - 1, 1.5, 2.0};
      const Polynomial f_y{n_y - 1, 2.5, 1.5};
      const auto xi_source = logical_coordinates(source_mesh);
      const DataVector f_source =
          f_x(get<0>(xi_source)) * f_y(get<1>(xi_source));
      const DataVector f_expected =
          f_x(get<0>(xi_target)) * f_y(get<1>(xi_target));
      const intrp::Irregular<3> interpolator(source_mesh, xi_target);
      const DataVector f_interpolated = interpolator.interpolate(f_source);
      CHECK_ITERABLE_APPROX(f_interpolated, f_expected);
    }
  }
}

#ifdef SPECTRE_DEBUG
void test_errors() {
  const tnsr::I<DataVector, 2, Frame::ElementLogical> target_coords_2d{
      {{{0.5, 1.0}, {0.0, 1.5}}}};
  const tnsr::I<DataVector, 3, Frame::ElementLogical> target_coords_3d{
      {{{0.5, 1.0}, {0.0, 1.5}, {0.1, 0.8}}}};
  {
    INFO("Testing SphericalHarmonic basis consistency assertion for 2D");
    CHECK_THROWS_WITH(
        (intrp::Irregular<2>{
            Mesh<2>{
                {3, 5},
                {Spectral::Basis::SphericalHarmonic, Spectral::Basis::Legendre},
                {Spectral::Quadrature::Gauss, Spectral::Quadrature::Gauss}},
            target_coords_2d}),
        Catch::Matchers::ContainsSubstring(
            "Expected both dimensions to have spherical harmonic basis"));
  }
  {
    INFO("Testing ZernikeB2 basis consistency assertion for 2D");
    CHECK_THROWS_WITH(
        (intrp::Irregular<2>{
            Mesh<2>{{3, 3},
                    {Spectral::Basis::ZernikeB2, Spectral::Basis::Legendre},
                    {Spectral::Quadrature::GaussRadauUpper,
                     Spectral::Quadrature::Gauss}},
            target_coords_2d}),
        Catch::Matchers::ContainsSubstring("Unexpected basis combination"));
  }
  {
    INFO("Testing SphericalHarmonic basis consistency assertion for 3D");
    CHECK_THROWS_WITH(
        (intrp::Irregular<3>{
            Mesh<3>{
                {3, 3, 5},
                {Spectral::Basis::Legendre, Spectral::Basis::SphericalHarmonic,
                 Spectral::Basis::Legendre},
                {Spectral::Quadrature::Gauss, Spectral::Quadrature::Gauss,
                 Spectral::Quadrature::Gauss}},
            target_coords_3d}),
        Catch::Matchers::ContainsSubstring(
            "Expected last two dimensions to each have spherical harmonic "
            "basis"));
  }
  {
    INFO("Testing ZernikeB2 basis consistency assertion for 3D");
    CHECK_THROWS_WITH(
        (intrp::Irregular<3>{
            Mesh<3>{{3, 3, 3},
                    {Spectral::Basis::ZernikeB2, Spectral::Basis::Legendre,
                     Spectral::Basis::Legendre},
                    {Spectral::Quadrature::GaussRadauUpper,
                     Spectral::Quadrature::Gauss, Spectral::Quadrature::Gauss}},
            target_coords_3d}),
        Catch::Matchers::ContainsSubstring("Unexpected basis combination"));
  }
  {
    INFO("Testing N_phi odd assertion for ZernikeB2");
    CHECK_THROWS_WITH(
        (intrp::Irregular<2>{
            Mesh<2>{{3, 4},
                    {Spectral::Basis::ZernikeB2, Spectral::Basis::ZernikeB2},
                    {Spectral::Quadrature::GaussRadauUpper,
                     Spectral::Quadrature::Equiangular}},
            target_coords_2d}),
        Catch::Matchers::ContainsSubstring(
            "Need N_phi to be odd for stability"));
  }
  {
    INFO("Testing mixed FD and DG bases assertion for 2D");
    CHECK_THROWS_WITH(
        (intrp::Irregular<2>{Mesh<2>{{3, 3},
                                     {Spectral::Basis::FiniteDifference,
                                      Spectral::Basis::Legendre},
                                     {Spectral::Quadrature::FaceCentered,
                                      Spectral::Quadrature::Gauss}},
                             target_coords_2d}),
        Catch::Matchers::ContainsSubstring(
            "Mixed FD and DG bases are not supported"));
  }
  {
    INFO("Testing fd_to_fd_interp_order nullopt assertion for non-FD mesh");
    CHECK_THROWS_WITH(
        (intrp::Irregular<2>{
            Mesh<2>{{3, 3},
                    {Spectral::Basis::Legendre, Spectral::Basis::Legendre},
                    {Spectral::Quadrature::Gauss, Spectral::Quadrature::Gauss}},
            target_coords_2d, std::optional<size_t>{2}}),
        Catch::Matchers::ContainsSubstring(
            "fd_to_fd_interp_order only applies to FD meshes"));
  }
  {
    INFO("Testing mixed FD and DG bases assertion for 3D");
    CHECK_THROWS_WITH(
        (intrp::Irregular<3>{Mesh<3>{{3, 3, 3},
                                     {Spectral::Basis::FiniteDifference,
                                      Spectral::Basis::FiniteDifference,
                                      Spectral::Basis::Legendre},
                                     {Spectral::Quadrature::FaceCentered,
                                      Spectral::Quadrature::FaceCentered,
                                      Spectral::Quadrature::Gauss}},
                             target_coords_3d}),
        Catch::Matchers::ContainsSubstring(
            "Mixed FD and DG bases are not supported"));
  }
}
#endif
}  // namespace

// [[TimeOut, 10]]
SPECTRE_TEST_CASE("Unit.Numerical.Interpolation.IrregularInterpolant",
                  "[Unit][NumericalAlgorithms]") {
  test_irregular_interpolant<Spectral::Basis::Legendre,
                             Spectral::Quadrature::GaussLobatto>();
  test_irregular_interpolant<Spectral::Basis::Legendre,
                             Spectral::Quadrature::Gauss>();
  test_irregular_interpolant_mixed_quadrature();
  for (size_t max_degree = 1; max_degree <= 4; ++max_degree) {
    test_polynomial_interpolant<1>({{11}}, max_degree);
    test_polynomial_interpolant<2>({{11, 11}}, max_degree);
    test_polynomial_interpolant<2>({{11, 9}}, max_degree);
    test_polynomial_interpolant<3>({{11, 11, 11}}, max_degree);
    test_polynomial_interpolant<3>({{11, 9, 11}}, max_degree);
    test_polynomial_interpolant<3>({{11, 11, 9}}, max_degree);
    test_polynomial_interpolant<3>({{11, 9, 9}}, max_degree);
    test_polynomial_interpolant<3>({{11, 9, 13}}, max_degree);
    for (const bool specified_interp_order : {false, true}) {
      test_tov(max_degree, specified_interp_order);
    }
  }
  test_fd_interpolation();
  test_fd_coordinate_reuse();
  test_fd_interpolation_errors();
  MAKE_GENERATOR(generator);
  test_2d_spherical(make_not_null(&generator));
  test_3d_spherical(make_not_null(&generator));
  test_2d_disk(make_not_null(&generator));
  test_3d_cylinder(make_not_null(&generator));
  test_2d_hollow_disk(make_not_null(&generator));
  test_3d_hollow_cylinder(make_not_null(&generator));
  test_cartoon_spherical(make_not_null(&generator));
  test_cartoon_axial(make_not_null(&generator));
  test_cartoon_fd(make_not_null(&generator));
#ifdef SPECTRE_DEBUG
  test_errors();
#endif
}
