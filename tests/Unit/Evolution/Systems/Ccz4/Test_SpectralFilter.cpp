// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <random>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Ccz4/ApplyTensorYlmFilter.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Evolution/Systems/Ccz4/SpectralFilter.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Hypercube.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackCache.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/TensorYlmFilter.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {

using full_list = Ccz4::fd::System::variables_tag_list;
using nine_list = Ccz4::filter_detail::ccz4_vars_list<Frame::Inertial>;
using trailing_list =
    tmpl::list_difference<full_list, nine_list>;  // aux + boundary tags

constexpr size_t ell_max = 4;
constexpr size_t num_modes_to_kill = 2;

// Exact (bitwise) equality of two tensors of the same type.
template <typename TensorType>
bool exactly_equal(const TensorType& lhs, const TensorType& rhs) {
  for (size_t c = 0; c < lhs.size(); ++c) {
    for (size_t i = 0; i < lhs[c].size(); ++i) {
      if (lhs[c][i] != rhs[c][i]) {
        return false;
      }
    }
  }
  return true;
}

template <typename TensorType>
double max_abs_difference(const TensorType& lhs, const TensorType& rhs) {
  double result = 0.0;
  for (size_t c = 0; c < lhs.size(); ++c) {
    for (size_t i = 0; i < lhs[c].size(); ++i) {
      result = std::max(result, std::abs(lhs[c][i] - rhs[c][i]));
    }
  }
  return result;
}

// Random diagonally-dominant (invertible) Jacobian pair, same idiom as
// Test_ApplyTensorYlmFilter.
template <typename Generator>
auto make_jacobians(const size_t num_points,
                    const gsl::not_null<Generator*> generator) {
  std::uniform_real_distribution<double> dist{-1.0, 1.0};
  std::uniform_real_distribution<double> positive_dist{0.5, 1.0};
  InverseJacobian<DataVector, 3, Frame::Inertial, Frame::Grid>
      jac_inertial_to_grid(num_points);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      jac_inertial_to_grid.get(i, j) = 0.05 * dist(*generator);
    }
    jac_inertial_to_grid.get(i, i) += positive_dist(*generator);
  }
  InverseJacobian<DataVector, 3, Frame::Grid, Frame::Inertial>
      jac_grid_to_inertial(num_points);
  Scalar<DataVector> det(num_points);
  determinant_and_inverse(make_not_null(&det),
                          make_not_null(&jac_grid_to_inertial),
                          jac_inertial_to_grid);
  return std::make_pair(std::move(jac_inertial_to_grid),
                        std::move(jac_grid_to_inertial));
}

void test_fill_matrices() {
  ylm::TensorYlm::FilterMatrixHolder holder{};
  ylm::TensorYlm::fill_tensor_ylm_filters<full_list>(
      make_not_null(&holder), ell_max, num_modes_to_kill, std::nullopt,
      ylm::TensorYlm::CoefficientNormalization::Spherepack);

  // Exactly the ranks present in the 9 filtered CCZ4 variables are filled.
  CHECK(holder.scalar.has_value());
  CHECK(holder.i.has_value());
  CHECK(holder.ii.has_value());
  CHECK_FALSE(holder.ij.has_value());
  CHECK_FALSE(holder.kii.has_value());

  // The caching parameters record the fill.
  CHECK(holder.number_of_ell_modes_to_kill == num_modes_to_kill);
  CHECK_FALSE(holder.half_power.has_value());
  CHECK(holder.coefficient_normalization ==
        ylm::TensorYlm::CoefficientNormalization::Spherepack);

  // Refill with different parameters updates the recorded parameters.
  ylm::TensorYlm::fill_tensor_ylm_filters<full_list>(
      make_not_null(&holder), ell_max, num_modes_to_kill + 1, std::nullopt,
      ylm::TensorYlm::CoefficientNormalization::Spherepack);
  CHECK(holder.number_of_ell_modes_to_kill == num_modes_to_kill + 1);
}

// Pass-through + equivalence for a given number of radial points.
// radial_extents == 1 exercises the boundary (spherical-slice) path.
template <typename Generator>
void test_filter(const size_t radial_extents,
                 const gsl::not_null<Generator*> generator) {
  INFO("radial_extents = " << radial_extents);
  const auto& ylm = ::ylm::get_spherepack_cache(ell_max);
  const size_t num_points = ylm.physical_size() * radial_extents;

  std::uniform_real_distribution<double> dist{-1.0, 1.0};
  Variables<full_list> full_vars(num_points);
  for (size_t i = 0; i < full_vars.size(); ++i) {
    full_vars.data()[i] = dist(*generator);
  }
  const Variables<full_list> original_full_vars = full_vars;

  const auto [jac_inertial_to_grid, jac_grid_to_inertial] =
      make_jacobians(num_points, generator);

  ylm::TensorYlm::FilterMatrixHolder holder{};
  ylm::TensorYlm::fill_tensor_ylm_filters<full_list>(
      make_not_null(&holder), ell_max, num_modes_to_kill, std::nullopt,
      ylm::TensorYlm::CoefficientNormalization::Spherepack);

  // Temp storage sized as the SphericalShell filter sizes it: the full
  // Variables with radial_extents * spectral_size grid points.
  Variables<full_list> temp_storage(radial_extents * ylm.spectral_size());

  ylm::TensorYlm::apply_tensor_ylm_filter<full_list>(
      make_not_null(&full_vars), make_not_null(&temp_storage),
      jac_inertial_to_grid, jac_grid_to_inertial, holder, ell_max,
      radial_extents);

  {
    INFO("Trailing auxiliary and boundary variables are bitwise untouched");
    tmpl::for_each<trailing_list>([&full_vars,
                                   &original_full_vars]<typename Tag>(
                                      const tmpl::type_<Tag> /*meta*/) {
      CHECK(exactly_equal(get<Tag>(full_vars), get<Tag>(original_full_vars)));
    });
  }

  {
    INFO("Each of the 9 filtered variables visibly changed");
    tmpl::for_each<nine_list>([&full_vars, &original_full_vars]<typename Tag>(
                                  const tmpl::type_<Tag> /*meta*/) {
      CHECK(max_abs_difference(get<Tag>(full_vars),
                               get<Tag>(original_full_vars)) > 1.0e-12);
    });
  }

  {
    INFO("Equivalence with the production Ccz4 filter on the 9 variables");
    // Same input data and identical matrices, through the pre-existing
    // production entry point on a standalone 9-variable Variables.
    Variables<nine_list> nine_vars(num_points);
    tmpl::for_each<nine_list>([&nine_vars, &original_full_vars]<typename Tag>(
                                  const tmpl::type_<Tag> /*meta*/) {
      get<Tag>(nine_vars) = get<Tag>(original_full_vars);
    });
    Variables<nine_list> nine_temp_storage(0);
    Ccz4::apply_tensor_ylm_filter(
        make_not_null(&nine_vars), make_not_null(&nine_temp_storage),
        jac_inertial_to_grid, jac_grid_to_inertial, holder.scalar.value(),
        holder.i.value(), holder.ii.value(), ell_max, radial_extents);

    tmpl::for_each<nine_list>(
        [&full_vars, &nine_vars]<typename Tag>(const tmpl::type_<Tag>
                                               /*meta*/) {
          CHECK_ITERABLE_APPROX(get<Tag>(full_vars), get<Tag>(nine_vars));
        });
  }

  {
    INFO("Different filter parameters give a different result");
    ylm::TensorYlm::FilterMatrixHolder stronger_holder{};
    ylm::TensorYlm::fill_tensor_ylm_filters<full_list>(
        make_not_null(&stronger_holder), ell_max, num_modes_to_kill + 1,
        std::nullopt, ylm::TensorYlm::CoefficientNormalization::Spherepack);
    Variables<full_list> stronger_vars = original_full_vars;
    ylm::TensorYlm::apply_tensor_ylm_filter<full_list>(
        make_not_null(&stronger_vars), make_not_null(&temp_storage),
        jac_inertial_to_grid, jac_grid_to_inertial, stronger_holder, ell_max,
        radial_extents);
    bool some_nine_component_differs = false;
    tmpl::for_each<nine_list>([&full_vars, &stronger_vars,
                               &some_nine_component_differs]<typename Tag>(
                                  const tmpl::type_<Tag> /*meta*/) {
      if (max_abs_difference(get<Tag>(full_vars), get<Tag>(stronger_vars)) >
          1.0e-12) {
        some_nine_component_differs = true;
      }
    });
    CHECK(some_nine_component_differs);
  }
}

// No filter may modify the trailing auxiliary and boundary second-order
// variables: Filters::runtime::FilteredTags restricts the Hypercube
// exponential filter to the 9 original variables (matching the tag list
// of the legacy filter action in EvolveCcz4.hpp). Random data, so an
// unrestricted filter fails the bitwise pass-through checks.
template <typename Generator>
void test_hypercube_exp_passthrough(const gsl::not_null<Generator*> generator) {
  INFO("hypercube_exp_passthrough");
  const auto hypercube =
      TestHelpers::test_creation<Filters::runtime::Hypercube<3, full_list>>(
          "HalfPower: 2\n"
          "Enable: True\n"
          "BlocksToFilter: All\n"
          "VolumeFilterOnSubstep: False\n"
          "BoundaryCorrectionFilterOnSubstep: False\n"
          "VolumeFilterEveryNSteps: None\n"
          "BoundaryCorrectionFilterEveryNSteps: None\n");
  std::uniform_real_distribution<double> dist{-1.0, 1.0};
  {
    INFO("volume");
    const Mesh<3> mesh{5, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
    Variables<full_list> vars(mesh.number_of_grid_points());
    for (size_t i = 0; i < vars.size(); ++i) {
      vars.data()[i] = dist(*generator);
    }
    const Variables<full_list> original = vars;
    hypercube.apply_in_volume(make_not_null(&vars), mesh, std::nullopt,
                              std::nullopt);
    tmpl::for_each<trailing_list>(
        [&vars, &original]<typename Tag>(const tmpl::type_<Tag> /*meta*/) {
          CHECK(exactly_equal(get<Tag>(vars), get<Tag>(original)));
        });
    tmpl::for_each<nine_list>([&vars, &original]<typename Tag>(
                                  const tmpl::type_<Tag> /*meta*/) {
      CHECK(max_abs_difference(get<Tag>(vars), get<Tag>(original)) > 1.0e-12);
    });
  }
  {
    INFO("boundary");
    const Mesh<2> face_mesh{5, Spectral::Basis::Legendre,
                            Spectral::Quadrature::GaussLobatto};
    Variables<full_list> vars(face_mesh.number_of_grid_points());
    for (size_t i = 0; i < vars.size(); ++i) {
      vars.data()[i] = dist(*generator);
    }
    const Variables<full_list> original = vars;
    hypercube.apply_on_boundary(make_not_null(&vars), face_mesh, std::nullopt,
                                std::nullopt);
    tmpl::for_each<trailing_list>(
        [&vars, &original]<typename Tag>(const tmpl::type_<Tag> /*meta*/) {
          CHECK(exactly_equal(get<Tag>(vars), get<Tag>(original)));
        });
    tmpl::for_each<nine_list>([&vars, &original]<typename Tag>(
                                  const tmpl::type_<Tag> /*meta*/) {
      CHECK(max_abs_difference(get<Tag>(vars), get<Tag>(original)) > 1.0e-12);
    });
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Ccz4.SpectralFilter",
                  "[Unit][Evolution]") {
  test_fill_matrices();
  MAKE_GENERATOR(generator);
  // Volume (shell) path and single-slice (boundary) path.
  test_filter(3, make_not_null(&generator));
  test_filter(1, make_not_null(&generator));
  test_hypercube_exp_passthrough(make_not_null(&generator));
}
