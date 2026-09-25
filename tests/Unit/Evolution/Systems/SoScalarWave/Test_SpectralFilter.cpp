// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/Systems/SoScalarWave/ProjectSpectralFilter.hpp"
#include "Evolution/Systems/SoScalarWave/SpectralFilter.hpp"
#include "Evolution/Systems/SoScalarWave/System.hpp"
#include "Evolution/Systems/SoScalarWave/Tags.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Factory.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Filter.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Hypercube.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/None.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/SphericalShell.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Tag.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackCache.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackIterator.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/TensorYlmFilter.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {

using tags_1d = typename SoScalarWave::System<1>::variables_tag::tags_list;
using tags_2d = typename SoScalarWave::System<2>::variables_tag::tags_list;
using tags_3d = typename SoScalarWave::System<3>::variables_tag::tags_list;

using Psi = SoScalarWave::Tags::Psi;
using Pi = SoScalarWave::Tags::Pi;
using Phi = SoScalarWave::Tags::Phi<3>;
using BoundaryPsi = SoScalarWave::Tags::BoundaryPsi;

using scalar_norm = ylm::TensorYlm::CoefficientNormalization;

// The filtered leading block and the untouched trailing block.
using filtered_tags = tmpl::list<Psi, Pi>;
using trailing_tags = tmpl::list<Phi, BoundaryPsi>;

// ---------------------------------------------------------------------------
// Compile-time contract: all_runtime_filters registers SphericalShell only in
// 3D; 1D/2D see only None + Hypercube.
// ---------------------------------------------------------------------------
static_assert(tmpl::size<SoScalarWave::all_runtime_filters<1>>::value == 2,
              "1D must register exactly None + Hypercube.");
static_assert(tmpl::size<SoScalarWave::all_runtime_filters<2>>::value == 2,
              "2D must register exactly None + Hypercube.");
static_assert(tmpl::size<SoScalarWave::all_runtime_filters<3>>::value == 3,
              "3D must register None + Hypercube + SphericalShell.");
static_assert(
    tmpl::list_contains_v<SoScalarWave::all_runtime_filters<3>,
                          Filters::runtime::SphericalShell<tags_3d>>,
    "3D must contain the SphericalShell filter for the SoScalarWave list.");
static_assert(
    not tmpl::list_contains_v<SoScalarWave::all_runtime_filters<1>,
                              Filters::runtime::SphericalShell<tags_1d>>,
    "1D must not register the SphericalShell filter.");

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

template <typename Generator>
void fill_random(const gsl::not_null<Variables<tags_3d>*> vars,
                 const gsl::not_null<Generator*> generator) {
  std::uniform_real_distribution<double> dist{-1.0, 1.0};
  for (size_t i = 0; i < vars->size(); ++i) {
    vars->data()[i] = dist(*generator);
  }
}

// Jacobians are IGNORED by the SoScalarWave (scalar) specialization. Build both
// frame-pair inverse Jacobians filled with a chosen constant per-component so
// that "identity" and "garbage" variants can be compared.
template <typename InvJacType>
InvJacType make_invjac(const size_t num_points, const bool garbage) {
  InvJacType jac(num_points);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      if (garbage) {
        // Wildly non-identity, deliberately including large / negative
        // off-diagonal entries.
        jac.get(i, j) =
            13.0 * static_cast<double>(i) - 7.5 * static_cast<double>(j) - 3.25;
      } else {
        jac.get(i, j) = (i == j) ? 1.0 : 0.0;
      }
    }
  }
  return jac;
}

auto make_ig(const size_t np, const bool garbage) {
  return make_invjac<
      InverseJacobian<DataVector, 3, Frame::Inertial, Frame::Grid>>(np,
                                                                    garbage);
}
auto make_gi(const size_t np, const bool garbage) {
  return make_invjac<
      InverseJacobian<DataVector, 3, Frame::Grid, Frame::Inertial>>(np,
                                                                    garbage);
}

// Independent Heaviside-filter reference for a single scalar nodal component:
// phys_to_spec -> zero the top `num_modes_to_kill` ell modes -> spec_to_phys.
// Uses ONLY Spherepack primitives (never the code under test's pipeline nor
// fill_filter): for a Heaviside cutoff the exact effect is to zero every
// coefficient with l > ell_max - num_modes_to_kill and leave the rest intact.
DataVector heaviside_reference_scalar(const DataVector& nodal,
                                      const ylm::Spherepack& ylm,
                                      const size_t radial_extents,
                                      const size_t ell_max,
                                      const size_t num_modes_to_kill) {
  std::vector<double> spectral(radial_extents * ylm.spectral_size(), 0.0);
  ylm.phys_to_spec_all_offsets(make_not_null(spectral.data()),
                               make_not_null(nodal.data()), radial_extents);
  ylm::SpherepackIterator it(ell_max, ell_max, radial_extents);
  for (size_t offset = 0; offset < radial_extents; ++offset) {
    for (it.reset(); it; ++it) {
      // l > ell_max - num_modes_to_kill, written to avoid size_t underflow.
      if (it.l() + num_modes_to_kill > ell_max) {
        spectral[it() + offset] = 0.0;
      }
    }
  }
  DataVector result(nodal.size());
  ylm.spec_to_phys_all_offsets(make_not_null(result.data()),
                               make_not_null(spectral.data()), radial_extents);
  return result;
}

// Build a single-Ylm-mode (l_target, m=0) nodal scalar with a per-radial-slice
// amplitude. Uses only spec_to_phys.
DataVector pure_mode_scalar(const ylm::Spherepack& ylm,
                            const size_t radial_extents, const size_t ell_max,
                            const size_t l_target,
                            const std::vector<double>& radial_amplitude) {
  std::vector<double> spectral(radial_extents * ylm.spectral_size(), 0.0);
  ylm::SpherepackIterator it(ell_max, ell_max, radial_extents);
  for (size_t offset = 0; offset < radial_extents; ++offset) {
    for (it.reset(); it; ++it) {
      if (it.l() == l_target and it.m() == 0 and
          it.coefficient_array() ==
              ylm::SpherepackIterator::CoefficientArray::a) {
        spectral[it() + offset] = radial_amplitude[offset];
      }
    }
  }
  DataVector nodal(radial_extents * ylm.physical_size());
  ylm.spec_to_phys_all_offsets(make_not_null(nodal.data()),
                               make_not_null(spectral.data()), radial_extents);
  return nodal;
}

// Drive the SoScalarWave apply specialization directly, with a freshly-filled
// matrix holder.
void run_apply(const gsl::not_null<Variables<tags_3d>*> vars,
               const size_t ell_max, const size_t radial_extents,
               const size_t num_modes_to_kill,
               const std::optional<size_t> half_power = std::nullopt,
               const bool garbage_jacobians = false) {
  const auto& ylm = ::ylm::get_spherepack_cache(ell_max);
  ylm::TensorYlm::FilterMatrixHolder holder{};
  ylm::TensorYlm::fill_tensor_ylm_filters<tags_3d>(
      make_not_null(&holder), ell_max, num_modes_to_kill, half_power,
      scalar_norm::Spherepack);
  const size_t np = vars->number_of_grid_points();
  Variables<tags_3d> temp(radial_extents * ylm.spectral_size());
  ylm::TensorYlm::apply_tensor_ylm_filter<tags_3d>(
      vars, make_not_null(&temp), make_ig(np, garbage_jacobians),
      make_gi(np, garbage_jacobians), holder, ell_max, radial_extents);
}

constexpr size_t ell_max = 8;

// ===========================================================================
// Attack 1 + 2: pass-through (bitwise) of Phi/BoundaryPsi and correct filtering
// of Psi/Pi against an independent Spherepack reference.
// ===========================================================================
template <typename Generator>
void test_passthrough_and_filtering(const size_t radial_extents,
                                    const gsl::not_null<Generator*> generator) {
  INFO("passthrough_and_filtering, radial_extents = " << radial_extents);
  const size_t num_modes_to_kill = 2;
  const auto& ylm = ::ylm::get_spherepack_cache(ell_max);
  const size_t np = ylm.physical_size() * radial_extents;

  Variables<tags_3d> vars(np);
  fill_random(make_not_null(&vars), generator);
  const Variables<tags_3d> original = vars;

  run_apply(make_not_null(&vars), ell_max, radial_extents, num_modes_to_kill);

  {
    INFO("Phi and BoundaryPsi are bitwise untouched");
    tmpl::for_each<trailing_tags>(
        [&vars, &original]<typename Tag>(const tmpl::type_<Tag> /*meta*/) {
          CHECK(exactly_equal(get<Tag>(vars), get<Tag>(original)));
        });
  }
  {
    INFO("Psi and Pi visibly changed");
    tmpl::for_each<filtered_tags>([&vars, &original]<typename Tag>(
                                      const tmpl::type_<Tag> /*meta*/) {
      CHECK(max_abs_difference(get<Tag>(vars), get<Tag>(original)) > 1.0e-12);
    });
  }
  {
    INFO("Psi and Pi match the independent Heaviside reference");
    const DataVector psi_ref =
        heaviside_reference_scalar(get(get<Psi>(original)), ylm, radial_extents,
                                   ell_max, num_modes_to_kill);
    const DataVector pi_ref =
        heaviside_reference_scalar(get(get<Pi>(original)), ylm, radial_extents,
                                   ell_max, num_modes_to_kill);
    CHECK_ITERABLE_APPROX(get(get<Psi>(vars)), psi_ref);
    CHECK_ITERABLE_APPROX(get(get<Pi>(vars)), pi_ref);
  }
}

// ===========================================================================
// Attack 3: analytic kill test.
// ===========================================================================
void test_analytic_kill() {
  INFO("analytic_kill");
  const size_t num_modes_to_kill = 1;
  const auto& ylm = ::ylm::get_spherepack_cache(ell_max);

  // (a) pure Y_{ell_max, 0} mode times a radial profile is annihilated.
  for (const size_t radial_extents : {size_t{1}, size_t{3}}) {
    INFO("annihilation of top mode, radial_extents = " << radial_extents);
    std::vector<double> amplitude(radial_extents);
    for (size_t r = 0; r < radial_extents; ++r) {
      amplitude[r] = 1.0 + 0.5 * static_cast<double>(r);
    }
    const DataVector top_mode =
        pure_mode_scalar(ylm, radial_extents, ell_max, ell_max, amplitude);
    // Sanity: the mode is actually nontrivial before filtering.
    CHECK(max(abs(top_mode)) > 0.1);

    Variables<tags_3d> vars(ylm.physical_size() * radial_extents, 0.0);
    get(get<Psi>(vars)) = top_mode;
    get(get<Pi>(vars)) = top_mode;
    run_apply(make_not_null(&vars), ell_max, radial_extents, num_modes_to_kill);
    CHECK(max(abs(get(get<Psi>(vars)))) < 1.0e-11);
    CHECK(max(abs(get(get<Pi>(vars)))) < 1.0e-11);
  }

  // (b) a low-l (l = 1) mode is preserved.
  {
    INFO("preservation of low-l mode");
    const DataVector low_mode =
        pure_mode_scalar(ylm, 1, ell_max, 1, std::vector<double>{1.0});
    Variables<tags_3d> vars(ylm.physical_size(), 0.0);
    get(get<Psi>(vars)) = low_mode;
    run_apply(make_not_null(&vars), ell_max, 1, num_modes_to_kill);
    CHECK_ITERABLE_APPROX(get(get<Psi>(vars)), low_mode);
  }

  // (c) a constant is preserved exactly (to roundoff).
  {
    INFO("preservation of a constant");
    const DataVector constant_field(ylm.physical_size(), 3.7);
    Variables<tags_3d> vars(ylm.physical_size(), 0.0);
    get(get<Psi>(vars)) = constant_field;
    run_apply(make_not_null(&vars), ell_max, 1, num_modes_to_kill);
    CHECK_ITERABLE_APPROX(get(get<Psi>(vars)), constant_field);
  }
}

// ===========================================================================
// Attack 4: Psi and Pi filtered independently and identically (no cross-talk).
// ===========================================================================
template <typename Generator>
void test_psi_pi_independence(const size_t radial_extents,
                              const gsl::not_null<Generator*> generator) {
  INFO("psi_pi_independence, radial_extents = " << radial_extents);
  const size_t num_modes_to_kill = 2;
  const auto& ylm = ::ylm::get_spherepack_cache(ell_max);
  const size_t np = ylm.physical_size() * radial_extents;

  Variables<tags_3d> both(np);
  fill_random(make_not_null(&both), generator);
  const DataVector field_a = get(get<Psi>(both));
  const DataVector field_b = get(get<Pi>(both));

  // Same field in both -> identical filtered output.
  {
    INFO("Same input field in Psi and Pi gives identical output");
    Variables<tags_3d> same(np, 0.0);
    get(get<Psi>(same)) = field_a;
    get(get<Pi>(same)) = field_a;
    run_apply(make_not_null(&same), ell_max, radial_extents, num_modes_to_kill);
    CHECK(exactly_equal(get<Psi>(same), get<Pi>(same)));
  }

  // Different fields -> no cross-talk: each output depends only on its input.
  {
    INFO("Distinct inputs do not cross-contaminate");
    Variables<tags_3d> combined(np, 0.0);
    get(get<Psi>(combined)) = field_a;
    get(get<Pi>(combined)) = field_b;
    run_apply(make_not_null(&combined), ell_max, radial_extents,
              num_modes_to_kill);

    Variables<tags_3d> psi_only(np, 0.0);
    get(get<Psi>(psi_only)) = field_a;  // Pi left zero
    run_apply(make_not_null(&psi_only), ell_max, radial_extents,
              num_modes_to_kill);

    Variables<tags_3d> pi_only(np, 0.0);
    get(get<Pi>(pi_only)) = field_b;  // Psi left zero
    run_apply(make_not_null(&pi_only), ell_max, radial_extents,
              num_modes_to_kill);

    CHECK(exactly_equal(get<Psi>(combined), get<Psi>(psi_only)));
    CHECK(exactly_equal(get<Pi>(combined), get<Pi>(pi_only)));
  }
}

// ===========================================================================
// Attack 5: Jacobians are ignored (scalar contract).
// ===========================================================================
template <typename Generator>
void test_jacobian_ignored(const size_t radial_extents,
                           const gsl::not_null<Generator*> generator) {
  INFO("jacobian_ignored, radial_extents = " << radial_extents);
  const size_t num_modes_to_kill = 3;
  const auto& ylm = ::ylm::get_spherepack_cache(ell_max);
  const size_t np = ylm.physical_size() * radial_extents;

  Variables<tags_3d> vars(np);
  fill_random(make_not_null(&vars), generator);

  Variables<tags_3d> with_identity = vars;
  run_apply(make_not_null(&with_identity), ell_max, radial_extents,
            num_modes_to_kill, std::nullopt, false);

  Variables<tags_3d> with_garbage = vars;
  run_apply(make_not_null(&with_garbage), ell_max, radial_extents,
            num_modes_to_kill, std::nullopt, true);

  tmpl::for_each<tags_3d>(
      [&with_identity, &with_garbage]<typename Tag>(const tmpl::type_<Tag>
                                                    /*meta*/) {
        CHECK(exactly_equal(get<Tag>(with_identity), get<Tag>(with_garbage)));
      });
}

// ===========================================================================
// Attack 6: fill_tensor_ylm_filters caching / only-scalar-filled.
// ===========================================================================
template <typename Generator>
void test_fill_matrices(const gsl::not_null<Generator*> generator) {
  INFO("fill_matrices");
  const size_t radial_extents = 2;
  const auto& ylm = ::ylm::get_spherepack_cache(ell_max);
  const size_t np = ylm.physical_size() * radial_extents;

  ylm::TensorYlm::FilterMatrixHolder holder{};
  ylm::TensorYlm::fill_tensor_ylm_filters<tags_3d>(make_not_null(&holder),
                                                   ell_max, 2, std::nullopt,
                                                   scalar_norm::Spherepack);

  // Only the scalar matrix is filled; the tensor-rank matrices remain unset.
  CHECK(holder.scalar.has_value());
  CHECK_FALSE(holder.i.has_value());
  CHECK_FALSE(holder.ii.has_value());
  CHECK_FALSE(holder.ij.has_value());
  CHECK_FALSE(holder.kii.has_value());
  CHECK(holder.number_of_ell_modes_to_kill == 2);
  CHECK_FALSE(holder.half_power.has_value());
  CHECK(holder.coefficient_normalization == scalar_norm::Spherepack);

  // Locate diagonal matrix positions for representative ell values (the fill
  // uses a stride-1, zero_m_is_real=false iterator).
  ylm::SpherepackIterator finder(ell_max, ell_max, 1, false);
  const auto position_for_l = [&finder](const size_t l_value) {
    for (finder.reset(); finder; ++finder) {
      if (finder.l() == l_value) {
        return finder();
      }
    }
    ERROR("no position for l");
  };
  const size_t pos_l0 = position_for_l(0);
  const size_t pos_l6 = position_for_l(6);
  const size_t pos_l8 = position_for_l(8);

  // Heaviside with kill=2: lcut^+ = 6, so l in {7,8} are zeroed (diag = -1),
  // l <= 6 are retained (diag = 0).
  CHECK(holder.scalar.value()(pos_l0, pos_l0) == 0.0);
  CHECK(holder.scalar.value()(pos_l6, pos_l6) == 0.0);
  CHECK(holder.scalar.value()(pos_l8, pos_l8) == -1.0);

  // Refill with the SAME parameters: matrix unchanged (reuse path).
  ylm::TensorYlm::fill_tensor_ylm_filters<tags_3d>(make_not_null(&holder),
                                                   ell_max, 2, std::nullopt,
                                                   scalar_norm::Spherepack);
  CHECK(holder.scalar.value()(pos_l6, pos_l6) == 0.0);
  CHECK(holder.scalar.value()(pos_l8, pos_l8) == -1.0);

  // Refill with a DIFFERENT NumModesToKill: matrix is refilled; l=6 now killed.
  ylm::TensorYlm::fill_tensor_ylm_filters<tags_3d>(make_not_null(&holder),
                                                   ell_max, 3, std::nullopt,
                                                   scalar_norm::Spherepack);
  CHECK(holder.number_of_ell_modes_to_kill == 3);
  CHECK(holder.scalar.value()(pos_l6, pos_l6) == -1.0);
  CHECK(holder.scalar.value()(pos_l8, pos_l8) == -1.0);
  CHECK(holder.scalar.value()(pos_l0, pos_l0) == 0.0);
  CHECK_FALSE(holder.i.has_value());
  CHECK_FALSE(holder.ii.has_value());

  // Observable effect: same params give identical results; changed params
  // give different results.
  Variables<tags_3d> data(np);
  fill_random(make_not_null(&data), generator);

  const auto apply_with = [&](const size_t num_modes_to_kill) {
    ylm::TensorYlm::FilterMatrixHolder local{};
    ylm::TensorYlm::fill_tensor_ylm_filters<tags_3d>(
        make_not_null(&local), ell_max, num_modes_to_kill, std::nullopt,
        scalar_norm::Spherepack);
    Variables<tags_3d> vars = data;
    Variables<tags_3d> temp(radial_extents * ylm.spectral_size());
    ylm::TensorYlm::apply_tensor_ylm_filter<tags_3d>(
        make_not_null(&vars), make_not_null(&temp), make_ig(np, false),
        make_gi(np, false), local, ell_max, radial_extents);
    return vars;
  };
  const Variables<tags_3d> two_a = apply_with(2);
  const Variables<tags_3d> two_b = apply_with(2);
  const Variables<tags_3d> three = apply_with(3);
  CHECK(exactly_equal(get<Psi>(two_a), get<Psi>(two_b)));
  CHECK(max_abs_difference(get<Psi>(two_a), get<Psi>(three)) > 1.0e-12);
}

// ===========================================================================
// Attack 7: temp_storage sizing edge.
// ===========================================================================
template <typename Generator>
void test_temp_storage_sizing(const size_t radial_extents,
                              const gsl::not_null<Generator*> generator) {
  INFO("temp_storage_sizing, radial_extents = " << radial_extents);
  const size_t num_modes_to_kill = 2;
  const auto& ylm = ::ylm::get_spherepack_cache(ell_max);
  const size_t np = ylm.physical_size() * radial_extents;

  Variables<tags_3d> data(np);
  fill_random(make_not_null(&data), generator);

  ylm::TensorYlm::FilterMatrixHolder holder{};
  ylm::TensorYlm::fill_tensor_ylm_filters<tags_3d>(
      make_not_null(&holder), ell_max, num_modes_to_kill, std::nullopt,
      scalar_norm::Spherepack);

  // The specialization needs 2 * radial_extents * spectral_size doubles of
  // scratch. A Variables<tags_3d>(g) holds 6*g doubles, so the smallest
  // conforming grid-point count is ceil(radial*spectral / 3).
  const size_t spectral_doubles = 2 * radial_extents * ylm.spectral_size();
  const size_t min_grid_points = (radial_extents * ylm.spectral_size() + 2) / 3;
  REQUIRE(6 * min_grid_points >= spectral_doubles);

  const auto apply_with_temp = [&](const size_t temp_grid_points) {
    Variables<tags_3d> vars = data;
    Variables<tags_3d> temp(temp_grid_points);
    ylm::TensorYlm::apply_tensor_ylm_filter<tags_3d>(
        make_not_null(&vars), make_not_null(&temp), make_ig(np, false),
        make_gi(np, false), holder, ell_max, radial_extents);
    return vars;
  };

  const Variables<tags_3d> minimal = apply_with_temp(min_grid_points);
  const Variables<tags_3d> comfortable =
      apply_with_temp(3 * radial_extents * ylm.spectral_size());
  tmpl::for_each<tags_3d>(
      [&minimal, &comfortable]<typename Tag>(const tmpl::type_<Tag> /*meta*/) {
        CHECK(exactly_equal(get<Tag>(minimal), get<Tag>(comfortable)));
      });
}

// ===========================================================================
// Attack 8: ProjectSpectralFilter -- every AMR overload errors out.
// ===========================================================================
void test_project_spectral_filter() {
  INFO("project_spectral_filter");
  using Projector = SoScalarWave::ProjectSpectralFilter<3>;
  using filter_tag = Filters::runtime::Tags::SpectralFilter<3, tags_3d>;
  using filter_type = typename filter_tag::type;

  filter_type filter{};

  // p-refinement overload.
  {
    const std::pair<Mesh<3>, Element<3>> old_mesh_and_element{
        Mesh<3>{2, Spectral::Basis::Legendre,
                Spectral::Quadrature::GaussLobatto},
        Element<3>{}};
    CHECK_THROWS_WITH(
        Projector::apply(make_not_null(&filter), old_mesh_and_element),
        Catch::Matchers::ContainsSubstring(
            "AMR is not supported for SoScalarWave"));
  }

  // h-refinement split overload.
  {
    const tuples::TaggedTuple<filter_tag> parent_items{};
    CHECK_THROWS_WITH(Projector::apply(make_not_null(&filter), parent_items),
                      Catch::Matchers::ContainsSubstring(
                          "AMR is not supported for SoScalarWave"));
  }

  // h-refinement join overload.
  {
    const std::unordered_map<ElementId<3>, tuples::TaggedTuple<filter_tag>>
        children_items{};
    CHECK_THROWS_WITH(Projector::apply(make_not_null(&filter), children_items),
                      Catch::Matchers::ContainsSubstring(
                          "AMR is not supported for SoScalarWave"));
  }
}

// ===========================================================================
// Attack 9: end-to-end through the SphericalShell framework.
// ===========================================================================
// The mesh of a Ylm shell block: Legendre radially, SphericalHarmonic in the
// angular directions (Gauss colatitude, Equiangular longitude).
Mesh<3> make_shell_mesh(const size_t radial_extents, const size_t l_max) {
  return Mesh<3>{
      {{radial_extents, l_max + 1, (2 * l_max) + 1}},
      {{Spectral::Basis::Legendre, Spectral::Basis::SphericalHarmonic,
        Spectral::Basis::SphericalHarmonic}},
      {{Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::Gauss,
        Spectral::Quadrature::Equiangular}}};
}

template <typename Generator>
void test_end_to_end(const gsl::not_null<Generator*> generator) {
  INFO("end_to_end");
  const size_t radial_extents = 4;
  const auto& ylm = ::ylm::get_spherepack_cache(ell_max);
  const size_t np = ylm.physical_size() * radial_extents;
  const Mesh<3> mesh = make_shell_mesh(radial_extents, ell_max);
  REQUIRE(mesh.number_of_grid_points() == np);

  // Heaviside-only angular filter (no smooth roll-off, no radial filter) so the
  // framework path equals the pure angular reference.
  const auto filter =
      TestHelpers::test_creation<Filters::runtime::SphericalShell<tags_3d>>(
          "NumModesToKill: 2\n"
          "AngularHalfPower: None\n"
          "RadialHalfPower: None\n"
          "Enable: True\n"
          "BlocksToFilter: All\n"
          "VolumeFilterOnSubstep: False\n"
          "BoundaryCorrectionFilterOnSubstep: False\n"
          "VolumeFilterEveryNSteps: None\n"
          "BoundaryCorrectionFilterEveryNSteps: None\n");

  CHECK(filter.supports_mesh(mesh));
  CHECK_FALSE(filter.supports_mesh(
      Mesh<3>{{{radial_extents, ell_max + 1, (2 * ell_max) + 1}},
              Spectral::Basis::Legendre,
              Spectral::Quadrature::GaussLobatto}));

  Variables<tags_3d> vars(np);
  fill_random(make_not_null(&vars), generator);
  const Variables<tags_3d> original = vars;

  const std::optional<
      InverseJacobian<DataVector, 3, Frame::Grid, Frame::Inertial>>
      inv_jac_grid_to_inertial = make_gi(np, false);
  const std::optional<Jacobian<DataVector, 3, Frame::Grid, Frame::Inertial>>
      jac_grid_to_inertial = [np]() {
        Jacobian<DataVector, 3, Frame::Grid, Frame::Inertial> jac(np);
        for (size_t i = 0; i < 3; ++i) {
          for (size_t j = 0; j < 3; ++j) {
            jac.get(i, j) = (i == j) ? 1.0 : 0.0;
          }
        }
        return jac;
      }();

  filter.apply_in_volume(make_not_null(&vars), mesh, inv_jac_grid_to_inertial,
                         jac_grid_to_inertial);

  {
    INFO("Phi and BoundaryPsi are bitwise untouched by the framework path");
    tmpl::for_each<trailing_tags>(
        [&vars, &original]<typename Tag>(const tmpl::type_<Tag> /*meta*/) {
          CHECK(exactly_equal(get<Tag>(vars), get<Tag>(original)));
        });
  }
  {
    INFO("Psi and Pi are filtered to the independent reference");
    tmpl::for_each<filtered_tags>([&vars, &original]<typename Tag>(
                                      const tmpl::type_<Tag> /*meta*/) {
      CHECK(max_abs_difference(get<Tag>(vars), get<Tag>(original)) > 1.0e-12);
    });
    const DataVector psi_ref = heaviside_reference_scalar(
        get(get<Psi>(original)), ylm, radial_extents, ell_max, 2);
    const DataVector pi_ref = heaviside_reference_scalar(
        get(get<Pi>(original)), ylm, radial_extents, ell_max, 2);
    CHECK_ITERABLE_APPROX(get(get<Psi>(vars)), psi_ref);
    CHECK_ITERABLE_APPROX(get(get<Pi>(vars)), pi_ref);
  }
}

}  // namespace

// The exponential (Hypercube) filter and the SphericalShell radial
// roll-off must never modify the LDG auxiliary variable Phi (rebuilt from
// Psi every step with lifted boundary corrections) or the boundary-only
// evolved variable BoundaryPsi: Filters::runtime::FilteredTags restricts
// them to Psi and Pi. Random data, so an unrestricted filter fails the
// bitwise pass-through checks.
template <typename Generator>
void test_exp_filter_passthrough(const gsl::not_null<Generator*> generator) {
  INFO("exp_filter_passthrough");
  const auto hypercube =
      TestHelpers::test_creation<Filters::runtime::Hypercube<3, tags_3d>>(
          "HalfPower: 2\n"
          "Enable: True\n"
          "BlocksToFilter: All\n"
          "VolumeFilterOnSubstep: False\n"
          "BoundaryCorrectionFilterOnSubstep: False\n"
          "VolumeFilterEveryNSteps: None\n"
          "BoundaryCorrectionFilterEveryNSteps: None\n");
  {
    INFO("Hypercube volume");
    const Mesh<3> mesh{5, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
    Variables<tags_3d> vars(mesh.number_of_grid_points());
    fill_random(make_not_null(&vars), generator);
    const Variables<tags_3d> original = vars;
    hypercube.apply_in_volume(make_not_null(&vars), mesh, std::nullopt,
                              std::nullopt);
    tmpl::for_each<trailing_tags>(
        [&vars, &original]<typename Tag>(const tmpl::type_<Tag> /*meta*/) {
          CHECK(exactly_equal(get<Tag>(vars), get<Tag>(original)));
        });
    tmpl::for_each<filtered_tags>([&vars, &original]<typename Tag>(
                                      const tmpl::type_<Tag> /*meta*/) {
      CHECK(max_abs_difference(get<Tag>(vars), get<Tag>(original)) > 1.0e-12);
    });
  }
  {
    INFO("Hypercube boundary");
    const Mesh<2> face_mesh{5, Spectral::Basis::Legendre,
                            Spectral::Quadrature::GaussLobatto};
    Variables<tags_3d> vars(face_mesh.number_of_grid_points());
    std::uniform_real_distribution<double> dist{-1.0, 1.0};
    for (size_t i = 0; i < vars.size(); ++i) {
      vars.data()[i] = dist(*generator);
    }
    const Variables<tags_3d> original = vars;
    hypercube.apply_on_boundary(make_not_null(&vars), face_mesh, std::nullopt,
                                std::nullopt);
    tmpl::for_each<trailing_tags>(
        [&vars, &original]<typename Tag>(const tmpl::type_<Tag> /*meta*/) {
          CHECK(exactly_equal(get<Tag>(vars), get<Tag>(original)));
        });
    tmpl::for_each<filtered_tags>([&vars, &original]<typename Tag>(
                                      const tmpl::type_<Tag> /*meta*/) {
      CHECK(max_abs_difference(get<Tag>(vars), get<Tag>(original)) > 1.0e-12);
    });
  }
  {
    INFO("SphericalShell radial roll-off");
    const size_t radial_extents = 4;
    const auto& ylm = ::ylm::get_spherepack_cache(ell_max);
    const size_t np = ylm.physical_size() * radial_extents;
    const Mesh<3> mesh = make_shell_mesh(radial_extents, ell_max);
    const auto shell =
        TestHelpers::test_creation<Filters::runtime::SphericalShell<tags_3d>>(
            "NumModesToKill: 1\n"
            "AngularHalfPower: None\n"
            "RadialHalfPower: 2\n"
            "Enable: True\n"
            "BlocksToFilter: All\n"
            "VolumeFilterOnSubstep: False\n"
            "BoundaryCorrectionFilterOnSubstep: False\n"
            "VolumeFilterEveryNSteps: None\n"
            "BoundaryCorrectionFilterEveryNSteps: None\n");
    Variables<tags_3d> vars(np);
    fill_random(make_not_null(&vars), generator);
    const Variables<tags_3d> original = vars;
    shell.apply_in_volume(
        make_not_null(&vars), mesh, make_gi(np, false), [np]() {
          Jacobian<DataVector, 3, Frame::Grid, Frame::Inertial> jac(np);
          for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
              jac.get(i, j) = (i == j) ? 1.0 : 0.0;
            }
          }
          return jac;
        }());
    tmpl::for_each<trailing_tags>(
        [&vars, &original]<typename Tag>(const tmpl::type_<Tag> /*meta*/) {
          CHECK(exactly_equal(get<Tag>(vars), get<Tag>(original)));
        });
    tmpl::for_each<filtered_tags>([&vars, &original]<typename Tag>(
                                      const tmpl::type_<Tag> /*meta*/) {
      CHECK(max_abs_difference(get<Tag>(vars), get<Tag>(original)) > 1.0e-12);
    });
  }
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.SoScalarWave.SpectralFilter",
                  "[Unit][Evolution]") {
  MAKE_GENERATOR(generator);
  // Boundary-face path (radial_extents == 1) and volume/shell path (> 1).
  test_passthrough_and_filtering(1, make_not_null(&generator));
  test_passthrough_and_filtering(3, make_not_null(&generator));
  test_analytic_kill();
  test_psi_pi_independence(1, make_not_null(&generator));
  test_psi_pi_independence(3, make_not_null(&generator));
  test_jacobian_ignored(1, make_not_null(&generator));
  test_jacobian_ignored(3, make_not_null(&generator));
  test_fill_matrices(make_not_null(&generator));
  test_temp_storage_sizing(1, make_not_null(&generator));
  test_temp_storage_sizing(3, make_not_null(&generator));
  test_project_spectral_filter();
  test_end_to_end(make_not_null(&generator));
  test_exp_filter_passthrough(make_not_null(&generator));
}
