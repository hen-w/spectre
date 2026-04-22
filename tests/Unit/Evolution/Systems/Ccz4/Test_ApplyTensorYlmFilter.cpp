// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <memory>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/FrameTransform.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Structure/Topology.hpp"
#include "Evolution/Systems/Ccz4/ApplyTensorYlmFilter.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/NumericalAlgorithms/SphericalHarmonics/Test_ApplyTensorYlmFilter.hpp"
#include "NumericalAlgorithms/LinearOperators/Filter.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/ApplyTensorYlmFilter.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackCache.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackIterator.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/TensorYlmFilter.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace Ccz4 {
namespace {
struct Metavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<Filters::Filter, tmpl::list<TensorYlmFilter>>>;
  };
};

void test_transform_spatial_tensors_to_different_frame() {
  constexpr size_t mesh_size = 10;

  Variables<filter_detail::ccz4_vars_list<Frame::Inertial>> inertial_vars(
      mesh_size);

  // Fill inertial_vars with random numbers
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<double> dist{-1.0, 1.0};
  for (size_t i = 0; i < inertial_vars.size(); ++i) {
    inertial_vars.data()[i] = dist(generator);
  }

  // Create a diagonally-dominant random Jacobian (invertible).
  std::uniform_real_distribution<double> positive_dist{0.5, 1.0};
  InverseJacobian<DataVector, 3, Frame::Inertial, Frame::Grid>
      jac_inertial_to_grid(mesh_size);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      jac_inertial_to_grid.get(i, j) = 0.05 * dist(generator);
    }
    jac_inertial_to_grid.get(i, i) += positive_dist(generator);
  }

  // Compute the inverse
  InverseJacobian<DataVector, 3, Frame::Grid, Frame::Inertial>
      jac_grid_to_inertial(mesh_size);
  Scalar<DataVector> det(mesh_size);
  determinant_and_inverse(make_not_null(&det),
                          make_not_null(&jac_grid_to_inertial),
                          jac_inertial_to_grid);

  // Roundtrip: Inertial -> Grid -> Inertial
  Variables<filter_detail::ccz4_vars_list<Frame::Grid>> grid_vars(mesh_size);
  filter_detail::transform_spatial_tensors_to_different_frame_without_hessians(
      make_not_null(&grid_vars), inertial_vars, jac_inertial_to_grid,
      jac_grid_to_inertial);
  Variables<filter_detail::ccz4_vars_list<Frame::Inertial>>
      test_inertial_vars(mesh_size);
  filter_detail::transform_spatial_tensors_to_different_frame_without_hessians(
      make_not_null(&test_inertial_vars), grid_vars, jac_grid_to_inertial,
      jac_inertial_to_grid);

  // Roundoff differences from Jacobian multiplication
  for (size_t i = 0; i < inertial_vars.size(); ++i) {
    CHECK(inertial_vars.data()[i] == approx(test_inertial_vars.data()[i]));
  }
}

void test_modal_nodal_invertibility() {
  constexpr size_t radial_extents = 3;
  constexpr size_t ell_max = 4;

  const auto& ylm = ::ylm::get_spherepack_cache(ell_max);
  const size_t spectral_mesh_size = ylm.spectral_size() * radial_extents;
  const size_t physical_mesh_size = ylm.physical_size() * radial_extents;

  // Fill modal variables with random numbers in each mode.
  Variables<filter_detail::ccz4_vars_list<Frame::Grid>> modal_vars(
      spectral_mesh_size);
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<double> dist{-1.0, 1.0};
  ylm::SpherepackIterator it(ell_max, ell_max, radial_extents, true);
  tmpl::for_each<filter_detail::ccz4_vars_list<Frame::Grid>>(
      [&modal_vars, &it, &dist,
       &generator]<class Tag>(const tmpl::type_<Tag> /*meta*/) {
        auto& tensor = get<Tag>(modal_vars);
        for (auto& component : tensor) {
          for (size_t offset = 0; offset < radial_extents; ++offset) {
            for (it.reset(); it; ++it) {
              component[it() + offset] = dist(generator);
            }
          }
        }
      });

  // Roundtrip: modal -> nodal -> modal
  Variables<filter_detail::ccz4_vars_list<Frame::Grid>> nodal_vars(
      physical_mesh_size);
  ylm::TensorYlm::filter_detail::modal_to_nodal_ylm(
      make_not_null(&nodal_vars), modal_vars, ylm, radial_extents);
  Variables<filter_detail::ccz4_vars_list<Frame::Grid>> test_modal_vars(
      spectral_mesh_size, 0.0);
  ylm::TensorYlm::filter_detail::nodal_to_modal_ylm(
      make_not_null(&test_modal_vars), nodal_vars, ylm, radial_extents);

  tmpl::for_each<filter_detail::ccz4_vars_list<Frame::Grid>>(
      [&modal_vars, &test_modal_vars,
       &it]<class Tag>(const tmpl::type_<Tag> /*meta*/) {
        constexpr size_t num_independent_components =
            Tag::type::structure::size();
        const auto& tensor_a = get<Tag>(modal_vars);
        const auto& tensor_b = get<Tag>(test_modal_vars);
        for (size_t storage_index = 0;
             storage_index < num_independent_components; ++storage_index) {
          const auto& a = tensor_a[storage_index];
          const auto& b = tensor_b[storage_index];
          for (size_t offset = 0; offset < radial_extents; ++offset) {
            for (it.reset(); it; ++it) {
              CHECK(a[it() + offset] == approx(b[it() + offset]));
            }
          }
        }
      });
}

// Test the free function apply_tensor_ylm_filter with skip flags using
// band-limited data (initialized in spectral space to avoid projection
// artifacts from the nodal<->modal roundtrip).
void test_skip_external_boundary_faces() {
  constexpr size_t radial_extents = 5;
  constexpr size_t ell_max = 4;
  constexpr size_t num_modes_to_kill = 2;

  using vars_list = filter_detail::ccz4_vars_list<Frame::Inertial>;
  const auto& ylm = ::ylm::get_spherepack_cache(ell_max);
  const size_t spectral_mesh_size = ylm.spectral_size() * radial_extents;
  const size_t physical_mesh_size = ylm.physical_size() * radial_extents;

  // Fill spectral coefficients with random data, only for modes with
  // ell <= ell_max - rank so the data is exactly representable.
  Variables<vars_list> modal_vars(spectral_mesh_size, 0.0);
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<double> dist{-1.0, 1.0};
  ylm::SpherepackIterator it(ell_max, ell_max, radial_extents, true);
  tmpl::for_each<vars_list>(
      [&modal_vars, &it, &dist,
       &generator]<class Tag>(const tmpl::type_<Tag> /*meta*/) {
        auto& tensor = get<Tag>(modal_vars);
        for (auto& component : tensor) {
          for (size_t offset = 0; offset < radial_extents; ++offset) {
            for (it.reset(); it; ++it) {
              if (it.l() <= ell_max - tensor.rank()) {
                component[it() + offset] = dist(generator);
              }
            }
          }
        }
      });

  // Transform to nodal (physical) space — this data is band-limited,
  // so the nodal<->modal roundtrip is exact.
  Variables<vars_list> original_vars(physical_mesh_size);
  ylm::TensorYlm::filter_detail::modal_to_nodal_ylm(
      make_not_null(&original_vars), modal_vars, ylm, radial_extents);

  // Identity Jacobians (Grid == Inertial)
  InverseJacobian<DataVector, 3, Frame::Inertial, Frame::Grid>
      jac_inertial_to_grid(physical_mesh_size, 0.0);
  InverseJacobian<DataVector, 3, Frame::Grid, Frame::Inertial>
      jac_grid_to_inertial(physical_mesh_size, 0.0);
  for (size_t i = 0; i < 3; ++i) {
    jac_inertial_to_grid.get(i, i) = 1.0;
    jac_grid_to_inertial.get(i, i) = 1.0;
  }

  // Build filter matrices
  SimpleSparseMatrix filter_matrix_scalar{};
  SimpleSparseMatrix filter_matrix_i{};
  SimpleSparseMatrix filter_matrix_ii{};
  ylm::TensorYlm::fill_filter<Scalar<DataVector>::structure>(
      make_not_null(&filter_matrix_scalar), ell_max, num_modes_to_kill,
      std::nullopt,
      ylm::TensorYlm::CoefficientNormalization::Spherepack);
  ylm::TensorYlm::fill_filter<tnsr::i<DataVector, 3>::structure>(
      make_not_null(&filter_matrix_i), ell_max, num_modes_to_kill,
      std::nullopt,
      ylm::TensorYlm::CoefficientNormalization::Spherepack);
  ylm::TensorYlm::fill_filter<tnsr::ii<DataVector, 3>::structure>(
      make_not_null(&filter_matrix_ii), ell_max, num_modes_to_kill,
      std::nullopt,
      ylm::TensorYlm::CoefficientNormalization::Spherepack);

  // Apply filter WITH skipping both boundary slices
  auto filtered_skip = original_vars;
  Variables<vars_list> temp_storage(0);
  apply_tensor_ylm_filter(make_not_null(&filtered_skip),
                          make_not_null(&temp_storage), jac_inertial_to_grid,
                          jac_grid_to_inertial, filter_matrix_scalar,
                          filter_matrix_i, filter_matrix_ii, ell_max,
                          radial_extents, true, true);

  // Boundary slices (offset=0 and offset=radial_extents-1) should be
  // unchanged because skip=true and the data is band-limited.
  const size_t angular_size = ylm.physical_size();
  for (size_t comp = 0;
       comp < original_vars.number_of_independent_components; ++comp) {
    const auto* original_component =
        original_vars.data() + comp * physical_mesh_size;
    const auto* skip_component =
        filtered_skip.data() + comp * physical_mesh_size;
    for (size_t a = 0; a < angular_size; ++a) {
      CHECK(original_component[0 + radial_extents * a] ==
            approx(skip_component[0 + radial_extents * a]));
      CHECK(original_component[(radial_extents - 1) + radial_extents * a] ==
            approx(skip_component[(radial_extents - 1) +
                                  radial_extents * a]));
    }
  }

  // Interior slices should be modified by the filter (non-trivial check).
  bool interior_modified = false;
  for (size_t comp = 0;
       comp < original_vars.number_of_independent_components; ++comp) {
    const auto* original_component =
        original_vars.data() + comp * physical_mesh_size;
    const auto* skip_component =
        filtered_skip.data() + comp * physical_mesh_size;
    for (size_t a = 0; a < angular_size; ++a) {
      if (original_component[2 + radial_extents * a] !=
          approx(skip_component[2 + radial_extents * a])) {
        interior_modified = true;
        break;
      }
    }
    if (interior_modified) {
      break;
    }
  }
  CHECK(interior_modified);

  // Apply filter WITHOUT skipping — interior slices should match skip case,
  // boundary slices should differ.
  auto filtered_noskip = original_vars;
  apply_tensor_ylm_filter(make_not_null(&filtered_noskip),
                          make_not_null(&temp_storage), jac_inertial_to_grid,
                          jac_grid_to_inertial, filter_matrix_scalar,
                          filter_matrix_i, filter_matrix_ii, ell_max,
                          radial_extents, false, false);

  for (size_t comp = 0;
       comp < original_vars.number_of_independent_components; ++comp) {
    const auto* noskip_component =
        filtered_noskip.data() + comp * physical_mesh_size;
    const auto* skip_component =
        filtered_skip.data() + comp * physical_mesh_size;
    for (size_t a = 0; a < angular_size; ++a) {
      for (size_t r = 1; r < radial_extents - 1; ++r) {
        CHECK(noskip_component[r + radial_extents * a] ==
              approx(skip_component[r + radial_extents * a]));
      }
    }
  }

  bool boundary_differs = false;
  for (size_t comp = 0;
       comp < original_vars.number_of_independent_components; ++comp) {
    const auto* noskip_component =
        filtered_noskip.data() + comp * physical_mesh_size;
    const auto* skip_component =
        filtered_skip.data() + comp * physical_mesh_size;
    for (size_t a = 0; a < angular_size; ++a) {
      if (noskip_component[0 + radial_extents * a] !=
          approx(skip_component[0 + radial_extents * a])) {
        boundary_differs = true;
        break;
      }
    }
    if (boundary_differs) {
      break;
    }
  }
  CHECK(boundary_differs);
}

// Test that the ghost tensor frame transform is correct by comparing
// against transform::to_different_frame (independently written and tested
// in SpECTRE's core library) for each new tensor type.
void test_transform_ghost_tensors_to_different_frame() {
  constexpr size_t mesh_size = 10;

  Variables<filter_detail::ccz4_ghost_vars_list<Frame::Inertial>>
      inertial_vars(mesh_size);

  // Fill inertial_vars with random numbers
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<double> dist{-1.0, 1.0};
  for (size_t i = 0; i < inertial_vars.size(); ++i) {
    inertial_vars.data()[i] = dist(generator);
  }

  // Create a diagonally-dominant random Jacobian (invertible).
  std::uniform_real_distribution<double> positive_dist{0.5, 1.0};
  InverseJacobian<DataVector, 3, Frame::Inertial, Frame::Grid>
      jac_inertial_to_grid(mesh_size);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      jac_inertial_to_grid.get(i, j) = 0.05 * dist(generator);
    }
    jac_inertial_to_grid.get(i, i) += positive_dist(generator);
  }

  // Compute the inverse
  InverseJacobian<DataVector, 3, Frame::Grid, Frame::Inertial>
      jac_grid_to_inertial(mesh_size);
  Scalar<DataVector> det(mesh_size);
  determinant_and_inverse(make_not_null(&det),
                          make_not_null(&jac_grid_to_inertial),
                          jac_inertial_to_grid);

  // One-way transform: Inertial -> Grid
  Variables<filter_detail::ccz4_ghost_vars_list<Frame::Grid>> grid_vars(
      mesh_size);
  filter_detail::transform_ghost_tensors_to_different_frame(
      make_not_null(&grid_vars), inertial_vars, jac_inertial_to_grid,
      jac_grid_to_inertial);

  // Compare each new tensor type against transform::to_different_frame.
  // Jacobian<DV,3,Grid,Inertial> is the same type as
  // InverseJacobian<DV,3,Inertial,Grid>, so we can pass jac_inertial_to_grid
  // directly. Similarly InverseJacobian<DV,3,Grid,Inertial> =
  // jac_grid_to_inertial.

  // tnsr::i (FieldA)
  {
    tnsr::i<DataVector, 3, Frame::Grid> expected(mesh_size);
    transform::to_different_frame(
        make_not_null(&expected),
        get<Tags::FieldA<DataVector, 3, Frame::Inertial>>(inertial_vars),
        jac_inertial_to_grid, jac_grid_to_inertial);
    const auto& actual =
        get<Tags::FieldA<DataVector, 3, Frame::Grid>>(grid_vars);
    for (size_t i = 0; i < 3; ++i) {
      CHECK_ITERABLE_APPROX(actual.get(i), expected.get(i));
    }
  }
  // tnsr::i (FieldP)
  {
    tnsr::i<DataVector, 3, Frame::Grid> expected(mesh_size);
    transform::to_different_frame(
        make_not_null(&expected),
        get<Tags::FieldP<DataVector, 3, Frame::Inertial>>(inertial_vars),
        jac_inertial_to_grid, jac_grid_to_inertial);
    const auto& actual =
        get<Tags::FieldP<DataVector, 3, Frame::Grid>>(grid_vars);
    for (size_t i = 0; i < 3; ++i) {
      CHECK_ITERABLE_APPROX(actual.get(i), expected.get(i));
    }
  }
  // tnsr::iJ (FieldB)
  {
    tnsr::iJ<DataVector, 3, Frame::Grid> expected(mesh_size);
    transform::to_different_frame(
        make_not_null(&expected),
        get<Tags::FieldB<DataVector, 3, Frame::Inertial>>(inertial_vars),
        jac_inertial_to_grid, jac_grid_to_inertial);
    const auto& actual =
        get<Tags::FieldB<DataVector, 3, Frame::Grid>>(grid_vars);
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        CHECK_ITERABLE_APPROX(actual.get(i, j), expected.get(i, j));
      }
    }
  }
  // tnsr::ijj (FieldD)
  {
    tnsr::ijj<DataVector, 3, Frame::Grid> expected(mesh_size);
    transform::to_different_frame(
        make_not_null(&expected),
        get<Tags::FieldD<DataVector, 3, Frame::Inertial>>(inertial_vars),
        jac_inertial_to_grid, jac_grid_to_inertial);
    const auto& actual =
        get<Tags::FieldD<DataVector, 3, Frame::Grid>>(grid_vars);
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        for (size_t k = j; k < 3; ++k) {
          CHECK_ITERABLE_APPROX(actual.get(i, j, k), expected.get(i, j, k));
        }
      }
    }
  }

  // Also test roundtrip: Inertial -> Grid -> Inertial
  Variables<filter_detail::ccz4_ghost_vars_list<Frame::Inertial>>
      test_inertial_vars(mesh_size);
  filter_detail::transform_ghost_tensors_to_different_frame(
      make_not_null(&test_inertial_vars), grid_vars, jac_grid_to_inertial,
      jac_inertial_to_grid);
  for (size_t i = 0; i < inertial_vars.size(); ++i) {
    CHECK(inertial_vars.data()[i] == approx(test_inertial_vars.data()[i]));
  }
}

// Test the TensorYlmFilter::operator() with SkipExternalBoundaryFaces option
// using an Element with external boundaries.
void test_skip_boundary_faces_via_operator() {
  constexpr size_t radial_extents = 5;
  constexpr size_t ell_max = 4;
  constexpr size_t num_modes_to_kill = 2;

  using vars_list = filter_detail::ccz4_vars_list<Frame::Inertial>;
  const auto& ylm = ::ylm::get_spherepack_cache(ell_max);
  const size_t spectral_mesh_size = ylm.spectral_size() * radial_extents;
  const size_t physical_mesh_size = ylm.physical_size() * radial_extents;

  // Band-limited data (same pattern as above)
  Variables<vars_list> modal_vars(spectral_mesh_size, 0.0);
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<double> dist{-1.0, 1.0};
  ylm::SpherepackIterator it(ell_max, ell_max, radial_extents, true);
  tmpl::for_each<vars_list>(
      [&modal_vars, &it, &dist,
       &generator]<class Tag>(const tmpl::type_<Tag> /*meta*/) {
        auto& tensor = get<Tag>(modal_vars);
        for (auto& component : tensor) {
          for (size_t offset = 0; offset < radial_extents; ++offset) {
            for (it.reset(); it; ++it) {
              if (it.l() <= ell_max - tensor.rank()) {
                component[it() + offset] = dist(generator);
              }
            }
          }
        }
      });

  Variables<vars_list> original_vars(physical_mesh_size);
  ylm::TensorYlm::filter_detail::modal_to_nodal_ylm(
      make_not_null(&original_vars), modal_vars, ylm, radial_extents);

  // Mesh: radial (Legendre-GL) x angular (SphericalHarmonic)
  const Mesh<3> mesh(
      {{radial_extents, ell_max + 1, 2 * ell_max + 1}},
      {{Spectral::Basis::Legendre, Spectral::Basis::SphericalHarmonic,
        Spectral::Basis::SphericalHarmonic}},
      {{Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::Gauss,
        Spectral::Quadrature::Equiangular}});

  // Element with external boundary at upper_xi (outer face).
  // Uses spherical_shell topology so angular directions are topological
  // (not external boundaries).
  const ElementId<3> element_id(0);
  const auto aligned = OrientationMap<3>::create_aligned();
  Element<3> element_with_outer_boundary(
      element_id,
      Element<3>::Neighbors_t{
          {Direction<3>::lower_xi(), Neighbors<3>(element_id, aligned)}},
      domain::topologies::spherical_shell);

  // Element with no external boundaries (interior element)
  Element<3> interior_element(
      element_id,
      Element<3>::Neighbors_t{
          {Direction<3>::lower_xi(), Neighbors<3>(element_id, aligned)},
          {Direction<3>::upper_xi(), Neighbors<3>(element_id, aligned)}},
      domain::topologies::spherical_shell);

  // No time-dependent map (identity Jacobian)
  const std::optional<std::tuple<
      tnsr::I<DataVector, 3, Frame::Inertial>,
      InverseJacobian<DataVector, 3, Frame::Grid, Frame::Inertial>,
      Jacobian<DataVector, 3, Frame::Grid, Frame::Inertial>,
      tnsr::I<DataVector, 3, Frame::Inertial>>>
      grid_to_inertial_quantities = std::nullopt;

  // Filter with SkipExternalBoundaryFaces: true
  TensorYlmFilter filter_skip(num_modes_to_kill, std::nullopt, std::nullopt,
                               true);

  // Filter with SkipExternalBoundaryFaces: false
  TensorYlmFilter filter_noskip(num_modes_to_kill, std::nullopt, std::nullopt,
                                 false);

  // Apply filter_skip to element with outer boundary
  auto vars_skip = original_vars;
  filter_skip(make_not_null(&vars_skip), mesh, element_with_outer_boundary,
              grid_to_inertial_quantities);

  // Apply filter_noskip to same element
  auto vars_noskip = original_vars;
  filter_noskip(make_not_null(&vars_noskip), mesh, element_with_outer_boundary,
                grid_to_inertial_quantities);

  // Apply filter_skip to interior element (no external boundaries, so skip
  // has no effect)
  auto vars_skip_interior = original_vars;
  filter_skip(make_not_null(&vars_skip_interior), mesh, interior_element,
              grid_to_inertial_quantities);

  const size_t angular_size = ylm.physical_size();

  // For the element with outer boundary:
  // Last radial slice (upper_xi) should be preserved with skip=true
  for (size_t comp = 0;
       comp < original_vars.number_of_independent_components; ++comp) {
    const auto* original_component =
        original_vars.data() + comp * physical_mesh_size;
    const auto* skip_component =
        vars_skip.data() + comp * physical_mesh_size;
    for (size_t a = 0; a < angular_size; ++a) {
      CHECK(original_component[(radial_extents - 1) + radial_extents * a] ==
            approx(skip_component[(radial_extents - 1) +
                                  radial_extents * a]));
    }
  }

  // First radial slice (lower_xi = internal boundary, has neighbor) should
  // still be filtered even with skip=true
  bool inner_face_modified = false;
  for (size_t comp = 0;
       comp < original_vars.number_of_independent_components; ++comp) {
    const auto* original_component =
        original_vars.data() + comp * physical_mesh_size;
    const auto* skip_component =
        vars_skip.data() + comp * physical_mesh_size;
    for (size_t a = 0; a < angular_size; ++a) {
      if (original_component[0 + radial_extents * a] !=
          approx(skip_component[0 + radial_extents * a])) {
        inner_face_modified = true;
        break;
      }
    }
    if (inner_face_modified) {
      break;
    }
  }
  CHECK(inner_face_modified);

  // For the interior element, skip=true should have no effect (no external
  // boundaries), so the result should match noskip.
  for (size_t i = 0; i < vars_skip_interior.size(); ++i) {
    CHECK(vars_skip_interior.data()[i] == approx(vars_noskip.data()[i]));
  }
}

// [[TimeOut, 20]]
SPECTRE_TEST_CASE("Unit.Evolution.Systems.Ccz4.ApplyTensorYlmFilter",
                  "[NumericalAlgorithms][Unit]") {
  register_factory_classes_with_charm<Metavariables>();

  // Test creation with BlocksToFilter: All, SkipExternalBoundaryFaces: false
  const auto created_filter = TestHelpers::test_creation<
      std::unique_ptr<Filters::Filter>, Metavariables>(
      "TensorYlmFilter:\n"
      "  NumModesToKill: 2\n"
      "  HalfPower: 5\n"
      "  BlocksToFilter: All\n"
      "  SkipExternalBoundaryFaces: false");
  const auto& concrete_filter =
      dynamic_cast<const TensorYlmFilter&>(*created_filter);
  CHECK(concrete_filter == TensorYlmFilter{2, 5});
  CHECK(concrete_filter.blocks_to_filter() == std::nullopt);

  const auto deserialized_filter = serialize_and_deserialize(created_filter);
  CHECK(dynamic_cast<const TensorYlmFilter&>(*deserialized_filter) ==
        concrete_filter);

  // Test creation with specific blocks and SkipExternalBoundaryFaces: true
  const auto block_filter = TestHelpers::test_creation<
      std::unique_ptr<Filters::Filter>, Metavariables>(
      "TensorYlmFilter:\n"
      "  NumModesToKill: 3\n"
      "  HalfPower: None\n"
      "  BlocksToFilter:\n"
      "    - InnerShell\n"
      "    - OuterShell\n"
      "  SkipExternalBoundaryFaces: true");
  const auto& concrete_block_filter =
      dynamic_cast<const TensorYlmFilter&>(*block_filter);
  CHECK(concrete_block_filter ==
        TensorYlmFilter{3, std::nullopt,
                        std::vector<std::string>{"InnerShell", "OuterShell"},
                        true});
  CHECK(concrete_block_filter.blocks_to_filter().has_value());
  CHECK(concrete_block_filter.blocks_to_filter()->size() == 2);
  CHECK(concrete_block_filter.blocks_to_filter()->count("InnerShell") == 1);
  CHECK(concrete_block_filter.blocks_to_filter()->count("OuterShell") == 1);

  // Test serialization with blocks and skip boundary faces
  const auto deserialized_block_filter =
      serialize_and_deserialize(block_filter);
  CHECK(dynamic_cast<const TensorYlmFilter&>(*deserialized_block_filter) ==
        concrete_block_filter);

  // Test that SkipExternalBoundaryFaces affects equality
  CHECK_FALSE(concrete_filter == concrete_block_filter);

  test_transform_spatial_tensors_to_different_frame();
  test_transform_ghost_tensors_to_different_frame();
  test_modal_nodal_invertibility();
  test_skip_external_boundary_faces();
  test_skip_boundary_faces_via_operator();

  const auto apply_filter =
      [](const auto vars_nodal, const auto vars_storage,
         const auto& jac_inertial_to_grid, const auto& jac_grid_to_inertial,
         const auto& filter_matrices, const size_t ell_max,
         const size_t radial_extents) {
        apply_tensor_ylm_filter(vars_nodal, vars_storage, jac_inertial_to_grid,
                                jac_grid_to_inertial, filter_matrices.scalar,
                                filter_matrices.i, filter_matrices.ii, ell_max,
                                radial_extents);
      };
  ylm::TensorYlm::test_apply_filter<
      filter_detail::ccz4_vars_list<Frame::Inertial>, true>(0, apply_filter);
  ylm::TensorYlm::test_apply_filter<
      filter_detail::ccz4_vars_list<Frame::Inertial>, true>(5, apply_filter);

  const auto apply_ghost_filter =
      [](const auto vars_nodal, const auto vars_storage,
         const auto& jac_inertial_to_grid, const auto& jac_grid_to_inertial,
         const auto& filter_matrices, const size_t ell_max,
         const size_t radial_extents) {
        apply_tensor_ylm_filter_ghost(
            vars_nodal, vars_storage, jac_inertial_to_grid,
            jac_grid_to_inertial, filter_matrices.scalar, filter_matrices.i,
            filter_matrices.ii, filter_matrices.ij, filter_matrices.kii,
            ell_max, radial_extents);
      };
  ylm::TensorYlm::test_apply_filter<
      filter_detail::ccz4_ghost_vars_list<Frame::Inertial>, true>(
      0, apply_ghost_filter);
  ylm::TensorYlm::test_apply_filter<
      filter_detail::ccz4_ghost_vars_list<Frame::Inertial>, true>(
      5, apply_ghost_filter);
}
}  // namespace
}  // namespace Ccz4
