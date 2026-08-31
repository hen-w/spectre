// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <vector>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/BoundaryVariables.hpp"
#include "Domain/BoundaryVariablesTag.hpp"
#include "Domain/Creators/Tags/ExternalBoundaryConditions.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DiscontinuousGalerkin/BoundaryEvolvedVariables.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/BoundaryEvolvedVariables.hpp"
#include "Framework/ActionTesting.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/ProjectToBoundary.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Time/LtsMode.hpp"
#include "Time/Tags/LtsMode.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace {
// Interior source fields. `Psi` is the source whose boundary twin
// `BoundaryValue<Psi>` we store and time-integrate; `Pi` is a second volume
// field with no boundary twin, present to keep the volume Variables
// heterogeneous (mirrors SecondOrderScalarWave::System's volume_vars).
struct Psi : db::SimpleTag {
  using type = Scalar<DataVector>;
};
struct Pi : db::SimpleTag {
  using type = Scalar<DataVector>;
};

using evolution::dg::Tags::BoundaryValue;

// Mock system in the shape SecondOrderScalarWave::System uses: a list-valued
// `variables_tag` whose front is the volume Variables and whose back is the
// `::Tags::BoundaryVariables` entry holding the boundary-evolved twins.
template <size_t Dim>
struct MockSystem {
  using volume_vars = tmpl::list<Psi, Pi>;
  using boundary_vars = tmpl::list<BoundaryValue<Psi>>;
  using variables_tag =
      tmpl::list<::Tags::Variables<volume_vars>,
                 ::Tags::BoundaryVariables<Dim, boundary_vars>>;
};

// A boundary condition that opts in: its only distinguishing feature is that
// it declares the member `boundary_field_time_derivatives`, which is the
// opt-in the initializer detects. The signature is irrelevant -- only the
// member's presence matters -- so a trivial one suffices. Derives from the
// domain base class so the initializer's typeid resolution against
// `ExternalBoundaryConditions` is against a real polymorphic hierarchy.
template <size_t Dim>
class MockOptingBc : public domain::BoundaryConditions::BoundaryCondition {
 public:
  MockOptingBc() = default;
  MockOptingBc(MockOptingBc&&) = default;
  MockOptingBc& operator=(MockOptingBc&&) = default;
  MockOptingBc(const MockOptingBc&) = default;
  MockOptingBc& operator=(const MockOptingBc&) = default;
  ~MockOptingBc() override = default;

  explicit MockOptingBc(CkMigrateMessage* msg)
      : domain::BoundaryConditions::BoundaryCondition(msg) {}

  WRAPPED_PUPable_decl_base_template(
      domain::BoundaryConditions::BoundaryCondition, MockOptingBc);

  auto get_clone() const -> std::unique_ptr<
      domain::BoundaryConditions::BoundaryCondition> override {
    return std::make_unique<MockOptingBc<Dim>>(*this);
  }

  // NOLINTNEXTLINE
  void pup(PUP::er& p) override {
    domain::BoundaryConditions::BoundaryCondition::pup(p);
  }

  // The opt-in marker; the method itself is never called by the
  // initializer.
  static constexpr bool evolves_boundary_variables = true;
  void boundary_field_time_derivatives() const {}
};

template <size_t Dim>
// NOLINTNEXTLINE
PUP::able::PUP_ID MockOptingBc<Dim>::my_PUP_ID = 0;

// A boundary condition that does NOT opt in: no
// `boundary_field_time_derivatives`.
template <size_t Dim>
class MockNonOptingBc : public domain::BoundaryConditions::BoundaryCondition {
 public:
  MockNonOptingBc() = default;
  MockNonOptingBc(MockNonOptingBc&&) = default;
  MockNonOptingBc& operator=(MockNonOptingBc&&) = default;
  MockNonOptingBc(const MockNonOptingBc&) = default;
  MockNonOptingBc& operator=(const MockNonOptingBc&) = default;
  ~MockNonOptingBc() override = default;

  explicit MockNonOptingBc(CkMigrateMessage* msg)
      : domain::BoundaryConditions::BoundaryCondition(msg) {}

  WRAPPED_PUPable_decl_base_template(
      domain::BoundaryConditions::BoundaryCondition, MockNonOptingBc);

  auto get_clone() const -> std::unique_ptr<
      domain::BoundaryConditions::BoundaryCondition> override {
    return std::make_unique<MockNonOptingBc<Dim>>(*this);
  }

  // NOLINTNEXTLINE
  void pup(PUP::er& p) override {
    domain::BoundaryConditions::BoundaryCondition::pup(p);
  }
};

template <size_t Dim>
// NOLINTNEXTLINE
PUP::able::PUP_ID MockNonOptingBc<Dim>::my_PUP_ID = 0;

// The `::Tags::BoundaryVariables` container tag and its `::Tags::dt` twin, as
// the initializer names them (front/back of the list-valued variables_tag).
template <size_t Dim>
using boundary_variables_tag =
    evolution::dg::boundary_variables_tag<MockSystem<Dim>>;
template <size_t Dim>
using dt_boundary_variables_tag =
    db::add_tag_prefix<::Tags::dt, boundary_variables_tag<Dim>>;
template <size_t Dim>
using volume_variables_tag =
    tmpl::front<typename MockSystem<Dim>::variables_tag>;

template <typename Metavariables>
struct MockComponent {
  static constexpr size_t Dim = Metavariables::volume_dim;
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  // The DataBox seen by the action: element, mesh, the volume Variables entry
  // (filled with known data so projection is verifiable), the default
  // `BoundaryVariables` container the action initializes, and its default
  // `::Tags::dt` twin the action resizes. `ExternalBoundaryConditions` is a
  // const global cache tag, supplied to the runner directly.
  using simple_tags =
      tmpl::list<::Tags::LtsMode, domain::Tags::Element<Dim>,
                 domain::Tags::Mesh<Dim>, volume_variables_tag<Dim>,
                 boundary_variables_tag<Dim>, dt_boundary_variables_tag<Dim>>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             tmpl::list<ActionTesting::InitializeDataBox<
                                 simple_tags, tmpl::list<>>>>,
      Parallel::PhaseActions<
          Parallel::Phase::Testing,
          tmpl::list<evolution::dg::Initialization::BoundaryEvolvedVariables<
              Dim, MockSystem<Dim>,
              tmpl::list<MockOptingBc<Dim>, MockNonOptingBc<Dim>>>>>>;
};

struct MockMetavars {
  static constexpr size_t volume_dim = 2;
  using component_list = tmpl::list<MockComponent<MockMetavars>>;
};

// Builds an element with two opting external faces (lower_xi, lower_eta) on
// different sliced dimensions, one non-opting external face (upper_xi), and
// one internal face (upper_eta, which has a neighbor). Returns the element and
// the per-block ExternalBoundaryConditions, keyed by block id. If
// `any_opts_in` is false, both external "opting" faces instead carry the
// non-opting condition, so nothing opts in.
template <size_t Dim>
auto make_element_and_bcs(const bool any_opts_in) {
  const ElementId<Dim> self_id{0, {{{0, 0}, {1, 0}}}};
  const ElementId<Dim> neighbor_eta_id{0, {{{0, 0}, {1, 1}}}};
  const OrientationMap<Dim> orientation = OrientationMap<Dim>::create_aligned();
  typename Element<Dim>::Neighbors_t neighbors{};
  neighbors[Direction<Dim>::upper_eta()] =
      Neighbors<Dim>{{neighbor_eta_id}, orientation};
  Element<Dim> element{self_id, neighbors};

  std::vector<DirectionMap<
      Dim, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
      external_boundary_conditions{1};
  const std::array<Direction<Dim>, 2> opting_directions{
      Direction<Dim>::lower_xi(), Direction<Dim>::lower_eta()};
  for (const auto& direction : opting_directions) {
    if (any_opts_in) {
      external_boundary_conditions[0][direction] =
          std::make_unique<MockOptingBc<Dim>>();
    } else {
      external_boundary_conditions[0][direction] =
          std::make_unique<MockNonOptingBc<Dim>>();
    }
  }
  external_boundary_conditions[0][Direction<Dim>::upper_xi()] =
      std::make_unique<MockNonOptingBc<Dim>>();

  return std::make_pair(std::move(element),
                        std::move(external_boundary_conditions));
}

void test_two_opting_faces(const Spectral::Quadrature quadrature) {
  INFO("BoundaryEvolvedVariables: two opting external faces");
  CAPTURE(quadrature);
  constexpr size_t Dim = 2;
  using component = MockComponent<MockMetavars>;

  // Anisotropic mesh so the two opting faces slice different dimensions and
  // therefore have different face sizes: lower_xi slices xi -> 4 face points
  // (the eta extent); lower_eta slices eta -> 3 face points (the xi extent).
  const Mesh<Dim> mesh{{{3, 4}}, Spectral::Basis::Legendre, quadrature};

  auto [element, external_boundary_conditions] =
      make_element_and_bcs<Dim>(true);

  // Known, per-node-distinct volume data so the projection check is meaningful.
  Variables<tmpl::list<Psi, Pi>> volume_vars{mesh.number_of_grid_points()};
  for (size_t i = 0; i < mesh.number_of_grid_points(); ++i) {
    get(get<Psi>(volume_vars))[i] = 1.0 + 0.75 * static_cast<double>(i);
    get(get<Pi>(volume_vars))[i] =
        -2.0 + 0.5 * static_cast<double>(i) * static_cast<double>(i);
  }

  ActionTesting::MockRuntimeSystem<MockMetavars> runner{
      {std::move(external_boundary_conditions)}};
  ActionTesting::emplace_component_and_initialize<component>(
      make_not_null(&runner), 0,
      {LtsMode::Off, element, mesh, volume_vars,
       typename boundary_variables_tag<Dim>::type{},
       typename dt_boundary_variables_tag<Dim>::type{}});
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);
  ActionTesting::next_action<component>(make_not_null(&runner), 0);

  const auto& boundary_vars =
      ActionTesting::get_databox_tag<component, boundary_variables_tag<Dim>>(
          runner, 0);
  const auto& dt_boundary_vars =
      ActionTesting::get_databox_tag<component, dt_boundary_variables_tag<Dim>>(
          runner, 0);

  const std::array<Direction<Dim>, 2> opting_directions{
      Direction<Dim>::lower_xi(), Direction<Dim>::lower_eta()};

  // Exactly the two opting external faces have entries; the non-opting
  // external face and the interior face do not.
  const auto& per_direction = boundary_vars.points_per_direction();
  CHECK(per_direction.size() == 2);
  CHECK(per_direction.count(Direction<Dim>::lower_xi()) == 1);
  CHECK(per_direction.count(Direction<Dim>::lower_eta()) == 1);
  CHECK(per_direction.count(Direction<Dim>::upper_xi()) == 0);
  CHECK(per_direction.count(Direction<Dim>::upper_eta()) == 0);

  // The dt twin was resized to match the values' per-face point counts.
  CHECK(dt_boundary_vars.points_per_direction() ==
        boundary_vars.points_per_direction());

  for (const auto& direction : opting_directions) {
    CAPTURE(direction);
    const size_t num_face_pts =
        mesh.slice_away(direction.dimension()).number_of_grid_points();
    // Per-face sizes: lower_xi -> 4, lower_eta -> 3.
    CHECK(per_direction.at(direction) == num_face_pts);

    const auto& face_values = boundary_vars.variables().at(direction);
    CHECK(face_values.number_of_grid_points() == num_face_pts);

    // The face value is the volume Psi projected onto this face.
    const auto expected =
        dg::project_tensor_to_boundary(get<Psi>(volume_vars), mesh, direction);
    CHECK_ITERABLE_APPROX(get(get<BoundaryValue<Psi>>(face_values)),
                          get(expected));
  }
}

void test_no_opting_faces(const Spectral::Quadrature quadrature) {
  INFO("BoundaryEvolvedVariables: no face opts in");
  CAPTURE(quadrature);
  constexpr size_t Dim = 2;
  using component = MockComponent<MockMetavars>;

  const Mesh<Dim> mesh{{{3, 4}}, Spectral::Basis::Legendre, quadrature};

  auto [element, external_boundary_conditions] =
      make_element_and_bcs<Dim>(false);

  Variables<tmpl::list<Psi, Pi>> volume_vars{mesh.number_of_grid_points()};
  for (size_t i = 0; i < mesh.number_of_grid_points(); ++i) {
    get(get<Psi>(volume_vars))[i] = 1.0 + 0.75 * static_cast<double>(i);
    get(get<Pi>(volume_vars))[i] = -2.0 + static_cast<double>(i);
  }

  ActionTesting::MockRuntimeSystem<MockMetavars> runner{
      {std::move(external_boundary_conditions)}};
  ActionTesting::emplace_component_and_initialize<component>(
      make_not_null(&runner), 0,
      {LtsMode::Off, element, mesh, volume_vars,
       typename boundary_variables_tag<Dim>::type{},
       typename dt_boundary_variables_tag<Dim>::type{}});
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);
  ActionTesting::next_action<component>(make_not_null(&runner), 0);

  const auto& boundary_vars =
      ActionTesting::get_databox_tag<component, boundary_variables_tag<Dim>>(
          runner, 0);
  const auto& dt_boundary_vars =
      ActionTesting::get_databox_tag<component, dt_boundary_variables_tag<Dim>>(
          runner, 0);

  // No opting faces -> empty container and empty dt twin.
  CHECK(boundary_vars.variables().empty());
  CHECK(boundary_vars.points_per_direction().empty());
  CHECK(dt_boundary_vars.points_per_direction().empty());
}

// The two loud guards: boundary-evolved variables are unverified with local
// time stepping, and the volume variables must be allocated (and set)
// before this action projects them onto the faces.
void test_error_branches() {
  INFO("BoundaryEvolvedVariables: error branches");
  constexpr size_t Dim = 2;
  using component = MockComponent<MockMetavars>;
  const Mesh<Dim> mesh{
      {{3, 4}}, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto};
  {
    auto [element, external_boundary_conditions] =
        make_element_and_bcs<Dim>(true);
    Variables<tmpl::list<Psi, Pi>> volume_vars{mesh.number_of_grid_points(),
                                               0.0};
    ActionTesting::MockRuntimeSystem<MockMetavars> runner{
        {std::move(external_boundary_conditions)}};
    ActionTesting::emplace_component_and_initialize<component>(
        make_not_null(&runner), 0,
        {LtsMode::Conservative, element, mesh, volume_vars,
         typename boundary_variables_tag<Dim>::type{},
         typename dt_boundary_variables_tag<Dim>::type{}});
    ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);
    CHECK_THROWS_WITH(
        ActionTesting::next_action<component>(make_not_null(&runner), 0),
        Catch::Matchers::ContainsSubstring(
            "unverified with local time stepping"));
  }
  {
    auto [element, external_boundary_conditions] =
        make_element_and_bcs<Dim>(true);
    // Unallocated volume variables: the action must refuse to project.
    Variables<tmpl::list<Psi, Pi>> volume_vars{};
    ActionTesting::MockRuntimeSystem<MockMetavars> runner{
        {std::move(external_boundary_conditions)}};
    ActionTesting::emplace_component_and_initialize<component>(
        make_not_null(&runner), 0,
        {LtsMode::Off, element, mesh, volume_vars,
         typename boundary_variables_tag<Dim>::type{},
         typename dt_boundary_variables_tag<Dim>::type{}});
    ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);
    CHECK_THROWS_WITH(
        ActionTesting::next_action<component>(make_not_null(&runner), 0),
        Catch::Matchers::ContainsSubstring("not allocated to the mesh size"));
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Dg.Initialization.BoundaryEvolvedVariables",
                  "[Unit][Evolution]") {
  register_classes_with_charm<MockNonOptingBc<2>, MockOptingBc<2>>();

  test_two_opting_faces(Spectral::Quadrature::Gauss);
  test_two_opting_faces(Spectral::Quadrature::GaussLobatto);
  test_no_opting_faces(Spectral::Quadrature::Gauss);
  test_no_opting_faces(Spectral::Quadrature::GaussLobatto);
  test_error_branches();
}
