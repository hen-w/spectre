// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <pup.h>
#include <string>
#include <unordered_map>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Block.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Creators/Tags/ExternalBoundaryConditions.hpp"
#include "Domain/Creators/Tags/FunctionsOfTime.hpp"
#include "Domain/Creators/Tags/InitialExtents.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Structure/BlockNeighbors.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/BoundaryConditions/Type.hpp"
#include "Evolution/BoundaryCorrection.hpp"
#include "Evolution/BoundaryCorrectionTags.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/SendAuxiliaryBoundaryData.hpp"
#include "Evolution/DiscontinuousGalerkin/InboxTags.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/Mortars.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/QuadratureTag.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarData.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarDataHolder.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarTags.hpp"
#include "Framework/ActionTesting.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/ProjectToBoundary.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Time/LtsMode.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags/HistoryEvolvedVariables.hpp"
#include "Time/Tags/LtsMode.hpp"
#include "Time/Tags/Time.hpp"
#include "Time/Tags/TimeStep.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/Tags/TimeStepper.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/AdamsBashforth.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/CloneUniquePtrs.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace {
// Toy evolved variable for a flux-free LDG-like nonconservative system.
struct Psi : db::SimpleTag {
  using type = Scalar<DataVector>;
};

// Auxiliary package field: twice the evolved variable projected to the face.
// The toy correction scales by 2 so the packaged result is distinguishable
// from a plain copy of the projected field.
struct TwoTimesPsi : db::SimpleTag {
  using type = Scalar<DataVector>;
};

// A flux-free (nonconservative) boundary correction providing the LDG
// auxiliary package-data interface used by the auxiliary send. It also
// declares a physical package-data interface, which the auxiliary send never
// invokes but which must exist so the action's tag dispatch compiles.
template <size_t Dim>
struct AuxiliaryCorrection final : public ::evolution::BoundaryCorrection {
  explicit AuxiliaryCorrection(CkMigrateMessage* /*unused*/) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(AuxiliaryCorrection);  // NOLINT
  AuxiliaryCorrection() = default;
  AuxiliaryCorrection(const AuxiliaryCorrection&) = default;
  AuxiliaryCorrection& operator=(const AuxiliaryCorrection&) = default;
  AuxiliaryCorrection(AuxiliaryCorrection&&) = default;
  AuxiliaryCorrection& operator=(AuxiliaryCorrection&&) = default;
  ~AuxiliaryCorrection() override = default;

  std::unique_ptr<BoundaryCorrection> get_clone() const override {
    return std::make_unique<AuxiliaryCorrection>(*this);
  }

  void pup(PUP::er& p) override {  // NOLINT
    BoundaryCorrection::pup(p);
  }

  static constexpr bool need_normal_vector = false;

  // Physical interface. Not used by the auxiliary send, but the action's
  // typeid dispatch loop instantiates code referring to it, so it must exist.
  using dg_package_field_tags = tmpl::list<TwoTimesPsi>;
  using dg_package_data_temporary_tags = tmpl::list<>;
  using dg_package_data_primitive_tags = tmpl::list<>;
  using dg_package_data_volume_tags = tmpl::list<>;
  using dg_boundary_terms_volume_tags = tmpl::list<>;

  double dg_package_data(
      const gsl::not_null<Scalar<DataVector>*> two_times_psi,
      const Scalar<DataVector>& psi,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& /*normal_covector*/,
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
      /*mesh_velocity*/,
      const std::optional<Scalar<DataVector>>& /*normal_dot_mesh_velocity*/)
      const {
    get(*two_times_psi) = 2.0 * get(psi);
    return 1.0;
  }

  // Auxiliary interface used by the auxiliary send. The packaged field is the
  // evolved variable projected to the face, scaled to make the result distinct
  // from a copy.
  using dg_auxiliary_package_field_tags = tmpl::list<TwoTimesPsi>;
  using dg_auxiliary_package_data_temporary_tags = tmpl::list<>;
  using dg_auxiliary_package_data_volume_tags = tmpl::list<>;
  using dg_auxiliary_boundary_terms_volume_tags = tmpl::list<>;

  double dg_auxiliary_package_data(
      const gsl::not_null<Scalar<DataVector>*> two_times_psi,
      const Scalar<DataVector>& psi,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& /*normal_covector*/,
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
      /*mesh_velocity*/,
      const std::optional<Scalar<DataVector>>& /*normal_dot_mesh_velocity*/)
      const {
    get(*two_times_psi) = 2.0 * get(psi);
    return 1.0;
  }
};

template <size_t Dim>
PUP::able::PUP_ID AuxiliaryCorrection<Dim>::my_PUP_ID = 0;

// A trivial external boundary condition. The self element's lower-xi face is an
// external boundary, so this is invoked there during the auxiliary send, but
// the DemandOutgoingCharSpeeds type returns early without lifting any boundary
// terms. It mainly has to satisfy the factory machinery.
template <size_t Dim>
class BoundaryCondition : public domain::BoundaryConditions::BoundaryCondition {
 public:
  BoundaryCondition() = default;
  BoundaryCondition(BoundaryCondition&&) = default;
  BoundaryCondition& operator=(BoundaryCondition&&) = default;
  BoundaryCondition(const BoundaryCondition&) = default;
  BoundaryCondition& operator=(const BoundaryCondition&) = default;
  ~BoundaryCondition() override = default;
  explicit BoundaryCondition(CkMigrateMessage* msg)
      : domain::BoundaryConditions::BoundaryCondition(msg) {}

  void pup(PUP::er& p) override {
    domain::BoundaryConditions::BoundaryCondition::pup(p);
  }
};

template <size_t Dim>
class DemandOutgoingCharSpeeds : public BoundaryCondition<Dim> {
 public:
  DemandOutgoingCharSpeeds() = default;
  DemandOutgoingCharSpeeds(DemandOutgoingCharSpeeds&&) = default;
  DemandOutgoingCharSpeeds& operator=(DemandOutgoingCharSpeeds&&) = default;
  DemandOutgoingCharSpeeds(const DemandOutgoingCharSpeeds&) = default;
  DemandOutgoingCharSpeeds& operator=(const DemandOutgoingCharSpeeds&) =
      default;
  ~DemandOutgoingCharSpeeds() override = default;

  explicit DemandOutgoingCharSpeeds(CkMigrateMessage* msg)
      : BoundaryCondition<Dim>(msg) {}

  WRAPPED_PUPable_decl_base_template(
      domain::BoundaryConditions::BoundaryCondition, DemandOutgoingCharSpeeds);

  auto get_clone() const -> std::unique_ptr<
      domain::BoundaryConditions::BoundaryCondition> override {
    return std::make_unique<DemandOutgoingCharSpeeds<Dim>>(*this);
  }

  static constexpr ::evolution::BoundaryConditions::Type bc_type =
      ::evolution::BoundaryConditions::Type::DemandOutgoingCharSpeeds;

  void pup(PUP::er& p) override { BoundaryCondition<Dim>::pup(p); }

  using dg_interior_evolved_variables_tags = tmpl::list<>;
  using dg_interior_primitive_variables_tags = tmpl::list<>;
  using dg_interior_temporary_tags = tmpl::list<>;
  using dg_gridless_tags = tmpl::list<>;

  static std::optional<std::string> dg_demand_outgoing_char_speeds(
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
      /*face_mesh_velocity*/,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
      /*outward_directed_normal_covector*/) {
    return std::nullopt;
  }
};

template <size_t Dim>
PUP::able::PUP_ID DemandOutgoingCharSpeeds<Dim>::my_PUP_ID = 0;

template <size_t Dim>
struct TimeDerivativeTerms {
  using temporary_tags = tmpl::list<>;
  using argument_tags = tmpl::list<>;
};

template <size_t Dim>
struct System {
  static constexpr bool has_primitive_and_conservative_vars = false;
  static constexpr size_t volume_dim = Dim;

  using boundary_conditions_base = BoundaryCondition<Dim>;

  using variables_tag = Tags::Variables<tmpl::list<Psi>>;
  using flux_variables = tmpl::list<>;
  using gradient_variables = tmpl::list<Psi>;

  using compute_volume_time_derivative_terms = TimeDerivativeTerms<Dim>;
};

constexpr bool use_nodegroup_dg_elements = false;

struct Metavariables;

template <size_t Dim>
struct component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<Dim>;

  using variables_tag = typename System<Dim>::variables_tag;

  using simple_tags = tmpl::list<
      ::Tags::TimeStepId, ::Tags::Next<::Tags::TimeStepId>, ::Tags::TimeStep,
      ::Tags::Time, ::evolution::dg::Tags::Quadrature, variables_tag,
      ::Tags::HistoryEvolvedVariables<variables_tag>, domain::Tags::Mesh<Dim>,
      ::domain::Tags::FunctionsOfTimeInitialize,
      domain::CoordinateMaps::Tags::CoordinateMap<Dim, Frame::Grid,
                                                  Frame::Inertial>,
      domain::Tags::Element<Dim>, domain::Tags::NeighborMesh<Dim>,
      domain::Tags::Coordinates<Dim, Frame::Inertial>,
      domain::Tags::InverseJacobian<Dim, Frame::ElementLogical,
                                    Frame::Inertial>,
      domain::Tags::MeshVelocity<Dim>, domain::Tags::DivMeshVelocity,
      domain::Tags::ElementMap<Dim, Frame::Grid>,
      ::Tags::ConcreteTimeStepper<TimeStepper>>;

  using compute_tags =
      tmpl::list<domain::Tags::JacobianCompute<Dim, Frame::ElementLogical,
                                               Frame::Inertial>,
                 domain::Tags::DetInvJacobianCompute<Dim, Frame::ElementLogical,
                                                     Frame::Inertial>>;

  // Declare the physical (IsAuxiliary == false) inbox channel on the component
  // as well, so the test can verify it stays empty. The auxiliary channel is
  // contributed by the SendAuxiliaryBoundaryData action below.
  using inbox_tags =
      tmpl::list<::evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<
          Dim, use_nodegroup_dg_elements, /*IsAuxiliary=*/false>>;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          Parallel::Phase::Initialization,
          tmpl::flatten<tmpl::list<
              ActionTesting::InitializeDataBox<simple_tags, compute_tags>,
              ::evolution::dg::Initialization::Mortars<Dim>>>>,
      Parallel::PhaseActions<
          Parallel::Phase::Testing,
          tmpl::list<::evolution::dg::Actions::SendAuxiliaryBoundaryData<
              Dim, System<Dim>, false, use_nodegroup_dg_elements>>>>;
};

struct Metavariables {
  static constexpr size_t volume_dim = 1;
  static constexpr bool local_time_stepping = false;
  static constexpr bool use_nodegroup_dg_elements = ::use_nodegroup_dg_elements;
  using system = System<1>;
  using const_global_cache_tags =
      tmpl::list<domain::Tags::InitialExtents<1>, domain::Tags::Domain<1>,
                 ::Tags::LtsMode>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<BoundaryCondition<1>,
                             tmpl::list<DemandOutgoingCharSpeeds<1>>>,
                  tmpl::pair<::evolution::BoundaryCorrection,
                             tmpl::list<AuxiliaryCorrection<1>>>>;
  };
  using component_list = tmpl::list<component<1>>;
};

void test() {
  constexpr size_t Dim = 1;
  using metavars = Metavariables;
  using system = typename metavars::system;
  using variables_tag = typename system::variables_tag;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavars>;
  register_classes_with_charm<TimeSteppers::AdamsBashforth>();
  register_factory_classes_with_charm<metavars>();

  const Spectral::Quadrature quadrature = Spectral::Quadrature::GaussLobatto;
  const ::dg::Formulation dg_formulation = ::dg::Formulation::StrongInertial;

  // Two elements in a single block, with a shared internal boundary in +xi.
  const ElementId<Dim> self_id{0, {{{1, 0}}}};
  const ElementId<Dim> east_id{0, {{{1, 1}}}};
  DirectionMap<Dim, Neighbors<Dim>> neighbors{};
  neighbors[Direction<Dim>::upper_xi()] =
      Neighbors<Dim>{{east_id}, OrientationMap<Dim>::create_aligned()};
  const Element<Dim> element{self_id, neighbors};

  const auto grid_to_inertial_map =
      domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
          domain::CoordinateMaps::Identity<Dim>{});

  const std::array<size_t, Dim> extents{{2}};
  const Mesh<Dim> mesh{extents, Spectral::Basis::Legendre, quadrature};

  DirectionalIdMap<Dim, Mesh<Dim>> neighbor_mesh{};
  for (const auto& [direction, direction_neighbors] : neighbors) {
    for (const auto& neighbor : direction_neighbors) {
      const auto& neighbor_orientation =
          direction_neighbors.orientation(neighbor);
      neighbor_mesh.emplace(DirectionalId{direction, neighbor},
                            neighbor_orientation.inverse_map()(mesh));
    }
  }

  const Slab time_slab{0.2, 3.4};
  const TimeDelta time_step{time_slab, {1, 128}};
  const TimeStepId time_step_id{true, 3, Time{time_slab, {3, 128}}};
  const TimeStepId next_time_step_id = time_step_id.next_step(time_step);

  MockRuntimeSystem runner = [&dg_formulation, &extents,
                              &grid_to_inertial_map]() {
    std::vector<DirectionMap<
        Dim, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
        boundary_conditions{1};
    std::vector<Block<Dim>> blocks{1};
    for (const auto& direction : Direction<Dim>::all_directions()) {
      boundary_conditions[0][direction] =
          std::make_unique<DemandOutgoingCharSpeeds<Dim>>();
    }
    blocks[0] = Block<Dim>{
        domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
            domain::CoordinateMaps::Identity<Dim>{}),
        0,
        {}};
    Domain<Dim> domain{std::move(blocks)};
    domain.inject_time_dependent_map_for_block(
        0, grid_to_inertial_map->get_clone());
    return MockRuntimeSystem{
        {std::vector<std::array<size_t, Dim>>{extents, extents},
         std::move(domain), ::LtsMode::Off, dg_formulation,
         std::make_unique<AuxiliaryCorrection<Dim>>(),
         std::move(boundary_conditions)}};
  }();

  const auto get_tag = [&runner, &self_id](auto tag_v) -> decltype(auto) {
    using tag = std::decay_t<decltype(tag_v)>;
    return ActionTesting::get_databox_tag<component<Dim>, tag>(runner, self_id);
  };

  ::InverseJacobian<DataVector, Dim, Frame::ElementLogical, Frame::Inertial>
      inv_jac{mesh.number_of_grid_points(), 0.0};
  inv_jac.get(0, 0) = 1.0;

  const auto logical_coords = logical_coordinates(mesh);
  tnsr::I<DataVector, Dim, Frame::Inertial> inertial_coords{};
  for (size_t i = 0; i < logical_coords.size(); ++i) {
    inertial_coords[i] = logical_coords[i];
  }

  const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>
      mesh_velocity{};
  const std::optional<Scalar<DataVector>> div_mesh_velocity{};

  Variables<tmpl::list<Psi>> evolved_vars{mesh.number_of_grid_points()};
  get(get<Psi>(evolved_vars)) = DataVector{2.0, 5.0};
  const ::TimeSteppers::History<decltype(evolved_vars)> history{1};

  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};

  const auto emplace = [&](const ElementId<Dim>& id) {
    ActionTesting::emplace_component_and_initialize<component<Dim>>(
        &runner, id,
        {time_step_id, next_time_step_id, time_step,
         time_step_id.step_time().value(), quadrature, evolved_vars, history,
         mesh, clone_unique_ptrs(functions_of_time),
         grid_to_inertial_map->get_clone(), element, neighbor_mesh,
         inertial_coords, inv_jac, mesh_velocity, div_mesh_velocity,
         ElementMap<Dim, Frame::Grid>{
             id,
             domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(
                 domain::CoordinateMaps::Identity<Dim>{})},
         std::make_unique<TimeSteppers::AdamsBashforth>(5)});
  };
  emplace(self_id);
  emplace(east_id);

  // Initialize the mortars.
  ActionTesting::next_action<component<Dim>>(make_not_null(&runner), self_id);

  // Run the auxiliary send.
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);
  ActionTesting::next_action<component<Dim>>(make_not_null(&runner), self_id);

  // Compute the expected auxiliary mortar data: project Psi to the +xi face and
  // call dg_auxiliary_package_data (which scales by 2).
  const DirectionalId<Dim> mortar_id_east{Direction<Dim>::upper_xi(), east_id};
  const Mesh<Dim - 1> face_mesh =
      mesh.slice_away(Direction<Dim>::upper_xi().dimension());
  Variables<tmpl::list<Psi>> psi_on_face{face_mesh.number_of_grid_points()};
  ::dg::project_contiguous_data_to_boundary(make_not_null(&psi_on_face),
                                            evolved_vars, mesh,
                                            Direction<Dim>::upper_xi());
  Variables<tmpl::list<TwoTimesPsi>> expected_packaged_data{
      face_mesh.number_of_grid_points()};
  get(get<TwoTimesPsi>(expected_packaged_data)) =
      2.0 * get(get<Psi>(psi_on_face));
  const DataVector expected_mortar_data{
      get(get<TwoTimesPsi>(expected_packaged_data))};

  // The local auxiliary mortar data lives in MortarData.
  CHECK_ITERABLE_APPROX(get_tag(::evolution::dg::Tags::MortarData<Dim>{})
                            .at(mortar_id_east)
                            .local()
                            .mortar_data.value(),
                        expected_mortar_data);

  // The neighbor receives the data on the auxiliary inbox channel.
  const DirectionalId<Dim> east_neighbor_mortar_id{Direction<Dim>::lower_xi(),
                                                   self_id};
  const auto& aux_messages =
      ActionTesting::get_inbox_tag<
          component<Dim>,
          ::evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<
              Dim, use_nodegroup_dg_elements, /*IsAuxiliary=*/true>>(runner,
                                                                     east_id)
          .messages;
  REQUIRE(aux_messages.count(time_step_id) == 1);
  const auto& aux_messages_at_time = aux_messages.at(time_step_id);
  const auto received_entry =
      alg::find_if(aux_messages_at_time, [&](const auto& entry) {
        return entry.first == east_neighbor_mortar_id;
      });
  REQUIRE(received_entry != aux_messages_at_time.end());
  const auto& received = received_entry->second;
  REQUIRE(received.boundary_correction_data.has_value());
  CHECK_ITERABLE_APPROX(received.boundary_correction_data.value(),
                        expected_mortar_data);
  CHECK(received.validity_range == next_time_step_id);

  // The physical inbox channel (IsAuxiliary == false) must stay empty.
  const auto& physical_messages =
      ActionTesting::get_inbox_tag<
          component<Dim>,
          ::evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<
              Dim, use_nodegroup_dg_elements, /*IsAuxiliary=*/false>>(runner,
                                                                      east_id)
          .messages;
  CHECK(physical_messages.empty());
}

SPECTRE_TEST_CASE("Unit.Evolution.DG.SendAuxiliaryBoundaryData",
                  "[Unit][Evolution][Actions]") {
  PUPable_reg(
      SINGLE_ARG(domain::CoordinateMap<Frame::Grid, Frame::Inertial,
                                       domain::CoordinateMaps::Identity<1>>));
  PUPable_reg(
      SINGLE_ARG(domain::CoordinateMap<Frame::BlockLogical, Frame::Inertial,
                                       domain::CoordinateMaps::Identity<1>>));
  PUPable_reg(
      SINGLE_ARG(domain::CoordinateMap<Frame::BlockLogical, Frame::Grid,
                                       domain::CoordinateMaps::Identity<1>>));
  test();
}
}  // namespace
