// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SliceVariables.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Domain/CoordinateMaps/Distribution.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/Rectilinear.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Creators/Tags/ExternalBoundaryConditions.hpp"
#include "Domain/Creators/Tags/FunctionsOfTime.hpp"
#include "Domain/Creators/Tags/InitialExtents.hpp"
#include "Domain/Creators/Tags/InitialRefinementLevels.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Actions/RunEventsAndDenseTriggers.hpp"
#include "Evolution/BoundaryCorrection.hpp"
#include "Evolution/BoundaryCorrectionTags.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ApplyBoundaryCorrections.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ComputeTimeDerivative.hpp"
#include "Evolution/DiscontinuousGalerkin/InboxTags.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/Mortars.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/QuadratureTag.hpp"
#include "Evolution/Initialization/DgDomain.hpp"
#include "Evolution/Systems/Ccz4/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/Ccz4/Ccz4WrappedGr.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/EnforceConstrainedEvolution.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/BoundaryConditions/DirichletAnalytic.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/BoundaryCorrections/UpwindGradientProjection.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Chart.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/FinalizeTimeDerivative.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Initialize.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/LocalOperators.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/PrepareChartFields.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/StepActions.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/System.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Tags.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/TimeDerivative.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/Pypp.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags/Formulation.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/None.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Tag.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/Actions/InitializeItems.hpp"
#include "ParallelAlgorithms/Actions/MutateApply.hpp"
#include "ParallelAlgorithms/EventsAndDenseTriggers/DenseTrigger.hpp"
#include "ParallelAlgorithms/EventsAndDenseTriggers/EventsAndDenseTriggers.hpp"
#include "ParallelAlgorithms/EventsAndDenseTriggers/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/EventsAndTriggers.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/WhenToCheck.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Tags/InitialData.hpp"
#include "Time/Actions/SelfStartActions.hpp"
#include "Time/AdvanceTime.hpp"
#include "Time/History.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags/AdaptiveSteppingDiagnostics.hpp"
#include "Time/Tags/HistoryEvolvedVariables.hpp"
#include "Time/Tags/StepNumberWithinSlab.hpp"
#include "Time/Tags/Time.hpp"
#include "Time/Tags/TimeStep.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/Tags/TimeStepper.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/AdamsBashforth.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeVector.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace {
namespace gp = Ccz4::GradientProjection;
using system = gp::System;
using variables_tag = system::variables_tag;
using variables_tags = variables_tag::tags_list;
using dt_variables_tag = db::add_tag_prefix<::Tags::dt, variables_tag>;
using history_tag = ::Tags::HistoryEvolvedVariables<variables_tag>;
using Vars = variables_tag::type;
using DtVars = dt_variables_tag::type;
using physical_tags = system::physical_tags;
using chart_tags = system::chart_tags;
using PhysicalVars = Variables<physical_tags>;
using dt_physical_tags = db::wrap_tags_in<::Tags::dt, physical_tags>;
using DtPhysicalVars = Variables<dt_physical_tags>;
using filter_tag = Filters::runtime::Tags::SpectralFilter<3, variables_tags>;
using inbox_tag =
    evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<3, false>;
using Minkowski = Ccz4::Solutions::Ccz4WrappedGr<gr::Solutions::Minkowski<3>>;

// A dense trigger that fires once, at a fixed time.
class FireOnce : public DenseTrigger {
 public:
  FireOnce() = default;
  explicit FireOnce(CkMigrateMessage* const msg) : DenseTrigger(msg) {}
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
  WRAPPED_PUPable_decl_template(FireOnce);  // NOLINT
#pragma GCC diagnostic pop
  explicit FireOnce(const double trigger_time) : trigger_time_(trigger_time) {}

  using is_triggered_return_tags = tmpl::list<>;
  using is_triggered_argument_tags = tmpl::list<::Tags::Time>;
  template <typename Metavariables, typename ArrayIndex, typename Component>
  std::optional<bool> is_triggered(
      Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const Component* /*component*/,
      const double time) const {
    return time == trigger_time_;
  }

  using next_check_time_return_tags = tmpl::list<>;
  using next_check_time_argument_tags = tmpl::list<::Tags::Time>;
  template <typename Metavariables, typename ArrayIndex, typename Component>
  std::optional<double> next_check_time(
      Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const Component* /*component*/,
      const double time) const {
    return time < trigger_time_ ? trigger_time_
                                : std::numeric_limits<double>::infinity();
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override {
    DenseTrigger::pup(p);
    p | trigger_time_;
  }

 private:
  double trigger_time_ = std::numeric_limits<double>::signaling_NaN();
};

PUP::able::PUP_ID FireOnce::my_PUP_ID = 0;  // NOLINT

// An event recording the time and the evolved variables it observes.
struct RecordState : public Event {
  RecordState() = default;
  explicit RecordState(CkMigrateMessage* const /*msg*/) {}
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
  WRAPPED_PUPable_decl_template(RecordState);  // NOLINT
#pragma GCC diagnostic pop

  using compute_tags_for_observation_box = tmpl::list<>;
  using return_tags = tmpl::list<>;
  using argument_tags = tmpl::list<::Tags::Time, variables_tag>;

  template <typename Metavariables, typename ArrayIndex, typename Component>
  void operator()(const double time, const Vars& vars,
                  Parallel::GlobalCache<Metavariables>& /*cache*/,
                  const ArrayIndex& /*array_index*/,
                  const Component* const /*meta*/,
                  const ObservationValue& /*observation_value*/) const {
    calls.emplace_back(time, vars);
  }

  using is_ready_argument_tags = tmpl::list<>;
  template <typename Metavariables, typename ArrayIndex, typename Component>
  bool is_ready(Parallel::GlobalCache<Metavariables>& /*cache*/,
                const ArrayIndex& /*array_index*/,
                const Component* const /*meta*/) const {
    return true;
  }

  bool needs_evolved_variables() const override { return true; }

  static std::vector<std::pair<double, Vars>> calls;
};

std::vector<std::pair<double, Vars>> RecordState::calls{};
PUP::able::PUP_ID RecordState::my_PUP_ID = 0;  // NOLINT

template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<3>;
  using simple_tags = tmpl::list<
      domain::Tags::InitialExtents<3>, domain::Tags::InitialRefinementLevels<3>,
      evolution::dg::Tags::Quadrature, ::Tags::TimeStepId,
      ::Tags::Next<::Tags::TimeStepId>, ::Tags::TimeStep, ::Tags::Time,
      ::Tags::StepNumberWithinSlab, ::Tags::AdaptiveSteppingDiagnostics,
      variables_tag, dt_variables_tag, history_tag, filter_tag,
      Ccz4::Tags::Kappa1, Ccz4::Tags::Kappa2, Ccz4::Tags::Eta<DataVector>,
      Ccz4::Tags::K0<DataVector>, ::Tags::EventsAndDenseTriggers,
      ::Tags::PreviousTriggerTime>;
  using compute_tags = time_stepper_ref_tags<TimeStepper>;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          Parallel::Phase::Initialization,
          tmpl::list<
              ActionTesting::InitializeDataBox<simple_tags, compute_tags>,
              Initialization::Actions::InitializeItems<
                  evolution::dg::Initialization::Domain<Metavariables>>,
              evolution::dg::Initialization::Mortars<3>,
              Initialization::Actions::InitializeItems<
                  gp::InitializeLocalOperators>,
              gp::Actions::SetInitialData>>,
      Parallel::PhaseActions<
          Parallel::Phase::InitializeTimeStepperHistory,
          SelfStart::self_start_procedure<gp::step_actions, system>>,
      Parallel::PhaseActions<
          Parallel::Phase::Evolve,
          tmpl::push_back<gp::step_actions,
                          ::Actions::MutateApply<AdvanceTime<>>>>>;
};

struct Metavariables {
  static constexpr size_t volume_dim = 3;
  static constexpr bool local_time_stepping = false;
  using system = gp::System;
  using component_list = tmpl::list<Component<Metavariables>>;
  using const_global_cache_tags =
      tmpl::list<::Tags::ConcreteTimeStepper<TimeStepper>, Ccz4::Tags::Kappa3,
                 evolution::initial_data::Tags::InitialData,
                 ::Tags::EventsAndTriggers<Triggers::WhenToCheck::AtSlabs>>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<
            evolution::BoundaryCorrection,
            tmpl::list<gp::BoundaryCorrections::UpwindGradientProjection>>,
        tmpl::pair<
            Ccz4::BoundaryConditions::BoundaryCondition,
            tmpl::list<gp::BoundaryConditions::DirichletAnalytic,
                       domain::BoundaryConditions::Periodic<
                           Ccz4::BoundaryConditions::BoundaryCondition>>>,
        tmpl::pair<evolution::initial_data::InitialData, tmpl::list<Minkowski>>,
        tmpl::pair<DenseTrigger, tmpl::list<FireOnce>>,
        tmpl::pair<Event, tmpl::list<RecordState>>>;
  };
};

using component = Component<Metavariables>;
using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<Metavariables>;

template <typename Tag>
const auto& get_tag(const MockRuntimeSystem& runner, const ElementId<3>& id) {
  return ActionTesting::get_databox_tag<component, Tag>(runner, id);
}

template <typename Tags, typename VarsType>
Variables<Tags> part(const VarsType& vars) {
  Variables<Tags> result{vars.number_of_grid_points()};
  tmpl::for_each<Tags>(
      [&result, &vars]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
        get<Tag>(result) = get<Tag>(vars);
      });
  return result;
}

// The physical state of the given chart fields; it lies on the
// algebraic-constraint surface. The chart scratch fields hold garbage, which
// must be overwritten before use.
Vars physical_from_chart(const gp::Tags::ChartU::type& u,
                         const gp::Tags::ChartW::type& w) {
  const size_t n = get<0>(u).size();
  Vars vars{n, 0.0};
  for (size_t s = 0; s < n; ++s) {
    gp::PotentialVector u_point{};
    gp::VelocityVector w_point{};
    for (size_t a = 0; a < gp::number_of_potentials; ++a) {
      gsl::at(u_point, a) = u.get(a)[s];
    }
    for (size_t a = 0; a < gp::number_of_velocities; ++a) {
      gsl::at(w_point, a) = w.get(a)[s];
    }
    const gp::LogMetricChart chart{
        gp::stf_decode(&u_point[gp::PotentialIndex::log_metric])};
    const gp::Matrix3 g = chart.metric();
    const gp::Matrix3 a_tilde =
        0.5 * chart.tangent(gp::stf_decode(&w_point[gp::VelocityIndex::a]));
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = i; j < 3; ++j) {
        get<Ccz4::Tags::ConformalMetric<DataVector, 3>>(vars).get(i, j)[s] =
            g(i, j);
        get<Ccz4::Tags::ATilde<DataVector, 3>>(vars).get(i, j)[s] =
            a_tilde(i, j);
      }
      get<gr::Tags::Shift<DataVector, 3>>(vars).get(i)[s] =
          gsl::at(u_point, gp::PotentialIndex::shift + i);
      get<Ccz4::Tags::GammaHat<DataVector, 3>>(vars).get(i)[s] =
          gsl::at(w_point, gp::VelocityIndex::gamma_hat + i);
      get<Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>>(vars).get(i)[s] =
          gsl::at(w_point, gp::VelocityIndex::b + i);
    }
    get(get<Ccz4::Tags::ConformalFactor<DataVector>>(vars))[s] =
        std::exp(u_point[gp::PotentialIndex::log_conformal_factor]);
    get(get<gr::Tags::Lapse<DataVector>>(vars))[s] =
        u_point[gp::PotentialIndex::lapse];
    get(get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(vars))[s] =
        w_point[gp::VelocityIndex::trace_extrinsic_curvature];
    get(get<Ccz4::Tags::Theta<DataVector>>(vars))[s] =
        w_point[gp::VelocityIndex::theta];
  }
  get<gp::Tags::ChartU>(vars) = gp::Tags::ChartU::type{n, 1.0e3};
  get<gp::Tags::ChartW>(vars) = gp::Tags::ChartW::type{n, -2.0e3};
  get<gp::Tags::ChartQ>(vars) = gp::Tags::ChartQ::type{n, 3.0e3};
  return vars;
}

// A smooth periodic perturbation of Minkowski.
Vars initial_state(const tnsr::I<DataVector, 3>& x,
                   const std::array<double, 3>& lengths) {
  const size_t n = get<0>(x).size();
  const DataVector phase = 2.0 * M_PI *
                           (get<0>(x) / lengths[0] + get<1>(x) / lengths[1] -
                            get<2>(x) / lengths[2]);
  gp::Tags::ChartU::type u{n};
  gp::Tags::ChartW::type w{n};
  for (size_t a = 0; a < gp::number_of_potentials; ++a) {
    u.get(a) = 0.05 * sin(phase + 0.3 * static_cast<double>(a));
  }
  u.get(gp::PotentialIndex::lapse) += 1.0;
  for (size_t a = 0; a < gp::number_of_velocities; ++a) {
    w.get(a) = 0.04 * cos(phase + 0.7 * static_cast<double>(a));
  }
  return physical_from_chart(u, w);
}

// The integrals over [start, end] of the Lagrange basis polynomials through
// the three derivative times, by two-point Gauss-Legendre quadrature (exact
// for the quadratic interpolant). Adams-Bashforth 3 with arbitrary (including
// the non-monotonic self-start) history times integrates this interpolant.
std::array<double, 3> adams_weights(const std::array<double, 3>& times,
                                    const double start, const double end) {
  const double mid = 0.5 * (start + end);
  const double half = 0.5 * (end - start);
  std::array<double, 3> weights{};
  for (const double t :
       {mid - half / std::sqrt(3.0), mid + half / std::sqrt(3.0)}) {
    for (size_t j = 0; j < 3; ++j) {
      double basis = 1.0;
      for (size_t k = 0; k < 3; ++k) {
        if (k != j) {
          basis *=
              (t - gsl::at(times, k)) / (gsl::at(times, j) - gsl::at(times, k));
        }
      }
      gsl::at(weights, j) += half * basis;
    }
  }
  return weights;
}

std::array<double, 3> history_times(const history_tag::type& history) {
  REQUIRE(history.size() == 3);
  return {{history[0].time_step_id.substep_time(),
           history[1].time_step_id.substep_time(),
           history[2].time_step_id.substep_time()}};
}

// The physical state at the step start plus the weighted recorded finalized
// physical rates.
PhysicalVars adams_physical_state(const history_tag::type& history,
                                  const std::array<double, 3>& weights) {
  REQUIRE(history.size() == 3);
  PhysicalVars result = part<physical_tags>(*history.back().value);
  tmpl::for_each<physical_tags>(
      [&result, &history, &weights]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
        auto& field = get<Tag>(result);
        for (size_t entry = 0; entry < 3; ++entry) {
          const auto& rate = get<::Tags::dt<Tag>>(history[entry].derivative);
          for (size_t c = 0; c < field.size(); ++c) {
            field[c] += gsl::at(weights, entry) * rate[c];
          }
        }
      });
  return result;
}

void check_physical_rates_zero(const DtVars& dt_vars) {
  tmpl::for_each<physical_tags>(
      [&dt_vars]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
        for (const auto& component : get<::Tags::dt<Tag>>(dt_vars)) {
          CHECK(max(abs(component)) == 0.0);
        }
      });
}

// The corrected targets expected from the volume targets: on every face of the
// periodic element the exterior state is the opposite face of the same
// element, the interior normal is the outward Euclidean unit covector, and the
// correction is lifted with the inverse LGL end weight 1 / w_0 = 3 (three
// points) times the face-normal magnitude 2 / L.
DtVars expected_corrected_targets(
    const DtVars& volume_dt, const Vars& vars, const Mesh<3>& mesh,
    const std::array<double, 3>& lengths,
    const gp::BoundaryCorrections::UpwindGradientProjection& correction) {
  using ChartVars = Variables<chart_tags>;
  REQUIRE(mesh.extents() == Index<3>(3));
  const ChartVars chart = part<chart_tags>(vars);
  DtVars result = volume_dt;
  const size_t last = mesh.extents(0) - 1;
  for (size_t axis = 0; axis < 3; ++axis) {
    for (const double sign : {-1.0, 1.0}) {
      const size_t index = sign > 0.0 ? last : 0;
      const size_t opposite = sign > 0.0 ? 0 : last;
      const ChartVars interior =
          data_on_slice(chart, mesh.extents(), axis, index);
      const ChartVars exterior =
          data_on_slice(chart, mesh.extents(), axis, opposite);
      const size_t face_points = interior.number_of_grid_points();
      tnsr::i<DataVector, 3> normal{face_points, 0.0};
      normal.get(axis) = sign;
      tnsr::i<DataVector, 3> exterior_normal{face_points, 0.0};
      exterior_normal.get(axis) = -sign;
      Vars terms{face_points};
      correction.dg_boundary_terms<false>(
          make_not_null(&get<gp::Tags::ChartW>(terms)),
          make_not_null(&get<gp::Tags::ChartQ>(terms)),
          make_not_null(&get<gp::Tags::ChartU>(terms)),
          make_not_null(
              &get<Ccz4::Tags::ConformalMetric<DataVector, 3>>(terms)),
          make_not_null(&get<Ccz4::Tags::ConformalFactor<DataVector>>(terms)),
          make_not_null(&get<Ccz4::Tags::ATilde<DataVector, 3>>(terms)),
          make_not_null(
              &get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(terms)),
          make_not_null(&get<Ccz4::Tags::Theta<DataVector>>(terms)),
          make_not_null(&get<Ccz4::Tags::GammaHat<DataVector, 3>>(terms)),
          make_not_null(&get<gr::Tags::Lapse<DataVector>>(terms)),
          make_not_null(&get<gr::Tags::Shift<DataVector, 3>>(terms)),
          make_not_null(
              &get<Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>>(terms)),
          get<gp::Tags::ChartW>(interior), get<gp::Tags::ChartQ>(interior),
          get<gp::Tags::ChartU>(interior), normal,
          get<gp::Tags::ChartW>(exterior), get<gp::Tags::ChartQ>(exterior),
          get<gp::Tags::ChartU>(exterior), exterior_normal,
          dg::Formulation::StrongInertial);
      tmpl::for_each<physical_tags>(
          [&terms]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
            for (const auto& component : get<Tag>(terms)) {
              CHECK(max(abs(component)) == 0.0);
            }
          });
      terms *= -3.0 * 2.0 / gsl::at(lengths, axis);
      add_slice_to_data(make_not_null(&result), terms, mesh.extents(), axis,
                        index);
    }
  }
  return result;
}

using CacheTuple = tuples::TaggedTuple<
    domain::Tags::Domain<3>, ::Tags::ConcreteTimeStepper<TimeStepper>,
    Ccz4::Tags::Kappa3,
    ::Tags::EventsAndTriggers<Triggers::WhenToCheck::AtSlabs>,
    evolution::Tags::BoundaryCorrection, ::dg::Tags::Formulation,
    domain::Tags::ExternalBoundaryConditions<3>,
    evolution::initial_data::Tags::InitialData>;
using MutableCacheTuple =
    tuples::TaggedTuple<domain::Tags::FunctionsOfTimeInitialize>;

const std::array<double, 3> element_lengths{{1.0, 1.3, 0.8}};
constexpr double kappa_3 = 0.5;
constexpr size_t number_of_points = 27;

// A periodic brick of 2^refinement_x elements along x, each of size
// element_lengths, with three LGL points per dimension.
domain::creators::Brick brick(const size_t refinement_x) {
  const double elements_x = static_cast<double>(two_to_the(refinement_x));
  return {{{0.0, 0.0, 0.0}},
          {{elements_x * element_lengths[0], element_lengths[1],
            element_lengths[2]}},
          {{refinement_x, 0, 0}},
          {{3, 3, 3}},
          {{true, true, true}}};
}

ElementId<3> element_id(const size_t refinement_x, const size_t index_x) {
  return {
      0,
      {{SegmentId{refinement_x, index_x}, SegmentId{0, 0}, SegmentId{0, 0}}}};
}

// The excised Sphere with one layer of six wedges and three LGL points per
// dimension, with analytic Minkowski data on both boundaries.
domain::creators::Sphere sphere() {
  const auto analytic = []() {
    return std::make_unique<gp::BoundaryConditions::DirichletAnalytic>(
        std::make_unique<Minkowski>());
  };
  return {1.0,
          2.0,
          domain::creators::Sphere::Excision{analytic()},
          size_t{0},
          size_t{3},
          true,
          std::nullopt,
          {},
          domain::CoordinateMaps::Distribution::Linear,
          ShellWedges::All,
          std::nullopt,
          analytic()};
}

CacheTuple cache(const size_t order, const DomainCreator<3>& creator,
                 const evolution::BoundaryCorrection& correction) {
  return {creator.create_domain(),
          std::make_unique<TimeSteppers::AdamsBashforth>(order),
          kappa_3,
          EventsAndTriggers{},
          correction.get_clone(),
          ::dg::Formulation::StrongInertial,
          creator.external_boundary_conditions(),
          std::make_unique<Minkowski>()};
}

// Emplaces and initializes an element with the damping parameters kappa_1 and
// eta (the Gamma-driver damping, possibly varying over the points).
void emplace_element(
    const gsl::not_null<MockRuntimeSystem*> runner,
    const DomainCreator<3>& creator, const ElementId<3>& id,
    const TimeStepId& time_step_id, const TimeStepId& next_time_step_id,
    const TimeDelta& step, const double kappa_1, const DataVector& eta,
    EventsAndDenseTriggers::ConstructionType events_and_dense_triggers) {
  const size_t n = number_of_points;
  ActionTesting::emplace_array_component_and_initialize<component>(
      runner, ActionTesting::NodeId{0}, ActionTesting::LocalCoreId{0}, id,
      {creator.initial_extents(), creator.initial_refinement_levels(),
       Spectral::Quadrature::GaussLobatto, time_step_id, next_time_step_id,
       step, time_step_id.substep_time(), uint64_t{0},
       ::Tags::AdaptiveSteppingDiagnostics::type{}, Vars{n, 777.0},
       DtVars{n, 0.0}, history_tag::type{1},
       std::unique_ptr<Filters::runtime::Filter<3, variables_tags>>{
           std::make_unique<Filters::runtime::None<3, variables_tags>>(
               std::nullopt)},
       Scalar<DataVector>{n, kappa_1}, Scalar<DataVector>{n, 0.0},
       Scalar<DataVector>{eta}, Scalar<DataVector>{n, 0.0},
       EventsAndDenseTriggers(std::move(events_and_dense_triggers)),
       std::optional<double>{}});
  // Domain, Mortars, InitializeLocalOperators, SetInitialData
  for (size_t i = 0; i < 4; ++i) {
    ActionTesting::next_action<component>(runner, id);
  }
  // Exercise the production initial-data action, including clearing all
  // scratch slots and assigning every physical field on bricks and wedges.
  const auto& vars = get_tag<variables_tag>(*runner, id);
  const auto exact = Minkowski{}.variables(
      get_tag<domain::Tags::Coordinates<3, Frame::Inertial>>(*runner, id), 0.0,
      physical_tags{});
  tmpl::for_each<physical_tags>(
      [&vars, &exact]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
        CHECK_ITERABLE_APPROX(get<Tag>(vars), get<Tag>(exact));
      });
  tmpl::for_each<chart_tags>([&vars]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
    for (const auto& component : get<Tag>(vars)) {
      CHECK(max(abs(component)) == 0.0);
    }
  });
}

// The step actions, without and with a spatially varying Gamma-driver damping
// eta (which the volume targets and FinalizeTimeDerivative take from the
// DataBox).
void test_step_actions(const bool damped) {
  CAPTURE(damped);
  RecordState::calls.clear();
  const size_t order = 3;
  const gp::BoundaryCorrections::UpwindGradientProjection correction{1.0};
  MockRuntimeSystem runner{cache(order, brick(0), correction),
                           MutableCacheTuple{}};
  const ElementId<3> id = element_id(0, 0);
  const Slab slab{0.0, 0.03};
  const TimeDelta step = slab.duration();
  const double dt = step.value();
  const double trigger_time = 0.4 * dt;
  const size_t n = number_of_points;
  DataVector eta_values{n, 0.0};
  if (damped) {
    for (size_t s = 0; s < n; ++s) {
      eta_values[s] = 0.4 + 0.03 * static_cast<double>(s);
    }
  }
  const Scalar<DataVector> eta{eta_values};
  EventsAndDenseTriggers::ConstructionType events_and_dense_triggers{};
  events_and_dense_triggers.emplace_back(
      std::make_unique<FireOnce>(trigger_time),
      make_vector<std::unique_ptr<Event>>(std::make_unique<RecordState>()));
  emplace_element(
      make_not_null(&runner), brick(0), id, TimeStepId{},
      TimeStepId{true, 1 - static_cast<int64_t>(order), slab.start()}, step,
      0.001, eta_values, std::move(events_and_dense_triggers));
  {
    auto& box =
        ActionTesting::get_databox<component>(make_not_null(&runner), id);
    const auto& x = db::get<domain::Tags::Coordinates<3, Frame::Inertial>>(box);
    db::mutate<variables_tag>(
        [&x](const gsl::not_null<Vars*> vars) {
          *vars = initial_state(x, element_lengths);
        },
        make_not_null(&box));
  }
  const PhysicalVars initial_physical =
      part<physical_tags>(get_tag<variables_tag>(runner, id));
  const auto& mesh = get_tag<domain::Tags::Mesh<3>>(runner, id);
  const auto& inverse_jacobian = get_tag<
      domain::Tags::InverseJacobian<3, Frame::ElementLogical, Frame::Inertial>>(
      runner, id);
  const auto& operators = get_tag<gp::Tags::LocalOperators>(runner, id);
  const auto next = [&runner, &id]() {
    REQUIRE(ActionTesting::next_action_if_ready<component>(
        make_not_null(&runner), id));
  };

  {
    INFO("Self-start");
    ActionTesting::set_phase(make_not_null(&runner),
                             Parallel::Phase::InitializeTimeStepperHistory);
    size_t actions = 0;
    while (not ActionTesting::get_terminate<component>(runner, id)) {
      next();
      REQUIRE(++actions < 1000);
    }
    // The procedure restores the initial state and leaves a history of
    // order - 1 entries for the order-3 integration.
    CHECK(part<physical_tags>(get_tag<variables_tag>(runner, id)) ==
          initial_physical);
    CHECK(get_tag<::Tags::TimeStepId>(runner, id) ==
          TimeStepId(true, 0, slab.start()));
    CHECK(get_tag<history_tag>(runner, id).integration_order() == order);
    CHECK(get_tag<history_tag>(runner, id).size() == order - 1);
    // Dense triggers are not run during self-start.
    CHECK(RecordState::calls.empty());
  }

  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Evolve);
  {
    auto& box =
        ActionTesting::get_databox<component>(make_not_null(&runner), id);
    db::mutate<gp::Tags::ChartU, gp::Tags::ChartW, gp::Tags::ChartQ>(
        [](const auto u, const auto w, const auto q) {
          for (auto& component : *u) {
            component = 1.0e3;
          }
          for (auto& component : *w) {
            component = -2.0e3;
          }
          for (auto& component : *q) {
            component = 3.0e3;
          }
        },
        make_not_null(&box));
  }

  {
    INFO("PrepareChartFields overwrites the chart from the physical state");
    next();
    const auto& u = get_tag<gp::Tags::ChartU>(runner, id);
    gp::Tags::ChartQ::type expected_q{n};
    gp::potential_gradients(make_not_null(&expected_q), u, mesh,
                            inverse_jacobian);
    CHECK_ITERABLE_APPROX(get_tag<gp::Tags::ChartQ>(runner, id), expected_q);
    CHECK_ITERABLE_APPROX(
        u.get(gp::PotentialIndex::lapse),
        get(get_tag<gr::Tags::Lapse<DataVector>>(runner, id)));
    CHECK_ITERABLE_APPROX(
        u.get(gp::PotentialIndex::log_conformal_factor),
        log(get(get_tag<Ccz4::Tags::ConformalFactor<DataVector>>(runner, id))));
    CHECK_ITERABLE_APPROX(
        get_tag<gp::Tags::ChartW>(runner, id).get(gp::VelocityIndex::theta),
        get(get_tag<Ccz4::Tags::Theta<DataVector>>(runner, id)));
  }

  const TimeStepId time_step_id = get_tag<::Tags::TimeStepId>(runner, id);
  {
    INFO("ComputeTimeDerivative: volume targets and the single exchange");
    next();
    const auto& inbox =
        ActionTesting::get_inbox_tag<component, inbox_tag>(runner, id);
    REQUIRE(inbox.messages.size() == 1);
    CHECK(inbox.messages.begin()->first == time_step_id);
    CHECK(inbox.messages.begin()->second.size() == 6);
    const auto& dt_vars = get_tag<dt_variables_tag>(runner, id);
    check_physical_rates_zero(dt_vars);
    const auto derivatives = partial_derivatives<system::gradient_variables>(
        part<system::gradient_variables>(get_tag<variables_tag>(runner, id)),
        mesh, inverse_jacobian);
    gp::Tags::ChartW::type dt_w{n};
    gp::Tags::ChartQ::type dt_q{n};
    gp::Tags::ChartU::type dt_u{n};
    gp::volume_targets(
        make_not_null(&dt_w), make_not_null(&dt_q), make_not_null(&dt_u),
        get_tag<gp::Tags::ChartU>(runner, id),
        get_tag<gp::Tags::ChartW>(runner, id),
        get_tag<gp::Tags::ChartQ>(runner, id),
        get<gp::DerivChartW>(derivatives), get<gp::DerivChartQ>(derivatives),
        get_tag<Ccz4::Tags::Kappa1>(runner, id),
        get_tag<Ccz4::Tags::Kappa2>(runner, id), kappa_3,
        get_tag<Ccz4::Tags::Eta<DataVector>>(runner, id),
        get_tag<Ccz4::Tags::K0<DataVector>>(runner, id), Ccz4::fd::System::f);
    CHECK_ITERABLE_APPROX(get<::Tags::dt<gp::Tags::ChartW>>(dt_vars), dt_w);
    CHECK_ITERABLE_APPROX(get<::Tags::dt<gp::Tags::ChartQ>>(dt_vars), dt_q);
    CHECK_ITERABLE_APPROX(get<::Tags::dt<gp::Tags::ChartU>>(dt_vars), dt_u);
  }
  const DtVars volume_dt = get_tag<dt_variables_tag>(runner, id);

  {
    INFO("ApplyBoundaryCorrections lifts the target corrections");
    next();
    CHECK(
        ActionTesting::get_inbox_tag<component, inbox_tag>(runner, id).empty());
    const auto& dt_vars = get_tag<dt_variables_tag>(runner, id);
    check_physical_rates_zero(dt_vars);
    const DtVars expected = expected_corrected_targets(
        volume_dt, get_tag<variables_tag>(runner, id), mesh, element_lengths,
        correction);
    const auto custom = Approx::custom().epsilon(1.0e-12).scale(1.0);
    CHECK_VARIABLES_CUSTOM_APPROX(dt_vars, expected, custom);
    CHECK(get<::Tags::dt<gp::Tags::ChartU>>(dt_vars) !=
          get<::Tags::dt<gp::Tags::ChartU>>(volume_dt));
  }
  const DtVars corrected_dt = get_tag<dt_variables_tag>(runner, id);

  {
    INFO("FinalizeTimeDerivative writes only the physical rates");
    next();
    const auto& dt_vars = get_tag<dt_variables_tag>(runner, id);
    using dt_chart_tags = db::wrap_tags_in<::Tags::dt, chart_tags>;
    CHECK(part<dt_chart_tags>(dt_vars) == part<dt_chart_tags>(corrected_dt));
    const auto finalize = [&](const Scalar<DataVector>& damping) {
      DtPhysicalVars result{n};
      gp::FinalizeTimeDerivative::apply(
          make_not_null(
              &get<::Tags::dt<Ccz4::Tags::ConformalMetric<DataVector, 3>>>(
                  result)),
          make_not_null(&get<::Tags::dt<Ccz4::Tags::ConformalFactor<DataVector>>>(
              result)),
          make_not_null(
              &get<::Tags::dt<Ccz4::Tags::ATilde<DataVector, 3>>>(result)),
          make_not_null(
              &get<::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataVector>>>(
                  result)),
          make_not_null(
              &get<::Tags::dt<Ccz4::Tags::Theta<DataVector>>>(result)),
          make_not_null(
              &get<::Tags::dt<Ccz4::Tags::GammaHat<DataVector, 3>>>(result)),
          make_not_null(&get<::Tags::dt<gr::Tags::Lapse<DataVector>>>(result)),
          make_not_null(
              &get<::Tags::dt<gr::Tags::Shift<DataVector, 3>>>(result)),
          make_not_null(
              &get<::Tags::dt<Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>>>(
                  result)),
          get_tag<gp::Tags::ChartU>(runner, id),
          get_tag<gp::Tags::ChartW>(runner, id),
          get_tag<gp::Tags::ChartQ>(runner, id),
          get<::Tags::dt<gp::Tags::ChartU>>(corrected_dt),
          get<::Tags::dt<gp::Tags::ChartW>>(corrected_dt),
          get<::Tags::dt<gp::Tags::ChartQ>>(corrected_dt), mesh, inverse_jacobian,
          operators, damping);
      return result;
    };
    const DtPhysicalVars expected = finalize(eta);
    CHECK(part<dt_physical_tags>(dt_vars) == expected);
    if (damped) {
      // Keep the volume damping and omit only the fitted-rate correction.
      const DtPhysicalVars without_correction =
          finalize(Scalar<DataVector>{n, 0.0});
      CHECK(get<::Tags::dt<Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>>>(
                without_correction) !=
            get<::Tags::dt<Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>>>(
                expected));
      CHECK(get<::Tags::dt<Ccz4::Tags::GammaHat<DataVector, 3>>>(
                without_correction) ==
            get<::Tags::dt<Ccz4::Tags::GammaHat<DataVector, 3>>>(expected));
    }
  }

  // The remaining actions of a step, checked on three consecutive steps: the
  // first uses the non-monotonic self-start history, the third the mature
  // equally spaced history.
  const auto custom = Approx::custom().epsilon(1.0e-12).scale(1.0);
  for (size_t step_number = 0; step_number < 3; ++step_number) {
    CAPTURE(step_number);
    if (step_number > 0) {
      // PrepareChartFields, ComputeTimeDerivative, ApplyBoundaryCorrections,
      // FinalizeTimeDerivative
      for (size_t i = 0; i < 4; ++i) {
        next();
      }
    }
    const double step_start = get_tag<::Tags::Time>(runner, id);
    CHECK(step_start == approx(static_cast<double>(step_number) * dt));
    const Vars state = get_tag<variables_tag>(runner, id);
    const DtVars rates = get_tag<dt_variables_tag>(runner, id);

    // RecordTimeStepperData records the current state and finalized rates.
    next();
    const history_tag::type history = get_tag<history_tag>(runner, id);
    REQUIRE(history.size() == order);
    CHECK(history.back().time_step_id ==
          get_tag<::Tags::TimeStepId>(runner, id));
    CHECK(*history.back().value == state);
    CHECK(history.back().derivative == rates);

    // RunEventsAndDenseTriggers observes the dense output of the physical
    // fields at the trigger time and restores the state and time.
    RecordState::calls.clear();
    next();
    if (step_number == 0) {
      REQUIRE(RecordState::calls.size() == 1);
      CHECK(RecordState::calls[0].first == trigger_time);
      CHECK_VARIABLES_CUSTOM_APPROX(
          part<physical_tags>(RecordState::calls[0].second),
          adams_physical_state(
              history,
              adams_weights(history_times(history), step_start, trigger_time)),
          custom);
    } else {
      CHECK(RecordState::calls.empty());
    }
    CHECK(get_tag<variables_tag>(runner, id) == state);
    CHECK(get_tag<::Tags::Time>(runner, id) == step_start);
    CHECK(get_tag<history_tag>(runner, id).size() == history.size());

    // UpdateU advances the physical fields with the recorded physical rates.
    next();
    const auto weights =
        adams_weights(history_times(history), step_start, step_start + dt);
    if (step_number == 2) {
      // The mature equally spaced history gives the textbook coefficients.
      CHECK(weights[0] == approx(5.0 / 12.0 * dt));
      CHECK(weights[1] == approx(-16.0 / 12.0 * dt));
      CHECK(weights[2] == approx(23.0 / 12.0 * dt));
    }
    CHECK_VARIABLES_CUSTOM_APPROX(
        part<physical_tags>(get_tag<variables_tag>(runner, id)),
        adams_physical_state(history, weights), custom);

    // CleanHistory
    next();
    const history_tag::type cleaned_history = get_tag<history_tag>(runner, id);
    CHECK(cleaned_history.size() == order - 1);

    // FilterAndEnforceConstraints acts on the advanced state only.
    const PhysicalVars before_filter =
        part<physical_tags>(get_tag<variables_tag>(runner, id));
    next();
    PhysicalVars expected = before_filter;
    tmpl::for_each<tmpl::list<Ccz4::Tags::ATilde<DataVector, 3>,
                              gr::Tags::TraceExtrinsicCurvature<DataVector>,
                              Ccz4::Tags::Theta<DataVector>,
                              Ccz4::Tags::GammaHat<DataVector, 3>,
                              Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>>>(
        [&expected, &operators]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
          for (auto& component : get<Tag>(expected)) {
            operators.apply_filter(make_not_null(&component));
          }
        });
    Ccz4::fd::enforce_algebraic_constraints(
        make_not_null(
            &get<Ccz4::Tags::ConformalMetric<DataVector, 3>>(expected)),
        make_not_null(&get<Ccz4::Tags::ATilde<DataVector, 3>>(expected)));
    const PhysicalVars after_filter =
        part<physical_tags>(get_tag<variables_tag>(runner, id));
    CHECK_VARIABLES_CUSTOM_APPROX(after_filter, expected, custom);
    // The filter is active, and leaves the conformal factor, lapse and shift
    // untouched.
    CHECK(get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(after_filter) !=
          get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(before_filter));
    CHECK(get<Ccz4::Tags::ConformalFactor<DataVector>>(after_filter) ==
          get<Ccz4::Tags::ConformalFactor<DataVector>>(before_filter));
    CHECK(get<gr::Tags::Lapse<DataVector>>(after_filter) ==
          get<gr::Tags::Lapse<DataVector>>(before_filter));
    CHECK(get<gr::Tags::Shift<DataVector, 3>>(after_filter) ==
          get<gr::Tags::Shift<DataVector, 3>>(before_filter));
    CHECK_ITERABLE_APPROX(
        get(determinant(
            get<Ccz4::Tags::ConformalMetric<DataVector, 3>>(after_filter))),
        DataVector(n, 1.0));
    // The history is not modified by the filter.
    const auto& history_after_filter = get_tag<history_tag>(runner, id);
    REQUIRE(history_after_filter.size() == cleaned_history.size());
    for (size_t i = 0; i < cleaned_history.size(); ++i) {
      CHECK(history_after_filter[i].value == cleaned_history[i].value);
      CHECK(history_after_filter[i].derivative ==
            cleaned_history[i].derivative);
    }

    // AdvanceTime
    next();
    CHECK(get_tag<::Tags::Time>(runner, id) == approx(step_start + dt));
  }
}

// Advances every element to the next step: each right-hand side of the
// linearization checks is evaluated on a new step, as in the evolution (the
// background is static).
void advance_time(const gsl::not_null<MockRuntimeSystem*> runner,
                  const std::vector<ElementId<3>>& ids) {
  for (const auto& id : ids) {
    runner->force_next_action_to_be<component,
                                    ::Actions::MutateApply<AdvanceTime<>>>(id);
    ActionTesting::next_action<component>(runner, id);
  }
}

// Sets the chart state Minkowski + delta on every element (delta holds the 13
// velocities and then the 10 potentials, component-major over `total` model
// points; model_index(e, s) is the model point of point s of element e), runs
// PrepareChartFields, ComputeTimeDerivative, ApplyBoundaryCorrections and
// FinalizeTimeDerivative, each on all elements before the next, returns the
// finalized physical rates and advances time.
template <typename ModelIndex>
std::vector<DtPhysicalVars> finalized_rates(
    const gsl::not_null<MockRuntimeSystem*> runner,
    const std::vector<ElementId<3>>& ids, const DataVector& delta,
    const size_t total, const ModelIndex& model_index) {
  const size_t n = number_of_points;
  for (size_t e = 0; e < ids.size(); ++e) {
    gp::Tags::ChartU::type u{n};
    gp::Tags::ChartW::type w{n};
    for (size_t s = 0; s < n; ++s) {
      for (size_t a = 0; a < gp::number_of_velocities; ++a) {
        w.get(a)[s] = delta[a * total + model_index(e, s)];
      }
      for (size_t a = 0; a < gp::number_of_potentials; ++a) {
        u.get(a)[s] =
            delta[(gp::number_of_velocities + a) * total + model_index(e, s)];
      }
    }
    u.get(gp::PotentialIndex::lapse) += 1.0;
    auto& box = ActionTesting::get_databox<component>(runner, ids[e]);
    db::mutate<variables_tag>(
        [&u, &w](const gsl::not_null<Vars*> vars) {
          *vars = physical_from_chart(u, w);
        },
        make_not_null(&box));
    runner->force_next_action_to_be<
        component, ::Actions::MutateApply<gp::PrepareChartFields>>(ids[e]);
  }
  for (size_t i = 0; i < 4; ++i) {
    for (const auto& id : ids) {
      REQUIRE(ActionTesting::next_action_if_ready<component>(runner, id));
    }
  }
  std::vector<DtPhysicalVars> result{};
  for (const auto& id : ids) {
    result.push_back(
        part<dt_physical_tags>(get_tag<dt_variables_tag>(*runner, id)));
  }
  advance_time(runner, ids);
  return result;
}

// The central difference of the finalized physical rates along delta about
// Minkowski space, mapped to chart rates at the background, in the layout of
// delta. The difference error is O(epsilon^2) relative to the rates.
template <typename ModelIndex>
DataVector linearized_chart_rates(
    const gsl::not_null<MockRuntimeSystem*> runner,
    const std::vector<ElementId<3>>& ids, const DataVector& delta,
    const size_t total, const ModelIndex& model_index) {
  const size_t n = number_of_points;
  // Truncation ~1e-10 and rounding ~1e-11 relative to the rates.
  const double epsilon = 3.0e-6;
  const gp::LogMetricChart background{gp::Matrix3(0.0)};
  auto derivatives =
      finalized_rates(runner, ids, epsilon * delta, total, model_index);
  const auto minus =
      finalized_rates(runner, ids, -epsilon * delta, total, model_index);
  DataVector chart_rates(23 * total);
  const size_t potentials = gp::number_of_velocities * total;
  for (size_t e = 0; e < ids.size(); ++e) {
    auto& derivative = derivatives[e];
    derivative -= minus[e];
    derivative *= 0.5 / epsilon;
    for (size_t s = 0; s < n; ++s) {
      const size_t m = model_index(e, s);
      gp::Matrix3 metric_rate{};
      gp::Matrix3 a_tilde_rate{};
      for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
          metric_rate(i, j) =
              get<::Tags::dt<Ccz4::Tags::ConformalMetric<DataVector, 3>>>(
                  derivative)
                  .get(i, j)[s];
          a_tilde_rate(i, j) =
              get<::Tags::dt<Ccz4::Tags::ATilde<DataVector, 3>>>(derivative)
                  .get(i, j)[s];
        }
      }
      const auto h_rate =
          gp::stf_encode(background.inverse_tangent(metric_rate));
      const auto a_rate =
          gp::stf_encode(2.0 * background.inverse_tangent(a_tilde_rate));
      for (size_t a = 0; a < 5; ++a) {
        chart_rates[(gp::VelocityIndex::a + a) * total + m] =
            gsl::at(a_rate, a);
        chart_rates[potentials + (gp::PotentialIndex::log_metric + a) * total +
                    m] = gsl::at(h_rate, a);
      }
      chart_rates[gp::VelocityIndex::trace_extrinsic_curvature * total + m] =
          get(get<::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataVector>>>(
              derivative))[s];
      chart_rates[gp::VelocityIndex::theta * total + m] =
          get(get<::Tags::dt<Ccz4::Tags::Theta<DataVector>>>(derivative))[s];
      // The background conformal factor is one.
      chart_rates[potentials +
                  gp::PotentialIndex::log_conformal_factor * total + m] =
          get(get<::Tags::dt<Ccz4::Tags::ConformalFactor<DataVector>>>(
              derivative))[s];
      chart_rates[potentials + gp::PotentialIndex::lapse * total + m] =
          get(get<::Tags::dt<gr::Tags::Lapse<DataVector>>>(derivative))[s];
      for (size_t i = 0; i < 3; ++i) {
        chart_rates[(gp::VelocityIndex::gamma_hat + i) * total + m] =
            get<::Tags::dt<Ccz4::Tags::GammaHat<DataVector, 3>>>(derivative)
                .get(i)[s];
        chart_rates[(gp::VelocityIndex::b + i) * total + m] =
            get<::Tags::dt<Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>>>(
                derivative)
                .get(i)[s];
        chart_rates[potentials + (gp::PotentialIndex::shift + i) * total + m] =
            get<::Tags::dt<gr::Tags::Shift<DataVector, 3>>>(derivative)
                .get(i)[s];
      }
    }
  }
  return chart_rates;
}

DataVector random_vector(const gsl::not_null<std::mt19937*> gen,
                         const size_t size) {
  std::normal_distribution<double> normal(0.0, 1.0);
  DataVector result(size);
  for (auto& value : result) {
    value = normal(*gen);
  }
  return result;
}

// The step actions up to the finalized rates reproduce the linear model of the
// scheme (GradientProjectionModel.py) about Minkowski space on two periodic
// elements along x (on a single periodic element the jumps of the potential
// means cancel): the central difference of the physical rates along random
// chart perturbations, mapped to chart rates at the background, equals the
// model operator applied to the perturbation, for every combination of the
// damping parameters kappa_1 and eta (the latter including the fitted shift
// damping of FinalizeTimeDerivative).
void test_linearized_operator(const gsl::not_null<std::mt19937*> gen) {
  constexpr size_t refinement_x = 1;
  constexpr size_t elements_x = 2;
  const size_t total = number_of_points * elements_x;
  const std::vector<ElementId<3>> ids{element_id(refinement_x, 0),
                                      element_id(refinement_x, 1)};
  // The model index of point s of element e (x fastest over both elements).
  const auto model_index = [](const size_t e, const size_t s) {
    return 3 * e + s % 3 + 3 * elements_x * (s / 3);
  };
  for (const auto& [kappa_1, eta] :
       std::array<std::array<double, 2>, 6>{{{{0.0, 0.0}},
                                             {{0.3, 0.0}},
                                             {{0.0, 0.1}},
                                             {{0.3, 0.1}},
                                             {{0.0, 1.0}},
                                             {{0.3, 1.0}}}}) {
    CAPTURE(kappa_1);
    CAPTURE(eta);
    const gp::BoundaryCorrections::UpwindGradientProjection correction{1.0};
    const auto creator = brick(refinement_x);
    MockRuntimeSystem runner{cache(3, creator, correction),
                             MutableCacheTuple{}};
    const Slab slab{0.0, 0.03};
    for (const auto& id : ids) {
      emplace_element(make_not_null(&runner), creator, id, TimeStepId{},
                      TimeStepId{true, 0, slab.start()}, slab.duration(),
                      kappa_1, DataVector{number_of_points, eta}, {});
    }
    ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Evolve);
    advance_time(make_not_null(&runner), ids);
    for (size_t trial = 0; trial < 2; ++trial) {
      CAPTURE(trial);
      const DataVector delta = random_vector(gen, 23 * total);
      const DataVector chart_rates = linearized_chart_rates(
          make_not_null(&runner), ids, delta, total, model_index);
      const auto expected = pypp::call<DataVector>(
          "GradientProjectionModel", "linearized_rates", delta,
          element_lengths[0], element_lengths[1], element_lengths[2], kappa_1,
          eta, Ccz4::fd::System::f, static_cast<double>(elements_x));
      const auto custom =
          Approx::custom().epsilon(1.0e-8).scale(max(abs(expected)));
      CHECK_ITERABLE_CUSTOM_APPROX(chart_rates, expected, custom);
    }
  }
}

// The six excised wedges of the Sphere with analytic Minkowski data on the
// inner and outer boundaries: one layer of curved elements with rotated
// neighbor orientations on the four angular faces of each wedge, and the
// analytic ghost closure on its two radial faces.
std::vector<ElementId<3>> sphere_elements(
    const gsl::not_null<MockRuntimeSystem*> runner,
    const DomainCreator<3>& creator, const double kappa_1, const double eta) {
  std::vector<ElementId<3>> ids{};
  const Slab slab{0.0, 0.03};
  for (size_t block = 0; block < 6; ++block) {
    ids.emplace_back(block);
    emplace_element(runner, creator, ids.back(), TimeStepId{},
                    TimeStepId{true, 0, slab.start()}, slab.duration(), kappa_1,
                    DataVector{number_of_points, eta}, {});
  }
  ActionTesting::set_phase(runner, Parallel::Phase::Evolve);
  advance_time(runner, ids);
  return ids;
}

// Minkowski space with the analytic Minkowski exterior is preserved: every
// finalized rate vanishes to rounding through the actual actions.
void test_sphere_minkowski(const double eta) {
  CAPTURE(eta);
  const auto creator = sphere();
  const gp::BoundaryCorrections::UpwindGradientProjection correction{1.0};
  MockRuntimeSystem runner{cache(3, creator, correction), MutableCacheTuple{}};
  const auto ids = sphere_elements(make_not_null(&runner), creator, 0.3, eta);
  const size_t total = 6 * number_of_points;
  const auto rates = finalized_rates(
      make_not_null(&runner), ids, DataVector(23 * total, 0.0), total,
      [](const size_t e, const size_t s) { return e * number_of_points + s; });
  double largest = 0.0;
  for (const auto& element_rates : rates) {
    for (size_t i = 0; i < element_rates.size(); ++i) {
      largest = std::max(largest, std::abs(element_rates.data()[i]));
    }
  }
  CAPTURE(largest);
  CHECK(largest < 1.0e-12);
}

// On the six wedges, the linearization of the actual actions about Minkowski
// space reproduces the curved-mesh linear model (independently assembling the
// physical derivatives, face normals and lifts from the map's inverse
// Jacobian, and pairing internal faces by node coordinates) including the
// analytic exterior closure, for two external dissipation factors and three
// values of eta. The errors on the radial (external) face nodes and on all
// other nodes are checked separately.
void test_sphere_linearized(const gsl::not_null<std::mt19937*> gen) {
  const auto creator = sphere();
  const double kappa_1 = 0.3;
  const size_t n = number_of_points;
  const size_t total = 6 * n;
  const auto model_index = [](const size_t e, const size_t s) {
    return e * number_of_points + s;
  };
  for (const auto& [external_factor, eta] :
       std::array<std::array<double, 2>, 6>{{{{1.0, 0.0}},
                                             {{2.0, 0.0}},
                                             {{1.0, 0.1}},
                                             {{2.0, 0.1}},
                                             {{1.0, 1.0}},
                                             {{2.0, 1.0}}}}) {
    CAPTURE(external_factor);
    CAPTURE(eta);
    const gp::BoundaryCorrections::UpwindGradientProjection correction{
        external_factor};
    MockRuntimeSystem runner{cache(3, creator, correction),
                             MutableCacheTuple{}};
    const auto ids =
        sphere_elements(make_not_null(&runner), creator, kappa_1, eta);
    DataVector geometry(total * 12);
    for (size_t e = 0; e < 6; ++e) {
      const auto& x = get_tag<domain::Tags::Coordinates<3, Frame::Inertial>>(
          runner, ids[e]);
      const auto& inverse_jacobian =
          get_tag<domain::Tags::InverseJacobian<3, Frame::ElementLogical,
                                                Frame::Inertial>>(runner,
                                                                  ids[e]);
      for (size_t s = 0; s < n; ++s) {
        const size_t offset = (e * n + s) * 12;
        for (size_t i = 0; i < 3; ++i) {
          geometry[offset + i] = x.get(i)[s];
          for (size_t a = 0; a < 3; ++a) {
            geometry[offset + 3 + 3 * a + i] = inverse_jacobian.get(a, i)[s];
          }
        }
      }
    }
    const DataVector delta = random_vector(gen, 23 * total);
    const DataVector chart_rates = linearized_chart_rates(
        make_not_null(&runner), ids, delta, total, model_index);
    const auto expected = pypp::call<DataVector>(
        "GradientProjectionModel", "linearized_rates_on_mesh", delta, geometry,
        3.0, 6.0, kappa_1, eta, Ccz4::fd::System::f, external_factor);
    const double scale = max(abs(expected));
    // The radial direction of each wedge is the third logical direction.
    double external_error = 0.0;
    double other_error = 0.0;
    for (size_t c = 0; c < 23; ++c) {
      for (size_t e = 0; e < 6; ++e) {
        for (size_t s = 0; s < n; ++s) {
          const size_t index = c * total + model_index(e, s);
          const double error = std::abs(chart_rates[index] - expected[index]);
          const size_t radial = s / 9;
          double& largest =
              radial == 0 or radial == 2 ? external_error : other_error;
          largest = std::max(largest, error);
        }
      }
    }
    CAPTURE(scale);
    CAPTURE(external_error);
    CAPTURE(other_error);
    CHECK(external_error < 1.0e-8 * scale);
    CHECK(other_error < 1.0e-8 * scale);
  }
}
}  // namespace

// [[TimeOut, 60]]
SPECTRE_TEST_CASE("Unit.Evolution.Systems.Ccz4.GradientProjection.StepActions",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env(
      "Evolution/Systems/Ccz4/GradientProjection/");
  domain::creators::register_derived_with_charm();
  register_factory_classes_with_charm<Metavariables>();
  register_classes_with_charm<TimeSteppers::AdamsBashforth>();
  register_classes_with_charm<Filters::runtime::None<3, variables_tags>>();
  MAKE_GENERATOR(gen);
  test_step_actions(false);
  test_step_actions(true);
  test_linearized_operator(make_not_null(&gen));
  test_sphere_minkowski(0.0);
  test_sphere_minkowski(1.0);
  test_sphere_linearized(make_not_null(&gen));
}
