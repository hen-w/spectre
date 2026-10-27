// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
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
#include "DataStructures/Index.hpp"
#include "DataStructures/SliceVariables.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Domain/CoordinateMaps/Distribution.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/NonconformingSphericalShells.hpp"
#include "Domain/Creators/Rectilinear.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/Creators/SphericalShells.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Creators/Tags/ExternalBoundaryConditions.hpp"
#include "Domain/Creators/Tags/FunctionsOfTime.hpp"
#include "Domain/Creators/Tags/InitialExtents.hpp"
#include "Domain/Creators/Tags/InitialRefinementLevels.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
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
#include "Evolution/DiscontinuousGalerkin/MortarData.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarTags.hpp"
#include "Evolution/DiscontinuousGalerkin/NormalVectorTags.hpp"
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
#include "Evolution/Systems/Ccz4/GradientProjection/PointwiseFields.hpp"
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
#include "NumericalAlgorithms/DiscontinuousGalerkin/LiftFlux.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags/Formulation.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/None.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Tag.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
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
using gradient_tags = tmpl::list<gp::Tags::PotentialGradients>;
using targets_tag = gp::Tags::RhsTargets;
using Targets = targets_tag::type;
using PotentialChart = gp::Tags::PotentialRateTarget::type;
using VelocityChart = gp::Tags::VelocityTarget::type;
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

// The GP boundary correction with a recording shim of the pre-lift filter
// hook: every call records its face mesh, connection flag and number of
// points, then forwards to the production hook.
class RecordingCorrection final : public evolution::BoundaryCorrection {
 public:
  struct Call {
    Mesh<2> face_mesh;
    bool has_nonconforming_block_connection;
    size_t number_of_points;
  };

  RecordingCorrection() = default;
  explicit RecordingCorrection(
      gp::BoundaryCorrections::UpwindGradientProjection correction)
      : correction_(std::move(correction)) {}
  explicit RecordingCorrection(CkMigrateMessage* const msg)
      : BoundaryCorrection(msg) {}
  using PUP::able::register_constructor;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
  WRAPPED_PUPable_decl_template(RecordingCorrection);  // NOLINT
#pragma GCC diagnostic pop

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override {
    BoundaryCorrection::pup(p);
    p | correction_;
  }

  std::unique_ptr<BoundaryCorrection> get_clone() const override {
    return std::make_unique<RecordingCorrection>(*this);
  }

  using Upwind = gp::BoundaryCorrections::UpwindGradientProjection;
  using dg_package_field_tags = Upwind::dg_package_field_tags;
  using dg_package_data_temporary_tags = Upwind::dg_package_data_temporary_tags;
  using dg_package_data_volume_tags = Upwind::dg_package_data_volume_tags;
  using dg_boundary_terms_volume_tags = Upwind::dg_boundary_terms_volume_tags;
  using dg_auxiliary_package_field_tags =
      Upwind::dg_auxiliary_package_field_tags;
  using dg_auxiliary_package_data_temporary_tags =
      Upwind::dg_auxiliary_package_data_temporary_tags;
  using dg_auxiliary_package_data_volume_tags =
      Upwind::dg_auxiliary_package_data_volume_tags;
  using dg_auxiliary_boundary_terms_volume_tags =
      Upwind::dg_auxiliary_boundary_terms_volume_tags;

  template <typename... Args>
    requires requires(const Upwind& correction, const Args&... args) {
      correction.dg_package_data(args...);
    }
  double dg_package_data(const Args&... args) const {
    return correction_.dg_package_data(args...);
  }

  template <bool ForExternalBoundary, typename... Args>
    requires requires(const Upwind& correction, const Args&... args) {
      correction.template dg_boundary_terms<ForExternalBoundary>(args...);
    }
  void dg_boundary_terms(const Args&... args) const {
    correction_.dg_boundary_terms<ForExternalBoundary>(args...);
  }

  void dg_filter_boundary_correction(
      const gsl::not_null<gp::Tags::RhsTargets::type*> corrections,
      const Mesh<2>& face_mesh,
      const bool has_nonconforming_block_connection) const {
    calls.push_back({face_mesh, has_nonconforming_block_connection,
                     corrections->number_of_grid_points()});
    correction_.dg_filter_boundary_correction(
        corrections, face_mesh, has_nonconforming_block_connection);
  }

  static std::vector<Call> calls;

 private:
  Upwind correction_{};
};

std::vector<RecordingCorrection::Call> RecordingCorrection::calls{};
PUP::able::PUP_ID RecordingCorrection::my_PUP_ID = 0;  // NOLINT

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
        tmpl::pair<evolution::BoundaryCorrection,
                   tmpl::list<gp::BoundaryCorrections::UpwindGradientProjection,
                              RecordingCorrection>>,
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
// algebraic-constraint surface. The stored gradients hold large finite values,
// which must be overwritten before use.
Vars physical_from_chart(const PotentialChart& u, const VelocityChart& w) {
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
        u_point[gp::PotentialIndex::conformal_factor];
    get(get<gr::Tags::Lapse<DataVector>>(vars))[s] =
        u_point[gp::PotentialIndex::lapse];
    get(get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(vars))[s] =
        w_point[gp::VelocityIndex::trace_extrinsic_curvature];
    get(get<Ccz4::Tags::Theta<DataVector>>(vars))[s] =
        w_point[gp::VelocityIndex::theta];
  }
  get<gp::Tags::PotentialGradients>(vars) =
      gp::Tags::PotentialGradients::type{n, 3.0e3};
  return vars;
}

// A smooth periodic perturbation of Minkowski plus a ramp along x, so that
// the traces on the two x faces of a periodic element genuinely differ (with
// unequal conformal factors on the two sides).
Vars initial_state(const tnsr::I<DataVector, 3>& x,
                   const std::array<double, 3>& lengths) {
  const size_t n = get<0>(x).size();
  const DataVector phase = 2.0 * M_PI *
                           (get<0>(x) / lengths[0] + get<1>(x) / lengths[1] -
                            get<2>(x) / lengths[2]);
  const DataVector ramp = get<0>(x) / lengths[0];
  PotentialChart u{n};
  VelocityChart w{n};
  for (size_t a = 0; a < gp::number_of_potentials; ++a) {
    u.get(a) = 0.05 * sin(phase + 0.3 * static_cast<double>(a)) +
               0.02 * (1.0 + 0.1 * static_cast<double>(a)) * ramp;
  }
  u.get(gp::PotentialIndex::lapse) += 1.0;
  u.get(gp::PotentialIndex::conformal_factor) += 1.0;
  for (size_t a = 0; a < gp::number_of_velocities; ++a) {
    w.get(a) = 0.04 * cos(phase + 0.7 * static_cast<double>(a)) - 0.03 * ramp;
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

void check_rates_zero(const DtVars& dt_vars) {
  double largest = 0.0;
  for (size_t i = 0; i < dt_vars.size(); ++i) {
    largest = std::max(largest, std::abs(dt_vars.data()[i]));
  }
  CHECK(largest == 0.0);
}

void check_gradient_rates_zero(const DtVars& dt_vars) {
  for (const auto& component :
       get<::Tags::dt<gp::Tags::PotentialGradients>>(dt_vars)) {
    CHECK(max(abs(component)) == 0.0);
  }
}

// The packaged chart fields of the face values of `vars` (evolved
// variables) with the given normal.
using PackagedVars = Variables<
    gp::BoundaryCorrections::UpwindGradientProjection::dg_package_field_tags>;
PackagedVars package_face(
    const Vars& face_vars, const tnsr::i<DataVector, 3>& normal,
    const gp::BoundaryCorrections::UpwindGradientProjection& correction) {
  PackagedVars packaged{face_vars.number_of_grid_points()};
  correction.dg_package_data(
      make_not_null(&get<gp::Tags::PackagedVelocities>(packaged)),
      make_not_null(&get<gp::Tags::PackagedGradients>(packaged)),
      make_not_null(&get<gp::Tags::PackagedPotentials>(packaged)),
      make_not_null(&get<Ccz4::Tags::NormalCovector<DataVector, 3>>(packaged)),
      get<Ccz4::Tags::ConformalMetric<DataVector, 3>>(face_vars),
      get<Ccz4::Tags::ConformalFactor<DataVector>>(face_vars),
      get<Ccz4::Tags::ATilde<DataVector, 3>>(face_vars),
      get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(face_vars),
      get<Ccz4::Tags::Theta<DataVector>>(face_vars),
      get<Ccz4::Tags::GammaHat<DataVector, 3>>(face_vars),
      get<gr::Tags::Lapse<DataVector>>(face_vars),
      get<gr::Tags::Shift<DataVector, 3>>(face_vars),
      get<Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>>(face_vars),
      get<gp::Tags::PotentialGradients>(face_vars), normal, std::nullopt,
      std::nullopt, Direction<3>::upper_xi());
  return packaged;
}

// The corrected targets expected from the volume targets: on every face of the
// periodic element the exterior state is the opposite face of the same
// element, the interior normal is the outward Euclidean unit covector, and the
// correction is lifted with the inverse LGL end weight 1 / w_0 = 3 (three
// points) times the face-normal magnitude 2 / L.
Targets expected_corrected_targets(
    const Targets& volume_targets, const Vars& vars, const Mesh<3>& mesh,
    const std::array<double, 3>& lengths,
    const gp::BoundaryCorrections::UpwindGradientProjection& correction) {
  REQUIRE(mesh.extents() == Index<3>(3));
  Targets result = volume_targets;
  const size_t last = mesh.extents(0) - 1;
  for (size_t axis = 0; axis < 3; ++axis) {
    for (const double sign : {-1.0, 1.0}) {
      const size_t index = sign > 0.0 ? last : 0;
      const size_t opposite = sign > 0.0 ? 0 : last;
      const Vars interior_vars =
          data_on_slice(vars, mesh.extents(), axis, index);
      const Vars exterior_vars =
          data_on_slice(vars, mesh.extents(), axis, opposite);
      const size_t face_points = interior_vars.number_of_grid_points();
      tnsr::i<DataVector, 3> normal{face_points, 0.0};
      normal.get(axis) = sign;
      tnsr::i<DataVector, 3> exterior_normal{face_points, 0.0};
      exterior_normal.get(axis) = -sign;
      const PackagedVars interior =
          package_face(interior_vars, normal, correction);
      const PackagedVars exterior =
          package_face(exterior_vars, exterior_normal, correction);
      Targets terms{face_points};
      correction.dg_boundary_terms<false>(
          make_not_null(&get<gp::Tags::VelocityTarget>(terms)),
          make_not_null(&get<gp::Tags::GradientTarget>(terms)),
          make_not_null(&get<gp::Tags::PotentialRateTarget>(terms)),
          get<gp::Tags::PackagedVelocities>(interior),
          get<gp::Tags::PackagedGradients>(interior),
          get<gp::Tags::PackagedPotentials>(interior),
          get<Ccz4::Tags::NormalCovector<DataVector, 3>>(interior),
          get<gp::Tags::PackagedVelocities>(exterior),
          get<gp::Tags::PackagedGradients>(exterior),
          get<gp::Tags::PackagedPotentials>(exterior),
          get<Ccz4::Tags::NormalCovector<DataVector, 3>>(exterior),
          dg::Formulation::StrongInertial);
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
    evolution::initial_data::Tags::InitialData, gp::Tags::TopShellFilterGain,
    gp::Tags::ShellAngularModesToKill>;
using MutableCacheTuple =
    tuples::TaggedTuple<domain::Tags::FunctionsOfTimeInitialize>;

const std::array<double, 3> element_lengths{{1.0, 1.3, 0.8}};
constexpr double kappa_3 = 0.5;
constexpr double top_shell_filter_gain = 0.3;
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

// The analytic Minkowski boundary condition.
std::unique_ptr<gp::BoundaryConditions::DirichletAnalytic>
analytic_minkowski() {
  return std::make_unique<gp::BoundaryConditions::DirichletAnalytic>(
      std::make_unique<Minkowski>());
}

// Two spherical-harmonic shells with radii 3, 4.5 and 6, three radial points
// and the smallest L the creator's options allow (6), with analytic Minkowski
// data on both boundaries.
domain::creators::SphericalShells two_shells() {
  return {3.0,
          6.0,
          size_t{0},
          size_t{3},
          size_t{6},
          std::vector<double>{4.5},
          domain::CoordinateMaps::Distribution::Linear,
          std::nullopt,
          analytic_minkowski(),
          analytic_minkowski()};
}

// Six excised wedges (equidistant angular maps, three LGL points per
// dimension) inside one spherical-harmonic shell with the given L and radial
// points, with analytic Minkowski data on the inner and outer boundaries.
domain::creators::NonconformingSphericalShells wedge_shell(
    const size_t l_max, const size_t radial_points,
    const size_t angular_refinement = 0) {
  return {1.0,
          1.5,
          2.0,
          std::vector<double>{},
          std::vector<double>{},
          std::array<std::vector<domain::CoordinateMaps::Distribution>, 2>{
              {{domain::CoordinateMaps::Distribution::Linear},
               {domain::CoordinateMaps::Distribution::Linear}}},
          size_t{0},
          angular_refinement,
          radial_points,
          l_max,
          size_t{3},
          domain::creators::NonconformingSphericalShells::Excision{
              analytic_minkowski()},
          false,
          analytic_minkowski()};
}

// The velocity filter of ComputeVolumeTargets is active (gain below one) only
// in the step walkthrough; the linearization checks of the right-hand side
// run at unit gain, where it is the identity on tensor-product elements (its
// arithmetic is tested in Test_Filter). On spherical-harmonic shells it still
// removes the angular content outside the retained space of
// `shell_angular_modes_to_kill` removed bands.
CacheTuple cache(const size_t order, const DomainCreator<3>& creator,
                 const evolution::BoundaryCorrection& correction,
                 const double filter_gain = 1.0,
                 const size_t shell_angular_modes_to_kill = 4) {
  return {creator.create_domain(),
          std::make_unique<TimeSteppers::AdamsBashforth>(order),
          kappa_3,
          EventsAndTriggers{},
          correction.get_clone(),
          ::dg::Formulation::StrongInertial,
          creator.external_boundary_conditions(),
          std::make_unique<Minkowski>(),
          filter_gain,
          shell_angular_modes_to_kill};
}

// Emplaces and initializes an element with the damping parameters kappa_1 and
// eta (the Gamma-driver damping, possibly varying over the points).
void emplace_element(
    const gsl::not_null<MockRuntimeSystem*> runner,
    const DomainCreator<3>& creator, const ElementId<3>& id,
    const TimeStepId& time_step_id, const TimeStepId& next_time_step_id,
    const TimeDelta& step, const double kappa_1, const DataVector& eta,
    EventsAndDenseTriggers::ConstructionType events_and_dense_triggers,
    const size_t n = number_of_points) {
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
  // Exercise the production initial-data action, including clearing the
  // stored gradients and assigning every physical field on bricks and wedges.
  const auto& vars = get_tag<variables_tag>(*runner, id);
  const auto exact = Minkowski{}.variables(
      get_tag<domain::Tags::Coordinates<3, Frame::Inertial>>(*runner, id), 0.0,
      physical_tags{});
  tmpl::for_each<physical_tags>(
      [&vars, &exact]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
        CHECK_ITERABLE_APPROX(get<Tag>(vars), get<Tag>(exact));
      });
  for (const auto& component : get<gp::Tags::PotentialGradients>(vars)) {
    CHECK(max(abs(component)) == 0.0);
  }
  // The right-hand-side targets and the chart buffers exist (empty) after
  // initialization.
  CHECK(get_tag<targets_tag>(*runner, id).number_of_grid_points() == 0);
  CHECK(
      get_tag<gp::Tags::StepChartFields>(*runner, id).number_of_grid_points() ==
      0);
  CHECK(get_tag<gp::Tags::StepCharts>(*runner, id).empty());
}

// The step actions, without and with a spatially varying Gamma-driver damping
// eta (which the volume targets and FinalizeTimeDerivative take from the
// DataBox).
void test_step_actions(const bool damped) {
  CAPTURE(damped);
  RecordState::calls.clear();
  const size_t order = 3;
  const gp::BoundaryCorrections::UpwindGradientProjection correction{
      gp::BoundaryCorrections::ExternalCorrection::Upwind, 1.0, 1.0, 1.0,
      std::nullopt};
  MockRuntimeSystem runner{
      cache(order, brick(0), correction, top_shell_filter_gain),
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
    // Poison the scratch gradients, the targets and the rates with large
    // finite values: they must be overwritten before use.
    auto& box =
        ActionTesting::get_databox<component>(make_not_null(&runner), id);
    db::mutate<gp::Tags::PotentialGradients, targets_tag, dt_variables_tag>(
        [n](const auto q, const auto targets, const auto dt_vars) {
          for (auto& component : *q) {
            component = 3.0e3;
          }
          *targets = Targets{n, -2.0e3};
          *dt_vars = DtVars{n, 1.0e3};
        },
        make_not_null(&box));
  }

  {
    INFO("ComputeVolumeTargets recomputes q and the volume targets");
    const Vars before = get_tag<variables_tag>(runner, id);
    next();
    // The same mutator on a copy of the physical state with unpoisoned
    // scratch (its arithmetic is tested in Test_PrepareChartFields).
    Vars expected_vars = before;
    get<gp::Tags::PotentialGradients>(expected_vars) =
        gp::Tags::PotentialGradients::type{n, 0.0};
    DtVars expected_dt{};
    Targets expected_targets{};
    gp::ComputeVolumeTargets::apply(
        make_not_null(&expected_vars), make_not_null(&expected_dt),
        make_not_null(&expected_targets), mesh, inverse_jacobian,
        get_tag<Ccz4::Tags::Kappa1>(runner, id),
        get_tag<Ccz4::Tags::Kappa2>(runner, id), kappa_3,
        get_tag<Ccz4::Tags::Eta<DataVector>>(runner, id),
        get_tag<Ccz4::Tags::K0<DataVector>>(runner, id), operators,
        top_shell_filter_gain);
    const Vars after = get_tag<variables_tag>(runner, id);
    CHECK(after == expected_vars);
    CHECK(get_tag<targets_tag>(runner, id) == expected_targets);
    // The potentials are referenced, not modified; the velocities are
    // filtered in place with the gain from the global cache (its arithmetic
    // is tested in Test_Filter).
    tmpl::for_each<tmpl::list<Ccz4::Tags::ConformalMetric<DataVector, 3>,
                              Ccz4::Tags::ConformalFactor<DataVector>,
                              gr::Tags::Lapse<DataVector>,
                              gr::Tags::Shift<DataVector, 3>>>(
        [&after, &before]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
          CHECK(get<Tag>(after) == get<Tag>(before));
        });
    tmpl::for_each<tmpl::list<Ccz4::Tags::ATilde<DataVector, 3>,
                              gr::Tags::TraceExtrinsicCurvature<DataVector>,
                              Ccz4::Tags::Theta<DataVector>,
                              Ccz4::Tags::GammaHat<DataVector, 3>,
                              Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>>>(
        [&after, &before]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
          CHECK(get<Tag>(after) != get<Tag>(before));
        });
    check_rates_zero(get_tag<dt_variables_tag>(runner, id));
    // The chart fields and charts of the (filtered) state, kept for
    // FinalizeTimeDerivative: the charts of the unchanged metric are
    // identical, the local a is the filtered a decoded from the physical
    // ATilde.
    gp::LocalChartFields expected_chart_fields{};
    std::vector<gp::LogMetricChart> expected_charts{};
    gp::local_chart_fields(
        make_not_null(&expected_chart_fields), make_not_null(&expected_charts),
        get<Ccz4::Tags::ConformalMetric<DataVector, 3>>(after),
        get<Ccz4::Tags::ConformalFactor<DataVector>>(after),
        get<Ccz4::Tags::ATilde<DataVector, 3>>(after),
        get<gr::Tags::Lapse<DataVector>>(after));
    CHECK_VARIABLES_CUSTOM_APPROX(
        get_tag<gp::Tags::StepChartFields>(runner, id), expected_chart_fields,
        Approx::custom().epsilon(1.0e-12).scale(1.0));
    CHECK(get_tag<gp::Tags::StepCharts>(runner, id) == expected_charts);
  }
  const Targets volume_targets = get_tag<targets_tag>(runner, id);

  const TimeStepId time_step_id = get_tag<::Tags::TimeStepId>(runner, id);
  {
    INFO("ComputeTimeDerivative: the single exchange, no generic volume terms");
    next();
    const auto& inbox =
        ActionTesting::get_inbox_tag<component, inbox_tag>(runner, id);
    REQUIRE(inbox.messages.size() == 1);
    CHECK(inbox.messages.begin()->first == time_step_id);
    CHECK(inbox.messages.begin()->second.size() == 6);
    check_rates_zero(get_tag<dt_variables_tag>(runner, id));
    // The periodic element has no external face: the targets are unchanged.
    CHECK(get_tag<targets_tag>(runner, id) == volume_targets);
  }

  {
    INFO("ApplyBoundaryCorrections lifts the corrections into the targets");
    next();
    CHECK(
        ActionTesting::get_inbox_tag<component, inbox_tag>(runner, id).empty());
    check_rates_zero(get_tag<dt_variables_tag>(runner, id));
    const Targets expected = expected_corrected_targets(
        volume_targets, get_tag<variables_tag>(runner, id), mesh,
        element_lengths, correction);
    const auto custom = Approx::custom().epsilon(1.0e-12).scale(1.0);
    CHECK_VARIABLES_CUSTOM_APPROX(get_tag<targets_tag>(runner, id), expected,
                                  custom);
    // The x faces carry genuine jumps, including of the potentials.
    CHECK(max(abs(get<gp::Tags::PotentialRateTarget>(
                      get_tag<targets_tag>(runner, id))
                      .get(gp::PotentialIndex::conformal_factor) -
                  get<gp::Tags::PotentialRateTarget>(volume_targets)
                      .get(gp::PotentialIndex::conformal_factor))) > 1.0e-3);
  }
  const Targets corrected_targets = get_tag<targets_tag>(runner, id);

  {
    INFO("FinalizeTimeDerivative writes only the physical rates");
    next();
    const auto& dt_vars = get_tag<dt_variables_tag>(runner, id);
    check_gradient_rates_zero(dt_vars);
    CHECK(get_tag<targets_tag>(runner, id) == corrected_targets);
    const auto& vars = get_tag<variables_tag>(runner, id);
    const auto finalize = [&](const Scalar<DataVector>& damping) {
      DtPhysicalVars result{n};
      gp::FinalizeTimeDerivative::apply(
          make_not_null(
              &get<::Tags::dt<Ccz4::Tags::ConformalMetric<DataVector, 3>>>(
                  result)),
          make_not_null(
              &get<::Tags::dt<Ccz4::Tags::ConformalFactor<DataVector>>>(
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
          get_tag<gp::Tags::StepChartFields>(runner, id),
          get_tag<gp::Tags::StepCharts>(runner, id),
          get<Ccz4::Tags::ConformalFactor<DataVector>>(vars),
          get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(vars),
          get<Ccz4::Tags::Theta<DataVector>>(vars),
          get<Ccz4::Tags::GammaHat<DataVector, 3>>(vars),
          get<gr::Tags::Lapse<DataVector>>(vars),
          get<gr::Tags::Shift<DataVector, 3>>(vars),
          get<Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>>(vars),
          get<gp::Tags::PotentialGradients>(vars),
          get<gp::Tags::PotentialRateTarget>(corrected_targets),
          get<gp::Tags::VelocityTarget>(corrected_targets),
          get<gp::Tags::GradientTarget>(corrected_targets), mesh,
          inverse_jacobian, operators, damping);
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
      // ComputeVolumeTargets, ComputeTimeDerivative, ApplyBoundaryCorrections,
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
    // The recorded rate of the stored gradients is defined and zero.
    for (size_t entry = 0; entry < history.size(); ++entry) {
      check_gradient_rates_zero(history[entry].derivative);
    }

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

    // EnforceAlgebraicConstraints projects the advanced state and nothing
    // else (the velocity filter runs with the next right-hand side).
    const PhysicalVars before_projection =
        part<physical_tags>(get_tag<variables_tag>(runner, id));
    next();
    PhysicalVars expected = before_projection;
    Ccz4::fd::enforce_algebraic_constraints(
        make_not_null(
            &get<Ccz4::Tags::ConformalMetric<DataVector, 3>>(expected)),
        make_not_null(&get<Ccz4::Tags::ATilde<DataVector, 3>>(expected)));
    const PhysicalVars after_projection =
        part<physical_tags>(get_tag<variables_tag>(runner, id));
    CHECK_VARIABLES_CUSTOM_APPROX(after_projection, expected, custom);
    // The projection is active, and leaves the other fields untouched.
    CHECK(get<Ccz4::Tags::ConformalMetric<DataVector, 3>>(after_projection) !=
          get<Ccz4::Tags::ConformalMetric<DataVector, 3>>(before_projection));
    tmpl::for_each<tmpl::list<
        Ccz4::Tags::ConformalFactor<DataVector>,
        gr::Tags::TraceExtrinsicCurvature<DataVector>,
        Ccz4::Tags::Theta<DataVector>, Ccz4::Tags::GammaHat<DataVector, 3>,
        gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>,
        Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>>>(
        [&after_projection,
         &before_projection]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
          CHECK(get<Tag>(after_projection) == get<Tag>(before_projection));
        });
    CHECK_ITERABLE_APPROX(
        get(determinant(
            get<Ccz4::Tags::ConformalMetric<DataVector, 3>>(after_projection))),
        DataVector(n, 1.0));
    // The history is not modified by the projection.
    const auto& history_after_projection = get_tag<history_tag>(runner, id);
    REQUIRE(history_after_projection.size() == cleaned_history.size());
    for (size_t i = 0; i < cleaned_history.size(); ++i) {
      CHECK(history_after_projection[i].value == cleaned_history[i].value);
      CHECK(history_after_projection[i].derivative ==
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

// The constant background state of the linearizations: unit conformal
// metric, w = 0, and the given conformal factor, lapse and shift.
struct Background {
  double conformal_factor = 1.0;
  double lapse = 1.0;
  std::array<double, 3> shift{};
};

// Sets the chart state background + delta on every element (delta holds the
// 13 velocities and then the 10 potentials, component-major over `total`
// model points; model_index(e, s) is the model point of point s of element
// e), runs ComputeVolumeTargets, ComputeTimeDerivative,
// ApplyBoundaryCorrections and FinalizeTimeDerivative, each on all elements
// before the next, returns the finalized physical rates and advances time.
template <typename ModelIndex>
std::vector<DtPhysicalVars> finalized_rates(
    const gsl::not_null<MockRuntimeSystem*> runner,
    const std::vector<ElementId<3>>& ids, const DataVector& delta,
    const size_t total, const ModelIndex& model_index,
    const Background& background = {}) {
  for (size_t e = 0; e < ids.size(); ++e) {
    const size_t n =
        get_tag<domain::Tags::Mesh<3>>(*runner, ids[e]).number_of_grid_points();
    PotentialChart u{n};
    VelocityChart w{n};
    for (size_t s = 0; s < n; ++s) {
      for (size_t a = 0; a < gp::number_of_velocities; ++a) {
        w.get(a)[s] = delta[a * total + model_index(e, s)];
      }
      for (size_t a = 0; a < gp::number_of_potentials; ++a) {
        u.get(a)[s] =
            delta[(gp::number_of_velocities + a) * total + model_index(e, s)];
      }
    }
    u.get(gp::PotentialIndex::conformal_factor) += background.conformal_factor;
    u.get(gp::PotentialIndex::lapse) += background.lapse;
    for (size_t i = 0; i < 3; ++i) {
      u.get(gp::PotentialIndex::shift + i) += gsl::at(background.shift, i);
    }
    auto& box = ActionTesting::get_databox<component>(runner, ids[e]);
    db::mutate<variables_tag>(
        [&u, &w](const gsl::not_null<Vars*> vars) {
          *vars = physical_from_chart(u, w);
        },
        make_not_null(&box));
    runner->force_next_action_to_be<
        component, ::Actions::MutateApply<gp::ComputeVolumeTargets>>(ids[e]);
  }
  for (size_t i = 0; i < 4; ++i) {
    for (const auto& id : ids) {
      CAPTURE(i);
      CAPTURE(id);
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
// the background, mapped to chart rates at the background (unit conformal
// metric), in the layout of delta. The difference error is O(epsilon^2)
// relative to the rates.
template <typename ModelIndex>
DataVector linearized_chart_rates(
    const gsl::not_null<MockRuntimeSystem*> runner,
    const std::vector<ElementId<3>>& ids, const DataVector& delta,
    const size_t total, const ModelIndex& model_index,
    const Background& background_state = {}) {
  // Truncation ~1e-10 and rounding ~1e-11 relative to the rates.
  const double epsilon = 3.0e-6;
  const gp::LogMetricChart background{gp::Matrix3(0.0)};
  auto derivatives = finalized_rates(runner, ids, epsilon * delta, total,
                                     model_index, background_state);
  const auto minus = finalized_rates(runner, ids, -epsilon * delta, total,
                                     model_index, background_state);
  DataVector chart_rates(23 * total);
  const size_t potentials = gp::number_of_velocities * total;
  for (size_t e = 0; e < ids.size(); ++e) {
    auto& derivative = derivatives[e];
    derivative -= minus[e];
    derivative *= 0.5 / epsilon;
    for (size_t s = 0; s < derivative.number_of_grid_points(); ++s) {
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
      // The conformal factor is itself a potential.
      chart_rates[potentials + gp::PotentialIndex::conformal_factor * total +
                  m] =
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
// damping of FinalizeTimeDerivative) at the scalar jump penalty sigma = 1, and
// for sigma = 0 and 2.5 at one of them.
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
  for (const auto& [kappa_1, eta, sigma] :
       std::array<std::array<double, 3>, 8>{{{{0.0, 0.0, 1.0}},
                                             {{0.3, 0.0, 1.0}},
                                             {{0.0, 0.1, 1.0}},
                                             {{0.3, 0.1, 1.0}},
                                             {{0.0, 1.0, 1.0}},
                                             {{0.3, 1.0, 1.0}},
                                             {{0.3, 0.1, 0.0}},
                                             {{0.3, 0.1, 2.5}}}}) {
    CAPTURE(kappa_1);
    CAPTURE(eta);
    CAPTURE(sigma);
    const gp::BoundaryCorrections::UpwindGradientProjection correction{
        gp::BoundaryCorrections::ExternalCorrection::Upwind, sigma, sigma,
        sigma, std::nullopt};
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
          eta, Ccz4::fd::System::f, static_cast<double>(elements_x), sigma);
      const auto custom =
          Approx::custom().epsilon(1.0e-8).scale(max(abs(expected)));
      CHECK_ITERABLE_CUSTOM_APPROX(chart_rates, expected, custom);
    }
  }
}

// The same linearization about a constant state with a non-unit conformal
// factor and lapse and a nonzero shift, against the model assembled from the
// frozen log(phi) symbol conjugated by the scaling of the gradients of phi
// (and the lapse-weighted damping), for every combination of zero and nonzero
// kappa_1 and eta at the scalar jump penalty sigma = 1, and for sigma = 0 and
// 2.5 at one of them. About phi = 1 a missing factor of phi in R_phi, FieldP,
// P or the normal operator would be invisible.
void test_linearized_operator_about_background(
    const gsl::not_null<std::mt19937*> gen) {
  constexpr size_t refinement_x = 1;
  constexpr size_t elements_x = 2;
  const size_t total = number_of_points * elements_x;
  const std::vector<ElementId<3>> ids{element_id(refinement_x, 0),
                                      element_id(refinement_x, 1)};
  const auto model_index = [](const size_t e, const size_t s) {
    return 3 * e + s % 3 + 3 * elements_x * (s / 3);
  };
  const Background background{0.8, 0.7, {{0.05, -0.03, 0.02}}};
  for (const auto& [kappa_1, eta, sigma] :
       std::array<std::array<double, 3>, 6>{{{{0.0, 0.0, 1.0}},
                                             {{0.3, 0.0, 1.0}},
                                             {{0.0, 1.0, 1.0}},
                                             {{0.3, 1.0, 1.0}},
                                             {{0.3, 1.0, 0.0}},
                                             {{0.3, 1.0, 2.5}}}}) {
    CAPTURE(kappa_1);
    CAPTURE(eta);
    CAPTURE(sigma);
    const gp::BoundaryCorrections::UpwindGradientProjection correction{
        gp::BoundaryCorrections::ExternalCorrection::Upwind, sigma, sigma,
        sigma, std::nullopt};
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
    const DataVector delta = random_vector(gen, 23 * total);
    const DataVector chart_rates = linearized_chart_rates(
        make_not_null(&runner), ids, delta, total, model_index, background);
    const auto expected = pypp::call<DataVector>(
        "GradientProjectionModel", "linearized_rates_about", delta,
        element_lengths[0], element_lengths[1], element_lengths[2], kappa_1,
        eta, Ccz4::fd::System::f, static_cast<double>(elements_x),
        DataVector{background.conformal_factor, background.lapse,
                   background.shift[0], background.shift[1],
                   background.shift[2]},
        sigma);
    const auto custom =
        Approx::custom().epsilon(1.0e-8).scale(max(abs(expected)));
    CHECK_ITERABLE_CUSTOM_APPROX(chart_rates, expected, custom);
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
  const gp::BoundaryCorrections::UpwindGradientProjection correction{
      gp::BoundaryCorrections::ExternalCorrection::Upwind, 1.0, 1.0, 1.0,
      std::nullopt};
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

// On the wedges with a perturbed state, the external analytic boundary lift
// of ComputeTimeDerivative changes the velocity and gradient targets only on
// the external (radial) face points and never the potential-rate input (an
// external face has no value jump), and no time derivative.
void test_external_lift_isolation(const gsl::not_null<std::mt19937*> gen) {
  const auto creator = sphere();
  const gp::BoundaryCorrections::UpwindGradientProjection correction{
      gp::BoundaryCorrections::ExternalCorrection::Upwind, 1.0, 1.0, 1.0,
      std::nullopt};
  MockRuntimeSystem runner{cache(3, creator, correction), MutableCacheTuple{}};
  const auto ids = sphere_elements(make_not_null(&runner), creator, 0.3, 0.4);
  const size_t n = number_of_points;
  std::normal_distribution<double> normal(0.0, 1.0e-3);
  for (const auto& id : ids) {
    PotentialChart u{n};
    VelocityChart w{n};
    for (auto& component : u) {
      for (auto& value : component) {
        value = normal(*gen);
      }
    }
    for (auto& component : w) {
      for (auto& value : component) {
        value = normal(*gen);
      }
    }
    u.get(gp::PotentialIndex::conformal_factor) += 1.0;
    u.get(gp::PotentialIndex::lapse) += 1.0;
    auto& box =
        ActionTesting::get_databox<component>(make_not_null(&runner), id);
    db::mutate<variables_tag>(
        [&u, &w](const gsl::not_null<Vars*> vars) {
          *vars = physical_from_chart(u, w);
        },
        make_not_null(&box));
    runner.force_next_action_to_be<
        component, ::Actions::MutateApply<gp::ComputeVolumeTargets>>(id);
    ActionTesting::next_action<component>(make_not_null(&runner), id);
  }
  for (const auto& id : ids) {
    const Targets volume_targets = get_tag<targets_tag>(runner, id);
    ActionTesting::next_action<component>(make_not_null(&runner), id);
    const Targets& targets = get_tag<targets_tag>(runner, id);
    CHECK(get<gp::Tags::PotentialRateTarget>(targets) ==
          get<gp::Tags::PotentialRateTarget>(volume_targets));
    check_rates_zero(get_tag<dt_variables_tag>(runner, id));
    double largest_external_change = 0.0;
    for (size_t c = 0; c < targets.number_of_independent_components; ++c) {
      for (size_t s = 0; s < n; ++s) {
        const size_t index = c * n + s;
        const double change =
            std::abs(targets.data()[index] - volume_targets.data()[index]);
        // The radial direction of each wedge is the third logical direction.
        const size_t radial = s / 9;
        if (radial == 1) {
          CHECK(change == 0.0);
        } else {
          largest_external_change = std::max(largest_external_change, change);
        }
      }
    }
    CHECK(largest_external_change > 0.0);
  }
}

// On the six wedges, the linearization of the actual actions about Minkowski
// space reproduces the curved-mesh linear model (independently assembling the
// physical derivatives, face normals and lifts from the map's inverse
// Jacobian, and pairing internal faces by node coordinates) including the
// analytic exterior closure, for every combination of the two options of the
// boundary correction (two external dissipation factors and three internal
// scalar jump penalties sigma) at eta = 0.1, and for three values of eta at
// sigma = 1. The errors on the radial (external) face nodes and on all other
// nodes are checked separately.
void test_sphere_linearized(const gsl::not_null<std::mt19937*> gen) {
  const auto creator = sphere();
  const double kappa_1 = 0.3;
  const size_t n = number_of_points;
  const size_t total = 6 * n;
  const auto model_index = [](const size_t e, const size_t s) {
    return e * number_of_points + s;
  };
  for (const auto& [external_factor, eta, sigma] :
       std::array<std::array<double, 3>, 10>{{{{1.0, 0.0, 1.0}},
                                              {{2.0, 0.0, 1.0}},
                                              {{1.0, 0.1, 1.0}},
                                              {{2.0, 0.1, 1.0}},
                                              {{1.0, 1.0, 1.0}},
                                              {{2.0, 1.0, 1.0}},
                                              {{1.0, 0.1, 0.0}},
                                              {{2.0, 0.1, 0.0}},
                                              {{1.0, 0.1, 2.5}},
                                              {{2.0, 0.1, 2.5}}}}) {
    CAPTURE(external_factor);
    CAPTURE(eta);
    CAPTURE(sigma);
    const gp::BoundaryCorrections::UpwindGradientProjection correction{
        external_factor == 1.0
            ? gp::BoundaryCorrections::ExternalCorrection::Upwind
            : gp::BoundaryCorrections::ExternalCorrection::DoubledDissipation,
        sigma, sigma, sigma, std::nullopt};
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
        3.0, 6.0, kappa_1, eta, Ccz4::fd::System::f, external_factor, sigma);
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

// An option setting of the boundary correction and the damping for the
// linearization checks on shell domains: the external dissipation factor,
// eta, the internal scalar penalties of the propagating, stationary gradient
// and stationary GammaHat-b modes, and ShellBoundaryModesToKill.
struct LinearizationOptions {
  double external_factor;
  double eta;
  std::array<double, 3> penalties;
  std::optional<size_t> shell_boundary_modes_to_kill;
};

// On spherical-harmonic shells the velocity filter of ComputeVolumeTargets
// is not the identity at unit gain, so the model applies it too, with this
// many removed angular bands (fewer than the default four, so that most of
// the angular content of the perturbation is retained).
constexpr size_t linearization_shell_angular_modes_to_kill = 1;

// On every element of the domain, the linearization of the actual actions
// about Minkowski space reproduces the general curved-mesh linear model
// (GradientProjectionModel.linearized_rates_on_general_mesh), which
// independently assembles the spherical-harmonic derivatives, the shell
// gradient fit with its kernel completion, the shell velocity filter, the
// characteristic projectors of the split scalar penalties, the native
// wedge-shell transfers and the shell-side boundary-correction filter from
// the nodal coordinates and inverse Jacobians. The errors on the points of
// the external faces (at the inner and outer radius) and on all other points
// are checked separately.
void check_linearized_on_domain(
    const gsl::not_null<std::mt19937*> gen, const DomainCreator<3>& creator,
    const LinearizationOptions& options, const double inner_radius,
    const double outer_radius,
    const bool require_multiple_contributors = false) {
  const double kappa_1 = 0.3;
  const auto& [external_factor, eta, penalties, boundary_modes_to_kill] =
      options;
  CAPTURE(external_factor);
  CAPTURE(eta);
  CAPTURE(penalties);
  // The removed bands of the shell-side boundary filter, -1 without it.
  const double boundary_modes =
      boundary_modes_to_kill.has_value()
          ? static_cast<double>(*boundary_modes_to_kill)
          : -1.0;
  CAPTURE(boundary_modes);
  const gp::BoundaryCorrections::UpwindGradientProjection correction{
      external_factor == 1.0
          ? gp::BoundaryCorrections::ExternalCorrection::Upwind
          : gp::BoundaryCorrections::ExternalCorrection::DoubledDissipation,
      penalties[0], penalties[1], penalties[2], boundary_modes_to_kill};
  MockRuntimeSystem runner{cache(3, creator, correction, 1.0,
                                 linearization_shell_angular_modes_to_kill),
                           MutableCacheTuple{}};
  const auto ids = initial_element_ids(creator.initial_refinement_levels());
  const Slab slab{0.0, 0.03};
  std::vector<size_t> offsets{0};
  for (const auto& id : ids) {
    const auto extents = creator.initial_extents()[id.block_id()];
    const size_t n = extents[0] * extents[1] * extents[2];
    emplace_element(make_not_null(&runner), creator, id, TimeStepId{},
                    TimeStepId{true, 0, slab.start()}, slab.duration(), kappa_1,
                    DataVector{n, eta}, {}, n);
    offsets.push_back(offsets.back() + n);
  }
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Evolve);
  advance_time(make_not_null(&runner), ids);
  const size_t total = offsets.back();
  DataVector geometry(total * 12);
  DataVector layout(4 * ids.size());
  std::vector<bool> on_external_face(total);
  for (size_t e = 0; e < ids.size(); ++e) {
    const auto& mesh = get_tag<domain::Tags::Mesh<3>>(runner, ids[e]);
    layout[4 * e] =
        mesh.basis(1) == Spectral::Basis::SphericalHarmonic ? 1.0 : 0.0;
    for (size_t d = 0; d < 3; ++d) {
      layout[4 * e + 1 + d] = static_cast<double>(mesh.extents(d));
    }
    const auto& x =
        get_tag<domain::Tags::Coordinates<3, Frame::Inertial>>(runner, ids[e]);
    const auto& inverse_jacobian =
        get_tag<domain::Tags::InverseJacobian<3, Frame::ElementLogical,
                                              Frame::Inertial>>(runner, ids[e]);
    for (size_t s = 0; s < mesh.number_of_grid_points(); ++s) {
      const size_t offset = (offsets[e] + s) * 12;
      for (size_t i = 0; i < 3; ++i) {
        geometry[offset + i] = x.get(i)[s];
        for (size_t a = 0; a < 3; ++a) {
          geometry[offset + 3 + 3 * a + i] = inverse_jacobian.get(a, i)[s];
        }
      }
      const double radius = std::sqrt(
          square(get<0>(x)[s]) + square(get<1>(x)[s]) + square(get<2>(x)[s]));
      on_external_face[offsets[e] + s] =
          std::abs(radius - inner_radius) < 1.0e-10 or
          std::abs(radius - outer_radius) < 1.0e-10;
    }
  }
  const auto model_index = [&offsets](const size_t e, const size_t s) {
    return offsets[e] + s;
  };
  const DataVector delta = random_vector(gen, 23 * total);
  const DataVector chart_rates = linearized_chart_rates(
      make_not_null(&runner), ids, delta, total, model_index);
  const auto expected = pypp::call<DataVector>(
      "GradientProjectionModel", "linearized_rates_on_general_mesh", delta,
      geometry, layout, kappa_1, eta, Ccz4::fd::System::f, external_factor,
      DataVector{penalties[0], penalties[1], penalties[2]},
      static_cast<double>(linearization_shell_angular_modes_to_kill),
      boundary_modes, require_multiple_contributors ? 1.0 : 0.0);
  const double scale = max(abs(expected));
  double external_error = 0.0;
  double other_error = 0.0;
  for (size_t c = 0; c < 23; ++c) {
    for (size_t m = 0; m < total; ++m) {
      const size_t index = c * total + m;
      const double error = std::abs(chart_rates[index] - expected[index]);
      double& largest = on_external_face[m] ? external_error : other_error;
      largest = std::max(largest, error);
    }
  }
  CAPTURE(scale);
  CAPTURE(external_error);
  CAPTURE(other_error);
  CHECK(external_error < 1.0e-8 * scale);
  CHECK(other_error < 1.0e-8 * scale);
}

// Two spherical-harmonic shells: the shell derivatives, gradient fit and
// velocity filter, the conforming shell-shell interface and the analytic
// closure on both boundaries, for both external dissipation factors with
// equal, split and unequal scalar penalties at eta = 0, and at eta = 0.1.
void test_two_shell_linearized(const gsl::not_null<std::mt19937*> gen) {
  const auto creator = two_shells();
  for (const auto& options : std::array<LinearizationOptions, 10>{
           {{1.0, 0.0, {{1.0, 1.0, 1.0}}, std::nullopt},
            {2.0, 0.0, {{1.0, 1.0, 1.0}}, std::nullopt},
            {1.0, 0.0, {{1.0, 1.0, 0.0}}, std::nullopt},
            {2.0, 0.0, {{1.0, 1.0, 0.0}}, std::nullopt},
            {1.0, 0.0, {{0.0, 0.0, 0.0}}, std::nullopt},
            {2.0, 0.0, {{0.0, 0.0, 0.0}}, std::nullopt},
            {1.0, 0.0, {{0.5, 2.0, 1.5}}, std::nullopt},
            {2.0, 0.0, {{0.5, 2.0, 1.5}}, std::nullopt},
            {1.0, 0.1, {{1.0, 1.0, 0.0}}, std::nullopt},
            {2.0, 0.1, {{1.0, 1.0, 1.0}}, std::nullopt}}}) {
    check_linearized_on_domain(gen, creator, options, 3.0, 6.0);
  }
}

// Six wedges inside one spherical-harmonic shell: additionally the
// nonconforming wedge-shell exchange (the native interpolations in both
// directions, combined on the single host-labeled shell mortar) and the
// shell-side boundary-correction filter, without and with it, for both
// external dissipation factors with equal, split and unequal scalar
// penalties at eta = 0, and at eta = 0.1.
void test_wedge_shell_linearized(const gsl::not_null<std::mt19937*> gen) {
  const auto creator = wedge_shell(4, 3);
  for (const auto& options : std::array<LinearizationOptions, 12>{
           {{1.0, 0.0, {{1.0, 1.0, 1.0}}, std::nullopt},
            {2.0, 0.0, {{1.0, 1.0, 1.0}}, std::nullopt},
            {1.0, 0.0, {{1.0, 1.0, 0.0}}, std::nullopt},
            {2.0, 0.0, {{1.0, 1.0, 0.0}}, std::nullopt},
            {1.0, 0.0, {{0.0, 0.0, 0.0}}, std::nullopt},
            {2.0, 0.0, {{0.5, 2.0, 1.5}}, std::nullopt},
            {1.0, 0.0, {{1.0, 1.0, 1.0}}, size_t{2}},
            {1.0, 0.0, {{1.0, 1.0, 0.0}}, size_t{2}},
            {2.0, 0.0, {{1.0, 1.0, 0.0}}, size_t{2}},
            {1.0, 0.0, {{0.5, 2.0, 1.5}}, size_t{2}},
            {1.0, 0.1, {{1.0, 1.0, 0.0}}, size_t{2}},
            {2.0, 0.1, {{1.0, 1.0, 1.0}}, std::nullopt}}}) {
    check_linearized_on_domain(gen, creator, options, 1.0, 2.0);
  }
}

// Angularly refined wedges (2x2 patches per wedge) inside the shell: the
// refined patch boundaries lie on the phi = 0 mod pi/2 planes and the
// equator, which contain shell collocation points at even L, so single
// shell points receive data from two (and at the crossings four) wedge
// patches: the multi-contributor host-mortar averaging genuinely combines
// contributions (the model asserts this), on top of everything the
// unrefined test checks.
void test_refined_wedge_shell_linearized(
    const gsl::not_null<std::mt19937*> gen) {
  const auto creator = wedge_shell(4, 3, 1);
  check_linearized_on_domain(
      gen, creator, LinearizationOptions{1.0, 0.0, {{1.0, 1.0, 0.0}}, 2}, 1.0,
      2.0, true);
}

// A random chart state about Minkowski space with perturbations of size
// `size`, as physical fields.
Vars random_state(const gsl::not_null<std::mt19937*> gen, const size_t n,
                  const double size) {
  std::normal_distribution<double> normal(0.0, size);
  PotentialChart u{n};
  VelocityChart w{n};
  for (auto& component : u) {
    for (auto& value : component) {
      value = normal(*gen);
    }
  }
  for (auto& component : w) {
    for (auto& value : component) {
      value = normal(*gen);
    }
  }
  u.get(gp::PotentialIndex::conformal_factor) += 1.0;
  u.get(gp::PotentialIndex::lapse) += 1.0;
  return physical_from_chart(u, w);
}

// The pre-lift hook of the boundary correction is called exactly once per
// mortar in ApplyBoundaryCorrections, with the face mesh and the conforming
// flag of the block connection: on the conforming faces of two periodic brick
// elements it is told so, and the corrections lifted into the targets are
// bitwise those without the hook, even with an active shell cutoff.
void test_hook_invocation_conforming(const gsl::not_null<std::mt19937*> gen) {
  constexpr size_t refinement_x = 1;
  const std::vector<ElementId<3>> ids{element_id(refinement_x, 0),
                                      element_id(refinement_x, 1)};
  const auto creator = brick(refinement_x);
  const gp::BoundaryCorrections::UpwindGradientProjection plain{
      gp::BoundaryCorrections::ExternalCorrection::Upwind, 1.0, 1.0, 0.0,
      std::nullopt};
  const RecordingCorrection recording{
      gp::BoundaryCorrections::UpwindGradientProjection{
          gp::BoundaryCorrections::ExternalCorrection::Upwind, 1.0, 1.0, 0.0,
          size_t{2}}};
  std::vector<Vars> states{};
  for (size_t e = 0; e < ids.size(); ++e) {
    states.push_back(random_state(gen, number_of_points, 1.0e-2));
  }
  const auto corrected_targets = [&creator, &ids, &states](
                                     const evolution::BoundaryCorrection&
                                         correction,
                                     const bool recorded) {
    MockRuntimeSystem runner{cache(3, creator, correction),
                             MutableCacheTuple{}};
    const Slab slab{0.0, 0.03};
    for (const auto& id : ids) {
      emplace_element(make_not_null(&runner), creator, id, TimeStepId{},
                      TimeStepId{true, 0, slab.start()}, slab.duration(), 0.3,
                      DataVector{number_of_points, 0.1}, {});
    }
    ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Evolve);
    advance_time(make_not_null(&runner), ids);
    for (size_t e = 0; e < ids.size(); ++e) {
      auto& box =
          ActionTesting::get_databox<component>(make_not_null(&runner), ids[e]);
      db::mutate<variables_tag>(
          [&states, e](const gsl::not_null<Vars*> vars) { *vars = states[e]; },
          make_not_null(&box));
      runner.force_next_action_to_be<
          component, ::Actions::MutateApply<gp::ComputeVolumeTargets>>(ids[e]);
    }
    // ComputeVolumeTargets and ComputeTimeDerivative on all elements.
    for (size_t i = 0; i < 2; ++i) {
      for (const auto& id : ids) {
        REQUIRE(ActionTesting::next_action_if_ready<component>(
            make_not_null(&runner), id));
      }
    }
    std::vector<Targets> result{};
    for (const auto& id : ids) {
      RecordingCorrection::calls.clear();
      REQUIRE(ActionTesting::next_action_if_ready<component>(
          make_not_null(&runner), id));
      if (recorded) {
        // Six periodic conforming faces, one mortar each.
        REQUIRE(RecordingCorrection::calls.size() == 6);
        for (const auto& call : RecordingCorrection::calls) {
          CHECK_FALSE(call.has_nonconforming_block_connection);
          CHECK(call.face_mesh == Mesh<2>{3, Spectral::Basis::Legendre,
                                          Spectral::Quadrature::GaussLobatto});
          CHECK(call.number_of_points == 9);
        }
      } else {
        CHECK(RecordingCorrection::calls.empty());
      }
      result.push_back(get_tag<targets_tag>(runner, id));
    }
    return result;
  };
  const auto expected = corrected_targets(plain, false);
  const auto actual = corrected_targets(recording, true);
  REQUIRE(actual.size() == expected.size());
  for (size_t e = 0; e < actual.size(); ++e) {
    CHECK(actual[e] == expected[e]);
  }
}

// On the spherical-harmonic shell of a wedge-shell domain, the single host
// mortar of the six nonconforming wedge neighbors calls the hook exactly once
// with the nonconforming flag and the shell face mesh, before the lift: the
// lifted targets with ShellBoundaryModesToKill = 4 differ from those with
// None exactly by the lift of the filtered minus the unfiltered correction
// on the inner face, the potential-rate input is bitwise unchanged, and no
// other point changes. The mortar data are set directly (the wedges are not
// emplaced), and the action's mutator is applied to the shell's DataBox.
void test_hook_invocation_wedge_shell(const gsl::not_null<std::mt19937*> gen) {
  const size_t l_max = 4;
  const size_t radial_points = 3;
  const auto creator = wedge_shell(l_max, radial_points);
  const ElementId<3> shell_id{6};
  const Mesh<2> shell_face{
      {{l_max + 1, 2 * l_max + 1}},
      {{Spectral::Basis::SphericalHarmonic,
        Spectral::Basis::SphericalHarmonic}},
      {{Spectral::Quadrature::Gauss, Spectral::Quadrature::Equiangular}}};
  const size_t face_points = shell_face.number_of_grid_points();
  const size_t n = radial_points * face_points;
  const Index<3> extents{radial_points, l_max + 1, 2 * l_max + 1};
  const DirectionalId<3> mortar_id{Direction<3>::lower_xi(), shell_id};
  const Vars state = random_state(gen, n, 1.0e-2);
  const Vars neighbor_face_state = random_state(gen, face_points, 1.0e-2);
  const Targets initial_targets = [&gen, n]() {
    std::normal_distribution<double> normal(0.0, 1.0);
    Targets result{n};
    for (size_t i = 0; i < result.size(); ++i) {
      result.data()[i] = normal(*gen);
    }
    return result;
  }();
  Scalar<DataVector> normal_magnitude{face_points};
  {
    std::uniform_real_distribution<double> positive(0.5, 1.5);
    for (auto& value : get(normal_magnitude)) {
      value = positive(*gen);
    }
  }
  const auto owning_copy = [](const PackagedVars& packaged) {
    DataVector result(packaged.size());
    std::copy(packaged.data(),
              std::next(packaged.data(),
                        static_cast<std::ptrdiff_t>(packaged.size())),
              result.begin());
    return result;
  };

  struct Result {
    Targets targets;
    PackagedVars local;
    PackagedVars neighbor;
    tnsr::i<DataVector, 3> normal;
  };
  const auto apply_corrections = [&](const std::optional<size_t>
                                         modes_to_kill) {
    const gp::BoundaryCorrections::UpwindGradientProjection upwind{
        gp::BoundaryCorrections::ExternalCorrection::Upwind, 1.0, 1.0, 0.0,
        modes_to_kill};
    const RecordingCorrection correction{upwind};
    MockRuntimeSystem runner{cache(3, creator, correction),
                             MutableCacheTuple{}};
    const Slab slab{0.0, 0.03};
    emplace_element(make_not_null(&runner), creator, shell_id, TimeStepId{},
                    TimeStepId{true, 0, slab.start()}, slab.duration(), 0.3,
                    DataVector{n, 0.1}, {}, n);
    REQUIRE(get_tag<domain::Tags::Mesh<3>>(runner, shell_id).slice_away(0) ==
            shell_face);
    const auto& element = get_tag<domain::Tags::Element<3>>(runner, shell_id);
    CHECK_FALSE(
        element.neighbors().at(Direction<3>::lower_xi()).are_conforming());
    CHECK(element.neighbors().at(Direction<3>::lower_xi()).size() == 6);
    REQUIRE(get_tag<evolution::dg::Tags::MortarMesh<3>>(runner, shell_id)
                .at(mortar_id) == shell_face);
    const auto& x = get_tag<domain::Tags::Coordinates<3, Frame::Inertial>>(
        runner, shell_id);
    // The inner face is the first radial point of every angular point.
    Result result{initial_targets, PackagedVars{}, PackagedVars{},
                  tnsr::i<DataVector, 3>{face_points}};
    for (size_t j = 0; j < face_points; ++j) {
      const double radius = std::sqrt(square(get<0>(x)[radial_points * j]) +
                                      square(get<1>(x)[radial_points * j]) +
                                      square(get<2>(x)[radial_points * j]));
      for (size_t i = 0; i < 3; ++i) {
        result.normal.get(i)[j] = -x.get(i)[radial_points * j] / radius;
      }
    }
    tnsr::i<DataVector, 3> neighbor_normal{face_points};
    for (size_t i = 0; i < 3; ++i) {
      neighbor_normal.get(i) = -result.normal.get(i);
    }
    result.local = package_face(data_on_slice(state, extents, 0, 0),
                                result.normal, upwind);
    result.neighbor =
        package_face(neighbor_face_state, neighbor_normal, upwind);
    auto& box =
        ActionTesting::get_databox<component>(make_not_null(&runner), shell_id);
    db::mutate<variables_tag, targets_tag, evolution::dg::Tags::MortarData<3>,
               evolution::dg::Tags::NormalCovectorAndMagnitude<3>>(
        [&](const gsl::not_null<Vars*> vars,
            const gsl::not_null<Targets*> targets, const auto mortar_data,
            const auto normal_and_magnitude) {
          *vars = state;
          *targets = initial_targets;
          auto& data = mortar_data->at(mortar_id);
          data.local().mortar_data = owning_copy(result.local);
          data.local().mortar_mesh = shell_face;
          data.local().face_mesh = shell_face;
          data.neighbor().mortar_data = owning_copy(result.neighbor);
          data.neighbor().mortar_mesh = shell_face;
          data.neighbor().face_mesh = shell_face;
          Variables<tmpl::list<evolution::dg::Tags::MagnitudeOfNormal,
                               evolution::dg::Tags::NormalCovector<3>>>
              normal_vars{face_points};
          get<evolution::dg::Tags::MagnitudeOfNormal>(normal_vars) =
              normal_magnitude;
          get<evolution::dg::Tags::NormalCovector<3>>(normal_vars) =
              result.normal;
          (*normal_and_magnitude)[Direction<3>::lower_xi()] =
              std::move(normal_vars);
        },
        make_not_null(&box));
    RecordingCorrection::calls.clear();
    db::mutate_apply<evolution::dg::ApplyBoundaryCorrections<
        false, Metavariables, 3, false>>(make_not_null(&box));
    REQUIRE(RecordingCorrection::calls.size() == 1);
    CHECK(RecordingCorrection::calls[0].has_nonconforming_block_connection);
    CHECK(RecordingCorrection::calls[0].face_mesh == shell_face);
    CHECK(RecordingCorrection::calls[0].number_of_points == face_points);
    result.targets = get_tag<targets_tag>(runner, shell_id);
    return result;
  };

  const Result unfiltered = apply_corrections(std::nullopt);
  const Result filtered = apply_corrections(size_t{4});
  // The same packaged data on both runs.
  CHECK(unfiltered.local == filtered.local);
  CHECK(unfiltered.neighbor == filtered.neighbor);
  CHECK(get<gp::Tags::PotentialRateTarget>(filtered.targets) ==
        get<gp::Tags::PotentialRateTarget>(unfiltered.targets));

  // The expected change: the lift of the filtered minus the unfiltered
  // correction of the face.
  const gp::BoundaryCorrections::UpwindGradientProjection upwind{
      gp::BoundaryCorrections::ExternalCorrection::Upwind, 1.0, 1.0, 0.0,
      size_t{4}};
  Targets correction{face_points};
  upwind.dg_boundary_terms<false>(
      make_not_null(&get<gp::Tags::VelocityTarget>(correction)),
      make_not_null(&get<gp::Tags::GradientTarget>(correction)),
      make_not_null(&get<gp::Tags::PotentialRateTarget>(correction)),
      get<gp::Tags::PackagedVelocities>(filtered.local),
      get<gp::Tags::PackagedGradients>(filtered.local),
      get<gp::Tags::PackagedPotentials>(filtered.local),
      get<Ccz4::Tags::NormalCovector<DataVector, 3>>(filtered.local),
      get<gp::Tags::PackagedVelocities>(filtered.neighbor),
      get<gp::Tags::PackagedGradients>(filtered.neighbor),
      get<gp::Tags::PackagedPotentials>(filtered.neighbor),
      get<Ccz4::Tags::NormalCovector<DataVector, 3>>(filtered.neighbor),
      dg::Formulation::StrongInertial);
  Targets filtered_correction = correction;
  upwind.dg_filter_boundary_correction(make_not_null(&filtered_correction),
                                       shell_face, true);
  Targets change = filtered_correction;
  change -= correction;
  ::dg::lift_flux(make_not_null(&change), radial_points, normal_magnitude);
  CHECK(max(abs(get<gp::Tags::VelocityTarget>(change).get(0))) > 1.0e-6);
  Targets actual_change = filtered.targets;
  actual_change -= unfiltered.targets;
  const Targets face_change = data_on_slice(actual_change, extents, 0, 0);
  CHECK_VARIABLES_CUSTOM_APPROX(face_change, change,
                                Approx::custom().epsilon(1.0e-12).scale(1.0));
  for (size_t r = 1; r < radial_points; ++r) {
    CHECK(data_on_slice(filtered.targets, extents, 0, r) ==
          data_on_slice(initial_targets, extents, 0, r));
    CHECK(data_on_slice(unfiltered.targets, extents, 0, r) ==
          data_on_slice(initial_targets, extents, 0, r));
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
  test_linearized_operator_about_background(make_not_null(&gen));
  test_sphere_minkowski(0.0);
  test_sphere_minkowski(1.0);
  test_sphere_linearized(make_not_null(&gen));
  test_two_shell_linearized(make_not_null(&gen));
  test_wedge_shell_linearized(make_not_null(&gen));
  test_refined_wedge_shell_linearized(make_not_null(&gen));
  test_external_lift_isolation(make_not_null(&gen));
  test_hook_invocation_conforming(make_not_null(&gen));
  test_hook_invocation_wedge_shell(make_not_null(&gen));
}
