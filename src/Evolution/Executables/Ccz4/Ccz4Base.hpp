// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <vector>

#include "Evolution/Systems/Ccz4/System.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Factory.hpp"

/// \cond
namespace Frame {

struct Inertial;
}  // namespace Frame
namespace PUP {
class er;
}  // namespace PUP
namespace Parallel {
template <typename Metavariables>
class CProxy_GlobalCache;
}  // namespace Parallel
/// \endcond

namespace detail {
constexpr size_t volume_dim = 3;

struct ObserverTags {
  using system = Ccz4::fd::System;

  using variables_tag = typename system::variables_tag;
  using analytic_solution_fields = typename variables_tag::tags_list;

  using initial_data_list = Ccz4::Solutions::all_solutions;

  using analytic_compute = evolution::Tags::
      AnalyticSolutionsCompute</*using DgSubcell should be false?*/
                               volume_dim, analytic_solution_fields, false,
                               initial_data_list>;
  using deriv_compute = ::Tags::DerivCompute<
      variables_tag, domain::Tags::Mesh<volume_dim>,
      domain::Tags::InverseJacobian<volume_dim, Frame::ElementLogical,
                                    Frame::Inertial>,
      typename system::gradient_variables>;
  using error_compute = Tags::ErrorsCompute<analytic_solution_fields>;
  using error_tags = db::wrap_tags_in<Tags::Error, analytic_solution_fields>;

  using observe_fields =
      tmpl::append <
      tmpl::push_back<analytic_solution_fields
                          // following tags added to observe constraints
                          /* should we observse Theta and SpatialZ4Constraints
                             here? */
                          error_tags>;

  using non_tensor_compute_tags = tmpl::list<>;

  using field_observations =
      dg::Events::field_observations<volume_dim, observe_fields,
                                     non_tensor_compute_tags>;
};

struct FactoryCreation : tt::ConformsTo<Options::protocols::FactoryCreation> {
  using LocalTimeStepping = false;
  using system = Ccz4::fd::System<volume_dim>;

  using factory_classes = tmpl::map<
      /* how do I choose fd for SpatialDiscretization option?    */
      tmpl::pair<DenseTrigger, DenseTriggers::standard_dense_triggers>,
      /* can I use all the domains? */
      tmpl::pair<DomainCreator<volume_dim>, domain_creators<volume_dim>>,
      tmpl::pair<
          Event,
          tmpl::flatten<tmpl::list<
              Events::Completion, Events::MonitorMemory<volume_dim>,
              typename detail::ObserverTags<volume_dim>::field_observations,
              Events::time_events<system>,
              dg::Events::ObserveTimeStepVolume<system>>>>,
      /* need to write a Dirichlet analytic boundary condition class */
      tmpl::pair<
          Ccz4::BoundaryConditions::BoundaryCondition<volume_dim>,
          Ccz4::BoundaryConditions::standard_boundary_conditions<volume_dim>>,
      tmpl::pair<evolution::initial_data::InitialData,
                 tmpl::append<Ccz4::Solutions::all_solutions<volume_dim>>>,
      tmpl::pair<MathFunction<1, Frame::Inertial>,
                 MathFunctions::all_math_functions<1, Frame::Inertial>>,
      tmpl::pair<PhaseChange, PhaseControl::factory_creatable_classes>,
      tmpl::pair<StepChooser<StepChooserUse::LtsStep>,
                 StepChoosers::standard_step_choosers<system>>,
      tmpl::pair<
          StepChooser<StepChooserUse::Slab>,
          StepChoosers::standard_slab_choosers<system, LocalTimeStepping>>,
      /* what does this do? */
      tmpl::pair<TimeSequence<double>,
                 TimeSequences::all_time_sequences<double>>,
      tmpl::pair<TimeSequence<std::uint64_t>,
                 TimeSequences::all_time_sequences<std::uint64_t>>,
      tmpl::pair<TimeStepper, TimeSteppers::time_steppers>,
      tmpl::pair<Trigger, tmpl::append<Triggers::logical_triggers,
                                       Triggers::time_triggers>>>;
};
}  // namespace detail

struct Ccz4TemplateBase {
  static constexpr size_t volume_dim = 3;
  using LocalTimeStepping = false;
  using system = Ccz4::System<volume_dim>;
  using TimeStepperBase =
      tmpl::conditional_t<LocalTimeStepping, LtsTimeStepper, TimeStepper>;

  static constexpr bool local_time_stepping =
      TimeStepperBase::local_time_stepping;
  static constexpr bool use_dg_element_collection = false;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}

  using factory_creation =
      detail::FactoryCreation<volume_dim, local_time_stepping>;

  using observed_reduction_data_tags =
      observers::collect_reduction_data_tags<tmpl::push_back<
          tmpl::at<typename factory_creation::factory_classes, Event>>>;

  using initialize_initial_data_dependent_quantities_actions =
      tmpl::list<Parallel::Actions::TerminatePhase>;

  // A tmpl::list of tags to be added to the GlobalCache by the
  using const_global_cache_tags = tmpl::list<>;

  using dg_registration_list =
      tmpl::list<observers::Actions::RegisterEventsWithObservers>;

  // Register needs to be before InitializeTimeStepperHistory so that CCE is
  // properly registered when the self-start happens
  static constexpr auto default_phase_order =
      std::array{Parallel::Phase::Initialization,
                 Parallel::Phase::RegisterWithElementDataReader,
                 Parallel::Phase::ImportInitialData,
                 Parallel::Phase::InitializeInitialDataDependentQuantities,
                 Parallel::Phase::Register,
                 Parallel::Phase::InitializeTimeStepperHistory,
                 Parallel::Phase::CheckDomain,
                 Parallel::Phase::Evolve,
                 Parallel::Phase::Exit};

  template <typename ControlSystems>
  using step_actions = tmpl::list<
      /* send data for reconstruction? */
      /* this TimeDerivative function does not respect the doc?? */
      evolution::dg::subcell::fd::Actions::TakeTimeStep<
          Burgers::subcell::TimeDerivative>,
      // do we need filtering?
      dg::Actions::Filter<
          Filters::Exponential<0>,
          tmpl::list<gr::Tags::SpacetimeMetric<DataVector, volume_dim>,
                     gh::Tags::Pi<DataVector, volume_dim>,
                     gh::Tags::Phi<DataVector, volume_dim>>>>;

  template <typename DerivedMetavars, bool UseControlSystems>
  using initialization_actions =
      tmpl::list <
      Initialization::Actions::InitializeItems<
          Initialization::TimeStepping<DerivedMetavars, TimeStepperBase>,
          evolution::dg::Initialization::Domain<DerivedMetavars,
                                                UseControlSystems>,
          /* what does this do? */
          Initialization::Actions::NonconservativeSystem<system>,
          Initialization::Actions::AddComputeTags<::Tags::DerivCompute<
              typename system::variables_tag, domain::Tags::Mesh<volume_dim>,
              domain::Tags::InverseJacobian<volume_dim, Frame::ElementLogical,
                                            Frame::Inertial>,
              typename system::gradient_variables>>,
          Initialization::Actions::AddComputeTags<
              /* what does this do? */
              tmpl::push_back<StepChoosers::step_chooser_compute_tags<
                  GeneralizedHarmonicTemplateBase, local_time_stepping>>>,
          /* do I need mortars? How to communicate ghost pts? */
          ::evolution::dg::Initialization::Mortars<volume_dim, system>,
          evolution::Actions::InitializeRunEventsAndDenseTriggers,
          Parallel::Actions::TerminatePhase>;
};
