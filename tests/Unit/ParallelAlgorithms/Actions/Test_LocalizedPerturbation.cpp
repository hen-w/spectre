// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Domain/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Parallel/Phase.hpp"
#include "ParallelAlgorithms/Actions/LocalizedPerturbation.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {

struct PsiTag : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct PiTag : db::SimpleTag {
  using type = Scalar<DataVector>;
};

using tags_list = tmpl::list<PsiTag, PiTag>;
using VariablesTag = ::Tags::Variables<tags_list>;

struct PerturbInitialData {};

using PerturbAction =
    Actions::LocalizedPerturbation<VariablesTag, PerturbInitialData>;

template <typename Metavariables>
struct ElementArray {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<3>;
  using const_global_cache_tags = tmpl::list<>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          Parallel::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<
              tmpl::list<VariablesTag,
                         domain::Tags::Coordinates<3, Frame::Inertial>>>>>,
      Parallel::PhaseActions<Parallel::Phase::Testing,
                             tmpl::list<PerturbAction>>>;
};

struct Metavariables {
  using component_list = tmpl::list<ElementArray<Metavariables>>;
  using const_global_cache_tags = tmpl::list<>;
};

void test_selective_perturbation() {
  const size_t num_points = 5;
  const double amplitude = 1.0e-6;
  const double width = 5.0;
  const std::vector<double> center{10.0, 0.0, 0.0};

  // Set up known coordinates
  tnsr::I<DataVector, 3, Frame::Inertial> coords{num_points};
  get<0>(coords) = DataVector{0.0, 5.0, 10.0, 15.0, 20.0};
  get<1>(coords) = DataVector{0.0, 0.0, 0.0, 0.0, 0.0};
  get<2>(coords) = DataVector{0.0, 0.0, 0.0, 0.0, 0.0};

  // Initialize variables to zero
  Variables<tags_list> fields{num_points, 0.0};

  const ElementId<3> element_id{0, {{SegmentId{2, 1}, SegmentId{2, 1},
                                     SegmentId{2, 1}}}};

  PerturbAction::PerturbationParameters params{
      std::vector<std::string>{"PsiTag"}, amplitude, width, center,
      std::nullopt};

  using element_array = ElementArray<Metavariables>;
  ActionTesting::MockRuntimeSystem<Metavariables> runner{
      {std::optional<PerturbAction::PerturbationParameters>{params}}};
  ActionTesting::emplace_component_and_initialize<element_array>(
      &runner, element_id, {fields, coords});
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);
  ActionTesting::next_action<element_array>(make_not_null(&runner), element_id);

  const auto get_tag = [&runner, &element_id](auto tag_v) -> const auto& {
    using tag = std::decay_t<decltype(tag_v)>;
    return ActionTesting::get_databox_tag<element_array, tag>(runner,
                                                              element_id);
  };

  // Compute expected perturbation
  DataVector expected{num_points};
  for (size_t i = 0; i < num_points; ++i) {
    const double dx = get<0>(coords)[i] - center[0];
    const double dy = get<1>(coords)[i] - center[1];
    const double dz = get<2>(coords)[i] - center[2];
    const double r_sq = dx * dx + dy * dy + dz * dz;
    expected[i] = amplitude * exp(-r_sq / (width * width));
  }

  // PsiTag should be bitwise equal to expected (0.0 + x == x)
  const DataVector& psi_data = get(get_tag(PsiTag{}));
  CHECK(psi_data == expected);

  // PiTag should be identically zero
  const DataVector& pi_data = get(get_tag(PiTag{}));
  CHECK(pi_data == DataVector{num_points, 0.0});
}

void test_disabled() {
  const size_t num_points = 3;

  tnsr::I<DataVector, 3, Frame::Inertial> coords{num_points, 0.0};
  Variables<tags_list> fields{num_points, 0.0};

  const ElementId<3> element_id{0, {{SegmentId{2, 1}, SegmentId{2, 1},
                                     SegmentId{2, 1}}}};

  using element_array = ElementArray<Metavariables>;
  ActionTesting::MockRuntimeSystem<Metavariables> runner{
      {std::optional<PerturbAction::PerturbationParameters>{std::nullopt}}};
  ActionTesting::emplace_component_and_initialize<element_array>(
      &runner, element_id, {fields, coords});
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);
  ActionTesting::next_action<element_array>(make_not_null(&runner), element_id);

  const auto get_tag = [&runner, &element_id](auto tag_v) -> const auto& {
    using tag = std::decay_t<decltype(tag_v)>;
    return ActionTesting::get_databox_tag<element_array, tag>(runner,
                                                              element_id);
  };

  CHECK(get(get_tag(PsiTag{})) == DataVector{num_points, 0.0});
  CHECK(get(get_tag(PiTag{})) == DataVector{num_points, 0.0});
}

void test_localization() {
  const size_t num_points = 3;
  const double amplitude = 1.0;
  const double width = 0.1;
  const std::vector<double> center{0.0, 0.0, 0.0};

  tnsr::I<DataVector, 3, Frame::Inertial> coords{num_points};
  // Point 0: at center, point 1: far away, point 2: very far away
  get<0>(coords) = DataVector{0.0, 10.0, 100.0};
  get<1>(coords) = DataVector{0.0, 0.0, 0.0};
  get<2>(coords) = DataVector{0.0, 0.0, 0.0};

  Variables<tags_list> fields{num_points, 0.0};

  const ElementId<3> element_id{0, {{SegmentId{2, 1}, SegmentId{2, 1},
                                     SegmentId{2, 1}}}};

  PerturbAction::PerturbationParameters params{
      std::vector<std::string>{"PsiTag"}, amplitude, width, center,
      std::nullopt};

  using element_array = ElementArray<Metavariables>;
  ActionTesting::MockRuntimeSystem<Metavariables> runner{
      {std::optional<PerturbAction::PerturbationParameters>{params}}};
  ActionTesting::emplace_component_and_initialize<element_array>(
      &runner, element_id, {fields, coords});
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);
  ActionTesting::next_action<element_array>(make_not_null(&runner), element_id);

  const DataVector& psi_data =
      get(ActionTesting::get_databox_tag<element_array, PsiTag>(runner,
                                                                 element_id));
  // At center: full amplitude
  CHECK(psi_data[0] == amplitude);
  // Far from center: effectively zero (exp(-10^2/0.1^2) ≈ 0)
  CHECK(psi_data[1] < std::numeric_limits<double>::epsilon());
  CHECK(psi_data[2] < std::numeric_limits<double>::epsilon());
}

void test_spherical_shell_gaussian() {
  const size_t num_points = 5;
  const double amplitude = 1.0e-4;
  const double width = 1.0;
  const std::vector<double> center{0.0, 0.0, 0.0};
  const double radial_center = 10.0;

  // Points at various radii from the origin
  tnsr::I<DataVector, 3, Frame::Inertial> coords{num_points};
  // r = 0, 5, 10, 15, 20
  get<0>(coords) = DataVector{0.0, 3.0, 6.0, 9.0, 12.0};
  get<1>(coords) = DataVector{0.0, 4.0, 8.0, 12.0, 16.0};
  get<2>(coords) = DataVector{0.0, 0.0, 0.0, 0.0, 0.0};

  Variables<tags_list> fields{num_points, 0.0};

  const ElementId<3> element_id{
      0, {{SegmentId{2, 1}, SegmentId{2, 1}, SegmentId{2, 1}}}};

  PerturbAction::PerturbationParameters params{
      std::vector<std::string>{"PsiTag"}, amplitude, width, center,
      radial_center};

  using element_array = ElementArray<Metavariables>;
  ActionTesting::MockRuntimeSystem<Metavariables> runner{
      {std::optional<PerturbAction::PerturbationParameters>{params}}};
  ActionTesting::emplace_component_and_initialize<element_array>(
      &runner, element_id, {fields, coords});
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);
  ActionTesting::next_action<element_array>(make_not_null(&runner),
                                            element_id);

  const auto get_tag = [&runner, &element_id](auto tag_v) -> const auto& {
    using tag = std::decay_t<decltype(tag_v)>;
    return ActionTesting::get_databox_tag<element_array, tag>(runner,
                                                              element_id);
  };

  // Compute expected: amplitude * exp(-(r - r0)^2 / width^2)
  DataVector expected{num_points};
  for (size_t i = 0; i < num_points; ++i) {
    const double r = sqrt(square(get<0>(coords)[i]) +
                          square(get<1>(coords)[i]) +
                          square(get<2>(coords)[i]));
    expected[i] = amplitude * exp(-square(r - radial_center) / square(width));
  }

  const DataVector& psi_data = get(get_tag(PsiTag{}));
  CHECK(psi_data == expected);

  // PiTag should be identically zero
  const DataVector& pi_data = get(get_tag(PiTag{}));
  CHECK(pi_data == DataVector{num_points, 0.0});

  // Verify spherical symmetry: the perturbation at r=10 should be the peak
  // Point at index 2 has r = sqrt(36+64) = 10, so it should get full amplitude
  CHECK(psi_data[2] == approx(amplitude));
}

}  // namespace

SPECTRE_TEST_CASE("Unit.ParallelAlgorithms.Actions.LocalizedPerturbation",
                  "[Unit][ParallelAlgorithms][Actions]") {
  test_selective_perturbation();
  test_disabled();
  test_localization();
  test_spherical_shell_gaussian();
}
