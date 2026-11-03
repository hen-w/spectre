// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <optional>
#include <string>
#include <vector>

#include "Domain/ElementDistribution.hpp"
#include "Domain/ExplicitElementPlacement.hpp"
#include "Domain/Tags/ElementDistribution.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Parallel/Tags/Parallelization.hpp"
#include "Utilities/GetOutput.hpp"

namespace {
template <bool UseLTS>
struct TestMetavars {
  static constexpr bool local_time_stepping = UseLTS;
};

template <bool UseLTS>
domain::ElementDistribution make_option(const std::string& option_string) {
  return TestHelpers::test_option_tag<domain::OptionTags::ElementDistribution,
                                      TestMetavars<UseLTS>>(option_string);
}

domain::ElementDistribution make_option_without_lts_metavars(
    const std::string& option_string) {
  return TestHelpers::test_option_tag<domain::OptionTags::ElementDistribution>(
      option_string);
}

void test_explicit_options() {
  const domain::ElementDistribution expected{domain::ExplicitElementPlacement{
      {{"[B0,(L1I0,L0I0)]", 0}, {"[B0,(L1I1,L0I0)]", 31}}}};
  for (const std::string& option : std::vector<std::string>{
           "Explicit:\n"
           "  \"[B0,(L1I0,L0I0)]\": 0\n"
           "  \"[B0,(L1I1,L0I0)]\": 31",
           "{Explicit: {'[B0,(L1I1,L0I0)]': 31, '[B0,(L1I0,L0I0)]': 0}}"}) {
    CAPTURE(option);
    CHECK(make_option<true>(option) == expected);
    CHECK(make_option<false>(option) == expected);
    CHECK(make_option_without_lts_metavars(option) == expected);
  }

  const auto check_error = [](const std::string& option,
                              const std::string& expected_message) {
    CAPTURE(option);
    CHECK_THROWS_WITH(make_option<false>(option),
                      Catch::Matchers::ContainsSubstring(expected_message));
  };
  check_error(
      "Explicit:\n  \"[B0,(L0I0)]\": 0\nExplicit:\n  \"[B0,(L0I0)]\": 1",
      "Option 'Explicit' specified twice.");
  check_error("Explict:\n  \"[B0,(L0I0)]\": 0",
              "Option 'Explict' is not a valid option.");
  check_error("Explicit:\n  \"[B0,(L0I0)]\": 0\nUniform: 0",
              "Option 'Uniform' is not a valid option.");
  check_error("{}", "You did not specify the option (Explicit)");
  for (const std::string& option :
       std::vector<std::string>{"Explicit:", "Explicit: {}",
                                "Explicit: Uniform", "Explicit: [0, 1]"}) {
    check_error(option,
                "Expected a nonempty map from quoted element names to "
                "processor numbers");
  }
  check_error("Explicit:\n  \"[B0,(L0I0)]\": 0\n  \"[B0,(L0I0)]\": 1",
              "Element [B0,(L0I0)] is specified twice.");
  check_error("Explicit:\n  [B0,(L0I0)]: 0",
              "Element names must be quoted strings");
  check_error("Explicit:\n  \"[B0,(L0I0)]\": -1",
              "While parsing the processor of element [B0,(L0I0)]");
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Tags.ElementDistribution", "[Unit][Domain]") {
  TestHelpers::db::test_simple_tag<domain::Tags::ElementDistribution>(
      "ElementDistribution");
  CHECK(make_option<true>("Uniform").element_weight() ==
        std::optional{domain::ElementWeight::Uniform});
  CHECK(make_option<true>("NumGridPoints").element_weight() ==
        std::optional{domain::ElementWeight::NumGridPoints});
  CHECK(make_option<true>("NumGridPointsAndGridSpacing").element_weight() ==
        std::optional{domain::ElementWeight::NumGridPointsAndGridSpacing});
  CHECK(make_option<true>("RoundRobin").element_weight() == std::nullopt);

  CHECK(make_option<false>("Uniform").element_weight() ==
        std::optional{domain::ElementWeight::Uniform});
  CHECK(make_option<false>("NumGridPoints").element_weight() ==
        std::optional{domain::ElementWeight::NumGridPoints});
  CHECK_THROWS_WITH(make_option<false>("NumGridPointsAndGridSpacing"),
                    Catch::Matchers::ContainsSubstring(
                        "When not using local time stepping") and
                        Catch::Matchers::ContainsSubstring(
                            "Please choose another element distribution."));
  CHECK(make_option<false>("RoundRobin").element_weight() == std::nullopt);

  CHECK(make_option_without_lts_metavars("Uniform").element_weight() ==
        std::optional{domain::ElementWeight::Uniform});
  CHECK(make_option_without_lts_metavars("NumGridPoints").element_weight() ==
        std::optional{domain::ElementWeight::NumGridPoints});
  CHECK_THROWS_WITH(
      make_option_without_lts_metavars("NumGridPointsAndGridSpacing"),
      Catch::Matchers::ContainsSubstring(
          "When not using local time stepping") and
          Catch::Matchers::ContainsSubstring(
              "Please choose another element distribution."));
  CHECK(make_option_without_lts_metavars("RoundRobin").element_weight() ==
        std::nullopt);

  // The test weights do not require local time stepping
  for (const auto weight : {domain::ElementWeight::TestInteriorCubeGridPoints,
                            domain::ElementWeight::TestCcz4InteriorCubeWork}) {
    const std::string option = get_output(weight);
    CHECK(make_option<true>(option).element_weight() == std::optional{weight});
    CHECK(make_option<false>(option).element_weight() == std::optional{weight});
    CHECK(make_option_without_lts_metavars(option).element_weight() ==
          std::optional{weight});
  }
  CHECK_THROWS_WITH(make_option<true>("Unifrom"),
                    Catch::Matchers::ContainsSubstring(
                        "ElementWeight must be 'Uniform', 'NumGridPoints', "
                        "'NumGridPointsAndGridSpacing', "
                        "'TestInteriorCubeGridPoints', or "
                        "'TestCcz4InteriorCubeWork'"));

  test_explicit_options();
}
