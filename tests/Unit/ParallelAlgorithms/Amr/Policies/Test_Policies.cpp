// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <sstream>

#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "ParallelAlgorithms/Amr/Policies/Isotropy.hpp"
#include "ParallelAlgorithms/Amr/Policies/Limits.hpp"
#include "ParallelAlgorithms/Amr/Policies/Policies.hpp"

namespace {
void test_equality() {
  INFO("Equality");
  CHECK(amr::Policies{amr::Isotropy::Anisotropic, amr::Limits{}} ==
        amr::Policies{amr::Isotropy::Anisotropic, amr::Limits{}});
  CHECK(amr::Policies{amr::Isotropy::Anisotropic, amr::Limits{}} !=
        amr::Policies{amr::Isotropy::Isotropic, amr::Limits{}});
}

void test_pup() {
  INFO("Serialization");
  test_serialization(amr::Policies{amr::Isotropy::Anisotropic, amr::Limits{}});
}

void test_option_parsing() {
  INFO("Option Parsing creation");
  const auto isotropies =
      std::array{amr::Isotropy::Anisotropic, amr::Isotropy::Isotropic};
  const amr::Limits default_limits{};
  const amr::Limits specified_limits{1, 5, 3, 7};
  for (const auto isotropy : isotropies) {
    CAPTURE(isotropy);
    {
      INFO("specified limits");
      std::ostringstream creation_string;
      creation_string << "Isotropy: " << isotropy << "\n";
      creation_string << "Limits:\n"
                      << "  RefinementLevel: [1, 5]\n"
                      << "  NumGridPoints: [3, 7]\n";
      const auto policies =
          TestHelpers::test_creation<amr::Policies>(creation_string.str());
      CHECK(policies == amr::Policies{isotropy, specified_limits});
    }
    {
      INFO("default limits");
      std::ostringstream creation_string;
      creation_string << "Isotropy: " << isotropy << "\n";
      creation_string << "Limits:\n"
                      << "  RefinementLevel: Auto\n"
                      << "  NumGridPoints: Auto\n";
      const auto policies =
          TestHelpers::test_creation<amr::Policies>(creation_string.str());
      CHECK(policies == amr::Policies{isotropy, default_limits});
    }
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.ParallelAlgorithms.Amr.Policies",
                  "[ParallelAlgorithms][Unit]") {
  test_equality();
  test_pup();
  test_option_parsing();
}
