// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "Domain/ExplicitElementPlacement.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/Literals.hpp"

namespace {
using domain::ExplicitElementPlacement;

void test_semantics() {
  const ExplicitElementPlacement default_placement{};
  CHECK(default_placement.proc_by_element_name().empty());
  const std::map<std::string, size_t> procs{{"[B0,(L1I0)]", 3},
                                            {"[B0,(L1I1)]", 0}};
  const ExplicitElementPlacement placement{procs};
  CHECK(placement.proc_by_element_name() == procs);
  CHECK(placement == ExplicitElementPlacement{procs});
  CHECK(placement != default_placement);
  CHECK(placement !=
        ExplicitElementPlacement{{{"[B0,(L1I0)]", 3}, {"[B0,(L1I1)]", 1}}});
  CHECK(placement != ExplicitElementPlacement{{{"[B0,(L1I0)]", 3}}});
  test_copy_semantics(placement);
  auto placement_to_move = placement;
  test_move_semantics(std::move(placement_to_move), placement);
  test_serialization(placement);
  test_serialization(default_placement);
}

void test_parsing() {
  CHECK(TestHelpers::test_creation<ExplicitElementPlacement>(
            "\"[B0,(L1I0,L0I0)]\": 0\n"
            "'[B1,(L1I1,L0I0)]': 31\n"
            "\"[B0,(L1I1,L0I0)]\": 0") ==
        ExplicitElementPlacement{{{"[B0,(L1I0,L0I0)]", 0},
                                  {"[B1,(L1I1,L0I0)]", 31},
                                  {"[B0,(L1I1,L0I0)]", 0}}});
  CHECK(TestHelpers::test_creation<ExplicitElementPlacement>(
            "{\"[B0,(L0I0)]\": 18446744073709551615}") ==
        ExplicitElementPlacement{
            {{"[B0,(L0I0)]", std::numeric_limits<size_t>::max()}}});

  const auto check_parse_error = [](const std::string& input,
                                    const std::string& expected_message) {
    CAPTURE(input);
    CHECK_THROWS_WITH(
        TestHelpers::test_creation<ExplicitElementPlacement>(input),
        Catch::Matchers::ContainsSubstring(expected_message));
  };
  // Duplicate names are an error even if they are assigned the same processor
  check_parse_error("\"[B0,(L0I0)]\": 0\n\"[B0,(L1I0)]\": 1\n'[B0,(L0I0)]': 2",
                    "Element [B0,(L0I0)] is specified twice.");
  check_parse_error("\"[B0,(L0I0)]\": 0\n\"[B0,(L0I0)]\": 0",
                    "Element [B0,(L0I0)] is specified twice.");
  // Unquoted, YAML reads the name as a sequence
  check_parse_error("[B0,(L0I0)]: 0", "Element names must be quoted strings");
  for (const std::string& not_a_nonempty_map : std::vector<std::string>{
           "", "{}", "3", "Uniform", "[\"[B0,(L0I0)]\", 0]"}) {
    check_parse_error(not_a_nonempty_map,
                      "Expected a nonempty map from quoted element names to "
                      "processor numbers");
  }
  // Negative, fractional, overflowing, and non-numeric processors
  for (const std::string& invalid_proc :
       std::vector<std::string>{"-1", "1.5", "100000000000000000000", "1.0e20",
                                "Zero", "", "[0, 1]", "{a: 0}"}) {
    check_parse_error("\"[B0,(L0I0)]\": " + invalid_proc,
                      "While parsing the processor of element [B0,(L0I0)]");
  }
}

// Expected processor of each initial element, keyed by ElementIds built from
// their segments rather than from their names
template <size_t Dim>
using ExpectedProcs = std::unordered_map<ElementId<Dim>, size_t>;

template <size_t Dim>
void check_procs(const std::map<std::string, size_t>& proc_by_element_name,
                 const std::vector<std::array<size_t, Dim>>& refinements,
                 const size_t number_of_procs,
                 const std::unordered_set<size_t>& procs_to_ignore,
                 const ExpectedProcs<Dim>& expected_procs) {
  CHECK(
      ExplicitElementPlacement{proc_by_element_name}.procs_for_initial_elements(
          refinements, number_of_procs, procs_to_ignore) == expected_procs);
}

void test_valid_placements() {
  // Two blocks with four and two elements. Processors 1, 2, 4, and 6 are
  // unused, and the unused processors 1 and 6 are excluded.
  check_procs<1>({{"[B0,(L2I0)]", 3},
                  {"[B0,(L2I1)]", 3},
                  {"[B0,(L2I2)]", 0},
                  {"[B0,(L2I3)]", 5},
                  {"[B1,(L1I0)]", 0},
                  {"[B1,(L1I1)]", 3}},
                 {{{2}}, {{1}}}, 7, {1, 6},
                 {{ElementId<1>{0, {{SegmentId{2, 0}}}}, 3},
                  {ElementId<1>{0, {{SegmentId{2, 1}}}}, 3},
                  {ElementId<1>{0, {{SegmentId{2, 2}}}}, 0},
                  {ElementId<1>{0, {{SegmentId{2, 3}}}}, 5},
                  {ElementId<1>{1, {{SegmentId{1, 0}}}}, 0},
                  {ElementId<1>{1, {{SegmentId{1, 1}}}}, 3}});
  // Three blocks with anisotropic refinement, all elements on one processor
  check_procs<2>({{"[B0,(L1I0,L0I0)]", 2},
                  {"[B0,(L1I1,L0I0)]", 2},
                  {"[B1,(L0I0,L1I0)]", 2},
                  {"[B1,(L0I0,L1I1)]", 2},
                  {"[B2,(L0I0,L0I0)]", 2}},
                 {{{1, 0}}, {{0, 1}}, {{0, 0}}}, 3, {0},
                 {{ElementId<2>{0, {{SegmentId{1, 0}, SegmentId{0, 0}}}}, 2},
                  {ElementId<2>{0, {{SegmentId{1, 1}, SegmentId{0, 0}}}}, 2},
                  {ElementId<2>{1, {{SegmentId{0, 0}, SegmentId{1, 0}}}}, 2},
                  {ElementId<2>{1, {{SegmentId{0, 0}, SegmentId{1, 1}}}}, 2},
                  {ElementId<2>{2, {{SegmentId{0, 0}, SegmentId{0, 0}}}}, 2}});
  // The layout of the option help: one block with four elements on processors
  // 0 and 31 of 32, plus a second block with one element
  check_procs<3>(
      {{"[B0,(L1I0,L1I0,L0I0)]", 0},
       {"[B0,(L1I1,L1I0,L0I0)]", 31},
       {"[B0,(L1I0,L1I1,L0I0)]", 31},
       {"[B0,(L1I1,L1I1,L0I0)]", 0},
       {"[B1,(L0I0,L0I0,L0I0)]", 17}},
      {{{1, 1, 0}}, {{0, 0, 0}}}, 32, {},
      {{ElementId<3>{0, {{SegmentId{1, 0}, SegmentId{1, 0}, SegmentId{0, 0}}}},
        0},
       {ElementId<3>{0, {{SegmentId{1, 1}, SegmentId{1, 0}, SegmentId{0, 0}}}},
        31},
       {ElementId<3>{0, {{SegmentId{1, 0}, SegmentId{1, 1}, SegmentId{0, 0}}}},
        31},
       {ElementId<3>{0, {{SegmentId{1, 1}, SegmentId{1, 1}, SegmentId{0, 0}}}},
        0},
       {ElementId<3>{1, {{SegmentId{0, 0}, SegmentId{0, 0}, SegmentId{0, 0}}}},
        17}});
}

void test_invalid_placements() {
  const std::vector<std::array<size_t, 2>> refinements{{{1, 0}}, {{0, 1}}};
  const std::map<std::string, size_t> valid_procs{{"[B0,(L1I0,L0I0)]", 1},
                                                  {"[B0,(L1I1,L0I0)]", 0},
                                                  {"[B1,(L0I0,L1I0)]", 3},
                                                  {"[B1,(L0I0,L1I1)]", 1}};
  const auto check_error =
      [&refinements](const std::map<std::string, size_t>& procs,
                     const size_t number_of_procs,
                     const std::unordered_set<size_t>& procs_to_ignore,
                     const std::string& expected_message) {
        CAPTURE(procs);
        CHECK_THROWS_WITH(
            ExplicitElementPlacement{procs}.procs_for_initial_elements(
                refinements, number_of_procs, procs_to_ignore),
            Catch::Matchers::ContainsSubstring(expected_message));
      };

  // Missing element
  auto missing_procs = valid_procs;
  missing_procs.erase("[B1,(L0I0,L1I1)]");
  check_error(missing_procs, 4, {},
              "The explicit element placement does not assign the initial "
              "element [B1,(L0I0,L1I1)]. It assigns 3 of the 4 initial "
              "elements, but must assign all of them.");

  // Names that are not initial elements: wrong dimension, block, refinement,
  // segment index, or grid index, and noncanonical spellings
  for (const std::string& name : std::vector<std::string>{
           "[B0,(L1I0)]", "[B0,(L1I0,L0I0,L0I0)]", "[B2,(L0I0,L0I0)]",
           "[B0,(L2I0,L0I0)]", "[B0,(L16I0,L0I0)]", "[B0,(L1I2,L0I0)]",
           "[B0,(L1I0,L0I0),G1]", "[B0, (L1I0, L0I0)]", "[b0,(L1I0,L0I0)]",
           "B0,(L1I0,L0I0)", ""}) {
    auto extra_procs = valid_procs;
    extra_procs.emplace(name, 0);
    check_error(extra_procs, 4, {},
                "The explicit element placement names '" + name +
                    "', which is not an initial element of the domain. "
                    "Elements are named as ElementId<2> prints them, e.g. "
                    "'[B0,(L0I0,L0I0)]'.");
  }
  // Replacing an element by a non-initial one is also caught
  auto replaced_procs = valid_procs;
  replaced_procs.erase("[B0,(L1I0,L0I0)]");
  replaced_procs.emplace("[B0,(L0I0,L0I0)]", 1);
  check_error(replaced_procs, 4, {},
              "names '[B0,(L0I0,L0I0)]', which is not an initial element");

  // Processors that do not exist
  for (const size_t invalid_proc :
       {4_st, 5_st, std::numeric_limits<size_t>::max()}) {
    auto invalid_procs = valid_procs;
    invalid_procs.at("[B1,(L0I0,L1I0)]") = invalid_proc;
    check_error(invalid_procs, 4, {},
                "The explicit element placement assigns element "
                "[B1,(L0I0,L1I0)] to processor " +
                    std::to_string(invalid_proc) +
                    ", but there are only 4 processors.");
  }
  check_error(valid_procs, 0, {}, "but there are only 0 processors.");

  // Excluded processors, including when all processors are excluded
  check_error(valid_procs, 4, {2, 3},
              "The explicit element placement assigns element "
              "[B1,(L0I0,L1I0)] to processor 3, which is excluded from "
              "element placement.");
  check_error(valid_procs, 4, {0, 1, 2, 3},
              "to processor 1, which is excluded from element placement.");

  // The valid placement passes the same checks
  CHECK(ExplicitElementPlacement{valid_procs}
            .procs_for_initial_elements(refinements, 4, {2})
            .size() == 4);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.ExplicitElementPlacement", "[Domain][Unit]") {
  test_semantics();
  test_parsing();
  test_valid_placements();
  test_invalid_placements();
}
