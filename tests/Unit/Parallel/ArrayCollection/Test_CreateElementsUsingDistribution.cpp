// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Domain/Creators/AlignedLattice.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementDistribution.hpp"
#include "Domain/ExplicitElementPlacement.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Parallel/ArrayCollection/CreateElementsUsingDistribution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct Metavariables {
  using component_list = tmpl::list<>;
};

// The processor and node passed to the callback for each element
template <size_t Dim>
using CreatedElements =
    std::unordered_map<ElementId<Dim>, std::array<size_t, 2>>;

template <size_t Dim>
void create_elements(const gsl::not_null<CreatedElements<Dim>*> created,
                     const domain::ElementDistribution& distribution,
                     const DomainCreator<Dim>& creator,
                     const Parallel::GlobalCache<Metavariables>& cache,
                     const std::unordered_set<size_t>& procs_to_ignore) {
  const auto domain = creator.create_domain();
  const size_t number_of_procs = Parallel::number_of_procs<size_t>(cache);
  Parallel::create_elements_using_distribution(
      [&created](const ElementId<Dim>& element_id, const size_t target_proc,
                 const size_t target_node) {
        // Each element is created exactly once
        CHECK(
            created->insert({element_id, {{target_proc, target_node}}}).second);
      },
      distribution, domain.blocks(), creator.initial_extents(),
      creator.initial_refinement_levels(), Spectral::Basis::Legendre,
      Spectral::Quadrature::GaussLobatto, procs_to_ignore, number_of_procs,
      Parallel::number_of_nodes<size_t>(cache),
      number_of_procs - procs_to_ignore.size(), cache, false);
}

// The node of each processor of the cache in `test()`
constexpr std::array<size_t, 5> node_of_proc{{0, 0, 1, 1, 1}};

template <size_t Dim>
void test_explicit_placement(
    const DomainCreator<Dim>& creator,
    const Parallel::GlobalCache<Metavariables>& cache) {
  const std::vector<ElementId<Dim>> element_ids =
      initial_element_ids(creator.initial_refinement_levels());
  // Several elements share a processor, processor 1 is unused, and the unused
  // processor 2 is excluded.
  const std::array<size_t, 4> procs_to_cycle{{4, 0, 4, 3}};
  const std::unordered_set<size_t> procs_to_ignore{2};
  std::map<std::string, size_t> proc_by_element_name{};
  CreatedElements<Dim> expected{};
  for (size_t i = 0; i < element_ids.size(); ++i) {
    const size_t proc = procs_to_cycle[i % procs_to_cycle.size()];
    proc_by_element_name.emplace(get_output(element_ids[i]), proc);
    expected.insert({element_ids[i], {{proc, node_of_proc[proc]}}});
  }

  CreatedElements<Dim> created{};
  create_elements(make_not_null(&created),
                  domain::ElementDistribution{
                      domain::ExplicitElementPlacement{proc_by_element_name}},
                  creator, cache, procs_to_ignore);
  CHECK(created == expected);

  // Invalid placements create no elements
  const auto check_invalid = [&cache, &creator, &created](
                                 const std::map<std::string, size_t>& procs,
                                 const std::unordered_set<size_t>& ignore,
                                 const std::string& expected_message) {
    created.clear();
    CHECK_THROWS_WITH(
        create_elements(make_not_null(&created),
                        domain::ElementDistribution{
                            domain::ExplicitElementPlacement{procs}},
                        creator, cache, ignore),
        Catch::Matchers::ContainsSubstring(expected_message));
    CHECK(created.empty());
  };
  auto missing_procs = proc_by_element_name;
  missing_procs.erase(get_output(element_ids.back()));
  check_invalid(missing_procs, procs_to_ignore,
                "does not assign the initial element");
  auto extra_procs = proc_by_element_name;
  extra_procs.emplace(get_output(ElementId<Dim>{99}), 0);
  check_invalid(extra_procs, procs_to_ignore,
                "which is not an initial element of the domain");
  // Excluding the assigned processor 4
  check_invalid(proc_by_element_name, {2, 4},
                "which is excluded from element placement");
  auto out_of_range_procs = proc_by_element_name;
  out_of_range_procs.at(get_output(element_ids.back())) = 5;
  check_invalid(out_of_range_procs, procs_to_ignore,
                "but there are only 5 processors");
}

// The legacy distributions through the same function: round robin skipping
// excluded processors, and the Z-curve distribution
template <size_t Dim>
void test_legacy_distributions(
    const DomainCreator<Dim>& creator,
    const Parallel::GlobalCache<Metavariables>& cache) {
  const std::unordered_set<size_t> procs_to_ignore{2};
  const auto domain = creator.create_domain();
  const auto refinements = creator.initial_refinement_levels();
  const auto extents = creator.initial_extents();
  const std::vector<ElementId<Dim>> element_ids =
      initial_element_ids(refinements);

  CreatedElements<Dim> created{};
  create_elements(make_not_null(&created), domain::ElementDistribution{},
                  creator, cache, procs_to_ignore);
  CreatedElements<Dim> expected{};
  const std::array<size_t, 4> round_robin_procs{{0, 1, 3, 4}};
  for (size_t i = 0; i < element_ids.size(); ++i) {
    const size_t proc = round_robin_procs[i % round_robin_procs.size()];
    expected.insert({element_ids[i], {{proc, node_of_proc[proc]}}});
  }
  CHECK(created == expected);

  const domain::BlockZCurveProcDistribution<Dim> z_curve{
      domain::get_element_costs(domain.blocks(), refinements, extents,
                                domain::ElementWeight::NumGridPoints,
                                std::nullopt, std::nullopt),
      4,
      domain.blocks(),
      refinements,
      extents,
      procs_to_ignore};
  created.clear();
  create_elements(make_not_null(&created),
                  domain::ElementDistribution{
                      std::optional{domain::ElementWeight::NumGridPoints}},
                  creator, cache, procs_to_ignore);
  expected.clear();
  for (const auto& element_id : element_ids) {
    const size_t proc = z_curve.get_proc_for_element(element_id);
    expected.insert({element_id, {{proc, node_of_proc[proc]}}});
  }
  CHECK(created == expected);
}

template <size_t Dim>
void test(const DomainCreator<Dim>& creator) {
  // Processors 0 and 1 are on node 0, processors 2, 3, and 4 on node 1
  const Parallel::GlobalCache<Metavariables> cache{
      tuples::TaggedTuple<>{}, {}, std::vector<size_t>{2, 3}};
  test_explicit_placement(creator, cache);
  test_legacy_distributions(creator, cache);
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Parallel.ArrayCollection.CreateElementsUsingDistribution",
    "[Unit][Parallel]") {
  // Two blocks in each dimension
  test(domain::creators::AlignedLattice<1>({{{0.0, 0.5, 1.0}}}, {{2}}, {{3}},
                                           {}, {}, {}));
  test(domain::creators::AlignedLattice<2>({{{0.0, 0.5, 1.0}, {0.0, 1.0}}},
                                           {{1, 2}}, {{3, 4}}, {}, {}, {}));
  test(domain::creators::AlignedLattice<3>(
      {{{0.0, 1.0}, {0.0, 0.5, 1.0}, {0.0, 1.0}}}, {{1, 0, 1}}, {{3, 4, 5}}, {},
      {}, {}));
}
