// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "Domain/Block.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Creators/AlignedLattice.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/NonconformingSphericalShells.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementDistribution.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "Domain/Structure/ZCurve.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"

namespace {
// Check the static FD set independently of block connectivity: the cube and
// radial segment zero in the first six wedge blocks. On ordinary Sphere
// domains also compare with the original name-based costs and assignments.
void test_subcell_costs(const DomainCreator<3>& creator,
                        const bool has_ylm_shells) {
  const auto domain = creator.create_domain();
  const auto& blocks = domain.blocks();
  const auto refinements = creator.initial_refinement_levels();
  const auto extents = creator.initial_extents();
  const auto costs = domain::get_element_costs(
      blocks, refinements, extents, domain::ElementWeight::NumGridPoints,
      std::nullopt, std::nullopt);
  std::unordered_map<ElementId<3>, double> legacy_costs{};
  size_t fd_count = 0;
  size_t ylm_count = 0;
  size_t cube_id = 0;
  for (const auto& [id, cost] : costs) {
    const auto& block = blocks[id.block_id()];
    const bool is_cube = block.name() == "InnerCube";
    const bool is_fd =
        is_cube or (id.block_id() < 6 and id.segment_id(2).index() == 0);
    const bool is_ylm = has_ylm_shells and block.name().starts_with("Shell");
    CHECK(cost == (is_fd ? 4913.0 : (is_ylm ? 2925.0 : 729.0)));
    fd_count += is_fd;
    ylm_count += is_ylm;
    if (is_cube) {
      cube_id = id.block_id();
    }
    if (not has_ylm_shells) {
      const bool legacy_fd = is_cube or (block.name().starts_with("Shell0") and
                                         id.segment_id(2).index() == 0);
      legacy_costs.emplace(id, legacy_fd ? 4913.0 : 729.0);
    }
  }
  CHECK(fd_count == 32);
  CHECK(ylm_count == (has_ylm_shells ? 30 : 0));
  if (not has_ylm_shells) {
    CHECK(costs == legacy_costs);
    for (const size_t num_procs : {62_st, 186_st, 372_st}) {
      const domain::BlockZCurveProcDistribution<3> distribution{
          costs, num_procs, blocks, refinements, extents, {}};
      const domain::BlockZCurveProcDistribution<3> legacy_distribution{
          legacy_costs, num_procs, blocks, refinements, extents, {}};
      for (const auto& [id, cost] : costs) {
        CHECK(distribution.get_proc_for_element(id) ==
              legacy_distribution.get_proc_for_element(id));
      }
    }
  }
  // Keep the fail-loud guards, while Uniform bypasses FD weighting.
  auto invalid_refinements = refinements;
  invalid_refinements[cube_id][0] = 0;
  CHECK_THROWS_WITH(
      domain::get_element_costs(blocks, invalid_refinements, extents,
                                domain::ElementWeight::NumGridPoints,
                                std::nullopt, std::nullopt),
      Catch::Matchers::ContainsSubstring(
          "InnerCube block has initial refinement"));
  invalid_refinements = refinements;
  invalid_refinements[0][2] = 0;
  CHECK_THROWS_WITH(
      domain::get_element_costs(blocks, invalid_refinements, extents,
                                domain::ElementWeight::NumGridPoints,
                                std::nullopt, std::nullopt),
      Catch::Matchers::ContainsSubstring("radial (zeta) refinement 1"));
  const auto uniform_costs = domain::get_element_costs(
      blocks, invalid_refinements, extents, domain::ElementWeight::Uniform,
      std::nullopt, std::nullopt);
  for (const auto& [id, cost] : uniform_costs) {
    CHECK(cost == 1.0);
  }
}

void test_filled_sphere_costs() {
  using domain::CoordinateMaps::Distribution;
  // Shell41 and the log-graded pure-wedge layout.
  for (const bool log_graded : {false, true}) {
    CAPTURE(log_graded);
    std::vector<double> partitions{};
    for (size_t i = 0; i < (log_graded ? 10_st : 41_st); ++i) {
      partitions.push_back(23.0 + 16.1 * i);
    }
    if (log_graded) {
      partitions.insert(partitions.end(), {354.880343, 515.938061, 622.093093});
    }
    const size_t num_shells = partitions.size() + 1;
    std::vector<std::array<size_t, 3>> refinements(6 * num_shells + 1,
                                                   {{1, 1, 1}});
    std::vector<Distribution> distributions(num_shells, Distribution::Linear);
    if (log_graded) {
      for (size_t shell = 10; shell < 14; ++shell) {
        distributions[shell] = Distribution::Logarithmic;
        for (size_t wedge = 0; wedge < 6; ++wedge) {
          refinements[6 * shell + wedge][2] = 14 - shell;
        }
      }
    }
    const domain::creators::Sphere sphere{
        6.9,
        683.1,
        domain::creators::Sphere::InnerCube{0.0},
        refinements,
        std::array<size_t, 3>{{9, 9, 9}},
        true,
        std::nullopt,
        partitions,
        distributions};
    test_subcell_costs(sphere, false);
  }
  // The Ylm creator calls its first spherical-harmonic block Shell0, which
  // must not be mistaken for an FD wedge. It has radial refinement in xi.
  const domain::creators::NonconformingSphericalShells ylm{
      6.9,
      167.9,
      683.1,
      {23.0, 39.1, 55.2, 71.3, 87.4, 103.5, 119.6, 135.7, 151.8},
      {184.36547733518756, 202.4456773855363, 222.29895143317333,
       244.0991798218503, 268.03729484802381, 294.32295299754816,
       323.18637117388278, 354.880343, 389.68245260668192, 427.89750648307671,
       469.86020240238878, 515.938061, 566.53464515765734, 622.093093},
      {{std::vector<Distribution>(10, Distribution::Linear),
        std::vector<Distribution>(15, Distribution::Logarithmic)}},
      1,
      1,
      9,
      12,
      9,
      domain::creators::NonconformingSphericalShells::InnerCube{0.0},
      true};
  test_subcell_costs(ylm, true);
}

// Test the weighting done by `domain::get_element_costs` for a uniform cost
// function
void test_uniform_cost_function() {
  // AlignedLattice with differently-refined Elements
  const auto domain_creator = domain::creators::AlignedLattice<2>(
      {{{{70, 71, 72, 73}}, {{90, 92, 95, 99}}}}, {{2, 5}}, {{3, 3}},
      {{{{{1, 0}}, {{3, 2}}, {{3, 5}}}, {{{2, 1}}, {{3, 3}}, {{4, 6}}}}},
      {{{{{1, 0}}, {{3, 2}}, {{4, 5}}}, {{{2, 1}}, {{3, 3}}, {{6, 7}}}}}, {});

  const auto domain = domain_creator.create_domain();
  const auto& blocks = domain.blocks();

  const auto costs = domain::get_element_costs(
      blocks, domain_creator.initial_refinement_levels(),
      domain_creator.initial_extents(), domain::ElementWeight::Uniform,
      std::nullopt, std::nullopt);

  for (const auto& element_id_and_cost : costs) {
    CHECK(element_id_and_cost.second == 1.0);
  }
}

// Test the weighting done by `domain::get_element_costs` for weighted cost
// functions
void test_weighted_cost_function(const domain::ElementWeight element_weight) {
  const auto domain_creator1 = domain::creators::AlignedLattice<3>(
      {{{{0.0, 1.0, 2.0}}, {{0.0, 1.0}}, {{0.0, 1.0}}}}, {{2, 1, 0}},
      {{4, 4, 4}}, {}, {}, {});
  const auto domain1 = domain_creator1.create_domain();
  const auto& blocks1 = domain1.blocks();

  // Block size and grid points are the same as blocks in `domain1`, but
  // refinement levels are different
  const auto domain_creator2 = domain::creators::AlignedLattice<3>(
      {{{{0.0, 1.0}}, {{0.0, 1.0}}, {{0.0, 1.0}}}}, {{2, 3, 2}}, {{4, 4, 4}},
      {}, {}, {});
  const auto domain2 = domain_creator2.create_domain();
  const auto& blocks2 = domain2.blocks();

  // Block size and refinement levels are the same as blocks in `domain1`, but
  // grid points are different
  const auto domain_creator3 = domain::creators::AlignedLattice<3>(
      {{{{0.0, 1.0}}, {{0.0, 1.0}}, {{0.0, 1.0}}}}, {{2, 1, 0}}, {{4, 3, 2}},
      {}, {}, {});
  const auto domain3 = domain_creator3.create_domain();
  const auto& blocks3 = domain2.blocks();

  const auto costs1 = domain::get_element_costs(
      blocks1, domain_creator1.initial_refinement_levels(),
      domain_creator1.initial_extents(), element_weight,
      Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto);

  const auto costs2 = domain::get_element_costs(
      blocks2, domain_creator2.initial_refinement_levels(),
      domain_creator2.initial_extents(), element_weight,
      Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto);

  const auto costs3 = domain::get_element_costs(
      blocks3, domain_creator3.initial_refinement_levels(),
      domain_creator3.initial_extents(), element_weight,
      Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto);

  // check that all elements in each domain have the same cost

  auto elemental_cost_it1 = costs1.begin();
  const double elemental_cost1 = elemental_cost_it1->second;
  elemental_cost_it1++;
  while (elemental_cost_it1 != costs1.end()) {
    CHECK(elemental_cost1 == approx(elemental_cost_it1->second));
    elemental_cost_it1++;
  }

  auto elemental_cost_it2 = costs2.begin();
  const double elemental_cost2 = elemental_cost_it2->second;
  elemental_cost_it2++;
  while (elemental_cost_it2 != costs1.end()) {
    CHECK(elemental_cost2 == approx(elemental_cost_it2->second));
    elemental_cost_it2++;
  }

  auto elemental_cost_it3 = costs3.begin();
  const double elemental_cost3 = elemental_cost_it3->second;
  elemental_cost_it3++;
  while (elemental_cost_it3 != costs3.end()) {
    CHECK(elemental_cost3 == approx(elemental_cost_it3->second));
    elemental_cost_it3++;
  }

  if (element_weight == domain::ElementWeight::NumGridPoints) {
    // check that varying refinement doesn't affect the cost
    CHECK(elemental_cost2 == elemental_cost1);
  } else {
    // element_weight == domain::ElementWeight::NumGridPointsAndGridSpacing

    // The highest refinement for the first test domain is 2 while the highest
    // refinement for the second test domain is 3, grid points held constant.
    // Since the minimum grid spacing of the second domain is half the minimum
    // grid spacing of the first and since
    // elemental cost = (# of grid points) / sqrt(min grid spacing), the
    // elemental cost of the second domain should be a factor of sqrt(2) the
    // cost.
    CHECK(elemental_cost2 == approx(sqrt(2.0) * elemental_cost1));
  }

  // The minimum grid spacing for the first and third domain are equal, but the
  // number of grid points in an element in the first is 64 while the number of
  // grid points in an element in the third is 24. Since elemental cost for
  // either domain::Elementweight::NumGridPoints or
  // domain::Elementweight::NumGridPointsAndGridSpacing should only scale by
  // the # of grid points, the elemental cost of the third domain should be a
  // factor of 24/64 = 3/8 the cost of the first domain.
  CHECK(elemental_cost3 == elemental_cost1 * 3.0 / 8.0);
}

// Test the processor distribution logic of the
// `domain::BlockZCurveProcDistribution` constructor for an unweighted element
// distribution
template <size_t Dim>
void test_uniform_element_distribution_construction(
    const DomainCreator<Dim>& domain_creator,
    const size_t number_of_procs_with_elements,
    const std::unordered_set<size_t>& global_procs_to_ignore = {}) {
  const auto domain = domain_creator.create_domain();
  const auto& blocks = domain.blocks();
  const auto initial_refinement_levels =
      domain_creator.initial_refinement_levels();
  const auto initial_extents = domain_creator.initial_extents();

  const size_t num_blocks = blocks.size();
  size_t num_elements = 0;
  std::vector<size_t> num_elements_by_block(num_blocks, 0);
  for (size_t i = 0; i < num_blocks; i++) {
    size_t num_elements_this_block =
        two_to_the(gsl::at(initial_refinement_levels[i], 0));
    for (size_t j = 1; j < Dim; j++) {
      num_elements_this_block *=
          two_to_the(gsl::at(initial_refinement_levels[i], j));
    }
    num_elements_by_block[i] = num_elements_this_block;
    num_elements += num_elements_this_block;
  }

  const auto costs = domain::get_element_costs(
      blocks, initial_refinement_levels, initial_extents,
      domain::ElementWeight::Uniform, std::nullopt, std::nullopt);

  const domain::BlockZCurveProcDistribution<Dim> element_distribution(
      costs, number_of_procs_with_elements, blocks, initial_refinement_levels,
      initial_extents, global_procs_to_ignore);
  const auto proc_map = element_distribution.block_element_distribution();

  const size_t total_procs =
      number_of_procs_with_elements + global_procs_to_ignore.size();
  std::vector<size_t> num_elements_by_proc(total_procs, 0);
  std::vector<size_t> actual_num_elements_by_block_in_dist(num_blocks, 0);
  size_t actual_num_elements_in_dist = 0;
  for (size_t block_number = 0; block_number < proc_map.size();
       block_number++) {
    for (const auto& proc_allowance : proc_map[block_number]) {
      const size_t proc_number = proc_allowance.first;
      const size_t element_allowance = proc_allowance.second;
      num_elements_by_proc[proc_number] += element_allowance;
      actual_num_elements_by_block_in_dist[block_number] += element_allowance;
    }
    // check that the number of elements in this block accounted for in the
    // distribution matches the expected number of total elements for this block
    CHECK(actual_num_elements_by_block_in_dist[block_number] ==
          num_elements_by_block[block_number]);
    actual_num_elements_in_dist +=
        actual_num_elements_by_block_in_dist[block_number];
  }
  // check that the number of elements accounted for in the distribution matches
  // the expected number of total elements
  CHECK(actual_num_elements_in_dist == num_elements);

  size_t lowest_proc_with_elements = 0;
  while (global_procs_to_ignore.count(lowest_proc_with_elements) == 1) {
    lowest_proc_with_elements++;
  }
  const size_t num_elements_on_lowest_proc =
      num_elements_by_proc[lowest_proc_with_elements];

  size_t num_elements_so_far = num_elements_on_lowest_proc;
  size_t proc_num = lowest_proc_with_elements + 1;
  while (proc_num < total_procs and num_elements_so_far < num_elements) {
    const size_t num_elements_this_proc = num_elements_by_proc[proc_num];
    if (global_procs_to_ignore.count(proc_num) == 1) {
      CHECK(num_elements_this_proc == 0);
    } else {
      // check that the distribution is near-uniform
      CHECK((num_elements_this_proc == num_elements_on_lowest_proc or
             num_elements_this_proc == num_elements_on_lowest_proc + 1 or
             num_elements_this_proc == num_elements_on_lowest_proc - 1));
    }
    num_elements_so_far += num_elements_this_proc;
    proc_num++;
  }

  // check that any remainder of processors we didn't need do indeed have 0
  // elements assigned to them
  while (proc_num < total_procs) {
    CHECK(num_elements_by_proc[proc_num] == 0);
    proc_num++;
  }
}

// Test the processor distribution logic of the
// `domain::BlockZCurveProcDistribution` constructor for weighted element
// distributions
template <size_t Dim>
void test_weighted_element_distribution_construction(
    const domain::ElementWeight element_weight,
    const DomainCreator<Dim>& domain_creator,
    const size_t number_of_procs_with_elements,
    const std::unordered_set<size_t>& global_procs_to_ignore = {}) {
  const auto domain = domain_creator.create_domain();
  const auto& blocks = domain.blocks();
  const auto initial_refinement_levels =
      domain_creator.initial_refinement_levels();
  const auto initial_extents = domain_creator.initial_extents();

  const size_t num_blocks = blocks.size();
  size_t num_elements = 0;
  std::vector<size_t> num_elements_by_block(num_blocks, 0);
  for (size_t i = 0; i < num_blocks; i++) {
    size_t num_elements_this_block =
        two_to_the(gsl::at(initial_refinement_levels[i], 0));
    for (size_t j = 1; j < Dim; j++) {
      num_elements_this_block *=
          two_to_the(gsl::at(initial_refinement_levels[i], j));
    }
    num_elements_by_block[i] = num_elements_this_block;
    num_elements += num_elements_this_block;
  }

  const auto costs = domain::get_element_costs(
      blocks, initial_refinement_levels, initial_extents, element_weight,
      Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto);

  const domain::BlockZCurveProcDistribution<Dim> element_distribution(
      costs, number_of_procs_with_elements, blocks, initial_refinement_levels,
      initial_extents, global_procs_to_ignore);
  const auto proc_map = element_distribution.block_element_distribution();

  const size_t total_procs =
      number_of_procs_with_elements + global_procs_to_ignore.size();
  std::vector<size_t> num_elements_by_proc(total_procs, 0);
  std::vector<size_t> actual_num_elements_by_block_in_dist(num_blocks, 0);
  size_t actual_num_elements_in_dist = 0;
  for (size_t block_number = 0; block_number < proc_map.size();
       block_number++) {
    for (const auto& proc_allowance : proc_map[block_number]) {
      const size_t proc_number = proc_allowance.first;
      const size_t element_allowance = proc_allowance.second;
      num_elements_by_proc[proc_number] += element_allowance;
      actual_num_elements_by_block_in_dist[block_number] += element_allowance;
    }
    // check that the number of elements in this block accounted for in the
    // distribution matches the expected number of total elements for this block
    CHECK(actual_num_elements_by_block_in_dist[block_number] ==
          num_elements_by_block[block_number]);
    actual_num_elements_in_dist +=
        actual_num_elements_by_block_in_dist[block_number];
  }
  // check that the number of elements accounted for in the distribution matches
  // the expected number of total elements
  CHECK(actual_num_elements_in_dist == num_elements);

  std::vector<std::vector<ElementId<Dim>>> initial_element_ids_by_block(
      num_blocks);
  for (size_t i = 0; i < num_blocks; i++) {
    const size_t num_elements_this_block = two_to_the(alg::accumulate(
        initial_refinement_levels[i], 0_st, std::plus<size_t>()));
    initial_element_ids_by_block[i].reserve(num_elements_this_block);
    initial_element_ids_by_block[i] =
        initial_element_ids(blocks[i].id(), initial_refinement_levels[i]);
    alg::sort(initial_element_ids_by_block[i],
              [](const ElementId<Dim>& lhs, const ElementId<Dim>& rhs) {
                return domain::z_curve_index(lhs) < domain::z_curve_index(rhs);
              });
  }

  double total_cost = 0.0;
  for (const auto& element_id_and_cost : costs) {
    total_cost += element_id_and_cost.second;
  }

  // one flattened vector instead of vectors by Block
  std::vector<double> costs_flattened(num_elements);
  size_t cost_index = 0;
  for (size_t i = 0; i < num_blocks; i++) {
    const size_t num_elements_this_block =
        initial_element_ids_by_block[i].size();
    for (size_t j = 0; j < num_elements_this_block; j++) {
      const ElementId<Dim>& element_id = initial_element_ids_by_block[i][j];
      costs_flattened[cost_index] = costs.at(element_id);
      cost_index++;
    }
  }

  cost_index = 0;
  double cost_remaining = total_cost;
  size_t procs_skipped = 0;
  size_t proc_num = 0;
  // check that we distributed the right number of elements to each proc based
  // on the sum of their costs in Z-curve index order
  while (proc_num < total_procs) {
    if (global_procs_to_ignore.count(proc_num)) {
      procs_skipped++;
      proc_num++;
      continue;
    }

    if (cost_index < num_elements) {
      // if we haven't accounted for all elements yet, we should still have cost
      // left to account for
      CHECK(cost_remaining <= total_cost);
    } else {
      // if we've already accounted for all the elements, we shouldn't have any
      // cost left to account for, and it's the case that more procs were
      // requested than could be used, e.g. in the case of less elements than
      // procs
      Approx custom_approx = Approx::custom().epsilon(1.0e-9).scale(1.0);
      CHECK(cost_remaining == custom_approx(0.0));
      break;
    }

    // the average cost per proc that we're aiming for
    const double target_proc_cost =
        cost_remaining /
        (number_of_procs_with_elements - proc_num + procs_skipped);

    // total cost on the processor before adding the cost of the final element
    // assigned to this proc
    double proc_cost_without_final_element = 0.0;
    const size_t num_elements_this_proc = num_elements_by_proc[proc_num];
    // add up costs of all elements but the final one to add
    for (size_t j = 0; j < num_elements_this_proc - 1; j++) {
      const double this_cost = costs_flattened[cost_index + j];
      proc_cost_without_final_element += this_cost;
    }

    // the cost of all of the elements assigned to this proc
    const double proc_cost_with_final_element =
        proc_cost_without_final_element +
        costs_flattened[cost_index + num_elements_this_proc - 1];

    const double diff_without_final_element =
        abs(proc_cost_without_final_element - target_proc_cost);

    const double diff_with_final_element =
        abs(proc_cost_with_final_element - target_proc_cost);

    // if the elements assigned to this proc have a cost that is over the target
    // cost per proc, make sure that either it's because only one element is
    // being assigned to the proc or this cost is closer to the target than if
    // we omitted the final element, i.e. check that it's better to keep the
    // final element than to not
    if (proc_cost_with_final_element > target_proc_cost) {
      CHECK((num_elements_this_proc == 1 or
             diff_with_final_element == approx(diff_without_final_element) or
             diff_with_final_element < diff_without_final_element));
    }

    if (cost_index + num_elements_this_proc < num_elements) {
      // total cost on the processor if we were to add the cost of the next
      // element (one additional than the number chosen)
      const double proc_cost_with_extra_element =
          proc_cost_with_final_element +
          costs_flattened[cost_index + num_elements_this_proc];
      const double diff_with_extra_element =
          abs(proc_cost_with_extra_element - target_proc_cost);

      // if it appears better to add one more element, check that it's because
      // the distance from the target cost with or without the additional
      // element is about the same
      if (diff_with_extra_element < diff_with_final_element) {
        Approx custom_approx = Approx::custom().epsilon(1.0e-12).scale(1.0);
        CHECK(diff_with_extra_element ==
              custom_approx(diff_with_final_element));
      }
    }

    cost_index += num_elements_this_proc;
    cost_remaining -= proc_cost_with_final_element;
    proc_num++;
  }

  // check that any remainder of processors we didn't need do indeed have 0
  // elements assigned to them
  while (proc_num < total_procs) {
    CHECK(num_elements_by_proc[proc_num] == 0);
    proc_num++;
  }
}

// Test the retrieval of the assigned processor that is done by
// `domain::BlockZCurveProcDistribution::get_proc_for_element`
template <size_t Dim>
void test_proc_retrieval(
    const domain::ElementWeight element_weight,
    const DomainCreator<Dim>& domain_creator,
    const size_t number_of_procs_with_elements,
    const std::unordered_set<size_t>& global_procs_to_ignore = {}) {
  const auto domain = domain_creator.create_domain();
  const auto& blocks = domain.blocks();
  const auto initial_refinement_levels =
      domain_creator.initial_refinement_levels();
  const auto initial_extents = domain_creator.initial_extents();

  const size_t num_blocks = blocks.size();
  std::vector<std::vector<ElementId<Dim>>> element_ids_in_z_curve_order(
      num_blocks);
  for (size_t i = 0; i < num_blocks; i++) {
    element_ids_in_z_curve_order[i] =
        initial_element_ids(i, gsl::at(initial_refinement_levels, i), 0);
    alg::sort(element_ids_in_z_curve_order[i],
              [](const ElementId<Dim>& lhs, const ElementId<Dim>& rhs) {
                return domain::z_curve_index(lhs) < domain::z_curve_index(rhs);
              });
  }

  const auto costs = domain::get_element_costs(
      blocks, initial_refinement_levels, initial_extents, element_weight,
      Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto);

  const domain::BlockZCurveProcDistribution<Dim> element_distribution(
      costs, number_of_procs_with_elements, blocks, initial_refinement_levels,
      initial_extents, global_procs_to_ignore);
  const auto proc_map = element_distribution.block_element_distribution();

  const size_t total_number_of_procs =
      number_of_procs_with_elements + global_procs_to_ignore.size();

  // whether or not we've assigned elements to a proc
  std::vector<bool> proc_hit(total_number_of_procs, false);

  size_t highest_proc_assigned = 0;
  for (size_t i = 0; i < num_blocks; i++) {
    size_t element_index = 0;
    const std::vector<std::pair<size_t, size_t>>& proc_map_this_block =
        proc_map[i];
    const size_t num_procs_this_block = proc_map_this_block.size();

    for (size_t j = 0; j < num_procs_this_block; j++) {
      const size_t expected_proc = proc_map_this_block[j].first;
      const size_t proc_allowance = proc_map_this_block[j].second;

      for (size_t k = 0; k < proc_allowance; k++) {
        const size_t actual_proc = element_distribution.get_proc_for_element(
            element_ids_in_z_curve_order[i][element_index]);
        // check that the correct processor is returned for the `ElementId`
        CHECK(actual_proc == expected_proc);
        proc_hit[actual_proc] = true;
        if (highest_proc_assigned < actual_proc) {
          highest_proc_assigned = actual_proc;
        }
      }
      element_index += proc_allowance;
    }
  }

  // check that ignored procs were indeed skipped and that all other procs
  // up to the highest one assigned were hit
  for (size_t i = 0; i < highest_proc_assigned + 1; i++) {
    if (global_procs_to_ignore.count(i) == 0) {
      CHECK(proc_hit[i]);
    } else {
      CHECK(not proc_hit[i]);
    }
  }
}

template <bool LocalTimeStepping>
struct ElementWeightMetavars {
  static constexpr bool local_time_stepping = LocalTimeStepping;
};

void test_element_weight_options() {
  using domain::ElementWeight;
  for (const auto weight :
       {ElementWeight::Uniform, ElementWeight::NumGridPoints,
        ElementWeight::NumGridPointsAndGridSpacing,
        ElementWeight::TestInteriorCubeGridPoints,
        ElementWeight::TestCcz4InteriorCubeWork}) {
    CHECK(
        TestHelpers::test_creation<ElementWeight, ElementWeightMetavars<true>>(
            get_output(weight)) == weight);
    test_serialization(weight);
  }
  CHECK(get_output(ElementWeight::TestInteriorCubeGridPoints) ==
        "TestInteriorCubeGridPoints");
  // Unlike NumGridPointsAndGridSpacing, the test weighting does not require
  // local time stepping.
  CHECK(TestHelpers::test_creation<ElementWeight, ElementWeightMetavars<false>>(
            "TestInteriorCubeGridPoints") ==
        ElementWeight::TestInteriorCubeGridPoints);
  CHECK(get_output(ElementWeight::TestCcz4InteriorCubeWork) ==
        "TestCcz4InteriorCubeWork");
  CHECK(TestHelpers::test_creation<ElementWeight, ElementWeightMetavars<false>>(
            "TestCcz4InteriorCubeWork") ==
        ElementWeight::TestCcz4InteriorCubeWork);
  CHECK_THROWS_WITH(
      (TestHelpers::test_creation<ElementWeight, ElementWeightMetavars<true>>(
          "InteriorCube")),
      Catch::Matchers::ContainsSubstring(
          "ElementWeight must be 'Uniform', 'NumGridPoints', "
          "'NumGridPointsAndGridSpacing', 'TestInteriorCubeGridPoints', or "
          "'TestCcz4InteriorCubeWork'"));
}

// A filled sphere with the structure of the all-wedges-DG layout: a cube with
// isotropic refinement and extents inside three shells of wedges with
// anisotropic extents. The angular refinements of the inner two shells are
// given, so the cube-wedge interface may be conforming or not.
domain::creators::Sphere interior_cube_sphere(
    const size_t cube_refinement, const size_t cube_extents,
    const std::array<size_t, 2>& shell_angular_refinements) {
  using RefinementOrExtents =
      std::unordered_map<std::string, std::array<size_t, 3>>;
  return domain::creators::Sphere{
      2.424871130596428,
      515.938061,
      domain::creators::Sphere::InnerCube{0.0},
      RefinementOrExtents{
          {"InnerCube", {{cube_refinement, cube_refinement, cube_refinement}}},
          {"Shell0",
           {{shell_angular_refinements[0], shell_angular_refinements[0], 0}}},
          {"Shell1",
           {{shell_angular_refinements[1], shell_angular_refinements[1], 0}}},
          {"Shell2", {{1, 1, 1}}}},
      RefinementOrExtents{
          {"InnerCube", {{cube_extents, cube_extents, cube_extents}}},
          {"Shell0", {{8, 8, 11}}},
          {"Shell1", {{8, 8, 9}}},
          {"Shell2", {{7, 7, 10}}}},
      false,
      std::nullopt,
      {25.0, 100.0}};
}

// The InnerCube elements that `evolution::dg::subcell::Actions::Initialize`
// puts on FD when every other block is DG-only: an element stays on DG if its
// block is DG-only or if any of its neighbors is in a DG-only block. This uses
// the neighbors of the initial elements, not their segment indices.
std::unordered_set<ElementId<3>> fd_elements_with_dg_only_exterior(
    const std::vector<Block<3>>& blocks,
    const std::vector<std::array<size_t, 3>>& refinements) {
  std::unordered_set<ElementId<3>> fd_elements{};
  for (const auto& block : blocks) {
    if (block.name() != "InnerCube") {
      continue;
    }
    for (const auto& element_id :
         initial_element_ids(block.id(), refinements[block.id()])) {
      const Element<3> element =
          domain::create_initial_element(element_id, blocks, refinements);
      // Only the DG-only blocks touch the outer boundary, so the subcell
      // external-boundary rule cannot change the decision.
      CHECK(element.external_boundaries().empty());
      if (alg::none_of(element.neighbors(),
                       [&block](const auto& direction_and_neighbors) {
                         return alg::any_of(
                             direction_and_neighbors.second.ids(),
                             [&block](const ElementId<3>& neighbor_id) {
                               return neighbor_id.block_id() != block.id();
                             });
                       })) {
        fd_elements.insert(element_id);
      }
    }
  }
  return fd_elements;
}

// Compare the TestInteriorCubeGridPoints weights with the FD set found from the
// initial element neighbors, and check the element counts and costs expected
// for the layout.
void test_interior_cube_costs(const domain::creators::Sphere& sphere,
                              const size_t expected_num_fd,
                              const size_t expected_num_collar,
                              const double expected_fd_cost,
                              const double expected_collar_cost) {
  const auto domain = sphere.create_domain();
  const auto& blocks = domain.blocks();
  const auto refinements = sphere.initial_refinement_levels();
  const auto extents = sphere.initial_extents();
  const auto costs = domain::get_element_costs(
      blocks, refinements, extents,
      domain::ElementWeight::TestInteriorCubeGridPoints, std::nullopt,
      std::nullopt);
  const auto fd_elements =
      fd_elements_with_dg_only_exterior(blocks, refinements);
  CHECK(fd_elements.size() == expected_num_fd);
  size_t num_fd = 0;
  size_t num_collar = 0;
  std::unordered_set<size_t> dg_wedge_blocks{};
  for (const auto& [id, cost] : costs) {
    const auto& block = blocks[id.block_id()];
    if (fd_elements.contains(id)) {
      CHECK(cost == expected_fd_cost);
      ++num_fd;
    } else if (block.name() == "InnerCube") {
      CHECK(cost == expected_collar_cost);
      ++num_collar;
    } else {
      CHECK(block.name().starts_with("Shell"));
      const auto& block_extents = extents[id.block_id()];
      CHECK(cost == static_cast<double>(block_extents[0] * block_extents[1] *
                                        block_extents[2]));
      dg_wedge_blocks.insert(id.block_id());
    }
  }
  CHECK(num_fd == expected_num_fd);
  CHECK(num_collar == expected_num_collar);
  // All six wedges of all three shells
  CHECK(dg_wedge_blocks.size() == 18);
}

void test_interior_cube_errors(const domain::creators::Sphere& sphere,
                               const domain::ElementWeight weight) {
  CAPTURE(weight);
  const auto test_weight_costs = [weight](const auto& blocks,
                                          const auto& refinements,
                                          const auto& extents) {
    return domain::get_element_costs(blocks, refinements, extents, weight,
                                     std::nullopt, std::nullopt);
  };
  const auto domain = sphere.create_domain();
  const auto& blocks = domain.blocks();
  const auto refinements = sphere.initial_refinement_levels();
  const auto extents = sphere.initial_extents();
  REQUIRE(blocks.back().name() == "InnerCube");
  const size_t cube_id = blocks.back().id();

  // The NumGridPoints FD weighting still rejects this layout.
  CHECK_THROWS_WITH(
      domain::get_element_costs(blocks, refinements, extents,
                                domain::ElementWeight::NumGridPoints,
                                std::nullopt, std::nullopt),
      Catch::Matchers::ContainsSubstring(
          "InnerCube block has initial refinement"));

  for (const auto& cube_refinement :
       {std::array<size_t, 3>{{1, 1, 1}}, std::array<size_t, 3>{{3, 3, 2}}}) {
    auto invalid_refinements = refinements;
    invalid_refinements[cube_id] = cube_refinement;
    CHECK_THROWS_WITH(
        test_weight_costs(blocks, invalid_refinements, extents),
        Catch::Matchers::ContainsSubstring(
            "requires isotropic initial refinement of at least 2 in the "
            "InnerCube block"));
  }
  auto invalid_extents = extents;
  invalid_extents[cube_id] = {{4, 4, 5}};
  CHECK_THROWS_WITH(test_weight_costs(blocks, refinements, invalid_extents),
                    Catch::Matchers::ContainsSubstring(
                        "requires isotropic initial extents in the InnerCube "
                        "block"));

  const domain::creators::Sphere excised_sphere{
      2.424871130596428,
      515.938061,
      domain::creators::Sphere::Excision{},
      1_st,
      4_st,
      false};
  CHECK_THROWS_WITH(
      test_weight_costs(excised_sphere.create_domain().blocks(),
                        excised_sphere.initial_refinement_levels(),
                        excised_sphere.initial_extents()),
      Catch::Matchers::ContainsSubstring(
          "requires exactly one block named 'InnerCube', but found 0"));

  // A lone InnerCube block has external boundaries on all six faces.
  const auto lone_inner_cube = [](const size_t id) {
    return Block<3>{
        domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
            domain::CoordinateMaps::Identity<3>{}),
        id,
        {},
        "InnerCube"};
  };
  std::vector<Block<3>> lone_blocks{};
  lone_blocks.push_back(lone_inner_cube(0));
  CHECK_THROWS_WITH(
      test_weight_costs(lone_blocks,
                        std::vector<std::array<size_t, 3>>(1, {{2, 2, 2}}),
                        std::vector<std::array<size_t, 3>>(1, {{4, 4, 4}})),
      Catch::Matchers::ContainsSubstring(
          "requires the InnerCube block to be connected to other blocks on all "
          "6 faces"));
  lone_blocks.push_back(lone_inner_cube(1));
  CHECK_THROWS_WITH(
      test_weight_costs(lone_blocks,
                        std::vector<std::array<size_t, 3>>(2, {{2, 2, 2}}),
                        std::vector<std::array<size_t, 3>>(2, {{4, 4, 4}})),
      Catch::Matchers::ContainsSubstring(
          "requires exactly one block named 'InnerCube', but found 2"));

  const auto lattice_1d = domain::creators::AlignedLattice<1>(
      {{{{0.0, 1.0}}}}, {{2}}, {{4}}, {}, {}, {});
  const auto lattice_2d = domain::creators::AlignedLattice<2>(
      {{{{0.0, 1.0}}, {{0.0, 1.0}}}}, {{2, 2}}, {{4, 4}}, {}, {}, {});
  CHECK_THROWS_WITH(
      test_weight_costs(lattice_1d.create_domain().blocks(),
                        lattice_1d.initial_refinement_levels(),
                        lattice_1d.initial_extents()),
      Catch::Matchers::ContainsSubstring(
          "requires a 3D domain, but the domain has dimension 1"));
  CHECK_THROWS_WITH(
      test_weight_costs(lattice_2d.create_domain().blocks(),
                        lattice_2d.initial_refinement_levels(),
                        lattice_2d.initial_extents()),
      Catch::Matchers::ContainsSubstring(
          "requires a 3D domain, but the domain has dimension 2"));
}

void test_interior_cube_work_costs(const domain::creators::Sphere& sphere) {
  const auto domain = sphere.create_domain();
  const auto& blocks = domain.blocks();
  const auto refinements = sphere.initial_refinement_levels();
  const auto extents = sphere.initial_extents();
  const auto point_costs = domain::get_element_costs(
      blocks, refinements, extents,
      domain::ElementWeight::TestInteriorCubeGridPoints, std::nullopt,
      std::nullopt);
  const auto work_costs =
      domain::get_element_costs(blocks, refinements, extents,
                                domain::ElementWeight::TestCcz4InteriorCubeWork,
                                std::nullopt, std::nullopt);
  const auto fd_elements =
      fd_elements_with_dg_only_exterior(blocks, refinements);
  REQUIRE(work_costs.size() == point_costs.size());
  // Pin the temporary calibration at several isotropic and anisotropic meshes.
  const std::unordered_map<size_t, double> expected_dg_costs{
      {27, 49.626351},    {216, 418.198464},  {512, 1069.940736},
      {704, 1541.320704}, {576, 1222.815744}, {490, 1018.3719}};
  for (const auto& [id, cost] : work_costs) {
    CAPTURE(id);
    const auto& n = extents[id.block_id()];
    if (fd_elements.contains(id)) {
      CHECK(cost == point_costs.at(id));
      CHECK(cost == static_cast<double>((2 * n[0] - 1) * (2 * n[1] - 1) *
                                        (2 * n[2] - 1)));
    } else {
      CHECK(cost == approx(expected_dg_costs.at(n[0] * n[1] * n[2])));
      CHECK(cost > point_costs.at(id));
    }
  }
}

void test_interior_cube_weighting() {
  test_element_weight_options();
  // The production layout: cube refinement 3 with 4^3 DG points, inside
  // innermost wedges of angular refinement 2 (a nonconforming interface). The
  // refinement-2 cube has a conforming interface, the refinement-4 cube a
  // nonconforming one.
  const auto production_sphere = interior_cube_sphere(3, 4, {{2, 1}});
  test_interior_cube_costs(production_sphere, 216, 296, 343.0, 64.0);
  test_interior_cube_costs(interior_cube_sphere(2, 5, {{2, 1}}), 8, 56, 729.0,
                           125.0);
  test_interior_cube_costs(interior_cube_sphere(4, 3, {{3, 2}}), 2744, 1352,
                           125.0, 27.0);
  for (const size_t num_procs : {186_st, 372_st}) {
    test_weighted_element_distribution_construction(
        domain::ElementWeight::TestInteriorCubeGridPoints, production_sphere,
        num_procs);
    test_proc_retrieval(domain::ElementWeight::TestInteriorCubeGridPoints,
                        production_sphere, num_procs);
  }
  test_interior_cube_errors(production_sphere,
                            domain::ElementWeight::TestInteriorCubeGridPoints);

  const auto work_sphere = interior_cube_sphere(3, 6, {{2, 1}});
  test_interior_cube_work_costs(work_sphere);
  test_interior_cube_work_costs(interior_cube_sphere(2, 8, {{2, 1}}));
  test_interior_cube_work_costs(interior_cube_sphere(4, 3, {{3, 2}}));
  for (const size_t num_procs : {186_st, 372_st}) {
    // Exercise the unchanged allocator with the nonintegral work weights,
    // including reserved processors that must receive no elements.
    const std::unordered_set<size_t> ignored_procs{0, 31};
    test_weighted_element_distribution_construction(
        domain::ElementWeight::TestCcz4InteriorCubeWork, work_sphere, num_procs,
        ignored_procs);
    test_proc_retrieval(domain::ElementWeight::TestCcz4InteriorCubeWork,
                        work_sphere, num_procs, ignored_procs);
  }
  test_interior_cube_errors(work_sphere,
                            domain::ElementWeight::TestCcz4InteriorCubeWork);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.ElementDistribution", "[Domain][Unit]") {
  // Test cost functions
  test_filled_sphere_costs();
  test_uniform_cost_function();
  test_weighted_cost_function(domain::ElementWeight::NumGridPoints);
  test_weighted_cost_function(
      domain::ElementWeight::NumGridPointsAndGridSpacing);

  // Inputs for testing `BlockZCurveProcDistribution`

  // 1D, single block
  const auto lattice_1d = domain::creators::AlignedLattice<1>(
      {{{{0.0, 1.0}}}}, {{4}}, {{6}}, {}, {}, {});
  // 2D
  const auto lattice_2d = domain::creators::AlignedLattice<2>(
      {{{{0.0, 0.3}}, {{0.0, 0.8, 2.5, 4.9}}}}, {{2, 3}}, {{4, 5}}, {}, {}, {});
  // 3D
  const auto lattice_3d = domain::creators::AlignedLattice<3>(
      {{{{0.0, 0.6}}, {{0.0, 0.4, 0.7}}, {{0.0, 0.3}}}}, {{2, 1, 3}},
      {{5, 4, 5}}, {}, {}, {});

  // Test element distribution construction logic

  // uniform distribution, single proc, single block
  test_uniform_element_distribution_construction(lattice_1d, 1);
  // uniform distribution, multiple procs, multiple blocks
  test_uniform_element_distribution_construction(
      lattice_2d, 20, std::unordered_set<size_t>{4, 20});
  // weighted distribution, multiple procs
  test_weighted_element_distribution_construction(
      domain::ElementWeight::NumGridPointsAndGridSpacing, lattice_3d, 22,
      std::unordered_set<size_t>{3, 4});
  // weighted distribution, more procs than elements to distribute
  test_weighted_element_distribution_construction(
      domain::ElementWeight::NumGridPoints, lattice_2d, 100,
      std::unordered_set<size_t>{0, 9});

  // Test processor retrieval with ignored processors
  test_proc_retrieval(domain::ElementWeight::NumGridPointsAndGridSpacing,
                      lattice_2d, 19, std::unordered_set<size_t>{0, 8, 9, 21});
  // Test processor retrieval when there are more processors requested than
  // `Element`s in the domain
  test_proc_retrieval(domain::ElementWeight::NumGridPointsAndGridSpacing,
                      lattice_2d, 100, std::unordered_set<size_t>{17});

  test_interior_cube_weighting();
}
