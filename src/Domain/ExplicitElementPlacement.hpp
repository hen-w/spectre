// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

/// \cond
template <size_t Dim>
class ElementId;
namespace Options {
class Option;
template <typename T>
struct create_from_yaml;
}  // namespace Options
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace domain {
/*!
 * \brief An explicit placement of the initial elements on processors
 *
 * \details Holds the global processor (Charm++ PE) number on which each
 * initial element is created, keyed by the name of the element. The name is
 * the output of `operator<<` for its `ElementId`, e.g. `[B0,(L1I0,L1I0,L0I0)]`,
 * and only this spelling is recognized. In an input file the names must be
 * quoted, since YAML would otherwise read them as sequences:
 *
 * \code{.yaml}
 * ElementDistribution:
 *   Explicit:
 *     "[B0,(L1I0,L1I0,L0I0)]": 0
 *     "[B0,(L1I1,L1I0,L0I0)]": 31
 * \endcode
 *
 * The placement only chooses the processor on which each initial element is
 * created. Binding processors to cores remains the task of the job launcher.
 * Since every initial element must be placed, the placement has to be
 * regenerated whenever the domain, its initial refinement, or the processors
 * change.
 */
class ExplicitElementPlacement {
 public:
  ExplicitElementPlacement() = default;
  explicit ExplicitElementPlacement(
      std::map<std::string, size_t> proc_by_element_name);

  /// The global processor number of each element, keyed by element name
  const std::map<std::string, size_t>& proc_by_element_name() const {
    return proc_by_element_name_;
  }

  /*!
   * \brief The processor of each initial element
   *
   * \details The initial elements are those returned by
   * `initial_element_ids(initial_refinement_levels)`. It is an error if the
   * placement names anything that is not an initial element, if it does not
   * name every initial element, or if it assigns an element to a processor
   * that is not below `number_of_procs` or is in `procs_to_ignore`.
   */
  template <size_t Dim>
  std::unordered_map<ElementId<Dim>, size_t> procs_for_initial_elements(
      const std::vector<std::array<size_t, Dim>>& initial_refinement_levels,
      size_t number_of_procs,
      const std::unordered_set<size_t>& procs_to_ignore) const;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

 private:
  std::map<std::string, size_t> proc_by_element_name_;
};

bool operator==(const ExplicitElementPlacement& lhs,
                const ExplicitElementPlacement& rhs);

bool operator!=(const ExplicitElementPlacement& lhs,
                const ExplicitElementPlacement& rhs);
}  // namespace domain

template <>
struct Options::create_from_yaml<domain::ExplicitElementPlacement> {
  template <typename Metavariables>
  static domain::ExplicitElementPlacement create(
      const Options::Option& options) {
    return create<void>(options);
  }
};

template <>
domain::ExplicitElementPlacement
Options::create_from_yaml<domain::ExplicitElementPlacement>::create<void>(
    const Options::Option& options);
