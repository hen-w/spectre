// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/ExplicitElementPlacement.hpp"

#include <array>
#include <cstddef>
#include <map>
#include <pup.h>
#include <pup_stl.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <yaml-cpp/yaml.h>

#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "Options/Context.hpp"
#include "Options/ParseError.hpp"
#include "Options/ParseOptions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/GetOutput.hpp"

namespace domain {
ExplicitElementPlacement::ExplicitElementPlacement(
    std::map<std::string, size_t> proc_by_element_name)
    : proc_by_element_name_(std::move(proc_by_element_name)) {}

template <size_t Dim>
std::unordered_map<ElementId<Dim>, size_t>
ExplicitElementPlacement::procs_for_initial_elements(
    const std::vector<std::array<size_t, Dim>>& initial_refinement_levels,
    const size_t number_of_procs,
    const std::unordered_set<size_t>& procs_to_ignore) const {
  const std::vector<ElementId<Dim>> element_ids =
      initial_element_ids(initial_refinement_levels);
  std::unordered_map<std::string, ElementId<Dim>> element_ids_by_name{};
  for (const auto& element_id : element_ids) {
    element_ids_by_name.emplace(get_output(element_id), element_id);
  }

  std::unordered_map<ElementId<Dim>, size_t> procs{};
  for (const auto& [name, proc] : proc_by_element_name_) {
    const auto element = element_ids_by_name.find(name);
    if (element == element_ids_by_name.end()) {
      ERROR("The explicit element placement names '"
            << name
            << "', which is not an initial element of the domain. Elements "
               "are named as ElementId<"
            << Dim << "> prints them, e.g. '" << ElementId<Dim>{0} << "'.");
    }
    if (proc >= number_of_procs) {
      ERROR("The explicit element placement assigns element "
            << name << " to processor " << proc << ", but there are only "
            << number_of_procs << " processors.");
    }
    if (procs_to_ignore.contains(proc)) {
      ERROR("The explicit element placement assigns element "
            << name << " to processor " << proc
            << ", which is excluded from element placement.");
    }
    procs.emplace(element->second, proc);
  }
  for (const auto& element_id : element_ids) {
    if (not procs.contains(element_id)) {
      ERROR(
          "The explicit element placement does not assign the initial "
          "element "
          << element_id << ". It assigns " << procs.size() << " of the "
          << element_ids.size()
          << " initial elements, but must assign all of them.");
    }
  }
  return procs;
}

void ExplicitElementPlacement::pup(PUP::er& p) { p | proc_by_element_name_; }

bool operator==(const ExplicitElementPlacement& lhs,
                const ExplicitElementPlacement& rhs) {
  return lhs.proc_by_element_name() == rhs.proc_by_element_name();
}

bool operator!=(const ExplicitElementPlacement& lhs,
                const ExplicitElementPlacement& rhs) {
  return not(lhs == rhs);
}

#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                  \
  template std::unordered_map<ElementId<GET_DIM(data)>, size_t> \
  ExplicitElementPlacement::procs_for_initial_elements(         \
      const std::vector<std::array<size_t, GET_DIM(data)>>&     \
          initial_refinement_levels,                            \
      size_t number_of_procs,                                   \
      const std::unordered_set<size_t>& procs_to_ignore) const;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef GET_DIM
#undef INSTANTIATION
}  // namespace domain

template <>
domain::ExplicitElementPlacement
Options::create_from_yaml<domain::ExplicitElementPlacement>::create<void>(
    const Options::Option& options) {
  const YAML::Node& node = options.node();
  if (not node.IsMap() or node.size() == 0) {
    PARSE_ERROR(options.context(),
                "Expected a nonempty map from quoted element names to "
                "processor numbers, but got: "
                    << node);
  }
  std::map<std::string, size_t> proc_by_element_name{};
  for (const auto& name_and_proc : node) {
    Options::Context context = options.context();
    context.line = name_and_proc.first.Mark().line;
    context.column = name_and_proc.first.Mark().column;
    if (not name_and_proc.first.IsScalar()) {
      PARSE_ERROR(context,
                  "Element names must be quoted strings such as "
                  "\"[B0,(L1I0,L0I0)]\", but got: "
                      << name_and_proc.first);
    }
    const std::string& name = name_and_proc.first.Scalar();
    Options::Option proc(name_and_proc.second, options.context());
    proc.append_context("While parsing the processor of element " + name);
    if (not proc_by_element_name.emplace(name, proc.parse_as<size_t>())
                .second) {
      PARSE_ERROR(context, "Element " << name << " is specified twice.");
    }
  }
  return domain::ExplicitElementPlacement{std::move(proc_by_element_name)};
}
