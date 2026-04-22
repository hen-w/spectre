// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cmath>
#include <cstddef>
#include <optional>
#include <pup.h>
#include <pup_stl.h>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/DataBox/ValidateSelection.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Options/Auto.hpp"
#include "Options/Context.hpp"
#include "Options/String.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
struct GlobalCache;
}  // namespace Parallel
namespace tuples {
template <typename... Tags>
struct TaggedTuple;
}  // namespace tuples
/// \endcond

namespace Actions {

/*!
 * \brief Optionally add a deterministic, localized Gaussian perturbation to
 * selected variables.
 *
 * Adds a perturbation of the form
 * \f$\text{amplitude} \cdot \exp(-|x - \text{center}|^2 /
 * \text{width}^2)\f$
 * to all independent tensor components of each selected variable. The
 * perturbation is deterministic (no RNG) and spatially localized.
 *
 * The action can be disabled by setting the input-file option to `None`.
 */
template <typename VariablesTag, typename Label>
struct LocalizedPerturbation {
 public:
  using tags_list = typename VariablesTag::type::tags_list;

  struct PerturbationParameters {
    struct VariablesToPerturb {
      using type = std::vector<std::string>;
      static constexpr Options::String help =
          "Names of the variables to perturb.";
      static size_t lower_bound_on_size() { return 1; }
    };
    struct Amplitude {
      using type = double;
      static constexpr Options::String help =
          "Amplitude of the Gaussian perturbation.";
    };
    struct Width {
      using type = double;
      static constexpr Options::String help =
          "Width of the Gaussian perturbation.";
      static double lower_bound() { return 0.0; }
    };
    struct Center {
      using type = std::vector<double>;
      static constexpr Options::String help =
          "Center of the Gaussian perturbation.";
    };
    struct SphericalShellGaussian {
      using type = Options::Auto<double, Options::AutoLabel::None>;
      static constexpr Options::String help =
          "If set, use a spherically symmetric Gaussian localized in the "
          "radial direction, peaked at the given radius from Center. The "
          "perturbation is amplitude * exp(-(r - r0)^2 / width^2) where "
          "r = |x - Center| and r0 is this value. Set to None to use a "
          "standard Cartesian Gaussian.";
    };
    using options =
        tmpl::list<VariablesToPerturb, Amplitude, Width, Center,
                   SphericalShellGaussian>;
    static constexpr Options::String help =
        "Parameters for a localized Gaussian perturbation.";

    PerturbationParameters() = default;
    PerturbationParameters(
        std::vector<std::string> in_variables_to_perturb, double in_amplitude,
        double in_width, std::vector<double> in_center,
        std::optional<double> in_spherical_shell_radial_center,
        const Options::Context& context = {})
        : variables_to_perturb(std::move(in_variables_to_perturb)),
          amplitude(in_amplitude),
          width(in_width),
          center(std::move(in_center)),
          spherical_shell_radial_center(in_spherical_shell_radial_center) {
      db::validate_selection<tags_list>(variables_to_perturb, context);
    }

    void pup(PUP::er& p) {
      p | variables_to_perturb;
      p | amplitude;
      p | width;
      p | center;
      p | spherical_shell_radial_center;
    }

    std::vector<std::string> variables_to_perturb{};
    double amplitude{};
    double width{};
    std::vector<double> center{};
    std::optional<double> spherical_shell_radial_center{};
  };

  struct PerturbationParametersOptionTag {
    static std::string name() { return pretty_type::name<Label>(); }
    using type =
        Options::Auto<PerturbationParameters, Options::AutoLabel::None>;
    static constexpr Options::String help =
        "Add a localized Gaussian perturbation to selected variables.";
  };

  struct PerturbationParametersTag : db::SimpleTag {
    using type = std::optional<PerturbationParameters>;
    using option_tags = tmpl::list<PerturbationParametersOptionTag>;
    static constexpr bool pass_metavariables = false;
    static type create_from_options(const type& value) { return value; }
  };

  using const_global_cache_tags = tmpl::list<PerturbationParametersTag>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& /*element_id*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const std::optional<PerturbationParameters>& params =
        db::get<PerturbationParametersTag>(box);
    if (not params.has_value()) {
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    }
    const auto& p = params.value();
    ASSERT(p.center.size() == Dim,
           "Center has " << p.center.size()
                         << " components but the domain is " << Dim
                         << "-dimensional.");
    const auto& coords =
        db::get<domain::Tags::Coordinates<Dim, Frame::Inertial>>(box);
    // Compute r_squared = |x - center|^2
    DataVector r_squared(get<0>(coords).size(), 0.0);
    for (size_t d = 0; d < Dim; ++d) {
      r_squared += square(coords.get(d) - gsl::at(p.center, d));
    }
    DataVector window(get<0>(coords).size());
    if (p.spherical_shell_radial_center.has_value()) {
      // Spherically symmetric Gaussian in radial direction:
      // amplitude * exp(-(r - r0)^2 / width^2)
      const DataVector r = sqrt(r_squared);
      window = p.amplitude *
               exp(-square(r - p.spherical_shell_radial_center.value()) /
                   square(p.width));
    } else {
      // Standard Cartesian Gaussian: amplitude * exp(-r^2 / width^2)
      window = p.amplitude * exp(-r_squared / square(p.width));
    }
    // Add perturbation to selected variables
    const auto& vars_to_perturb = p.variables_to_perturb;
    db::mutate<VariablesTag>(
        [&vars_to_perturb, &window](const auto fields) {
          tmpl::for_each<tags_list>([&](const auto tag_v) {
            using tag = tmpl::type_from<std::decay_t<decltype(tag_v)>>;
            if (alg::found(vars_to_perturb, db::tag_name<tag>())) {
              auto& tensor = get<tag>(*fields);
              for (size_t i = 0; i < tensor.size(); ++i) {
                tensor[i] += window;
              }
            }
          });
        },
        make_not_null(&box));
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

}  // namespace Actions
