// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <optional>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/BoundaryVariablesTag.hpp"
#include "Evolution/DiscontinuousGalerkin/BoundaryEvolvedVariables.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/DataStructures/TestTags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/NoSuchType.hpp"
#include "Utilities/TMPL.hpp"

namespace {
using ScalarTag = TestHelpers::Tags::Scalar<DataVector>;
using Scalar2Tag = TestHelpers::Tags::Scalar2<DataVector>;
using BoundaryScalar = evolution::dg::Tags::BoundaryValue<ScalarTag>;

static_assert(std::is_same_v<BoundaryScalar::type, ScalarTag::type>);
static_assert(std::is_same_v<BoundaryScalar::tag, ScalarTag>);

// Systems for the boundary-variables detection: split with a
// BoundaryVariables entry, split without one (Cce-like), and single-tag.
struct SplitSystem {
  using variables_tag =
      tmpl::list<::Tags::Variables<tmpl::list<ScalarTag>>,
                 ::Tags::BoundaryVariables<2, tmpl::list<BoundaryScalar>>>;
};
struct CceLikeSystem {
  using variables_tag = tmpl::list<::Tags::Variables<tmpl::list<ScalarTag>>,
                                   ::Tags::Variables<tmpl::list<Scalar2Tag>>>;
};
struct SingleTagSystem {
  using variables_tag = ::Tags::Variables<tmpl::list<ScalarTag>>;
};

static_assert(evolution::dg::system_has_boundary_variables_v<SplitSystem>);
static_assert(
    not evolution::dg::system_has_boundary_variables_v<CceLikeSystem>);
static_assert(
    not evolution::dg::system_has_boundary_variables_v<SingleTagSystem>);
static_assert(
    std::is_same_v<evolution::dg::boundary_variables_tag<SplitSystem>,
                   ::Tags::BoundaryVariables<2, tmpl::list<BoundaryScalar>>>);
static_assert(
    std::is_same_v<evolution::dg::boundary_variables_tag<SingleTagSystem>,
                   NoSuchType>);

// Boundary conditions for the opt-in detection: the marker
// `evolves_boundary_variables` is the opt-in; the method's presence alone is
// not (an overloaded or templated method cannot be detected reliably, so
// the marker is authoritative).
struct OptingCondition {
  static constexpr bool evolves_boundary_variables = true;
  using boundary_field_time_derivatives_evolved_variables_tags =
      tmpl::list<ScalarTag>;
  using boundary_field_time_derivatives_temporary_tags = tmpl::list<Scalar2Tag>;
  std::optional<std::string> boundary_field_time_derivatives(
      gsl::not_null<Scalar<DataVector>*> /*dt_boundary_scalar*/) const {
    return std::nullopt;
  }
};
struct NonOptingCondition {
  std::optional<std::string> dg_ghost() const { return std::nullopt; }
};
// Defines the method but not the marker: does not opt in (the
// boundary-condition pass rejects this combination with a static_assert).
struct MethodWithoutMarkerCondition {
  std::optional<std::string> boundary_field_time_derivatives(
      gsl::not_null<Scalar<DataVector>*> /*dt_boundary_scalar*/) const {
    return std::nullopt;
  }
};
// Explicitly opts out.
struct MarkerFalseCondition {
  static constexpr bool evolves_boundary_variables = false;
};

static_assert(evolution::dg::evolves_boundary_variables_v<OptingCondition>);
static_assert(
    not evolution::dg::evolves_boundary_variables_v<NonOptingCondition>);
static_assert(not evolution::dg::evolves_boundary_variables_v<
              MethodWithoutMarkerCondition>);
static_assert(
    not evolution::dg::evolves_boundary_variables_v<MarkerFalseCondition>);

// The assembled interior inputs are ordered evolved, primitive, temporary,
// with each undeclared list defaulting to empty.
static_assert(
    std::is_same_v<evolution::dg::boundary_field_time_derivatives_interior_tags<
                       OptingCondition>,
                   tmpl::list<ScalarTag, Scalar2Tag>>);
static_assert(
    std::is_same_v<evolution::dg::boundary_field_time_derivatives_interior_tags<
                       NonOptingCondition>,
                   tmpl::list<>>);

SPECTRE_TEST_CASE("Unit.Evolution.Dg.BoundaryEvolvedVariables",
                  "[Unit][Evolution]") {
  TestHelpers::db::test_prefix_tag<BoundaryScalar>("BoundaryValue(Scalar)");
}
}  // namespace
