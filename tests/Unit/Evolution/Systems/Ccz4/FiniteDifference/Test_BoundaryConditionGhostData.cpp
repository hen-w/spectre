// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <pup.h>
#include <tuple>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/MetavariablesTag.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/IndexIterator.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Domain/Creators/Tags/ExternalBoundaryConditions.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Side.hpp"
#include "Evolution/BoundaryConditions/Type.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Systems/Ccz4/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/BoundaryConditionGhostData.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Derivatives.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Filter.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/UnlimitedDeg4Prim.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Evolution/Systems/Ccz4/PrimReconstructor.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Time/Tags/Time.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace {
using GhostVars = Variables<Ccz4::fd::System::variables_tag_list>;
constexpr size_t number_of_components =
    GhostVars::number_of_independent_components;
constexpr size_t number_of_physical_components =
    Variables<Ccz4::fd::System::original_evolved_variables_tags>::
        number_of_independent_components;
static_assert(number_of_components == 55 and
              number_of_physical_components == 25);
constexpr size_t ghost_zone_size = 3;
constexpr double gradient_poison = 3.0e3;
using LogicalCoords = tnsr::I<DataVector, 3, Frame::ElementLogical>;

// The physical component `c` (counted in the storage order of the
// Variables) at time `time`: a cubic polynomial in the logical coordinates,
// differentiated exactly by the fourth-order derivatives and annihilated by
// the sixth-order Kreiss-Oliger operator, plus a multiple of the time.
DataVector field(const size_t c, const LogicalCoords& x, const double time) {
  const double k = static_cast<double>(c);
  DataVector result = 1.0 + 0.1 * k + (0.3 + 0.01 * k) * get<0>(x) -
                      0.2 * get<1>(x) + (0.1 - 0.02 * k) * get<2>(x) +
                      0.05 * get<0>(x) * get<1>(x) -
                      0.04 * get<1>(x) * square(get<2>(x)) +
                      0.03 * cube(get<0>(x)) + (k + 1.0) * time;
  return result;
}

// The logical coordinates of the ghost cells of the face in `direction`,
// without the ghost-zone coordinate helper: `ghost_zone_size` cells continuing
// the uniform cell-centered grid outside the face, ordered like the volume
// cells, at the tangential coordinates of the cells.
LogicalCoords ghost_coordinates(const Mesh<3>& mesh,
                                const Direction<3>& direction) {
  const size_t dim = direction.dimension();
  Index<3> extents = mesh.extents();
  extents[dim] = ghost_zone_size;
  LogicalCoords coords{extents.product()};
  for (IndexIterator<3> index(extents); index; ++index) {
    for (size_t d = 0; d < 3; ++d) {
      const double spacing = 2.0 / static_cast<double>(mesh.extents(d));
      const auto i = static_cast<double>(index()[d]);
      if (d == dim) {
        const double depth =
            direction.side() == Side::Upper
                ? i + 0.5
                : static_cast<double>(ghost_zone_size) - i - 0.5;
        coords.get(d)[index.collapsed_index()] =
            direction.sign() * (1.0 + depth * spacing);
      } else {
        coords.get(d)[index.collapsed_index()] = -1.0 + (i + 0.5) * spacing;
      }
    }
  }
  return coords;
}

// The volume variables, and the ghost data sent by neighbors: `field` and
// poisoned gradients.
GhostVars volume_vars(const LogicalCoords& x, const double time) {
  GhostVars vars{get<0>(x).size(), gradient_poison};
  for (size_t c = 0; c < number_of_physical_components; ++c) {
    DataVector component(vars.data() + c * vars.number_of_grid_points(),
                         vars.number_of_grid_points());
    component = field(c, x, time);
  }
  return vars;
}

// Sets the physical fields of the ghost cells to `field` at the current time.
class PolynomialGhost final
    : public Ccz4::BoundaryConditions::BoundaryCondition {
 public:
  PolynomialGhost() = default;
  PolynomialGhost(PolynomialGhost&&) = default;
  PolynomialGhost& operator=(PolynomialGhost&&) = default;
  PolynomialGhost(const PolynomialGhost&) = default;
  PolynomialGhost& operator=(const PolynomialGhost&) = default;
  ~PolynomialGhost() override = default;
  explicit PolynomialGhost(CkMigrateMessage* msg) : BoundaryCondition(msg) {}
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
  // NOLINTNEXTLINE
  WRAPPED_PUPable_decl_base_template(
      domain::BoundaryConditions::BoundaryCondition, PolynomialGhost);
#pragma GCC diagnostic pop
  auto get_clone() const -> std::unique_ptr<
      domain::BoundaryConditions::BoundaryCondition> override {
    return std::make_unique<PolynomialGhost>(*this);
  }

  static constexpr evolution::BoundaryConditions::Type bc_type =
      evolution::BoundaryConditions::Type::Ghost;

  using fd_interior_evolved_variables_tags = tmpl::list<>;
  using fd_interior_temporary_tags =
      tmpl::list<evolution::dg::subcell::Tags::Mesh<3>>;
  using fd_gridless_tags =
      tmpl::list<::Tags::Time, Ccz4::fd::Tags::Reconstructor>;

  void fd_ghost(
      const gsl::not_null<tnsr::ii<DataVector, 3>*> conformal_metric,
      const gsl::not_null<Scalar<DataVector>*> conformal_factor,
      const gsl::not_null<tnsr::ii<DataVector, 3>*> a_tilde,
      const gsl::not_null<Scalar<DataVector>*> trace_extrinsic_curvature,
      const gsl::not_null<Scalar<DataVector>*> theta,
      const gsl::not_null<tnsr::I<DataVector, 3>*> gamma_hat,
      const gsl::not_null<Scalar<DataVector>*> lapse,
      const gsl::not_null<tnsr::I<DataVector, 3>*> shift,
      const gsl::not_null<tnsr::I<DataVector, 3>*> b,
      const Direction<3>& direction, const Mesh<3>& subcell_mesh,
      const double time, const Ccz4::fd::Reconstructor& reconstructor) const {
    CHECK(reconstructor.ghost_zone_size() == ghost_zone_size);
    const LogicalCoords coords = ghost_coordinates(subcell_mesh, direction);
    size_t c = 0;
    const auto set = [&c, &coords, &time](const auto tensor) {
      for (auto& component : *tensor) {
        component = field(c, coords, time);
        ++c;
      }
    };
    set(conformal_metric);
    set(conformal_factor);
    set(a_tilde);
    set(trace_extrinsic_curvature);
    set(theta);
    set(gamma_hat);
    set(lapse);
    set(shift);
    set(b);
    CHECK(c == number_of_physical_components);
  }
};

// A condition without finite-difference ghost data.
class TimeDerivativeCondition final
    : public Ccz4::BoundaryConditions::BoundaryCondition {
 public:
  TimeDerivativeCondition() = default;
  TimeDerivativeCondition(TimeDerivativeCondition&&) = default;
  TimeDerivativeCondition& operator=(TimeDerivativeCondition&&) = default;
  TimeDerivativeCondition(const TimeDerivativeCondition&) = default;
  TimeDerivativeCondition& operator=(const TimeDerivativeCondition&) = default;
  ~TimeDerivativeCondition() override = default;
  explicit TimeDerivativeCondition(CkMigrateMessage* msg)
      : BoundaryCondition(msg) {}
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
  // NOLINTNEXTLINE
  WRAPPED_PUPable_decl_base_template(
      domain::BoundaryConditions::BoundaryCondition, TimeDerivativeCondition);
#pragma GCC diagnostic pop
  auto get_clone() const -> std::unique_ptr<
      domain::BoundaryConditions::BoundaryCondition> override {
    return std::make_unique<TimeDerivativeCondition>(*this);
  }

  static constexpr evolution::BoundaryConditions::Type bc_type =
      evolution::BoundaryConditions::Type::TimeDerivative;
};

PUP::able::PUP_ID PolynomialGhost::my_PUP_ID = 0;          // NOLINT
PUP::able::PUP_ID TimeDerivativeCondition::my_PUP_ID = 0;  // NOLINT

using Periodic = domain::BoundaryConditions::Periodic<
    Ccz4::BoundaryConditions::BoundaryCondition>;

struct Metavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<tmpl::pair<
        Ccz4::BoundaryConditions::BoundaryCondition,
        tmpl::list<PolynomialGhost, TimeDerivativeCondition, Periodic>>>;
  };
};

using ExternalBoundaryConditions =
    domain::Tags::ExternalBoundaryConditions<3>::type;
using GhostDataMap = DirectionalIdMap<3, evolution::dg::subcell::GhostData>;

template <typename Condition>
ExternalBoundaryConditions conditions_on(
    const std::vector<Direction<3>>& external_directions) {
  ExternalBoundaryConditions conditions(1);
  for (const auto& direction : external_directions) {
    conditions[0][direction] = std::make_unique<Condition>();
  }
  return conditions;
}

auto make_box(ExternalBoundaryConditions conditions, const Mesh<3>& mesh,
              GhostDataMap ghost_data, const double time) {
  return db::create<db::AddSimpleTags<
      Parallel::Tags::MetavariablesImpl<Metavariables>,
      domain::Tags::ExternalBoundaryConditions<3>,
      evolution::dg::subcell::Tags::Mesh<3>,
      evolution::dg::subcell::Tags::GhostDataForReconstruction<3>,
      Ccz4::fd::Tags::Reconstructor, ::Tags::Time>>(
      Metavariables{}, std::move(conditions), mesh, std::move(ghost_data),
      std::unique_ptr<Ccz4::fd::Reconstructor>{
          std::make_unique<Ccz4::fd::UnlimitedDeg4Prim>()},
      time);
}

template <typename DbTagsList>
void apply(const gsl::not_null<db::DataBox<DbTagsList>*> box,
           const Element<3>& element) {
  Ccz4::fd::BoundaryConditionGhostData::apply(
      box, element, db::get<Ccz4::fd::Tags::Reconstructor>(*box));
}

DirectionalId<3> external_id(const Direction<3>& direction) {
  return {direction, ElementId<3>::external_boundary_id()};
}

// The neighbor entries are unchanged and the external entry in each external
// direction has the full ghost-data layout: the physical fields of
// `PolynomialGhost` at `time`, and default-initialized gradients (signaling
// NaN with SPECTRE_NAN_INIT, compared bitwise; otherwise uninitialized and not
// read).
void check_ghost_data(const GhostDataMap& ghost_data,
                      const GhostDataMap& neighbor_ghost_data,
                      const Element<3>& element, const Mesh<3>& mesh,
                      const double time) {
  CHECK(ghost_data.size() ==
        neighbor_ghost_data.size() + element.external_boundaries().size());
  for (const auto& [id, neighbor_entry] : neighbor_ghost_data) {
    CAPTURE(id);
    REQUIRE(ghost_data.contains(id));
    CHECK(ghost_data.at(id) == neighbor_entry);
  }
  for (const auto& direction : element.external_boundaries()) {
    CAPTURE(direction);
    REQUIRE(ghost_data.contains(external_id(direction)));
    const DataVector& data = ghost_data.at(external_id(direction))
                                 .neighbor_ghost_data_for_reconstruction();
    const size_t number_of_points =
        ghost_zone_size *
        mesh.extents().slice_away(direction.dimension()).product();
    REQUIRE(data.size() == number_of_components * number_of_points);
    const LogicalCoords coords = ghost_coordinates(mesh, direction);
    for (size_t c = 0; c < number_of_physical_components; ++c) {
      CAPTURE(c);
      const DataVector component(
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          const_cast<double*>(data.data()) + c * number_of_points,
          number_of_points);
      CHECK(component == field(c, coords, time));
    }
#ifdef SPECTRE_NAN_INIT
    const size_t tail_size =
        (number_of_components - number_of_physical_components) *
        number_of_points;
    std::vector<std::uint64_t> tail_bits(tail_size);
    std::memcpy(tail_bits.data(),
                data.data() + number_of_physical_components * number_of_points,
                tail_size * sizeof(double));
    constexpr auto signaling_nan_bits = std::bit_cast<std::uint64_t>(
        std::numeric_limits<double>::signaling_NaN());
    CHECK(std::all_of(
        tail_bits.begin(), tail_bits.end(),
        [](const std::uint64_t bits) { return bits == signaling_nan_bits; }));
#endif  // SPECTRE_NAN_INIT
  }
}

// The external entries are inserted next to the received neighbor entries,
// in all external directions, at the current time, and replaced by fresh ones
// on each call.
void test_ghost_data(const Mesh<3>& mesh,
                     const std::vector<Direction<3>>& external_directions) {
  CAPTURE(mesh);
  const Element<3> element =
      TestHelpers::Ccz4::fd::detail::set_element_brick(external_directions);
  REQUIRE(element.external_boundaries().size() == external_directions.size());
  const LogicalCoords coords = logical_coordinates(mesh);
  double time = 0.25;
  const auto neighbor_vars = [&time](const LogicalCoords& x) {
    return volume_vars(x, time);
  };
  const GhostDataMap neighbor_ghost_data =
      TestHelpers::Ccz4::fd::detail::compute_ghost_data(
          mesh, coords, element.neighbors(), ghost_zone_size, neighbor_vars);
  auto box = make_box(conditions_on<PolynomialGhost>(external_directions), mesh,
                      neighbor_ghost_data, time);
  apply(make_not_null(&box), element);
  const auto& ghost_data =
      db::get<evolution::dg::subcell::Tags::GhostDataForReconstruction<3>>(box);
  check_ghost_data(ghost_data, neighbor_ghost_data, element, mesh, time);

  // A later call replaces the external entries with those of the current
  // time.
  time = 0.75;
  db::mutate<::Tags::Time>(
      [&time](const gsl::not_null<double*> box_time) { *box_time = time; },
      make_not_null(&box));
  apply(make_not_null(&box), element);
  check_ghost_data(ghost_data, neighbor_ghost_data, element, mesh, time);

  // After an exchange, which replaces all ghost data with the neighbors'.
  db::mutate<evolution::dg::subcell::Tags::GhostDataForReconstruction<3>>(
      [&neighbor_ghost_data](const gsl::not_null<GhostDataMap*> all_data) {
        *all_data = neighbor_ghost_data;
      },
      make_not_null(&box));
  apply(make_not_null(&box), element);
  check_ghost_data(ghost_data, neighbor_ghost_data, element, mesh, time);
}

// The finite-difference derivatives and Kreiss-Oliger dissipation with the
// external ghost data (floating-point exceptions are enabled) are those with
// neighbors in all directions holding the same fields, and do not depend on
// the gradients of the external ghost data. (These require an isotropic
// mesh.)
void test_consumers(const Mesh<3>& mesh,
                    const std::vector<Direction<3>>& external_directions) {
  CAPTURE(mesh);
  const Element<3> element =
      TestHelpers::Ccz4::fd::detail::set_element_brick(external_directions);
  const LogicalCoords coords = logical_coordinates(mesh);
  const double time = 0.5;
  const auto neighbor_vars = [&time](const LogicalCoords& x) {
    return volume_vars(x, time);
  };
  auto box = make_box(
      conditions_on<PolynomialGhost>(external_directions), mesh,
      TestHelpers::Ccz4::fd::detail::compute_ghost_data(
          mesh, coords, element.neighbors(), ghost_zone_size, neighbor_vars),
      time);
  apply(make_not_null(&box), element);
  const auto& ghost_data =
      db::get<evolution::dg::subcell::Tags::GhostDataForReconstruction<3>>(box);
  REQUIRE(ghost_data.size() == 6);

  const GhostVars volume = volume_vars(coords, time);
  const GhostDataMap reference_ghost_data =
      TestHelpers::Ccz4::fd::detail::compute_ghost_data(
          mesh, coords,
          TestHelpers::Ccz4::fd::detail::set_element().neighbors(),
          ghost_zone_size, neighbor_vars);
  GhostDataMap changed_gradients_ghost_data = ghost_data;
  for (const auto& direction : element.external_boundaries()) {
    DataVector& data = changed_gradients_ghost_data.at(external_id(direction))
                           .neighbor_ghost_data_for_reconstruction();
    const size_t number_of_ghost_points = data.size() / number_of_components;
    std::fill(
        data.data() + number_of_physical_components * number_of_ghost_points,
        data.data() + data.size(), 1.0e300);
  }
  const size_t number_of_points = mesh.number_of_grid_points();
  InverseJacobian<DataVector, 3, Frame::ElementLogical, Frame::Inertial>
      inverse_jacobian{number_of_points, 0.0};
  for (size_t i = 0; i < 3; ++i) {
    inverse_jacobian.get(i, i) = 1.0;
  }
  const InverseHessian<DataVector, 3, Frame::ElementLogical, Frame::Inertial>
      inverse_hessian{number_of_points, 0.0};
  using gradients_tags = Ccz4::fd::System::gradients_tags;
  using DerivVars =
      Variables<db::wrap_tags_in<::Tags::deriv, gradients_tags, tmpl::size_t<3>,
                                 Frame::Inertial>>;
  using SecondDerivVars =
      Variables<db::wrap_tags_in<::Tags::second_deriv, gradients_tags,
                                 tmpl::size_t<3>, Frame::Inertial>>;
  const auto consumers = [&](const GhostDataMap& all_ghost_data) {
    DerivVars derivatives{number_of_points};
    Ccz4::fd::spacetime_derivatives(make_not_null(&derivatives), volume,
                                    all_ghost_data, 4, mesh, inverse_jacobian);
    SecondDerivVars second_derivatives{number_of_points};
    Ccz4::fd::second_spacetime_derivatives(make_not_null(&second_derivatives),
                                           volume, all_ghost_data, 4, mesh,
                                           inverse_jacobian, inverse_hessian);
    GhostVars filtered = volume;
    Ccz4::fd::ccz4_kreiss_oliger_filter(make_not_null(&filtered), volume,
                                        all_ghost_data, true, mesh, 6, 0.1);
    return std::tuple{std::move(derivatives), std::move(second_derivatives),
                      std::move(filtered)};
  };
  const auto [derivatives, second_derivatives, filtered] =
      consumers(ghost_data);
  const auto [reference_derivatives, reference_second_derivatives,
              reference_filtered] = consumers(reference_ghost_data);
  // The second derivatives amplify roundoff by 1/h^2.
  Approx second_derivative_approx =
      Approx::custom().epsilon(1.0e-12).scale(1.0);
  CHECK_VARIABLES_APPROX(derivatives, reference_derivatives);
  CHECK_VARIABLES_CUSTOM_APPROX(second_derivatives,
                                reference_second_derivatives,
                                second_derivative_approx);
  CHECK_VARIABLES_APPROX(filtered, reference_filtered);
  CHECK_VARIABLES_APPROX(filtered, volume);
  const auto [changed_derivatives, changed_second_derivatives,
              changed_filtered] = consumers(changed_gradients_ghost_data);
  CHECK(changed_derivatives == derivatives);
  CHECK(changed_second_derivatives == second_derivatives);
  CHECK(changed_filtered == filtered);

  // The values against the exact derivatives of the cubic, for the
  // components of the lapse (c = 18) and of the first shift component
  // (c = 19).
  const auto& x = get<0>(coords);
  const auto& y = get<1>(coords);
  const auto& z = get<2>(coords);
  for (const size_t c : {18_st, 19_st}) {
    const double k = static_cast<double>(c);
    const auto& d_u =
        c == 18
            ? get<0>(get<::Tags::deriv<gr::Tags::Lapse<DataVector>,
                                       tmpl::size_t<3>, Frame::Inertial>>(
                  derivatives))
            : get<0, 0>(get<::Tags::deriv<gr::Tags::Shift<DataVector, 3>,
                                          tmpl::size_t<3>, Frame::Inertial>>(
                  derivatives));
    const DataVector expected_d_u =
        0.3 + 0.01 * k + 0.05 * y + 0.09 * square(x);
    CHECK_ITERABLE_APPROX(d_u, expected_d_u);
  }
  const auto& dd_lapse =
      get<::Tags::second_deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<3>,
                               Frame::Inertial>>(second_derivatives);
  const DataVector expected_dd_xx = 0.18 * x;
  const DataVector expected_dd_xy{number_of_points, 0.05};
  const DataVector expected_dd_yz = -0.08 * z;
  const DataVector expected_dd_zz = -0.08 * y;
  CHECK_ITERABLE_CUSTOM_APPROX(dd_lapse.get(0, 0), expected_dd_xx,
                               second_derivative_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(dd_lapse.get(0, 1), expected_dd_xy,
                               second_derivative_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(dd_lapse.get(1, 2), expected_dd_yz,
                               second_derivative_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(dd_lapse.get(2, 2), expected_dd_zz,
                               second_derivative_approx);
}

// No external boundaries: no change (and the boundary conditions are not
// accessed).
void test_no_external_boundaries(const Mesh<3>& mesh) {
  const Element<3> element = TestHelpers::Ccz4::fd::detail::set_element();
  const auto neighbor_vars = [](const LogicalCoords& x) {
    return volume_vars(x, 0.0);
  };
  const GhostDataMap neighbor_ghost_data =
      TestHelpers::Ccz4::fd::detail::compute_ghost_data(
          mesh, logical_coordinates(mesh), element.neighbors(), ghost_zone_size,
          neighbor_vars);
  auto box =
      make_box(ExternalBoundaryConditions{}, mesh, neighbor_ghost_data, 0.0);
  apply(make_not_null(&box), element);
  check_ghost_data(
      db::get<evolution::dg::subcell::Tags::GhostDataForReconstruction<3>>(box),
      neighbor_ghost_data, element, mesh, 0.0);
}

// Conditions without finite-difference ghost data are rejected, and periodic
// conditions are not dispatched.
void test_unsupported(const Mesh<3>& mesh) {
  const std::vector<Direction<3>> external_directions{
      Direction<3>::upper_eta()};
  const Element<3> element =
      TestHelpers::Ccz4::fd::detail::set_element_brick(external_directions);
  {
    auto box =
        make_box(conditions_on<TimeDerivativeCondition>(external_directions),
                 mesh, GhostDataMap{}, 0.0);
    CHECK_THROWS_WITH(apply(make_not_null(&box), element),
                      Catch::Matchers::ContainsSubstring(
                          "Unsupported boundary condition "
                          "TimeDerivativeCondition on an element using finite "
                          "differences."));
  }
  {
    auto box = make_box(conditions_on<Periodic>(external_directions), mesh,
                        GhostDataMap{}, 0.0);
    CHECK_THROWS_WITH(apply(make_not_null(&box), element),
                      Catch::Matchers::ContainsSubstring("is not registered"));
  }
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Ccz4.Fd.BoundaryConditionGhostData",
                  "[Unit][Evolution]") {
  const Mesh<3> anisotropic_mesh{{{5, 6, 7}},
                                 Spectral::Basis::FiniteDifference,
                                 Spectral::Quadrature::CellCentered};
  const Mesh<3> isotropic_mesh{7, Spectral::Basis::FiniteDifference,
                               Spectral::Quadrature::CellCentered};
  for (const auto& external_directions :
       {std::vector{Direction<3>::lower_zeta(), Direction<3>::upper_zeta()},
        std::vector{Direction<3>::upper_xi(), Direction<3>::lower_eta()}}) {
    test_ghost_data(anisotropic_mesh, external_directions);
    test_consumers(isotropic_mesh, external_directions);
  }
  test_no_external_boundaries(anisotropic_mesh);
  test_unsupported(anisotropic_mesh);
}
}  // namespace
