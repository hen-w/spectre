// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/BoundaryConditions/GetBoundaryConditionsBase.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Options/Context.hpp"
#include "Options/Options.hpp"
#include "Options/ParseError.hpp"
#include "Options/String.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <size_t Dim, typename T>
class DirectionMap;
template <size_t Dim>
class Domain;
namespace domain {
namespace CoordinateMaps {
class Affine;
class BulgedCube;
class Equiangular;
template <size_t Dim>
class Identity;
class Interval;
template <typename Map1, typename Map2>
class ProductOf2Maps;
template <typename Map1, typename Map2, typename Map3>
class ProductOf3Maps;
class SphericalToCartesianPfaffian;
template <size_t Dim>
class Wedge;
}  // namespace CoordinateMaps

template <typename SourceFrame, typename TargetFrame, typename... Maps>
class CoordinateMap;
}  // namespace domain
/// \endcond

namespace domain::creators::NonconformingSphericalShells_detail {

struct Excision {
  Excision() = default;
  Excision(std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
               in_boundary_condition)
      : boundary_condition(std::move(in_boundary_condition)) {}
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      boundary_condition = nullptr;
};

struct ExcisionFromOptions : Excision {
  static constexpr Options::String help = {
      "Excise the interior of the sphere, leaving a spherical shell."};
  template <typename BoundaryConditionsBase>
  struct BoundaryCondition {
    static std::string name() { return "ExciseWithBoundaryCondition"; }
    using type = std::unique_ptr<BoundaryConditionsBase>;
    static constexpr Options::String help = {
        "The boundary condition to impose on the excision surface."};
  };
  template <typename Metavariables>
  using options = tmpl::list<BoundaryCondition<
      domain::BoundaryConditions::get_boundary_conditions_base<
          typename Metavariables::system>>>;
  using Excision::Excision;
};

struct InnerCube {
  static constexpr Options::String help = {
      "Fill the interior of the sphere with a cube."};
  struct Sphericity {
    static std::string name() { return "FillWithSphericity"; }
    using type = double;
    static constexpr Options::String help = {
        "Sphericity of the inner cube. A sphericity of 0 uses a product "
        "of 1D maps as the map in the center. A sphericity > 0 uses a "
        "BulgedCube. A sphericity of exactly 1 is not allowed. See "
        "BulgedCube docs for why."};
    static double lower_bound() { return 0.0; }
    static double upper_bound() { return 1.0; }
  };
  using options = tmpl::list<Sphericity>;
  InnerCube() = default;
  explicit InnerCube(double sphericity_in) : sphericity(sphericity_in) {}
  double sphericity = std::numeric_limits<double>::signaling_NaN();
};

}  // namespace domain::creators::NonconformingSphericalShells_detail

template <>
struct Options::create_from_yaml<
    domain::creators::NonconformingSphericalShells_detail::Excision> {
  template <typename Metavariables>
  static domain::creators::NonconformingSphericalShells_detail::Excision create(
      const Options::Option& options) {
    if constexpr (domain::BoundaryConditions::has_boundary_conditions_base_v<
                      typename Metavariables::system>) {
      return options.parse_as<
          domain::creators::NonconformingSphericalShells_detail::
              ExcisionFromOptions,
          Metavariables>();
    } else {
      if (options.parse_as<std::string>() == "Excise") {
        return domain::creators::NonconformingSphericalShells_detail::
            Excision{};
      } else {
        PARSE_ERROR(options.context(), "Parse error");
      }
    }
  }
};

namespace domain::creators {
/*!
 * \brief A set of non-conforming concentric spherical shells
 *
 * \details The inner spherical shells are decomposed into six wedges
 * surrounding either an excised interior region or a central cube block.
 * The outer spherical shells will use a spherical harmonic basis which
 * cannot be used with subcell.
 *
 * This domain creator offers one grid anchor "Center" at the origin.
 *
 */
class NonconformingSphericalShells : public DomainCreator<3> {
 private:
  using Affine = CoordinateMaps::Affine;
  using Affine3D = CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  using Equiangular = CoordinateMaps::Equiangular;
  using Equiangular3D =
      CoordinateMaps::ProductOf3Maps<Equiangular, Equiangular, Equiangular>;
  using BulgedCube = CoordinateMaps::BulgedCube;

 public:
  using maps_list =
      tmpl::list<domain::CoordinateMap<Frame::BlockLogical, Frame::Inertial,
                                       BulgedCube>,
                 domain::CoordinateMap<Frame::BlockLogical, Frame::Inertial,
                                       Affine3D>,
                 domain::CoordinateMap<Frame::BlockLogical, Frame::Inertial,
                                       Equiangular3D>,
                 domain::CoordinateMap<
                     Frame::BlockLogical, Frame::Inertial,
                     domain::CoordinateMaps::ProductOf2Maps<
                         domain::CoordinateMaps::Affine,
                         domain::CoordinateMaps::Identity<2>>,
                     domain::CoordinateMaps::SphericalToCartesianPfaffian>,
                 domain::CoordinateMap<Frame::BlockLogical, Frame::Inertial,
                                       CoordinateMaps::Wedge<3>>>;

  using Excision = NonconformingSphericalShells_detail::Excision;
  using InnerCube = NonconformingSphericalShells_detail::InnerCube;

  struct InnerRadius {
    using type = double;
    static constexpr Options::String help = {
        "Inner radius of the inner wedges."};
  };

  struct InterfaceRadius {
    using type = double;
    static constexpr Options::String help = {
        "Radius of interface between the inner wedges and the outer spherical "
        "shells."};
  };

  struct OuterRadius {
    using type = double;
    static constexpr Options::String help = {
        "Outer radius of the outer spherical shell."};
  };

  struct InitialRadialRefinement {
    using type = size_t;
    static constexpr Options::String help = {
        "Initial radial refinement level for both the inner wedges and the "
        "outer spherical shells."};
  };

  struct InitialAngularRefinementOfWedges {
    using type = size_t;
    static constexpr Options::String help = {
        "Initial angular refinement levels of inner wedges."};
  };

  struct InitialNumberOfRadialGridPoints {
    using type = size_t;
    static constexpr Options::String help = {
        "Initial number of radial grid points for both the inner wedges and "
        "the outer spherical shells."};
  };

  struct InitialSphericalHarmonicL {
    using type = size_t;
    static constexpr Options::String help = {
        "Initial spherical harmonic resolution specified as the highest "
        "spherical harmonic represented on the grid."};
  };

  struct InitialNumberOfAngularGridPointsOfWedges {
    using type = size_t;
    static constexpr Options::String help = {
        "Initial angular refinement levels of inner wedges."};
  };

  struct Interior {
    using type = std::variant<Excision, InnerCube>;
    static constexpr Options::String help = {
        "Specify 'ExciseWithBoundaryCondition' and a boundary condition to "
        "excise the interior of the sphere, leaving a spherical shell "
        "(or just 'Excise' if boundary conditions are disabled). "
        "Or specify 'FillWithSphericity' to fill the interior with a cube."};
  };

  struct UseEquiangularMap {
    using type = bool;
    static constexpr Options::String help = {
        "Use equiangular instead of equidistant coordinates. Equiangular "
        "coordinates give better gridpoint spacings in the angular "
        "directions, while equidistant coordinates give better gridpoint "
        "spacings in the inner cube."};
  };

  template <typename BoundaryConditionsBase>
  struct OuterBoundaryCondition {
    static constexpr Options::String help =
        "Options for the boundary conditions at the outer radius.";
    using type = std::unique_ptr<BoundaryConditionsBase>;
  };

  using basic_options =
      tmpl::list<InnerRadius, InterfaceRadius, OuterRadius,
                 InitialRadialRefinement, InitialAngularRefinementOfWedges,
                 InitialNumberOfRadialGridPoints, InitialSphericalHarmonicL,
                 InitialNumberOfAngularGridPointsOfWedges, Interior,
                 UseEquiangularMap>;

  template <typename Metavariables>
  using options = tmpl::conditional_t<
      domain::BoundaryConditions::has_boundary_conditions_base_v<
          typename Metavariables::system>,
      tmpl::push_back<
          basic_options,
          OuterBoundaryCondition<
              domain::BoundaryConditions::get_boundary_conditions_base<
                  typename Metavariables::system>>>,
      basic_options>;

  static constexpr Options::String help{
      "A set of concentric spherical shells centered at the origin."};

  NonconformingSphericalShells(
      double inner_radius, double interface_radius, double outer_radius,
      size_t initial_radial_refinement,
      size_t initial_angular_refinement,
      size_t initial_number_of_radial_grid_points,
      size_t initial_spherical_harmonic_l,
      size_t initial_number_of_angular_grid_points_of_wedges,
      std::variant<Excision, InnerCube> interior,
      bool use_equiangular_map,
      std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
          outer_boundary_condition = nullptr,
      const Options::Context& context = {});

  NonconformingSphericalShells() = default;
  NonconformingSphericalShells(const NonconformingSphericalShells&) = delete;
  NonconformingSphericalShells(NonconformingSphericalShells&&) = default;
  NonconformingSphericalShells& operator=(const NonconformingSphericalShells&) =
      delete;
  NonconformingSphericalShells& operator=(NonconformingSphericalShells&&) =
      default;
  ~NonconformingSphericalShells() override = default;

  Domain<3> create_domain() const override;

  std::unordered_map<std::string, tnsr::I<double, 3, Frame::Grid>>
  grid_anchors() const override;

  std::vector<DirectionMap<
      3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
  external_boundary_conditions() const override;

  std::vector<std::string> block_names() const override;

  std::vector<std::array<size_t, 3>> initial_extents() const override;

  std::vector<std::array<size_t, 3>> initial_refinement_levels() const override;
 private:
  double inner_radius_{};
  double interface_radius_{};
  double outer_radius_{};
  size_t initial_radial_refinement_{};
  size_t initial_angular_refinement_{};
  size_t initial_number_of_radial_grid_points_{};
  size_t initial_spherical_harmonic_l_{};
  size_t initial_number_of_angular_grid_points_of_wedges_{};
  std::variant<Excision, InnerCube> interior_{};
  bool fill_interior_{};
  bool use_equiangular_map_{};
  std::vector<std::array<size_t, 3>> initial_refinement_levels_{};
  std::vector<std::array<size_t, 3>> initial_number_of_grid_points_{};
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      outer_boundary_condition_{};
  std::unordered_map<std::string, tnsr::I<double, 3, Frame::Grid>>
      grid_anchors_{};
};
}  // namespace domain::creators
