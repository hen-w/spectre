// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <pup.h>
#include <string>
#include <unordered_map>
#include <utility>

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Block.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/BoundaryConditions/Cartoon.hpp"
#include "Domain/BoundaryConditions/None.hpp"
#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Creators/Tags/ExternalBoundaryConditions.hpp"
#include "Domain/Creators/Tags/FunctionsOfTime.hpp"
#include "Domain/Domain.hpp"
#include "Evolution/BoundaryConditions/Type.hpp"
#include "Evolution/BoundaryCorrection.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/BoundaryConditionsImpl.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/Actions/SystemType.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/LiftFromBoundary.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/ProjectToBoundary.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags/Formulation.hpp"
#include "NumericalAlgorithms/Interpolation/RegularGridInterpolant.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Time/Tags/Time.hpp"
#include "Utilities/CloneUniquePtrs.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeVector.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace {
// We use offsets and then fill the Variables with offset+DataVector index
// This is a way to generate unique but known numbers
constexpr double offset_dt_evolved_vars = 1.0;
constexpr double offset_evolved_vars = 20.0;
constexpr double offset_temporaries = 50.0;
constexpr double offset_volume_fluxes = 200.0;
constexpr double offset_partial_derivs = 3000.0;
constexpr double offset_primitive_vars = 7000.0;
constexpr double offset_boundary_condition = 10000.0;
constexpr double offset_boundary_correction = 20000.0;
const std::array expected_velocities{1.2, -1.4, 0.3};

namespace Tags {
struct BoundaryCorrectionVolumeTag : db::SimpleTag {
  using type = double;
};

struct BoundaryCorrectionAuxiliaryVolumeTag : db::SimpleTag {
  using type = double;
};

struct BoundaryConditionVolumeTag : db::SimpleTag {
  using type = double;
};

struct Var1 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

template <size_t Dim>
struct Var2 : db::SimpleTag {
  using type = tnsr::I<DataVector, Dim, Frame::Inertial>;
};

struct Var3Squared : db::SimpleTag {
  using type = Scalar<DataVector>;
};

// Auxiliary variable tag
template <size_t Dim>
struct Var3 : db::SimpleTag {
  using type = tnsr::I<DataVector, Dim, Frame::Inertial>;
};

struct PrimVar1 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

template <size_t Dim>
struct PrimVar2 : db::SimpleTag {
  using type = tnsr::i<DataVector, Dim, Frame::Inertial>;
};

template <size_t Dim>
struct InverseSpatialMetric : db::SimpleTag {
  using type = tnsr::II<DataVector, Dim, Frame::Inertial>;
};
}  // namespace Tags

using SystemType = TestHelpers::evolution::dg::Actions::SystemType;

template <size_t Dim, bool HasPrims, SystemType SysType,
          bool HasInverseSpatialMetric>
struct BoundaryTerms final : public evolution::BoundaryCorrection {
  struct MaxAbsCharSpeed : db::SimpleTag {
    using type = Scalar<DataVector>;
  };

  explicit BoundaryTerms(CkMigrateMessage* /*unused*/) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(BoundaryTerms);  // NOLINT
  BoundaryTerms(const bool mesh_is_moving, const double sign_of_normal)
      : mesh_is_moving_(mesh_is_moving), sign_of_normal_(sign_of_normal) {}
  BoundaryTerms() = default;
  BoundaryTerms(const BoundaryTerms&) = default;
  BoundaryTerms& operator=(const BoundaryTerms&) = default;
  BoundaryTerms(BoundaryTerms&&) = default;
  BoundaryTerms& operator=(BoundaryTerms&&) = default;
  ~BoundaryTerms() override = default;

  using variables_tags = tmpl::list<Tags::Var1, Tags::Var2<Dim>>;
  using variables_tag = ::Tags::Variables<variables_tags>;

  std::unique_ptr<evolution::BoundaryCorrection> get_clone() const override {
    return std::make_unique<BoundaryTerms>(*this);
  }

  void pup(PUP::er& p) override {  // NOLINT
    BoundaryCorrection::pup(p);
    p | mesh_is_moving_;
    p | sign_of_normal_;
  }

  using dg_package_field_tags = tmpl::push_back<
      tmpl::push_back<
          tmpl::append<db::wrap_tags_in<::Tags::NormalDotFlux, variables_tags>,
                       variables_tags>,
          MaxAbsCharSpeed>,
      Tags::Var3<Dim>>;
  using dg_package_data_temporary_tags = tmpl::list<Tags::Var3Squared>;
  using dg_package_data_primitive_tags =
      tmpl::conditional_t<HasPrims, tmpl::list<Tags::PrimVar1>, tmpl::list<>>;
  using dg_package_data_volume_tags = tmpl::conditional_t<
      HasPrims, tmpl::list<Tags::BoundaryCorrectionVolumeTag>, tmpl::list<>>;
  using dg_boundary_terms_volume_tags = tmpl::conditional_t<
      HasPrims, tmpl::list<Tags::BoundaryCorrectionVolumeTag>, tmpl::list<>>;

  // Auxiliary variable is not used for auxiliary boundary correction
  using dg_auxiliary_package_field_tags = tmpl::push_back<
      tmpl::append<db::wrap_tags_in<::Tags::NormalDotFlux, variables_tags>,
                   variables_tags>,
      MaxAbsCharSpeed>;
  using dg_auxiliary_package_data_temporary_tags =
      tmpl::list<Tags::Var3Squared>;
  using dg_auxiliary_package_data_volume_tags =
      tmpl::list<Tags::BoundaryCorrectionAuxiliaryVolumeTag>;
  using dg_auxiliary_boundary_terms_volume_tags =
      tmpl::list<Tags::BoundaryCorrectionAuxiliaryVolumeTag>;

  // Conservative system, flat background
  double dg_package_data(
      const gsl::not_null<Scalar<DataVector>*> out_normal_dot_flux_var1,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          out_normal_dot_flux_var2,
      const gsl::not_null<Scalar<DataVector>*> out_var1,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> out_var2,
      const gsl::not_null<Scalar<DataVector>*> max_abs_char_speed,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> out_var3,

      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& var2,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& var3,

      const tnsr::I<DataVector, Dim, Frame::Inertial>& flux_var1,
      const tnsr::IJ<DataVector, Dim, Frame::Inertial>& flux_var2,

      const Scalar<DataVector>& var3_squared,

      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
          mesh_velocity,
      const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity) const {
    if (mesh_velocity.has_value()) {
      REQUIRE(normal_dot_mesh_velocity.has_value());
      CHECK_ITERABLE_APPROX(*normal_dot_mesh_velocity,
                            dot_product(normal_covector, *mesh_velocity));
    }

    *out_normal_dot_flux_var1 = dot_product(flux_var1, normal_covector);
    if (mesh_velocity.has_value()) {
      get(*out_normal_dot_flux_var1) -=
          get(var1) * get(dot_product(*mesh_velocity, normal_covector));
    }
    for (size_t i = 0; i < Dim; ++i) {
      out_normal_dot_flux_var2->get(i) =
          flux_var2.get(i, 0) * normal_covector.get(0);
      if (mesh_velocity.has_value()) {
        out_normal_dot_flux_var2->get(i) -=
            var2.get(i) * get<0>(*mesh_velocity) * normal_covector.get(0);
      }
      for (size_t j = 1; j < Dim; ++j) {
        out_normal_dot_flux_var2->get(i) +=
            flux_var2.get(i, j) * normal_covector.get(j);
        if (mesh_velocity.has_value()) {
          out_normal_dot_flux_var2->get(i) -=
              var2.get(i) * mesh_velocity->get(j) * normal_covector.get(j);
        }
      }
    }
    *out_var1 = var1;
    *out_var2 = var2;
    *out_var3 = var3;

    get(*max_abs_char_speed) = 2.0 * max(get(var3_squared));

    if (normal_dot_mesh_velocity.has_value()) {
      get(*max_abs_char_speed) += get(*normal_dot_mesh_velocity);
    }
    return max(get(*max_abs_char_speed));
  }

  // Conservative system, curved background
  double dg_package_data(
      const gsl::not_null<Scalar<DataVector>*> out_normal_dot_flux_var1,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          out_normal_dot_flux_var2,
      const gsl::not_null<Scalar<DataVector>*> out_var1,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> out_var2,
      const gsl::not_null<Scalar<DataVector>*> max_abs_char_speed,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> out_var3,

      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& var2,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& var3,

      const tnsr::I<DataVector, Dim, Frame::Inertial>& flux_var1,
      const tnsr::IJ<DataVector, Dim, Frame::Inertial>& flux_var2,

      const Scalar<DataVector>& var3_squared,

      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& normal_vector,
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
          mesh_velocity,
      const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity) const {
    CHECK_ITERABLE_APPROX(get(dot_product(normal_covector, normal_vector)),
                          DataVector(get(var1).size(), 1.0));
    return dg_package_data(
        out_normal_dot_flux_var1, out_normal_dot_flux_var2, out_var1, out_var2,
        max_abs_char_speed, out_var3, var1, var2, var3, flux_var1, flux_var2,
        var3_squared, normal_covector, mesh_velocity, normal_dot_mesh_velocity);
  }

  // Conservative system with prim vars, flat background
  double dg_package_data(
      const gsl::not_null<Scalar<DataVector>*> out_normal_dot_flux_var1,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          out_normal_dot_flux_var2,
      const gsl::not_null<Scalar<DataVector>*> out_var1,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> out_var2,
      const gsl::not_null<Scalar<DataVector>*> max_abs_char_speed,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> out_var3,

      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& var2,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& var3,

      const tnsr::I<DataVector, Dim, Frame::Inertial>& flux_var1,
      const tnsr::IJ<DataVector, Dim, Frame::Inertial>& flux_var2,

      const Scalar<DataVector>& var3_squared,
      const Scalar<DataVector>& prim_var1,

      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
          mesh_velocity,
      const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity,

      const double volume_number) const {
    dg_package_data(out_normal_dot_flux_var1, out_normal_dot_flux_var2,
                    out_var1, out_var2, max_abs_char_speed, out_var3, var1,
                    var2, var3, flux_var1, flux_var2, var3_squared,
                    normal_covector, mesh_velocity, normal_dot_mesh_velocity);
    get(*out_var1) += get(prim_var1) + volume_number;
    if (mesh_velocity.has_value()) {
      get(*out_normal_dot_flux_var1) -=
          (get(prim_var1) + volume_number) *
          get(dot_product(*mesh_velocity, normal_covector));
    }
    return max(get(*max_abs_char_speed));
  }

  // Conservative system with prim vars, curved background
  double dg_package_data(
      const gsl::not_null<Scalar<DataVector>*> out_normal_dot_flux_var1,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          out_normal_dot_flux_var2,
      const gsl::not_null<Scalar<DataVector>*> out_var1,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> out_var2,
      const gsl::not_null<Scalar<DataVector>*> max_abs_char_speed,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> out_var3,

      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& var2,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& var3,

      const tnsr::I<DataVector, Dim, Frame::Inertial>& flux_var1,
      const tnsr::IJ<DataVector, Dim, Frame::Inertial>& flux_var2,

      const Scalar<DataVector>& var3_squared,
      const Scalar<DataVector>& prim_var1,

      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& normal_vector,
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
          mesh_velocity,
      const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity,

      const double volume_number) const {
    CHECK_ITERABLE_APPROX(get(dot_product(normal_covector, normal_vector)),
                          DataVector(get(var1).size(), 1.0));
    return dg_package_data(out_normal_dot_flux_var1, out_normal_dot_flux_var2,
                           out_var1, out_var2, max_abs_char_speed, out_var3,
                           var1, var2, var3, flux_var1, flux_var2, var3_squared,
                           prim_var1, normal_covector, mesh_velocity,
                           normal_dot_mesh_velocity, volume_number);
  }

  // Nonconservative system, flat background
  double dg_package_data(
      const gsl::not_null<Scalar<DataVector>*> out_normal_dot_flux_var1,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          out_normal_dot_flux_var2,
      const gsl::not_null<Scalar<DataVector>*> out_var1,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> out_var2,
      const gsl::not_null<Scalar<DataVector>*> max_abs_char_speed,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> out_var3,

      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& var2,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& var3,

      const Scalar<DataVector>& var3_squared,

      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
          mesh_velocity,
      const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity) const {
    if (mesh_velocity.has_value()) {
      REQUIRE(normal_dot_mesh_velocity.has_value());
      CHECK_ITERABLE_APPROX(*normal_dot_mesh_velocity,
                            dot_product(normal_covector, *mesh_velocity));
    }

    get(*out_normal_dot_flux_var1) =
        get(var1) + get(dot_product(var2, normal_covector));
    if (mesh_velocity.has_value()) {
      get(*out_normal_dot_flux_var1) -=
          get(dot_product(*mesh_velocity, normal_covector));
    }
    for (size_t i = 0; i < Dim; ++i) {
      out_normal_dot_flux_var2->get(i) =
          normal_covector.get(i) * normal_covector.get(0) * var2.get(0);
      if (mesh_velocity.has_value()) {
        out_normal_dot_flux_var2->get(i) -=
            var2.get(i) * get<0>(*mesh_velocity) * normal_covector.get(0);
      }
      for (size_t j = 1; j < Dim; ++j) {
        out_normal_dot_flux_var2->get(i) +=
            normal_covector.get(i) * normal_covector.get(j) * var2.get(j);
        if (mesh_velocity.has_value()) {
          out_normal_dot_flux_var2->get(i) -=
              var2.get(i) * mesh_velocity->get(j) * normal_covector.get(j);
        }
      }
    }
    *out_var1 = var1;
    *out_var2 = var2;
    *out_var3 = var3;

    get(*max_abs_char_speed) = 2.0 * max(get(var3_squared));

    if (normal_dot_mesh_velocity.has_value()) {
      get(*max_abs_char_speed) += get(*normal_dot_mesh_velocity);
    }
    return max(get(*max_abs_char_speed));
  }

  // Nonconservative system, curved background
  double dg_package_data(
      const gsl::not_null<Scalar<DataVector>*> out_normal_dot_flux_var1,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          out_normal_dot_flux_var2,
      const gsl::not_null<Scalar<DataVector>*> out_var1,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> out_var2,
      const gsl::not_null<Scalar<DataVector>*> max_abs_char_speed,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> out_var3,

      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& var2,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& var3,

      const Scalar<DataVector>& var3_squared,

      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& normal_vector,
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
          mesh_velocity,
      const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity) const {
    CHECK_ITERABLE_APPROX(get(dot_product(normal_covector, normal_vector)),
                          DataVector(get(var1).size(), 1.0));
    return dg_package_data(out_normal_dot_flux_var1, out_normal_dot_flux_var2,
                           out_var1, out_var2, max_abs_char_speed, out_var3,
                           var1, var2, var3, var3_squared, normal_covector,
                           mesh_velocity, normal_dot_mesh_velocity);
  }

  // Mixed system, no prims, flat background
  double dg_package_data(
      const gsl::not_null<Scalar<DataVector>*> out_normal_dot_flux_var1,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          out_normal_dot_flux_var2,
      const gsl::not_null<Scalar<DataVector>*> out_var1,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> out_var2,
      const gsl::not_null<Scalar<DataVector>*> max_abs_char_speed,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> out_var3,

      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& var2,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& var3,

      const tnsr::IJ<DataVector, Dim, Frame::Inertial>& flux_var2,

      const Scalar<DataVector>& var3_squared,

      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
          mesh_velocity,
      const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity) const {
    if (mesh_velocity.has_value()) {
      REQUIRE(normal_dot_mesh_velocity.has_value());
      CHECK_ITERABLE_APPROX(*normal_dot_mesh_velocity,
                            dot_product(normal_covector, *mesh_velocity));
    }

    get(*out_normal_dot_flux_var1) =
        get(var1) + get(dot_product(var2, normal_covector));
    if (mesh_velocity.has_value()) {
      get(*out_normal_dot_flux_var1) -=
          get(dot_product(*mesh_velocity, normal_covector));
    }
    for (size_t i = 0; i < Dim; ++i) {
      out_normal_dot_flux_var2->get(i) =
          flux_var2.get(i, 0) * normal_covector.get(0);
      if (mesh_velocity.has_value()) {
        out_normal_dot_flux_var2->get(i) -=
            var2.get(i) * get<0>(*mesh_velocity) * normal_covector.get(0);
      }
      for (size_t j = 1; j < Dim; ++j) {
        out_normal_dot_flux_var2->get(i) +=
            flux_var2.get(i, j) * normal_covector.get(j);
        if (mesh_velocity.has_value()) {
          out_normal_dot_flux_var2->get(i) -=
              var2.get(i) * mesh_velocity->get(j) * normal_covector.get(j);
        }
      }
    }
    *out_var1 = var1;
    *out_var2 = var2;
    *out_var3 = var3;

    get(*max_abs_char_speed) = 2.0 * max(get(var3_squared));

    if (normal_dot_mesh_velocity.has_value()) {
      get(*max_abs_char_speed) += get(*normal_dot_mesh_velocity);
    }
    return max(get(*max_abs_char_speed));
  }

  // Mixed system, no prims, curved background
  double dg_package_data(
      const gsl::not_null<Scalar<DataVector>*> out_normal_dot_flux_var1,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          out_normal_dot_flux_var2,
      const gsl::not_null<Scalar<DataVector>*> out_var1,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> out_var2,
      const gsl::not_null<Scalar<DataVector>*> max_abs_char_speed,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> out_var3,

      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& var2,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& var3,

      const tnsr::IJ<DataVector, Dim, Frame::Inertial>& flux_var2,

      const Scalar<DataVector>& var3_squared,

      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& normal_vector,
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
          mesh_velocity,
      const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity) const {
    CHECK_ITERABLE_APPROX(get(dot_product(normal_covector, normal_vector)),
                          DataVector(get(var1).size(), 1.0));
    return dg_package_data(
        out_normal_dot_flux_var1, out_normal_dot_flux_var2, out_var1, out_var2,
        max_abs_char_speed, out_var3, var1, var2, var3, flux_var2, var3_squared,
        normal_covector, mesh_velocity, normal_dot_mesh_velocity);
  }

  // Mixed system with prims, flat background
  double dg_package_data(
      const gsl::not_null<Scalar<DataVector>*> out_normal_dot_flux_var1,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          out_normal_dot_flux_var2,
      const gsl::not_null<Scalar<DataVector>*> out_var1,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> out_var2,
      const gsl::not_null<Scalar<DataVector>*> max_abs_char_speed,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> out_var3,

      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& var2,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& var3,

      const tnsr::IJ<DataVector, Dim, Frame::Inertial>& flux_var2,

      const Scalar<DataVector>& var3_squared,

      const Scalar<DataVector>& prim_var1,

      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
          mesh_velocity,
      const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity,

      const double volume_number) const {
    dg_package_data(out_normal_dot_flux_var1, out_normal_dot_flux_var2,
                    out_var1, out_var2, max_abs_char_speed, out_var3, var1,
                    var2, var3, flux_var2, var3_squared, normal_covector,
                    mesh_velocity, normal_dot_mesh_velocity);
    get(*out_var1) += get(prim_var1) + volume_number;
    return max(get(*max_abs_char_speed));
  }

  // Mixed system with prims, curved background
  double dg_package_data(
      const gsl::not_null<Scalar<DataVector>*> out_normal_dot_flux_var1,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          out_normal_dot_flux_var2,
      const gsl::not_null<Scalar<DataVector>*> out_var1,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> out_var2,
      const gsl::not_null<Scalar<DataVector>*> max_abs_char_speed,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> out_var3,

      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& var2,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& var3,

      const tnsr::IJ<DataVector, Dim, Frame::Inertial>& flux_var2,

      const Scalar<DataVector>& var3_squared,

      const Scalar<DataVector>& prim_var1,

      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& normal_vector,
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
          mesh_velocity,
      const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity,

      const double volume_number) const {
    CHECK_ITERABLE_APPROX(get(dot_product(normal_covector, normal_vector)),
                          DataVector(get(var1).size(), 1.0));
    return dg_package_data(out_normal_dot_flux_var1, out_normal_dot_flux_var2,
                           out_var1, out_var2, max_abs_char_speed, out_var3,
                           var1, var2, var3, flux_var2, var3_squared, prim_var1,
                           normal_covector, mesh_velocity,
                           normal_dot_mesh_velocity, volume_number);
  }

  void dg_boundary_terms(
      const gsl::not_null<Scalar<DataVector>*> boundary_correction_var1,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          boundary_correction_var2,
      const Scalar<DataVector>& int_normal_dot_flux_var1,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& int_normal_dot_flux_var2,
      const Scalar<DataVector>& int_var1,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& int_var2,
      const Scalar<DataVector>& int_max_abs_char_speed,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& int_var3,
      const Scalar<DataVector>& ext_normal_dot_flux_var1,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& ext_normal_dot_flux_var2,
      const Scalar<DataVector>& ext_var1,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& ext_var2,
      const Scalar<DataVector>& ext_max_abs_char_speed,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& ext_var3,
      const dg::Formulation formulation) const {
    static_assert(Dim == 1,
                  "Flux dot normal assumes 1d, mostly because normal vector is "
                  "assumed to be 1d.");

    get(*boundary_correction_var1) =
        offset_boundary_correction *
        (formulation == dg::Formulation::WeakInertial ? 2.0 : 1.0);
    for (size_t i = 0; i < Dim; ++i) {
      boundary_correction_var2->get(i) = offset_boundary_correction + 1.0 + i;
    }
    const size_t num_pts = get(int_var1).size();

    const double mesh_velocity = mesh_is_moving_ ? 1.2 : 0.0;
    const double normalization_factor =
        HasInverseSpatialMetric ? sqrt(offset_temporaries + 1.0) : 1.0;
    if (SysType == SystemType::Conservative) {
      CHECK_ITERABLE_APPROX(
          get(int_normal_dot_flux_var1),
          DataVector(sign_of_normal_ / normalization_factor *
                     (offset_volume_fluxes - mesh_velocity * get(int_var1))));
    } else {
      CHECK_ITERABLE_APPROX(
          get(int_normal_dot_flux_var1),
          DataVector(offset_evolved_vars +
                     sign_of_normal_ / normalization_factor * get<0>(int_var2) -
                     sign_of_normal_ / normalization_factor * mesh_velocity));
    }

    if (SysType == SystemType::Conservative) {
      for (size_t i = 0; i < Dim; ++i) {
        CHECK_ITERABLE_APPROX(
            int_normal_dot_flux_var2.get(i),
            DataVector(sign_of_normal_ / normalization_factor *
                       (offset_volume_fluxes + 1.0 + i -
                        mesh_velocity * int_var2.get(i))));
      }
    } else if (SysType == SystemType::Mixed) {
      for (size_t i = 0; i < Dim; ++i) {
        CHECK_ITERABLE_APPROX(
            int_normal_dot_flux_var2.get(i),
            DataVector(
                sign_of_normal_ / normalization_factor *
                (offset_volume_fluxes + i - mesh_velocity * int_var2.get(i))));
      }
    } else {
      for (size_t i = 0; i < Dim; ++i) {
        DataVector expected{num_pts, 0.0};
        for (size_t j = 0; j < Dim; ++j) {
          expected += int_var2.get(j) /
                          square(normalization_factor)  // n_i n_j var2^j, bot
                                                        // n_i = (\pm 1) in 1d
                      - sign_of_normal_ / normalization_factor *
                            int_var2.get(i) * mesh_velocity;  // var2^i v^j n_j
        }
        CHECK_ITERABLE_APPROX(int_normal_dot_flux_var2.get(i), expected);
      }
    }
    CHECK_ITERABLE_APPROX(
        get(int_var1),
        DataVector(num_pts,
                   offset_evolved_vars +
                       (HasPrims ? offset_primitive_vars + 3.5 : 0.0)));
    for (size_t i = 0; i < Dim; ++i) {
      CHECK_ITERABLE_APPROX(int_var2.get(i),
                            DataVector(num_pts, offset_evolved_vars + 1.0 + i));
    }
    CHECK_ITERABLE_APPROX(
        get(int_max_abs_char_speed),
        DataVector(num_pts,
                   2.0 * offset_temporaries +
                       sign_of_normal_ / normalization_factor * mesh_velocity));

    // Check var3 is delivered correctly
    for (size_t i = 0; i < Dim; ++i) {
      CHECK_ITERABLE_APPROX(
          int_var3.get(i),
          DataVector(num_pts, offset_evolved_vars + static_cast<double>(i)));
    }

    if (SysType == SystemType::Conservative) {
      // The two comes from the dg_package_data also subtracting off the mesh
      // velocity.
      CHECK_ITERABLE_APPROX(
          get(ext_normal_dot_flux_var1),
          DataVector{-sign_of_normal_ / normalization_factor *
                     (offset_boundary_condition + 1.0 + Dim -
                      2 * mesh_velocity * offset_boundary_condition -
                      (HasPrims ? mesh_velocity * (offset_boundary_condition +
                                                   3.0 + 2 * Dim + 3.5)
                                : 0.0))});
    } else {
      CHECK_ITERABLE_APPROX(
          get(ext_normal_dot_flux_var1),
          DataVector(
              (offset_boundary_condition -
               sign_of_normal_ / normalization_factor * get<0>(ext_var2) +
               sign_of_normal_ / normalization_factor * mesh_velocity)));
    }
    if (SysType == SystemType::Conservative or SysType == SystemType::Mixed) {
      for (size_t i = 0; i < Dim; ++i) {
        CHECK_ITERABLE_APPROX(
            ext_normal_dot_flux_var2.get(i),
            DataVector(-sign_of_normal_ / normalization_factor *
                       (offset_boundary_condition + 2.0 + Dim + i -
                        2.0 * mesh_velocity * ext_var2.get(i))));
      }
    } else {
      static_assert(Dim == 1);
      CHECK_ITERABLE_APPROX(
          get<0>(ext_normal_dot_flux_var2),
          DataVector((get<0>(ext_var2) / square(normalization_factor) +
                      sign_of_normal_ / normalization_factor *
                          get<0>(ext_var2) * mesh_velocity)));
    }
    CHECK_ITERABLE_APPROX(
        get(ext_var1),
        DataVector(num_pts, offset_boundary_condition +
                                (HasPrims ? offset_boundary_condition +
                                                2.0 * Dim + 3.0 + 3.5
                                          : 0.0)));
    for (size_t i = 0; i < Dim; ++i) {
      CHECK_ITERABLE_APPROX(
          ext_var2.get(i),
          DataVector(num_pts, offset_boundary_condition + 1.0 + i));
    }
    CHECK_ITERABLE_APPROX(
        get(ext_max_abs_char_speed),
        DataVector(num_pts,
                   2.0 * (offset_boundary_condition + 2.0 + 2 * Dim) -
                       sign_of_normal_ / normalization_factor * mesh_velocity));

    // Check var3 is delivered correctly
    for (size_t i = 0; i < Dim; ++i) {
      CHECK_ITERABLE_APPROX(
          ext_var3.get(i), DataVector(num_pts, offset_boundary_condition + 2.0 +
                                                   static_cast<double>(i)));
    }
  }

  void dg_boundary_terms(
      const gsl::not_null<Scalar<DataVector>*> boundary_correction_var1,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          boundary_correction_var2,
      const Scalar<DataVector>& int_normal_dot_flux_var1,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& int_normal_dot_flux_var2,
      const Scalar<DataVector>& int_var1,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& int_var2,
      const Scalar<DataVector>& int_max_abs_char_speed,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& int_var3,
      const Scalar<DataVector>& ext_normal_dot_flux_var1,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& ext_normal_dot_flux_var2,
      const Scalar<DataVector>& ext_var1,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& ext_var2,
      const Scalar<DataVector>& ext_max_abs_char_speed,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& ext_var3,
      const dg::Formulation formulation, const double& volume_number) const {
    CHECK(volume_number == 3.5);
    dg_boundary_terms(boundary_correction_var1, boundary_correction_var2,
                      int_normal_dot_flux_var1, int_normal_dot_flux_var2,
                      int_var1, int_var2, int_max_abs_char_speed, int_var3,
                      ext_normal_dot_flux_var1, ext_normal_dot_flux_var2,
                      ext_var1, ext_var2, ext_max_abs_char_speed, ext_var3,
                      formulation);
  }

  // The auxiliary packaged fields are exactly the physical ones without Var3,
  // so each auxiliary overload forwards to the corresponding physical
  // `dg_package_data` overload with a scratch Var3 that is written and
  // discarded.

  // Conservative system, flat background
  double dg_auxiliary_package_data(
      const gsl::not_null<Scalar<DataVector>*> out_normal_dot_flux_var1,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          out_normal_dot_flux_var2,
      const gsl::not_null<Scalar<DataVector>*> out_var1,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> out_var2,
      const gsl::not_null<Scalar<DataVector>*> max_abs_char_speed,

      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& var2,

      const tnsr::I<DataVector, Dim, Frame::Inertial>& flux_var1,
      const tnsr::IJ<DataVector, Dim, Frame::Inertial>& flux_var2,

      const Scalar<DataVector>& var3_squared,

      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
          mesh_velocity,
      const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity,

      const double volume_number) const {
    CHECK(volume_number == 4.5);
    tnsr::I<DataVector, Dim, Frame::Inertial> var3_buffer{get(var1).size(),
                                                          0.0};
    return dg_package_data(out_normal_dot_flux_var1, out_normal_dot_flux_var2,
                           out_var1, out_var2, max_abs_char_speed,
                           make_not_null(&var3_buffer), var1, var2, var3_buffer,
                           flux_var1, flux_var2, var3_squared, normal_covector,
                           mesh_velocity, normal_dot_mesh_velocity);
  }

  // Conservative system, curved background
  double dg_auxiliary_package_data(
      const gsl::not_null<Scalar<DataVector>*> out_normal_dot_flux_var1,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          out_normal_dot_flux_var2,
      const gsl::not_null<Scalar<DataVector>*> out_var1,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> out_var2,
      const gsl::not_null<Scalar<DataVector>*> max_abs_char_speed,

      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& var2,

      const tnsr::I<DataVector, Dim, Frame::Inertial>& flux_var1,
      const tnsr::IJ<DataVector, Dim, Frame::Inertial>& flux_var2,

      const Scalar<DataVector>& var3_squared,

      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& normal_vector,
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
          mesh_velocity,
      const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity,

      const double volume_number) const {
    CHECK(volume_number == 4.5);
    tnsr::I<DataVector, Dim, Frame::Inertial> var3_buffer{get(var1).size(),
                                                          0.0};
    return dg_package_data(
        out_normal_dot_flux_var1, out_normal_dot_flux_var2, out_var1, out_var2,
        max_abs_char_speed, make_not_null(&var3_buffer), var1, var2,
        var3_buffer, flux_var1, flux_var2, var3_squared, normal_covector,
        normal_vector, mesh_velocity, normal_dot_mesh_velocity);
  }

  // Conservative system with prim vars, flat background
  double dg_auxiliary_package_data(
      const gsl::not_null<Scalar<DataVector>*> out_normal_dot_flux_var1,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          out_normal_dot_flux_var2,
      const gsl::not_null<Scalar<DataVector>*> out_var1,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> out_var2,
      const gsl::not_null<Scalar<DataVector>*> max_abs_char_speed,

      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& var2,

      const tnsr::I<DataVector, Dim, Frame::Inertial>& flux_var1,
      const tnsr::IJ<DataVector, Dim, Frame::Inertial>& flux_var2,

      const Scalar<DataVector>& var3_squared,
      const Scalar<DataVector>& prim_var1,

      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
          mesh_velocity,
      const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity,

      const double volume_number) const {
    CHECK(volume_number == 4.5);
    tnsr::I<DataVector, Dim, Frame::Inertial> var3_buffer{get(var1).size(),
                                                          0.0};
    // The physical overload folds its own volume-tag value
    // (`BoundaryCorrectionVolumeTag`, 3.5) into the packaged data, which is
    // what the shared `dg_boundary_terms` checks expect; the auxiliary volume
    // tag (4.5) is checked above and dropped.
    return dg_package_data(
        out_normal_dot_flux_var1, out_normal_dot_flux_var2, out_var1, out_var2,
        max_abs_char_speed, make_not_null(&var3_buffer), var1, var2,
        var3_buffer, flux_var1, flux_var2, var3_squared, prim_var1,
        normal_covector, mesh_velocity, normal_dot_mesh_velocity, 3.5);
  }

  // Conservative system with prim vars, curved background
  double dg_auxiliary_package_data(
      const gsl::not_null<Scalar<DataVector>*> out_normal_dot_flux_var1,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          out_normal_dot_flux_var2,
      const gsl::not_null<Scalar<DataVector>*> out_var1,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> out_var2,
      const gsl::not_null<Scalar<DataVector>*> max_abs_char_speed,

      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& var2,

      const tnsr::I<DataVector, Dim, Frame::Inertial>& flux_var1,
      const tnsr::IJ<DataVector, Dim, Frame::Inertial>& flux_var2,

      const Scalar<DataVector>& var3_squared,
      const Scalar<DataVector>& prim_var1,

      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& normal_vector,
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
          mesh_velocity,
      const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity,

      const double volume_number) const {
    CHECK(volume_number == 4.5);
    tnsr::I<DataVector, Dim, Frame::Inertial> var3_buffer{get(var1).size(),
                                                          0.0};
    // The physical overload folds its own volume-tag value
    // (`BoundaryCorrectionVolumeTag`, 3.5) into the packaged data, which is
    // what the shared `dg_boundary_terms` checks expect; the auxiliary volume
    // tag (4.5) is checked above and dropped.
    return dg_package_data(out_normal_dot_flux_var1, out_normal_dot_flux_var2,
                           out_var1, out_var2, max_abs_char_speed,
                           make_not_null(&var3_buffer), var1, var2, var3_buffer,
                           flux_var1, flux_var2, var3_squared, prim_var1,
                           normal_covector, normal_vector, mesh_velocity,
                           normal_dot_mesh_velocity, 3.5);
  }

  // Nonconservative system, flat background
  double dg_auxiliary_package_data(
      const gsl::not_null<Scalar<DataVector>*> out_normal_dot_flux_var1,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          out_normal_dot_flux_var2,
      const gsl::not_null<Scalar<DataVector>*> out_var1,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> out_var2,
      const gsl::not_null<Scalar<DataVector>*> max_abs_char_speed,

      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& var2,

      const Scalar<DataVector>& var3_squared,

      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
          mesh_velocity,
      const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity,

      const double volume_number) const {
    CHECK(volume_number == 4.5);
    tnsr::I<DataVector, Dim, Frame::Inertial> var3_buffer{get(var1).size(),
                                                          0.0};
    return dg_package_data(out_normal_dot_flux_var1, out_normal_dot_flux_var2,
                           out_var1, out_var2, max_abs_char_speed,
                           make_not_null(&var3_buffer), var1, var2, var3_buffer,
                           var3_squared, normal_covector, mesh_velocity,
                           normal_dot_mesh_velocity);
  }

  // Nonconservative system, curved background
  double dg_auxiliary_package_data(
      const gsl::not_null<Scalar<DataVector>*> out_normal_dot_flux_var1,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          out_normal_dot_flux_var2,
      const gsl::not_null<Scalar<DataVector>*> out_var1,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> out_var2,
      const gsl::not_null<Scalar<DataVector>*> max_abs_char_speed,

      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& var2,

      const Scalar<DataVector>& var3_squared,

      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& normal_vector,
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
          mesh_velocity,
      const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity,

      const double volume_number) const {
    CHECK(volume_number == 4.5);
    tnsr::I<DataVector, Dim, Frame::Inertial> var3_buffer{get(var1).size(),
                                                          0.0};
    return dg_package_data(out_normal_dot_flux_var1, out_normal_dot_flux_var2,
                           out_var1, out_var2, max_abs_char_speed,
                           make_not_null(&var3_buffer), var1, var2, var3_buffer,
                           var3_squared, normal_covector, normal_vector,
                           mesh_velocity, normal_dot_mesh_velocity);
  }

  // Mixed system, no prims, flat background
  double dg_auxiliary_package_data(
      const gsl::not_null<Scalar<DataVector>*> out_normal_dot_flux_var1,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          out_normal_dot_flux_var2,
      const gsl::not_null<Scalar<DataVector>*> out_var1,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> out_var2,
      const gsl::not_null<Scalar<DataVector>*> max_abs_char_speed,

      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& var2,

      const tnsr::IJ<DataVector, Dim, Frame::Inertial>& flux_var2,

      const Scalar<DataVector>& var3_squared,

      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
          mesh_velocity,
      const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity,

      const double volume_number) const {
    CHECK(volume_number == 4.5);
    tnsr::I<DataVector, Dim, Frame::Inertial> var3_buffer{get(var1).size(),
                                                          0.0};
    return dg_package_data(out_normal_dot_flux_var1, out_normal_dot_flux_var2,
                           out_var1, out_var2, max_abs_char_speed,
                           make_not_null(&var3_buffer), var1, var2, var3_buffer,
                           flux_var2, var3_squared, normal_covector,
                           mesh_velocity, normal_dot_mesh_velocity);
  }

  // Mixed system, no prims, curved background
  double dg_auxiliary_package_data(
      const gsl::not_null<Scalar<DataVector>*> out_normal_dot_flux_var1,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          out_normal_dot_flux_var2,
      const gsl::not_null<Scalar<DataVector>*> out_var1,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> out_var2,
      const gsl::not_null<Scalar<DataVector>*> max_abs_char_speed,

      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& var2,

      const tnsr::IJ<DataVector, Dim, Frame::Inertial>& flux_var2,

      const Scalar<DataVector>& var3_squared,

      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& normal_vector,
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
          mesh_velocity,
      const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity,

      const double volume_number) const {
    CHECK(volume_number == 4.5);
    tnsr::I<DataVector, Dim, Frame::Inertial> var3_buffer{get(var1).size(),
                                                          0.0};
    return dg_package_data(
        out_normal_dot_flux_var1, out_normal_dot_flux_var2, out_var1, out_var2,
        max_abs_char_speed, make_not_null(&var3_buffer), var1, var2,
        var3_buffer, flux_var2, var3_squared, normal_covector, normal_vector,
        mesh_velocity, normal_dot_mesh_velocity);
  }

  // Mixed system with prims, flat background
  double dg_auxiliary_package_data(
      const gsl::not_null<Scalar<DataVector>*> out_normal_dot_flux_var1,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          out_normal_dot_flux_var2,
      const gsl::not_null<Scalar<DataVector>*> out_var1,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> out_var2,
      const gsl::not_null<Scalar<DataVector>*> max_abs_char_speed,

      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& var2,

      const tnsr::IJ<DataVector, Dim, Frame::Inertial>& flux_var2,

      const Scalar<DataVector>& var3_squared,

      const Scalar<DataVector>& prim_var1,

      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
          mesh_velocity,
      const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity,

      const double volume_number) const {
    CHECK(volume_number == 4.5);
    tnsr::I<DataVector, Dim, Frame::Inertial> var3_buffer{get(var1).size(),
                                                          0.0};
    // The physical overload folds its own volume-tag value
    // (`BoundaryCorrectionVolumeTag`, 3.5) into the packaged data, which is
    // what the shared `dg_boundary_terms` checks expect; the auxiliary volume
    // tag (4.5) is checked above and dropped.
    return dg_package_data(out_normal_dot_flux_var1, out_normal_dot_flux_var2,
                           out_var1, out_var2, max_abs_char_speed,
                           make_not_null(&var3_buffer), var1, var2, var3_buffer,
                           flux_var2, var3_squared, prim_var1, normal_covector,
                           mesh_velocity, normal_dot_mesh_velocity, 3.5);
  }

  // Mixed system with prims, curved background
  double dg_auxiliary_package_data(
      const gsl::not_null<Scalar<DataVector>*> out_normal_dot_flux_var1,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          out_normal_dot_flux_var2,
      const gsl::not_null<Scalar<DataVector>*> out_var1,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> out_var2,
      const gsl::not_null<Scalar<DataVector>*> max_abs_char_speed,

      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& var2,

      const tnsr::IJ<DataVector, Dim, Frame::Inertial>& flux_var2,

      const Scalar<DataVector>& var3_squared,

      const Scalar<DataVector>& prim_var1,

      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& normal_vector,
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
          mesh_velocity,
      const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity,

      const double volume_number) const {
    CHECK(volume_number == 4.5);
    tnsr::I<DataVector, Dim, Frame::Inertial> var3_buffer{get(var1).size(),
                                                          0.0};
    // The physical overload folds its own volume-tag value
    // (`BoundaryCorrectionVolumeTag`, 3.5) into the packaged data, which is
    // what the shared `dg_boundary_terms` checks expect; the auxiliary volume
    // tag (4.5) is checked above and dropped.
    return dg_package_data(
        out_normal_dot_flux_var1, out_normal_dot_flux_var2, out_var1, out_var2,
        max_abs_char_speed, make_not_null(&var3_buffer), var1, var2,
        var3_buffer, flux_var2, var3_squared, prim_var1, normal_covector,
        normal_vector, mesh_velocity, normal_dot_mesh_velocity, 3.5);
  }

  // LDG auxiliary boundary terms.
  // Verify the framework delivered the correct interior/exterior packaged
  // data by reusing the physical `dg_boundary_terms` checks.
  // Its evolved-variable corrections are written to scratch buffers and
  // discarded. The Var3 passthrough fields it also checks are absent from
  // the auxiliary packaged fields, so they are supplied here with the values
  // expected, making those two checks vacuous in the auxiliary pass.
  void dg_auxiliary_boundary_terms(
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          boundary_correction_var3,
      const Scalar<DataVector>& int_normal_dot_flux_var1,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& int_normal_dot_flux_var2,
      const Scalar<DataVector>& int_var1,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& int_var2,
      const Scalar<DataVector>& int_max_abs_char_speed,
      const Scalar<DataVector>& ext_normal_dot_flux_var1,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& ext_normal_dot_flux_var2,
      const Scalar<DataVector>& ext_var1,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& ext_var2,
      const Scalar<DataVector>& ext_max_abs_char_speed,
      const dg::Formulation formulation, const double volume_number) const {
    CHECK(volume_number == 4.5);

    for (size_t i = 0; i < Dim; ++i) {
      boundary_correction_var3->get(i) =
          0.5 * (int_var2.get(i) + ext_var2.get(i));
    }

    const size_t num_pts = get(int_var1).size();
    Scalar<DataVector> correction_var1_buffer{num_pts};
    tnsr::I<DataVector, Dim, Frame::Inertial> correction_var2_buffer{num_pts};
    tnsr::I<DataVector, Dim, Frame::Inertial> int_var3{num_pts};
    tnsr::I<DataVector, Dim, Frame::Inertial> ext_var3{num_pts};
    for (size_t i = 0; i < Dim; ++i) {
      int_var3.get(i) = offset_evolved_vars + static_cast<double>(i);
      ext_var3.get(i) =
          offset_boundary_condition + 2.0 + static_cast<double>(i);
    }
    dg_boundary_terms(
        make_not_null(&correction_var1_buffer),
        make_not_null(&correction_var2_buffer), int_normal_dot_flux_var1,
        int_normal_dot_flux_var2, int_var1, int_var2, int_max_abs_char_speed,
        int_var3, ext_normal_dot_flux_var1, ext_normal_dot_flux_var2, ext_var1,
        ext_var2, ext_max_abs_char_speed, ext_var3, formulation);
  }

 private:
  bool mesh_is_moving_{false};
  double sign_of_normal_{0.0};
};

template <size_t Dim, bool HasPrims, SystemType SysType,
          bool HasInverseSpatialMetric>
PUP::able::PUP_ID
    // NOLINTNEXTLINE
    BoundaryTerms<Dim, HasPrims, SysType, HasInverseSpatialMetric>::my_PUP_ID =
        0;

// Forward declare different boundary conditions.
//
// We template them on the system so we can test that we have access to all the
// different tags that we should have access to.
template <typename System>
class DemandOutgoingCharSpeeds;
template <typename System>
class TimeDerivative;
template <typename System>
class Ghost;
template <typename System>
class GhostAndTimeDerivative;

template <typename System>
class BoundaryCondition : public domain::BoundaryConditions::BoundaryCondition {
 public:
  BoundaryCondition() = default;
  BoundaryCondition(BoundaryCondition&&) = default;
  BoundaryCondition& operator=(BoundaryCondition&&) = default;
  BoundaryCondition(const BoundaryCondition&) = default;
  BoundaryCondition& operator=(const BoundaryCondition&) = default;
  ~BoundaryCondition() override = default;
  explicit BoundaryCondition(CkMigrateMessage* msg)
      : domain::BoundaryConditions::BoundaryCondition(msg) {}

  void pup(PUP::er& p) override {
    domain::BoundaryConditions::BoundaryCondition::pup(p);
  }
};

template <typename System>
class DemandOutgoingCharSpeeds : public BoundaryCondition<System> {
 public:
  DemandOutgoingCharSpeeds() = default;
  explicit DemandOutgoingCharSpeeds(const bool mesh_is_moving)
      : mesh_is_moving_(mesh_is_moving) {}
  DemandOutgoingCharSpeeds(DemandOutgoingCharSpeeds&&) = default;
  DemandOutgoingCharSpeeds& operator=(DemandOutgoingCharSpeeds&&) = default;
  DemandOutgoingCharSpeeds(const DemandOutgoingCharSpeeds&) = default;
  DemandOutgoingCharSpeeds& operator=(const DemandOutgoingCharSpeeds&) =
      default;
  ~DemandOutgoingCharSpeeds() override = default;

  explicit DemandOutgoingCharSpeeds(CkMigrateMessage* msg)
      : BoundaryCondition<System>(msg) {}

  WRAPPED_PUPable_decl_base_template(
      domain::BoundaryConditions::BoundaryCondition, DemandOutgoingCharSpeeds);

  auto get_clone() const -> std::unique_ptr<
      domain::BoundaryConditions::BoundaryCondition> override {
    return std::make_unique<DemandOutgoingCharSpeeds<System>>(*this);
  }

  static constexpr ::evolution::BoundaryConditions::Type bc_type =
      ::evolution::BoundaryConditions::Type::DemandOutgoingCharSpeeds;

  // NOLINTNEXTLINE
  void pup(PUP::er& p) override {
    BoundaryCondition<System>::pup(p);
    p | mesh_is_moving_;
  }

  using dg_interior_evolved_variables_tags =
      tmpl::list<Tags::Var1, Tags::Var2<System::volume_dim>>;
  using dg_interior_primitive_variables_tags =
      tmpl::list<Tags::PrimVar1, Tags::PrimVar2<System::volume_dim>>;
  using dg_interior_temporary_tags = tmpl::list<Tags::Var3Squared>;
  using dg_interior_dt_vars_tags = tmpl::list<::Tags::dt<Tags::Var1>>;
  using dg_gridless_tags = tmpl::list<Tags::BoundaryConditionVolumeTag>;

  std::optional<std::string> dg_demand_outgoing_char_speeds(
      const std::optional<tnsr::I<DataVector, System::volume_dim,
                                  Frame::Inertial>>& face_mesh_velocity,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& var3_squared, const Scalar<DataVector>& dt_var1,
      const double volume_number) const {
    CHECK(volume_number == 2.5);
    const size_t num_pts = get(var1).size();
    CHECK_ITERABLE_APPROX(get(var3_squared),
                          DataVector(num_pts, offset_temporaries));
    CHECK_ITERABLE_APPROX(get(var1), DataVector(num_pts, offset_evolved_vars));
    for (size_t i = 0; i < System::volume_dim; ++i) {
      CHECK_ITERABLE_APPROX(var2.get(i),
                            DataVector(num_pts, offset_evolved_vars + 1 + i));
      for (size_t j = 0; j < num_pts; ++j) {
        // Catch doesn't allow `CHECK(a or b) so we do `CHECK((a or b))` instead
        if constexpr (System::volume_dim == 1) {
          const double normalization_factor =
              System::has_inverse_spatial_metric
                  ? sqrt(offset_temporaries + 1.0)
                  : 1.0;
          CHECK((approx(outward_directed_normal_covector.get(i)[j]) ==
                     1.0 / normalization_factor or
                 approx(outward_directed_normal_covector.get(i)[j]) ==
                     -1.0 / normalization_factor));
        } else {
          static_assert(not System::has_inverse_spatial_metric);
          CHECK((approx(outward_directed_normal_covector.get(i)[j]) == 1.0 or
                 approx(outward_directed_normal_covector.get(i)[j]) == -1.0 or
                 approx(outward_directed_normal_covector.get(i)[j]) == 0.0));
        }
      }
    }
    CHECK_ITERABLE_APPROX(get(dt_var1),
                          DataVector(num_pts, offset_dt_evolved_vars));
    REQUIRE(face_mesh_velocity.has_value() == mesh_is_moving_);
    if (mesh_is_moving_) {
      for (size_t i = 0; i < System::volume_dim; ++i) {
        CHECK_ITERABLE_APPROX(
            face_mesh_velocity->get(i),
            DataVector(num_pts, gsl::at(expected_velocities, i)));
      }
    }
    return std::nullopt;
  }

  std::optional<std::string> dg_demand_outgoing_char_speeds(
      const std::optional<tnsr::I<DataVector, System::volume_dim,
                                  Frame::Inertial>>& face_mesh_velocity,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_vector,
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& var3_squared, const Scalar<DataVector>& dt_var1,
      const double volume_number) const {
    dg_demand_outgoing_char_speeds(face_mesh_velocity,
                                   outward_directed_normal_covector, var1, var2,
                                   var3_squared, dt_var1, volume_number);
    CHECK_ITERABLE_APPROX(get(dot_product(outward_directed_normal_covector,
                                          outward_directed_normal_vector)),
                          DataVector(get(var1).size(), 1.0));
    return std::nullopt;
  }

  std::optional<std::string> dg_demand_outgoing_char_speeds(
      const std::optional<tnsr::I<DataVector, System::volume_dim,
                                  Frame::Inertial>>& face_mesh_velocity,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& prim_var1,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>& prim_var2,
      const Scalar<DataVector>& var3_squared, const Scalar<DataVector>& dt_var1,
      const double volume_number) const {
    dg_demand_outgoing_char_speeds(face_mesh_velocity,
                                   outward_directed_normal_covector, var1, var2,
                                   var3_squared, dt_var1, volume_number);
    const size_t num_pts = get(var1).size();
    CHECK_ITERABLE_APPROX(get(prim_var1),
                          DataVector(num_pts, offset_primitive_vars));
    for (size_t i = 0; i < System::volume_dim; ++i) {
      CHECK_ITERABLE_APPROX(prim_var2.get(i),
                            DataVector(num_pts, offset_primitive_vars + 1 + i));
    }
    return std::nullopt;
  }

  std::optional<std::string> dg_demand_outgoing_char_speeds(
      const std::optional<tnsr::I<DataVector, System::volume_dim,
                                  Frame::Inertial>>& face_mesh_velocity,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_vector,
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& prim_var1,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>& prim_var2,
      const Scalar<DataVector>& var3_squared, const Scalar<DataVector>& dt_var1,
      const double volume_number) const {
    dg_demand_outgoing_char_speeds(
        face_mesh_velocity, outward_directed_normal_covector, var1, var2,
        prim_var1, prim_var2, var3_squared, dt_var1, volume_number);
    CHECK_ITERABLE_APPROX(get(dot_product(outward_directed_normal_covector,
                                          outward_directed_normal_vector)),
                          DataVector(get(var1).size(), 1.0));
    return std::nullopt;
  }

 private:
  bool mesh_is_moving_{false};
};

template <typename System>
// NOLINTNEXTLINE
PUP::able::PUP_ID DemandOutgoingCharSpeeds<System>::my_PUP_ID = 0;

template <typename System>
class Ghost : public BoundaryCondition<System> {
 public:
  Ghost() = default;
  explicit Ghost(const bool mesh_is_moving) : mesh_is_moving_(mesh_is_moving) {}
  Ghost(Ghost&&) = default;
  Ghost& operator=(Ghost&&) = default;
  Ghost(const Ghost&) = default;
  Ghost& operator=(const Ghost&) = default;
  ~Ghost() override = default;

  explicit Ghost(CkMigrateMessage* msg) : BoundaryCondition<System>(msg) {}

  WRAPPED_PUPable_decl_base_template(
      domain::BoundaryConditions::BoundaryCondition, Ghost);

  auto get_clone() const -> std::unique_ptr<
      domain::BoundaryConditions::BoundaryCondition> override {
    return std::make_unique<Ghost<System>>(*this);
  }

  static constexpr ::evolution::BoundaryConditions::Type bc_type =
      ::evolution::BoundaryConditions::Type::Ghost;

  // NOLINTNEXTLINE
  void pup(PUP::er& p) override {
    BoundaryCondition<System>::pup(p);
    p | mesh_is_moving_;
  }

  using dg_interior_evolved_variables_tags =
      tmpl::list<Tags::Var1, Tags::Var2<System::volume_dim>>;
  using dg_interior_primitive_variables_tags =
      tmpl::list<Tags::PrimVar1, Tags::PrimVar2<System::volume_dim>>;
  using dg_interior_temporary_tags = tmpl::list<Tags::Var3Squared>;
  using dg_interior_dt_vars_tags = tmpl::list<::Tags::dt<Tags::Var1>>;
  using dg_gridless_tags = tmpl::list<Tags::BoundaryConditionVolumeTag>;

  // Nonconservative system, flat background
  std::optional<std::string> dg_ghost(
      const gsl::not_null<Scalar<DataVector>*> out_var1,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          out_var2,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          out_var3,
      const gsl::not_null<Scalar<DataVector>*> out_var3_squared,
      const std::optional<tnsr::I<DataVector, System::volume_dim,
                                  Frame::Inertial>>& face_mesh_velocity,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& var3_squared, const Scalar<DataVector>& dt_var1,
      const double volume_number) const {
    get(*out_var1) = offset_boundary_condition;
    for (size_t i = 0; i < System::volume_dim; ++i) {
      out_var2->get(i) =
          offset_boundary_condition + 1.0 + static_cast<double>(i);
    }
    for (size_t i = 0; i < System::volume_dim; ++i) {
      out_var3->get(i) =
          offset_boundary_condition + 2.0 + static_cast<double>(i);
    }
    get(*out_var3_squared) = offset_boundary_condition + 1.0 +
                             (2 + System::volume_dim) * System::volume_dim;

    CHECK(volume_number == 2.5);
    const size_t num_pts = get(var1).size();
    CHECK_ITERABLE_APPROX(get(var3_squared),
                          DataVector(num_pts, offset_temporaries));
    CHECK_ITERABLE_APPROX(get(var1), DataVector(num_pts, offset_evolved_vars));
    for (size_t i = 0; i < System::volume_dim; ++i) {
      CHECK_ITERABLE_APPROX(var2.get(i),
                            DataVector(num_pts, offset_evolved_vars + 1 + i));
      for (size_t j = 0; j < num_pts; ++j) {
        // Catch doesn't allow `CHECK(a or b) so we do `CHECK((a or b))` instead
        if constexpr (System::volume_dim == 1) {
          const double normalization_factor =
              System::has_inverse_spatial_metric
                  ? sqrt(offset_temporaries + 1.0)
                  : 1.0;
          CHECK((approx(outward_directed_normal_covector.get(i)[j]) ==
                     1.0 / normalization_factor or
                 approx(outward_directed_normal_covector.get(i)[j]) ==
                     -1.0 / normalization_factor));
        } else {
          static_assert(not System::has_inverse_spatial_metric);
          CHECK((approx(outward_directed_normal_covector.get(i)[j]) == 1.0 or
                 approx(outward_directed_normal_covector.get(i)[j]) == -1.0 or
                 approx(outward_directed_normal_covector.get(i)[j]) == 0.0));
        }
      }
    }
    CHECK_ITERABLE_APPROX(get(dt_var1),
                          DataVector(num_pts, offset_dt_evolved_vars));
    REQUIRE(face_mesh_velocity.has_value() == mesh_is_moving_);
    if (mesh_is_moving_) {
      for (size_t i = 0; i < System::volume_dim; ++i) {
        CHECK_ITERABLE_APPROX(
            face_mesh_velocity->get(i),
            DataVector(num_pts, gsl::at(expected_velocities, i)));
      }
    }
    return std::nullopt;
  }

  // Nonconservative system, curved background
  std::optional<std::string> dg_ghost(
      const gsl::not_null<Scalar<DataVector>*> out_var1,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          out_var2,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          out_var3,
      const gsl::not_null<Scalar<DataVector>*> out_var3_squared,
      const gsl::not_null<
          tnsr::II<DataVector, System::volume_dim, Frame::Inertial>*>
          inv_spatial_metric,

      const std::optional<tnsr::I<DataVector, System::volume_dim,
                                  Frame::Inertial>>& face_mesh_velocity,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_vector,
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& var3_squared, const Scalar<DataVector>& dt_var1,
      const double volume_number) const {
    check_normal_vector_set_inverse_spatial_metric(
        inv_spatial_metric, outward_directed_normal_covector,
        outward_directed_normal_vector);
    return dg_ghost(out_var1, out_var2, out_var3, out_var3_squared,
                    face_mesh_velocity, outward_directed_normal_covector, var1,
                    var2, var3_squared, dt_var1, volume_number);
  }

  // Mixed conservative non-conservative system, no prims, flat background
  std::optional<std::string> dg_ghost(
      const gsl::not_null<Scalar<DataVector>*> out_var1,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          out_var2,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          out_var3,
      const gsl::not_null<
          tnsr::IJ<DataVector, System::volume_dim, Frame::Inertial>*>
          flux_var2,
      const gsl::not_null<Scalar<DataVector>*> out_var3_squared,
      const std::optional<tnsr::I<DataVector, System::volume_dim,
                                  Frame::Inertial>>& face_mesh_velocity,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& var3_squared, const Scalar<DataVector>& dt_var1,
      const double volume_number) const {
    dg_ghost(out_var1, out_var2, out_var3, out_var3_squared, face_mesh_velocity,
             outward_directed_normal_covector, var1, var2, var3_squared,
             dt_var1, volume_number);
    for (size_t i = 0; i < System::volume_dim; ++i) {
      for (size_t j = 0; j < System::volume_dim; ++j) {
        flux_var2->get(i, j) = offset_boundary_condition + 1.0 +
                               static_cast<double>(i + 2 * System::volume_dim);
      }
    }
    return std::nullopt;
  }

  // Mixed conservative non-conservative system, no prims, curved background
  std::optional<std::string> dg_ghost(
      const gsl::not_null<Scalar<DataVector>*> out_var1,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          out_var2,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          out_var3,
      const gsl::not_null<
          tnsr::IJ<DataVector, System::volume_dim, Frame::Inertial>*>
          flux_var2,
      const gsl::not_null<Scalar<DataVector>*> out_var3_squared,
      const gsl::not_null<
          tnsr::II<DataVector, System::volume_dim, Frame::Inertial>*>
          inv_spatial_metric,

      const std::optional<tnsr::I<DataVector, System::volume_dim,
                                  Frame::Inertial>>& face_mesh_velocity,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_vector,
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& var3_squared, const Scalar<DataVector>& dt_var1,
      const double volume_number) const {
    check_normal_vector_set_inverse_spatial_metric(
        inv_spatial_metric, outward_directed_normal_covector,
        outward_directed_normal_vector);
    return dg_ghost(out_var1, out_var2, out_var3, flux_var2, out_var3_squared,
                    face_mesh_velocity, outward_directed_normal_covector, var1,
                    var2, var3_squared, dt_var1, volume_number);
  }

  // Mixed conservative non-conservative system, with prims, flat background
  std::optional<std::string> dg_ghost(
      const gsl::not_null<Scalar<DataVector>*> out_var1,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          out_var2,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          out_var3,
      const gsl::not_null<
          tnsr::IJ<DataVector, System::volume_dim, Frame::Inertial>*>
          flux_var2,
      const gsl::not_null<Scalar<DataVector>*> out_var3_squared,
      const gsl::not_null<Scalar<DataVector>*> out_prim_var1,

      const std::optional<tnsr::I<DataVector, System::volume_dim,
                                  Frame::Inertial>>& face_mesh_velocity,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& prim_var1,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>& prim_var2,
      const Scalar<DataVector>& var3_squared, const Scalar<DataVector>& dt_var1,
      const double volume_number) const {
    dg_ghost(out_var1, out_var2, out_var3, flux_var2, out_var3_squared,
             face_mesh_velocity, outward_directed_normal_covector, var1, var2,
             var3_squared, dt_var1, volume_number);
    get(*out_prim_var1) = get(*out_var3_squared) + 1.0;
    const size_t num_pts = get(var1).size();
    CHECK_ITERABLE_APPROX(get(prim_var1),
                          DataVector(num_pts, offset_primitive_vars));
    for (size_t i = 0; i < System::volume_dim; ++i) {
      CHECK_ITERABLE_APPROX(prim_var2.get(i),
                            DataVector(num_pts, offset_primitive_vars + 1 + i));
    }
    return std::nullopt;
  }

  // Mixed conservative non-conservative system, with prims, curved background
  std::optional<std::string> dg_ghost(
      const gsl::not_null<Scalar<DataVector>*> out_var1,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          out_var2,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          out_var3,
      const gsl::not_null<
          tnsr::IJ<DataVector, System::volume_dim, Frame::Inertial>*>
          flux_var2,
      const gsl::not_null<Scalar<DataVector>*> out_var3_squared,
      const gsl::not_null<Scalar<DataVector>*> out_prim_var1,
      const gsl::not_null<
          tnsr::II<DataVector, System::volume_dim, Frame::Inertial>*>
          inv_spatial_metric,

      const std::optional<tnsr::I<DataVector, System::volume_dim,
                                  Frame::Inertial>>& face_mesh_velocity,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_vector,
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& prim_var1,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>& prim_var2,
      const Scalar<DataVector>& var3_squared, const Scalar<DataVector>& dt_var1,
      const double volume_number) const {
    check_normal_vector_set_inverse_spatial_metric(
        inv_spatial_metric, outward_directed_normal_covector,
        outward_directed_normal_vector);
    return dg_ghost(out_var1, out_var2, out_var3, flux_var2, out_var3_squared,
                    out_prim_var1, face_mesh_velocity,
                    outward_directed_normal_covector, var1, var2, prim_var1,
                    prim_var2, var3_squared, dt_var1, volume_number);
  }

  // Conservative system, no prims, flat background
  std::optional<std::string> dg_ghost(
      const gsl::not_null<Scalar<DataVector>*> out_var1,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          out_var2,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          out_var3,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          flux_var1,
      const gsl::not_null<
          tnsr::IJ<DataVector, System::volume_dim, Frame::Inertial>*>
          flux_var2,
      const gsl::not_null<Scalar<DataVector>*> out_var3_squared,
      const std::optional<tnsr::I<DataVector, System::volume_dim,
                                  Frame::Inertial>>& face_mesh_velocity,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& var3_squared, const Scalar<DataVector>& dt_var1,
      const double volume_number) const {
    dg_ghost(out_var1, out_var2, out_var3, flux_var2, out_var3_squared,
             face_mesh_velocity, outward_directed_normal_covector, var1, var2,
             var3_squared, dt_var1, volume_number);
    for (size_t i = 0; i < System::volume_dim; ++i) {
      flux_var1->get(i) = offset_boundary_condition + 1.0 +
                          static_cast<double>(i + System::volume_dim);
    }
    return std::nullopt;
  }

  // Conservative system, no prims, curved background
  std::optional<std::string> dg_ghost(
      const gsl::not_null<Scalar<DataVector>*> out_var1,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          out_var2,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          out_var3,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          flux_var1,
      const gsl::not_null<
          tnsr::IJ<DataVector, System::volume_dim, Frame::Inertial>*>
          flux_var2,
      const gsl::not_null<Scalar<DataVector>*> out_var3_squared,
      const gsl::not_null<
          tnsr::II<DataVector, System::volume_dim, Frame::Inertial>*>
          inv_spatial_metric,
      const std::optional<tnsr::I<DataVector, System::volume_dim,
                                  Frame::Inertial>>& face_mesh_velocity,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_vector,
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& var3_squared, const Scalar<DataVector>& dt_var1,
      const double volume_number) const {
    check_normal_vector_set_inverse_spatial_metric(
        inv_spatial_metric, outward_directed_normal_covector,
        outward_directed_normal_vector);
    dg_ghost(out_var1, out_var2, out_var3, flux_var1, flux_var2,
             out_var3_squared, face_mesh_velocity,
             outward_directed_normal_covector, var1, var2, var3_squared,
             dt_var1, volume_number);
    return std::nullopt;
  }

  // Conservative system, with prims
  std::optional<std::string> dg_ghost(
      const gsl::not_null<Scalar<DataVector>*> out_var1,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          out_var2,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          out_var3,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          flux_var1,
      const gsl::not_null<
          tnsr::IJ<DataVector, System::volume_dim, Frame::Inertial>*>
          flux_var2,
      const gsl::not_null<Scalar<DataVector>*> out_var3_squared,
      const gsl::not_null<Scalar<DataVector>*> out_prim_var1,

      const std::optional<tnsr::I<DataVector, System::volume_dim,
                                  Frame::Inertial>>& face_mesh_velocity,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& prim_var1,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>& prim_var2,
      const Scalar<DataVector>& var3_squared, const Scalar<DataVector>& dt_var1,
      const double volume_number) const {
    dg_ghost(out_var1, out_var2, out_var3, flux_var1, flux_var2,
             out_var3_squared, face_mesh_velocity,
             outward_directed_normal_covector, var1, var2, var3_squared,
             dt_var1, volume_number);
    get(*out_prim_var1) = get(*out_var3_squared) + 1.0;
    const size_t num_pts = get(var1).size();
    CHECK_ITERABLE_APPROX(get(prim_var1),
                          DataVector(num_pts, offset_primitive_vars));
    for (size_t i = 0; i < System::volume_dim; ++i) {
      CHECK_ITERABLE_APPROX(prim_var2.get(i),
                            DataVector(num_pts, offset_primitive_vars + 1 + i));
    }
    return std::nullopt;
  }

  // Conservative system, with prims, curved background
  std::optional<std::string> dg_ghost(
      const gsl::not_null<Scalar<DataVector>*> out_var1,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          out_var2,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          out_var3,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          flux_var1,
      const gsl::not_null<
          tnsr::IJ<DataVector, System::volume_dim, Frame::Inertial>*>
          flux_var2,
      const gsl::not_null<Scalar<DataVector>*> out_var3_squared,
      const gsl::not_null<Scalar<DataVector>*> out_prim_var1,
      const gsl::not_null<
          tnsr::II<DataVector, System::volume_dim, Frame::Inertial>*>
          inv_spatial_metric,

      const std::optional<tnsr::I<DataVector, System::volume_dim,
                                  Frame::Inertial>>& face_mesh_velocity,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_vector,
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& prim_var1,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>& prim_var2,
      const Scalar<DataVector>& var3_squared, const Scalar<DataVector>& dt_var1,
      const double volume_number) const {
    check_normal_vector_set_inverse_spatial_metric(
        inv_spatial_metric, outward_directed_normal_covector,
        outward_directed_normal_vector);
    return dg_ghost(out_var1, out_var2, out_var3, flux_var1, flux_var2,
                    out_var3_squared, out_prim_var1, face_mesh_velocity,
                    outward_directed_normal_covector, var1, var2, prim_var1,
                    prim_var2, var3_squared, dt_var1, volume_number);
  }

  // public so that GhostAndTimeDerivative can call into this
  void check_normal_vector_set_inverse_spatial_metric(
      const gsl::not_null<
          tnsr::II<DataVector, System::volume_dim, Frame::Inertial>*>
          inv_spatial_metric,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_vector) const {
    CHECK_ITERABLE_APPROX(
        get(dot_product(outward_directed_normal_covector,
                        outward_directed_normal_vector)),
        DataVector(get<0>(outward_directed_normal_vector).size(), 1.0));
    for (size_t i = 0; i < inv_spatial_metric->size(); ++i) {
      (*inv_spatial_metric)[i] =
          DataVector{get<0>(outward_directed_normal_vector).size(),
                     (offset_temporaries + 1.0 + i)};
    }
  }

 private:
  bool mesh_is_moving_{false};
};

template <typename System>
// NOLINTNEXTLINE
PUP::able::PUP_ID Ghost<System>::my_PUP_ID = 0;

template <typename System>
class TimeDerivative : public BoundaryCondition<System> {
 public:
  TimeDerivative() = default;
  TimeDerivative(const bool mesh_is_moving, const double expected_dt_var1)
      : mesh_is_moving_(mesh_is_moving), expected_dt_var1_(expected_dt_var1) {}
  TimeDerivative(TimeDerivative&&) = default;
  TimeDerivative& operator=(TimeDerivative&&) = default;
  TimeDerivative(const TimeDerivative&) = default;
  TimeDerivative& operator=(const TimeDerivative&) = default;
  ~TimeDerivative() override = default;

  explicit TimeDerivative(CkMigrateMessage* msg)
      : BoundaryCondition<System>(msg) {}

  WRAPPED_PUPable_decl_base_template(
      domain::BoundaryConditions::BoundaryCondition, TimeDerivative);

  auto get_clone() const -> std::unique_ptr<
      domain::BoundaryConditions::BoundaryCondition> override {
    return std::make_unique<TimeDerivative<System>>(*this);
  }

  static constexpr ::evolution::BoundaryConditions::Type bc_type =
      ::evolution::BoundaryConditions::Type::TimeDerivative;

  // NOLINTNEXTLINE
  void pup(PUP::er& p) override {
    BoundaryCondition<System>::pup(p);
    p | mesh_is_moving_;
    p | expected_dt_var1_;
  }

  using dg_interior_evolved_variables_tags =
      tmpl::list<Tags::Var1, Tags::Var2<System::volume_dim>>;
  using dg_interior_primitive_variables_tags =
      tmpl::list<Tags::PrimVar1, Tags::PrimVar2<System::volume_dim>>;
  using dg_interior_temporary_tags = tmpl::list<Tags::Var3Squared>;
  using dg_interior_dt_vars_tags = tmpl::list<::Tags::dt<Tags::Var1>>;
  using dg_interior_deriv_vars_tags = tmpl::conditional_t<
      System::system_type == SystemType::Conservative, tmpl::list<>,
      tmpl::list<::Tags::deriv<Tags::Var1, tmpl::size_t<System::volume_dim>,
                               Frame::Inertial>>>;
  using dg_gridless_tags = tmpl::list<Tags::BoundaryConditionVolumeTag>;

  // Conservative, no prims, flat background
  std::optional<std::string> dg_time_derivative(
      const gsl::not_null<Scalar<DataVector>*> dt_correction_var1,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          dt_correction_var2,
      const std::optional<tnsr::I<DataVector, System::volume_dim,
                                  Frame::Inertial>>& face_mesh_velocity,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& var3_squared, const Scalar<DataVector>& dt_var1,
      const double volume_number) const {
    CHECK(volume_number == 2.5);
    const size_t num_pts = get(var1).size();
    CHECK_ITERABLE_APPROX(get(var3_squared),
                          DataVector(num_pts, offset_temporaries));
    CHECK_ITERABLE_APPROX(get(var1), DataVector(num_pts, offset_evolved_vars));
    for (size_t i = 0; i < System::volume_dim; ++i) {
      CHECK_ITERABLE_APPROX(var2.get(i),
                            DataVector(num_pts, offset_evolved_vars + 1 + i));
      for (size_t j = 0; j < num_pts; ++j) {
        // Catch doesn't allow `CHECK(a or b) so we do `CHECK((a or b))` instead
        if constexpr (System::volume_dim == 1) {
          const double normalization_factor =
              System::has_inverse_spatial_metric
                  ? sqrt(offset_temporaries + 1.0)
                  : 1.0;
          CHECK((approx(outward_directed_normal_covector.get(i)[j]) ==
                     1.0 / normalization_factor or
                 approx(outward_directed_normal_covector.get(i)[j]) ==
                     -1.0 / normalization_factor));
        } else {
          CHECK((approx(outward_directed_normal_covector.get(i)[j]) == 1.0 or
                 approx(outward_directed_normal_covector.get(i)[j]) == -1.0 or
                 approx(outward_directed_normal_covector.get(i)[j]) == 0.0));
        }
      }
    }
    CHECK_ITERABLE_APPROX(get(dt_var1), DataVector(num_pts, expected_dt_var1_));

    REQUIRE(face_mesh_velocity.has_value() == mesh_is_moving_);
    if (mesh_is_moving_) {
      for (size_t i = 0; i < System::volume_dim; ++i) {
        CHECK_ITERABLE_APPROX(
            face_mesh_velocity->get(i),
            DataVector(num_pts, gsl::at(expected_velocities, i)));
      }
    }
    get(*dt_correction_var1) = offset_boundary_condition;
    for (size_t i = 0; i < System::volume_dim; ++i) {
      dt_correction_var2->get(i) =
          offset_boundary_condition + 1.0 + static_cast<double>(i);
    }
    return std::nullopt;
  }

  // Conservative, no prims, curved background
  std::optional<std::string> dg_time_derivative(
      const gsl::not_null<Scalar<DataVector>*> dt_correction_var1,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          dt_correction_var2,
      const std::optional<tnsr::I<DataVector, System::volume_dim,
                                  Frame::Inertial>>& face_mesh_velocity,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_vector,
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& var3_squared, const Scalar<DataVector>& dt_var1,
      const double volume_number) const {
    check_normal_vector(outward_directed_normal_covector,
                        outward_directed_normal_vector);
    return dg_time_derivative(dt_correction_var1, dt_correction_var2,
                              face_mesh_velocity,
                              outward_directed_normal_covector, var1, var2,
                              var3_squared, dt_var1, volume_number);
  }

  // Mixed and non-conservative system, flat background
  std::optional<std::string> dg_time_derivative(
      const gsl::not_null<Scalar<DataVector>*> dt_correction_var1,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          dt_correction_var2,
      const std::optional<tnsr::I<DataVector, System::volume_dim,
                                  Frame::Inertial>>& face_mesh_velocity,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& var3_squared, const Scalar<DataVector>& dt_var1,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>& d_var1,
      const double volume_number) const {
    dg_time_derivative(dt_correction_var1, dt_correction_var2,
                       face_mesh_velocity, outward_directed_normal_covector,
                       var1, var2, var3_squared, dt_var1, volume_number);
    const size_t num_pts = get(var1).size();
    for (size_t i = 0; i < System::volume_dim; ++i) {
      CHECK_ITERABLE_APPROX(d_var1.get(i),
                            DataVector(num_pts, offset_partial_derivs + i));
    }
    return std::nullopt;
  }

  // Mixed and non-conservative system, curved background
  std::optional<std::string> dg_time_derivative(
      const gsl::not_null<Scalar<DataVector>*> dt_correction_var1,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          dt_correction_var2,
      const std::optional<tnsr::I<DataVector, System::volume_dim,
                                  Frame::Inertial>>& face_mesh_velocity,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_vector,
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& var3_squared, const Scalar<DataVector>& dt_var1,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>& d_var1,
      const double volume_number) const {
    check_normal_vector(outward_directed_normal_covector,
                        outward_directed_normal_vector);
    return dg_time_derivative(dt_correction_var1, dt_correction_var2,
                              face_mesh_velocity,
                              outward_directed_normal_covector, var1, var2,
                              var3_squared, dt_var1, d_var1, volume_number);
  }

  // Mixed system with primitive vars, flat background
  std::optional<std::string> dg_time_derivative(
      const gsl::not_null<Scalar<DataVector>*> dt_correction_var1,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          dt_correction_var2,
      const std::optional<tnsr::I<DataVector, System::volume_dim,
                                  Frame::Inertial>>& face_mesh_velocity,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& prim_var1,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>& prim_var2,
      const Scalar<DataVector>& var3_squared, const Scalar<DataVector>& dt_var1,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>& d_var1,
      const double volume_number) const {
    dg_time_derivative(dt_correction_var1, dt_correction_var2,
                       face_mesh_velocity, outward_directed_normal_covector,
                       var1, var2, prim_var1, prim_var2, var3_squared, dt_var1,
                       volume_number);
    // Sets the dt_correction again, but that's fine, values stay the same.
    dg_time_derivative(dt_correction_var1, dt_correction_var2,
                       face_mesh_velocity, outward_directed_normal_covector,
                       var1, var2, var3_squared, dt_var1, d_var1,
                       volume_number);
    return std::nullopt;
  }

  // Mixed system with primitive vars, curved background
  std::optional<std::string> dg_time_derivative(
      const gsl::not_null<Scalar<DataVector>*> dt_correction_var1,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          dt_correction_var2,
      const std::optional<tnsr::I<DataVector, System::volume_dim,
                                  Frame::Inertial>>& face_mesh_velocity,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_vector,
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& prim_var1,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>& prim_var2,
      const Scalar<DataVector>& var3_squared, const Scalar<DataVector>& dt_var1,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>& d_var1,
      const double volume_number) const {
    check_normal_vector(outward_directed_normal_covector,
                        outward_directed_normal_vector);
    return dg_time_derivative(
        dt_correction_var1, dt_correction_var2, face_mesh_velocity,
        outward_directed_normal_covector, var1, var2, prim_var1, prim_var2,
        var3_squared, dt_var1, d_var1, volume_number);
  }

  // Conservative system with primitive vars
  std::optional<std::string> dg_time_derivative(
      const gsl::not_null<Scalar<DataVector>*> dt_correction_var1,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          dt_correction_var2,
      const std::optional<tnsr::I<DataVector, System::volume_dim,
                                  Frame::Inertial>>& face_mesh_velocity,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& prim_var1,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>& prim_var2,
      const Scalar<DataVector>& var3_squared, const Scalar<DataVector>& dt_var1,
      const double volume_number) const {
    dg_time_derivative(dt_correction_var1, dt_correction_var2,
                       face_mesh_velocity, outward_directed_normal_covector,
                       var1, var2, var3_squared, dt_var1, volume_number);
    const size_t num_pts = get(var1).size();
    CHECK_ITERABLE_APPROX(get(prim_var1),
                          DataVector(num_pts, offset_primitive_vars));
    for (size_t i = 0; i < System::volume_dim; ++i) {
      CHECK_ITERABLE_APPROX(prim_var2.get(i),
                            DataVector(num_pts, offset_primitive_vars + 1 + i));
    }
    return std::nullopt;
  }

  // Conservative system with primitive vars, curved background
  std::optional<std::string> dg_time_derivative(
      const gsl::not_null<Scalar<DataVector>*> dt_correction_var1,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          dt_correction_var2,
      const std::optional<tnsr::I<DataVector, System::volume_dim,
                                  Frame::Inertial>>& face_mesh_velocity,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_vector,
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& prim_var1,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>& prim_var2,
      const Scalar<DataVector>& var3_squared, const Scalar<DataVector>& dt_var1,
      const double volume_number) const {
    check_normal_vector(outward_directed_normal_covector,
                        outward_directed_normal_vector);
    dg_time_derivative(dt_correction_var1, dt_correction_var2,
                       face_mesh_velocity, outward_directed_normal_covector,
                       var1, var2, prim_var1, prim_var2, var3_squared, dt_var1,
                       volume_number);
    return std::nullopt;
  }

 private:
  void check_normal_vector(
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_vector) const {
    CHECK_ITERABLE_APPROX(
        get(dot_product(outward_directed_normal_covector,
                        outward_directed_normal_vector)),
        DataVector(get<0>(outward_directed_normal_vector).size(), 1.0));
  }

  bool mesh_is_moving_{false};
  double expected_dt_var1_{std::numeric_limits<double>::signaling_NaN()};
};

template <typename System>
// NOLINTNEXTLINE
PUP::able::PUP_ID TimeDerivative<System>::my_PUP_ID = 0;

template <typename System>
class GhostAndTimeDerivative : public BoundaryCondition<System> {
 public:
  GhostAndTimeDerivative() = default;
  explicit GhostAndTimeDerivative(const bool mesh_is_moving)
      : ghost_{mesh_is_moving},
        time_derivative_{mesh_is_moving, offset_dt_evolved_vars} {}
  GhostAndTimeDerivative(GhostAndTimeDerivative&&) = default;
  GhostAndTimeDerivative& operator=(GhostAndTimeDerivative&&) = default;
  GhostAndTimeDerivative(const GhostAndTimeDerivative&) = default;
  GhostAndTimeDerivative& operator=(const GhostAndTimeDerivative&) = default;
  ~GhostAndTimeDerivative() override = default;

  explicit GhostAndTimeDerivative(CkMigrateMessage* msg)
      : BoundaryCondition<System>(msg) {}

  WRAPPED_PUPable_decl_base_template(
      domain::BoundaryConditions::BoundaryCondition, GhostAndTimeDerivative);

  auto get_clone() const -> std::unique_ptr<
      domain::BoundaryConditions::BoundaryCondition> override {
    return std::make_unique<GhostAndTimeDerivative<System>>(*this);
  }

  static constexpr ::evolution::BoundaryConditions::Type bc_type =
      ::evolution::BoundaryConditions::Type::GhostAndTimeDerivative;

  // NOLINTNEXTLINE
  void pup(PUP::er& p) override {
    BoundaryCondition<System>::pup(p);
    p | ghost_;
    p | time_derivative_;
  }

  using dg_interior_evolved_variables_tags =
      typename Ghost<System>::dg_interior_evolved_variables_tags;
  using dg_interior_primitive_variables_tags =
      typename Ghost<System>::dg_interior_primitive_variables_tags;
  using dg_interior_temporary_tags =
      typename Ghost<System>::dg_interior_temporary_tags;
  using dg_interior_dt_vars_tags =
      typename Ghost<System>::dg_interior_dt_vars_tags;
  using dg_interior_deriv_vars_tags =
      typename TimeDerivative<System>::dg_interior_deriv_vars_tags;
  using dg_gridless_tags = typename Ghost<System>::dg_gridless_tags;

  // Nonconservative, flat background
  std::optional<std::string> dg_ghost(
      const gsl::not_null<Scalar<DataVector>*> out_var1,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          out_var2,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          out_var3,
      const gsl::not_null<Scalar<DataVector>*> out_var3_squared,
      const std::optional<tnsr::I<DataVector, System::volume_dim,
                                  Frame::Inertial>>& face_mesh_velocity,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& var3_squared, const Scalar<DataVector>& dt_var1,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>& d_var1,
      const double volume_number) const {
    ghost_.dg_ghost(out_var1, out_var2, out_var3, out_var3_squared,
                    face_mesh_velocity, outward_directed_normal_covector, var1,
                    var2, var3_squared, dt_var1, volume_number);
    const size_t num_pts = get(var1).size();
    for (size_t i = 0; i < System::volume_dim; ++i) {
      CHECK_ITERABLE_APPROX(d_var1.get(i),
                            DataVector(num_pts, offset_partial_derivs + i));
    }
    return std::nullopt;
  }

  // Nonconservative, curved background
  std::optional<std::string> dg_ghost(
      const gsl::not_null<Scalar<DataVector>*> out_var1,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          out_var2,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          out_var3,
      const gsl::not_null<Scalar<DataVector>*> out_var3_squared,
      const gsl::not_null<
          tnsr::II<DataVector, System::volume_dim, Frame::Inertial>*>
          inv_spatial_metric,

      const std::optional<tnsr::I<DataVector, System::volume_dim,
                                  Frame::Inertial>>& face_mesh_velocity,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_vector,
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& var3_squared, const Scalar<DataVector>& dt_var1,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>& d_var1,
      const double volume_number) const {
    ghost_.check_normal_vector_set_inverse_spatial_metric(
        inv_spatial_metric, outward_directed_normal_covector,
        outward_directed_normal_vector);
    return dg_ghost(out_var1, out_var2, out_var3, out_var3_squared,
                    face_mesh_velocity, outward_directed_normal_covector, var1,
                    var2, var3_squared, dt_var1, d_var1, volume_number);
  }

  // Mixed conservative non-conservative system, no prims, flat background
  std::optional<std::string> dg_ghost(
      const gsl::not_null<Scalar<DataVector>*> out_var1,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          out_var2,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          out_var3,
      const gsl::not_null<
          tnsr::IJ<DataVector, System::volume_dim, Frame::Inertial>*>
          flux_var2,
      const gsl::not_null<Scalar<DataVector>*> out_var3_squared,
      const std::optional<tnsr::I<DataVector, System::volume_dim,
                                  Frame::Inertial>>& face_mesh_velocity,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& var3_squared, const Scalar<DataVector>& dt_var1,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>& d_var1,
      const double volume_number) const {
    ghost_.dg_ghost(out_var1, out_var2, out_var3, flux_var2, out_var3_squared,
                    face_mesh_velocity, outward_directed_normal_covector, var1,
                    var2, var3_squared, dt_var1, volume_number);
    const size_t num_pts = get(var1).size();
    for (size_t i = 0; i < System::volume_dim; ++i) {
      CHECK_ITERABLE_APPROX(d_var1.get(i),
                            DataVector(num_pts, offset_partial_derivs + i));
    }
    return std::nullopt;
  }

  // Mixed conservative non-conservative system, no prims, curved background
  std::optional<std::string> dg_ghost(
      const gsl::not_null<Scalar<DataVector>*> out_var1,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          out_var2,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          out_var3,
      const gsl::not_null<
          tnsr::IJ<DataVector, System::volume_dim, Frame::Inertial>*>
          flux_var2,
      const gsl::not_null<Scalar<DataVector>*> out_var3_squared,
      const gsl::not_null<
          tnsr::II<DataVector, System::volume_dim, Frame::Inertial>*>
          inv_spatial_metric,

      const std::optional<tnsr::I<DataVector, System::volume_dim,
                                  Frame::Inertial>>& face_mesh_velocity,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_vector,
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& var3_squared, const Scalar<DataVector>& dt_var1,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>& d_var1,
      const double volume_number) const {
    ghost_.check_normal_vector_set_inverse_spatial_metric(
        inv_spatial_metric, outward_directed_normal_covector,
        outward_directed_normal_vector);
    return dg_ghost(out_var1, out_var2, out_var3, flux_var2, out_var3_squared,
                    face_mesh_velocity, outward_directed_normal_covector, var1,
                    var2, var3_squared, dt_var1, d_var1, volume_number);
  }

  // Mixed conservative non-conservative system, prims, flat background
  std::optional<std::string> dg_ghost(
      const gsl::not_null<Scalar<DataVector>*> out_var1,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          out_var2,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          out_var3,
      const gsl::not_null<
          tnsr::IJ<DataVector, System::volume_dim, Frame::Inertial>*>
          flux_var2,
      const gsl::not_null<Scalar<DataVector>*> out_var3_squared,
      const gsl::not_null<Scalar<DataVector>*> out_prim_var1,
      const std::optional<tnsr::I<DataVector, System::volume_dim,
                                  Frame::Inertial>>& face_mesh_velocity,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& prim_var1,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>& prim_var2,
      const Scalar<DataVector>& var3_squared, const Scalar<DataVector>& dt_var1,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>& d_var1,
      const double volume_number) const {
    ghost_.dg_ghost(out_var1, out_var2, out_var3, flux_var2, out_var3_squared,
                    out_prim_var1, face_mesh_velocity,
                    outward_directed_normal_covector, var1, var2, prim_var1,
                    prim_var2, var3_squared, dt_var1, volume_number);
    const size_t num_pts = get(var1).size();
    for (size_t i = 0; i < System::volume_dim; ++i) {
      CHECK_ITERABLE_APPROX(d_var1.get(i),
                            DataVector(num_pts, offset_partial_derivs + i));
    }
    return std::nullopt;
  }

  // Mixed conservative non-conservative system, prims, curved background
  std::optional<std::string> dg_ghost(
      const gsl::not_null<Scalar<DataVector>*> out_var1,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          out_var2,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          out_var3,
      const gsl::not_null<
          tnsr::IJ<DataVector, System::volume_dim, Frame::Inertial>*>
          flux_var2,
      const gsl::not_null<Scalar<DataVector>*> out_var3_squared,
      const gsl::not_null<Scalar<DataVector>*> out_prim_var1,
      const gsl::not_null<
          tnsr::II<DataVector, System::volume_dim, Frame::Inertial>*>
          inv_spatial_metric,

      const std::optional<tnsr::I<DataVector, System::volume_dim,
                                  Frame::Inertial>>& face_mesh_velocity,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_vector,
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& prim_var1,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>& prim_var2,
      const Scalar<DataVector>& var3_squared, const Scalar<DataVector>& dt_var1,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>& d_var1,
      const double volume_number) const {
    ghost_.check_normal_vector_set_inverse_spatial_metric(
        inv_spatial_metric, outward_directed_normal_covector,
        outward_directed_normal_vector);
    return dg_ghost(out_var1, out_var2, out_var3, flux_var2, out_var3_squared,
                    out_prim_var1, face_mesh_velocity,
                    outward_directed_normal_covector, var1, var2, prim_var1,
                    prim_var2, var3_squared, dt_var1, d_var1, volume_number);
  }

  // Conservative system, no prims
  std::optional<std::string> dg_ghost(
      const gsl::not_null<Scalar<DataVector>*> out_var1,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          out_var2,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          out_var3,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          flux_var1,
      const gsl::not_null<
          tnsr::IJ<DataVector, System::volume_dim, Frame::Inertial>*>
          flux_var2,
      const gsl::not_null<Scalar<DataVector>*> out_var3_squared,
      const std::optional<tnsr::I<DataVector, System::volume_dim,
                                  Frame::Inertial>>& face_mesh_velocity,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& var3_squared, const Scalar<DataVector>& dt_var1,
      const double volume_number) const {
    ghost_.dg_ghost(out_var1, out_var2, out_var3, flux_var1, flux_var2,
                    out_var3_squared, face_mesh_velocity,
                    outward_directed_normal_covector, var1, var2, var3_squared,
                    dt_var1, volume_number);
    return std::nullopt;
  }

  // Conservative system, no prims, curved background
  std::optional<std::string> dg_ghost(
      const gsl::not_null<Scalar<DataVector>*> out_var1,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          out_var2,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          out_var3,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          flux_var1,
      const gsl::not_null<
          tnsr::IJ<DataVector, System::volume_dim, Frame::Inertial>*>
          flux_var2,
      const gsl::not_null<Scalar<DataVector>*> out_var3_squared,
      const gsl::not_null<
          tnsr::II<DataVector, System::volume_dim, Frame::Inertial>*>
          inv_spatial_metric,
      const std::optional<tnsr::I<DataVector, System::volume_dim,
                                  Frame::Inertial>>& face_mesh_velocity,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_vector,
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& var3_squared, const Scalar<DataVector>& dt_var1,
      const double volume_number) const {
    ghost_.check_normal_vector_set_inverse_spatial_metric(
        inv_spatial_metric, outward_directed_normal_covector,
        outward_directed_normal_vector);
    dg_ghost(out_var1, out_var2, out_var3, flux_var1, flux_var2,
             out_var3_squared, face_mesh_velocity,
             outward_directed_normal_covector, var1, var2, var3_squared,
             dt_var1, volume_number);
    return std::nullopt;
  }

  // Conservative system, with prims
  std::optional<std::string> dg_ghost(
      const gsl::not_null<Scalar<DataVector>*> out_var1,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          out_var2,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          out_var3,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          flux_var1,
      const gsl::not_null<
          tnsr::IJ<DataVector, System::volume_dim, Frame::Inertial>*>
          flux_var2,
      const gsl::not_null<Scalar<DataVector>*> out_var3_squared,
      const gsl::not_null<Scalar<DataVector>*> out_prim_var1,

      const std::optional<tnsr::I<DataVector, System::volume_dim,
                                  Frame::Inertial>>& face_mesh_velocity,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& prim_var1,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>& prim_var2,
      const Scalar<DataVector>& var3_squared, const Scalar<DataVector>& dt_var1,
      const double volume_number) const {
    ghost_.dg_ghost(out_var1, out_var2, out_var3, flux_var1, flux_var2,
                    out_var3_squared, out_prim_var1, face_mesh_velocity,
                    outward_directed_normal_covector, var1, var2, prim_var1,
                    prim_var2, var3_squared, dt_var1, volume_number);
    return std::nullopt;
  }

  // Conservative system, with prims, curved background
  std::optional<std::string> dg_ghost(
      const gsl::not_null<Scalar<DataVector>*> out_var1,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          out_var2,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          out_var3,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          flux_var1,
      const gsl::not_null<
          tnsr::IJ<DataVector, System::volume_dim, Frame::Inertial>*>
          flux_var2,
      const gsl::not_null<Scalar<DataVector>*> out_var3_squared,
      const gsl::not_null<Scalar<DataVector>*> out_prim_var1,
      const gsl::not_null<
          tnsr::II<DataVector, System::volume_dim, Frame::Inertial>*>
          inv_spatial_metric,

      const std::optional<tnsr::I<DataVector, System::volume_dim,
                                  Frame::Inertial>>& face_mesh_velocity,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_vector,
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& prim_var1,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>& prim_var2,
      const Scalar<DataVector>& var3_squared, const Scalar<DataVector>& dt_var1,
      const double volume_number) const {
    ghost_.check_normal_vector_set_inverse_spatial_metric(
        inv_spatial_metric, outward_directed_normal_covector,
        outward_directed_normal_vector);
    ghost_.dg_ghost(out_var1, out_var2, out_var3, flux_var1, flux_var2,
                    out_var3_squared, out_prim_var1, face_mesh_velocity,
                    outward_directed_normal_covector, var1, var2, prim_var1,
                    prim_var2, var3_squared, dt_var1, volume_number);
    return std::nullopt;
  }

  template <typename... Args>
  std::optional<std::string> dg_time_derivative(Args&&... args) const {
    return time_derivative_.dg_time_derivative(std::forward<Args>(args)...);
  }

 private:
  Ghost<System> ghost_;
  TimeDerivative<System> time_derivative_;
};

template <typename System>
// NOLINTNEXTLINE
PUP::able::PUP_ID GhostAndTimeDerivative<System>::my_PUP_ID = 0;

template <bool AddTypeAlias, size_t Dim>
struct InverseSpatialMetricTagImpl {
  using inverse_spatial_metric_tag = Tags::InverseSpatialMetric<Dim>;
};

template <size_t Dim>
struct InverseSpatialMetricTagImpl<false, Dim> {};

template <size_t Dim, SystemType SysType, bool HasPrimitiveVariables,
          bool HasInverseSpatialMetric>
struct System
    : public InverseSpatialMetricTagImpl<HasInverseSpatialMetric, Dim> {
  static constexpr SystemType system_type = SysType;
  static constexpr bool has_primitive_and_conservative_vars =
      HasPrimitiveVariables;
  static constexpr size_t volume_dim = Dim;
  static constexpr bool has_inverse_spatial_metric = HasInverseSpatialMetric;

  using boundary_conditions_base = BoundaryCondition<System>;

  using variables_tag =
      ::Tags::Variables<tmpl::list<Tags::Var1, Tags::Var2<Dim>>>;
  using auxiliary_variables = tmpl::list<Tags::Var3<Dim>>;
  using flux_variables = tmpl::conditional_t<
      system_type == SystemType::Conservative,
      tmpl::list<Tags::Var1, Tags::Var2<Dim>>,
      tmpl::conditional_t<system_type == SystemType::Nonconservative,
                          tmpl::list<>, tmpl::list<Tags::Var2<Dim>>>>;
  using gradient_variables = tmpl::conditional_t<
      system_type == SystemType::Conservative, tmpl::list<>,
      tmpl::conditional_t<system_type == SystemType::Nonconservative,
                          tmpl::list<Tags::Var1, Tags::Var2<Dim>>,
                          tmpl::list<Tags::Var1>>>;
  using primitive_variables_tag =
      ::Tags::Variables<tmpl::list<Tags::PrimVar1, Tags::PrimVar2<Dim>>>;

  struct compute_volume_time_derivative_terms {
    using temporary_tags = tmpl::append<
        tmpl::list<Tags::Var3Squared>,
        tmpl::conditional_t<HasInverseSpatialMetric,
                            tmpl::list<Tags::InverseSpatialMetric<Dim>>,
                            tmpl::list<>>>;
  };
};

// Note: DemandOutgoingCharSpeeds is intentionally first so it gets applied
// first. This makes the test easier because the other BCs modify the time
// derivatives.
template <typename System>
using standard_boundary_conditions =
    tmpl::list<DemandOutgoingCharSpeeds<System>, Ghost<System>,
               TimeDerivative<System>, GhostAndTimeDerivative<System>,
               domain::BoundaryConditions::Periodic<BoundaryCondition<System>>,
               domain::BoundaryConditions::None<BoundaryCondition<System>>>;

template <typename System>
using boundary_conditions_with_cartoon =
    tmpl::list<DemandOutgoingCharSpeeds<System>, Ghost<System>,
               TimeDerivative<System>, GhostAndTimeDerivative<System>,
               domain::BoundaryConditions::Periodic<BoundaryCondition<System>>,
               domain::BoundaryConditions::None<BoundaryCondition<System>>,
               domain::BoundaryConditions::Cartoon<BoundaryCondition<System>>>;

template <typename System>
struct Metavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<BoundaryCondition<System>,
                             standard_boundary_conditions<System>>>;
  };
};

template <typename System>
struct MetavariablesWithCartoon {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<BoundaryCondition<System>,
                             boundary_conditions_with_cartoon<System>>>;
  };
};

template <typename TagsList>
void fill_variables(const gsl::not_null<Variables<TagsList>*> variables,
                    const double offset) {
  double count = offset;
  tmpl::for_each<TagsList>([&count, &variables](auto tag_v) {
    using tag = tmpl::type_from<decltype(tag_v)>;
    auto& tensor = get<tag>(*variables);
    for (auto& tensor_component : tensor) {
      tensor_component = count;
      count += 1.0;
    }
  });
}

// Lifts a face-sized boundary correction into a zero-initialized volume-sized
// Variables, mirroring the lift in `apply_boundary_condition_on_face`
// (GaussLobatto: lift_flux + add_slice_to_data; Gauss:
// lift_boundary_terms_gauss_points), so a ghost correction's expected volume
// effect can be computed at the test site.
template <size_t Dim, typename TagsList>
Variables<TagsList> lifted_boundary_correction(
    Variables<TagsList> correction_on_face, const Mesh<Dim>& mesh,
    const Direction<Dim>& direction,
    const Scalar<DataVector>& volume_det_inv_jacobian,
    const Scalar<DataVector>& magnitude_of_face_normal) {
  Variables<TagsList> volume_correction{mesh.number_of_grid_points(), 0.0};
  if (mesh.quadrature(direction.dimension()) == Spectral::Quadrature::Gauss) {
    Scalar<DataVector> face_det_inv_jacobian{
        mesh.slice_away(direction.dimension()).number_of_grid_points()};
    const Matrix identity{};
    auto interpolation_matrices = make_array<Dim>(std::cref(identity));
    const std::pair<Matrix, Matrix>& matrices =
        Spectral::boundary_interpolation_matrices(
            mesh.slice_through(direction.dimension()));
    gsl::at(interpolation_matrices, direction.dimension()) =
        direction.side() == Side::Upper ? matrices.second : matrices.first;
    apply_matrices(make_not_null(&get(face_det_inv_jacobian)),
                   interpolation_matrices, get(volume_det_inv_jacobian),
                   mesh.extents());
    const Scalar<DataVector> face_det_jacobian{1.0 /
                                               get(face_det_inv_jacobian)};
    ::dg::lift_boundary_terms_gauss_points(
        make_not_null(&volume_correction), volume_det_inv_jacobian, mesh,
        direction, correction_on_face, magnitude_of_face_normal,
        face_det_jacobian);
  } else {
    ::dg::lift_flux(
        make_not_null(&correction_on_face), mesh.extents(direction.dimension()),
        magnitude_of_face_normal, mesh.basis(direction.dimension()));
    add_slice_to_data(make_not_null(&volume_correction), correction_on_face,
                      mesh.extents(), direction.dimension(),
                      index_to_slice_at(mesh.extents(), direction));
  }
  return volume_correction;
}

// Builds the volume DataBox shared by the boundary-condition tests: identity
// coordinate maps, time 1.2, a non-identity (2x) Jacobian (so index bugs
// cannot hide), placeholder normal storage on every external face, and the
// evolved/auxiliary/dt variables seeded from the file-scope offsets. The
// auxiliary variables are projected to the face and passed (unused) to
// `dg_package_data` in the physical pass; in the auxiliary pass they receive
// the lifted auxiliary correction. `ExtraTags` (paired one-to-one with
// `extra_values`) supplies the per-test volume-number and facility-storage
// tags, placed between the dt variables and the formulation.
template <typename Metavars, typename System, typename ExtraTags,
          size_t Dim = System::volume_dim, typename... ExtraValues>
auto make_boundary_conditions_box(
    const Mesh<Dim>& mesh, const Element<Dim>& element,
    std::vector<DirectionMap<
        Dim, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
        external_boundary_conditions,
    std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>> mesh_velocity,
    const dg::Formulation formulation, ExtraValues&&... extra_values) {
  using dt_variables_tag =
      db::add_tag_prefix<::Tags::dt, typename System::variables_tag>;
  ElementMap<Dim, Frame::Grid> element_map{
      element.id(),
      domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(
          domain::CoordinateMaps::Identity<Dim>{})};
  auto grid_to_inertial_map =
      domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
          domain::CoordinateMaps::Identity<Dim>{});
  const double time{1.2};
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};
  ::InverseJacobian<DataVector, Dim, Frame::ElementLogical, Frame::Inertial>
      inv_jacobian{mesh.number_of_grid_points(), 0.0};
  for (size_t i = 0; i < Dim; ++i) {
    inv_jacobian.get(i, i) = 2.0;
  }
  const auto det_inv_jacobian = determinant(inv_jacobian);
  DirectionMap<Dim, std::optional<Variables<
                        tmpl::list<evolution::dg::Tags::MagnitudeOfNormal,
                                   evolution::dg::Tags::NormalCovector<Dim>>>>>
      normal_covector_and_magnitude{};
  for (const auto& direction : element.external_boundaries()) {
    normal_covector_and_magnitude[direction] = std::nullopt;
  }
  Variables<
      db::wrap_tags_in<::Tags::dt, typename System::variables_tag::tags_list>>
      dt_evolved_vars{mesh.number_of_grid_points()};
  fill_variables(make_not_null(&dt_evolved_vars), offset_dt_evolved_vars);
  Variables<typename System::variables_tag::tags_list> evolved_vars{
      mesh.number_of_grid_points()};
  fill_variables(make_not_null(&evolved_vars), offset_evolved_vars);
  Variables<typename System::auxiliary_variables> auxiliary_vars{
      mesh.number_of_grid_points()};
  fill_variables(make_not_null(&auxiliary_vars), offset_evolved_vars);
  Domain<Dim> domain{make_vector(Block<Dim>{
      domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
          domain::CoordinateMaps::Identity<Dim>{}),
      0,
      {}})};
  domain.inject_time_dependent_map_for_block(0,
                                             grid_to_inertial_map->get_clone());

  using simple_tags = tmpl::append<
      tmpl::list<
          Parallel::Tags::MetavariablesImpl<Metavars>,
          domain::Tags::Domain<Dim>,
          domain::Tags::ExternalBoundaryConditions<Dim>,
          domain::Tags::Mesh<Dim>, domain::Tags::Element<Dim>,
          domain::Tags::ElementMap<Dim, Frame::Grid>,
          domain::CoordinateMaps::Tags::CoordinateMap<Dim, Frame::Grid,
                                                      Frame::Inertial>,
          ::Tags::Time, domain::Tags::FunctionsOfTimeInitialize,
          domain::Tags::MeshVelocity<Dim>,
          domain::Tags::InverseJacobian<Dim, Frame::ElementLogical,
                                        Frame::Inertial>,
          domain::Tags::DetInvJacobian<Frame::ElementLogical, Frame::Inertial>,
          evolution::dg::Tags::NormalCovectorAndMagnitude<Dim>,
          typename System::variables_tag,
          ::Tags::Variables<typename System::auxiliary_variables>,
          dt_variables_tag>,
      ExtraTags, tmpl::list<::dg::Tags::Formulation>>;
  using compute_tags = tmpl::list<>;

  return db::create<simple_tags, compute_tags>(
      Metavars{}, std::move(domain), std::move(external_boundary_conditions),
      mesh, element, std::move(element_map), grid_to_inertial_map->get_clone(),
      time, clone_unique_ptrs(functions_of_time), std::move(mesh_velocity),
      inv_jacobian, det_inv_jacobian, std::move(normal_covector_and_magnitude),
      std::move(evolved_vars), std::move(auxiliary_vars),
      std::move(dt_evolved_vars), std::forward<ExtraValues>(extra_values)...,
      formulation);
}

// Note: clang-8 wants us to capture Dim in the closures if we do `constexpr
// size_t Dim =...`, but then GCC-7 fails to build. Assigning `Dim` as a
// template parameter gets around that.
// When `IsAuxiliary` is true this exercises the LDG auxiliary communication
// pass (`ComputeAuxiliary=true`) instead of the physical pass. The two passes
// share almost all of their setup and structure; the auxiliary-specific
// deviations are folded in via `if constexpr (IsAuxiliary)` branches at the
// points where the passes genuinely differ.
template <typename System, bool IsAuxiliary = false,
          size_t Dim = System::volume_dim>
void test_1d(const bool moving_mesh, const dg::Formulation formulation,
             const Spectral::Quadrature quadrature) {
  CAPTURE(moving_mesh);
  CAPTURE(formulation);
  CAPTURE(quadrature);
  CAPTURE(System::has_primitive_and_conservative_vars);
  CAPTURE(System::system_type);
  // gcc-8 complains that there are no definitions for the destructors of None
  // and Periodic. We can generate them via explicit instantiations or by just
  // creating an empty dummy object.
  [[maybe_unused]] const domain::BoundaryConditions::None<
      BoundaryCondition<System>>
      instantiate_none_for_gcc_8{};
  [[maybe_unused]] const domain::BoundaryConditions::Periodic<
      BoundaryCondition<System>>
      instantiate_periodic_for_gcc_8{};
  static_assert(System::volume_dim == 1);

  using dt_variables_tag =
      db::add_tag_prefix<::Tags::dt, typename System::variables_tag>;
  const Mesh<Dim> mesh{5, Spectral::Basis::Legendre, quadrature};
  const ElementId<Dim> self_id{0, {{{1, 0}}}};
  const Element<Dim> element{self_id, {}};
  std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>> mesh_velocity{};
  if (moving_mesh) {
    const std::array<double, 3> velocities = {{1.2, -1.4, 0.3}};
    mesh_velocity =
        tnsr::I<DataVector, Dim, Frame::Inertial>{mesh.number_of_grid_points()};
    for (size_t i = 0; i < Dim; ++i) {
      mesh_velocity->get(i) = gsl::at(velocities, i);
    }
  }
  const double boundary_condition_volume_tag_number{2.5};
  const double boundary_correction_volume_tag_number{3.5};
  const double boundary_correction_auxiliary_volume_tag_number{4.5};

  // Pristine copies of the box's seeded variables (the builder fills the box
  // from the same offsets), used as baselines for the expected-value math
  // below.
  Variables<
      db::wrap_tags_in<::Tags::dt, typename System::variables_tag::tags_list>>
      dt_evolved_vars{mesh.number_of_grid_points()};
  fill_variables(make_not_null(&dt_evolved_vars), offset_dt_evolved_vars);
  Variables<typename System::variables_tag::tags_list> evolved_vars{
      mesh.number_of_grid_points()};
  fill_variables(make_not_null(&evolved_vars), offset_evolved_vars);
  Variables<typename System::auxiliary_variables> auxiliary_vars{
      mesh.number_of_grid_points()};
  fill_variables(make_not_null(&auxiliary_vars), offset_evolved_vars);
  Variables<
      typename System::compute_volume_time_derivative_terms::temporary_tags>
      temporaries{mesh.number_of_grid_points()};
  fill_variables(make_not_null(&temporaries), offset_temporaries);
  Variables<db::wrap_tags_in<::Tags::Flux, typename System::flux_variables,
                             tmpl::size_t<Dim>, Frame::Inertial>>
      volume_fluxes{mesh.number_of_grid_points()};
  fill_variables(make_not_null(&volume_fluxes), offset_volume_fluxes);
  Variables<db::wrap_tags_in<::Tags::deriv, typename System::gradient_variables,
                             tmpl::size_t<Dim>, Frame::Inertial>>
      partial_derivs{mesh.number_of_grid_points()};
  fill_variables(make_not_null(&partial_derivs), offset_partial_derivs);
  Variables<tmpl::conditional_t<
      System::has_primitive_and_conservative_vars,
      typename System::primitive_variables_tag::tags_list, tmpl::list<>>>
      primitive_vars{mesh.number_of_grid_points()};
  fill_variables(make_not_null(&primitive_vars), offset_primitive_vars);
  const Variables<tmpl::conditional_t<
      System::has_primitive_and_conservative_vars,
      typename System::primitive_variables_tag::tags_list, tmpl::list<>>>*
      primitive_vars_ptr =
          System::has_primitive_and_conservative_vars ? &primitive_vars
                                                      : nullptr;
  constexpr bool has_prims = System::has_primitive_and_conservative_vars;
  using BndryTerms = BoundaryTerms<Dim, has_prims, System::system_type,
                                   System::has_inverse_spatial_metric>;

  std::vector<DirectionMap<
      Dim, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
      external_boundary_conditions{1};
  // For the initial tests, set the boundary conditions to:
  // lower_xi: DemandOutgoingCharSpeeds
  // upper_xi: DemandOutgoingCharSpeeds
  external_boundary_conditions[0][Direction<Dim>::lower_xi()] =
      std::make_unique<DemandOutgoingCharSpeeds<System>>(moving_mesh);
  external_boundary_conditions[0][Direction<Dim>::upper_xi()] =
      std::make_unique<DemandOutgoingCharSpeeds<System>>(moving_mesh);

  auto box = make_boundary_conditions_box<
      Metavariables<System>, System,
      tmpl::list<Tags::BoundaryConditionVolumeTag,
                 Tags::BoundaryCorrectionVolumeTag,
                 Tags::BoundaryCorrectionAuxiliaryVolumeTag>>(
      mesh, element, std::move(external_boundary_conditions), mesh_velocity,
      formulation, boundary_condition_volume_tag_number,
      boundary_correction_volume_tag_number,
      boundary_correction_auxiliary_volume_tag_number);

  {
    INFO("DemandOutgoingCharSpeeds only");
    // DemandOutgoingCharSpeeds both sides: neither side uses a ghost
    // condition, so nothing is packaged or lifted in either pass. The evolved
    // variables, the auxiliary variables (Var3), and the time derivatives
    // are all unchanged.
    evolution::dg::Actions::detail::
        apply_boundary_conditions_on_all_external_faces<System, Dim,
                                                        IsAuxiliary>(
            make_not_null(&box), BndryTerms{moving_mesh, 0.0}, temporaries,
            volume_fluxes, partial_derivs, primitive_vars_ptr);
    CHECK_ITERABLE_APPROX(
        get(get<Tags::Var1>(box)),
        DataVector(mesh.number_of_grid_points(), offset_evolved_vars));
    for (size_t i = 0; i < Dim; ++i) {
      CHECK_ITERABLE_APPROX(
          get<Tags::Var2<Dim>>(box).get(i),
          DataVector(mesh.number_of_grid_points(),
                     offset_evolved_vars + 1.0 + static_cast<double>(i)));
    }
    for (size_t i = 0; i < Dim; ++i) {
      CHECK_ITERABLE_APPROX(
          get<Tags::Var3<Dim>>(box).get(i),
          DataVector(mesh.number_of_grid_points(),
                     offset_evolved_vars + static_cast<double>(i)));
    }
    CHECK_ITERABLE_APPROX(
        get(get<::Tags::dt<Tags::Var1>>(box)),
        DataVector(mesh.number_of_grid_points(), offset_dt_evolved_vars));
    for (size_t i = 0; i < Dim; ++i) {
      CHECK_ITERABLE_APPROX(get<::Tags::dt<Tags::Var2<Dim>>>(box).get(i),
                            DataVector(mesh.number_of_grid_points(),
                                       offset_dt_evolved_vars + 1 + i));
    }
  }

  const auto expected_ghost_dt_correction = [&box, &formulation, &mesh](
                                                const auto& ghost_direction) {
    Variables<tmpl::list<::Tags::dt<Tags::Var1>, ::Tags::dt<Tags::Var2<Dim>>>>
        expected_on_boundary{mesh.slice_away(ghost_direction.dimension())
                                 .number_of_grid_points()};
    get(get<::Tags::dt<Tags::Var1>>(expected_on_boundary)) =
        offset_boundary_correction *
        (formulation == dg::Formulation::WeakInertial ? 2.0 : 1.0);
    for (size_t i = 0; i < Dim; ++i) {
      get<::Tags::dt<Tags::Var2<Dim>>>(expected_on_boundary).get(i) =
          offset_boundary_correction + 1.0 + i;
    }
    return lifted_boundary_correction(
        std::move(expected_on_boundary), mesh, ghost_direction,
        db::get<domain::Tags::DetInvJacobian<Frame::ElementLogical,
                                             Frame::Inertial>>(box),
        get<evolution::dg::Tags::MagnitudeOfNormal>(
            *db::get<evolution::dg::Tags::NormalCovectorAndMagnitude<Dim>>(box)
                 .at(ghost_direction)));
  };

  // Auxiliary-pass analogue of `expected_ghost_dt_correction`. Expected lifted
  // auxiliary (Var3) correction added to the
  // ::Tags::Variables<auxiliary_variables> container by the auxiliary pass on a
  // ghost face. The correction on the face is the mock's
  // `dg_auxiliary_boundary_terms` output, `0.5 * (int_var2 + ext_var2)`. From
  // the packaged data, `int_var2.get(i) == offset_evolved_vars + 1 + i`
  // (projected volume Var2) and `ext_var2.get(i) == offset_boundary_condition +
  // 1 + i` (the Var2 the Ghost boundary condition supplies), so the face value
  // is
  //   Var3.get(i) = 0.5 * ((offset_evolved_vars + 1 + i) +
  //                        (offset_boundary_condition + 1 + i)).
  // The mock's toy auxiliary correction ignores its `formulation` argument, so
  // this expected value is formulation-independent. This is not a test gap
  // as we currently intend to support only the strong formulation with LDG.
  const auto expected_ghost_var3_correction = [&box, &mesh](
                                                  const auto& ghost_direction) {
    Variables<tmpl::list<Tags::Var3<Dim>>> expected_on_boundary{
        mesh.slice_away(ghost_direction.dimension()).number_of_grid_points()};
    for (size_t i = 0; i < Dim; ++i) {
      get<Tags::Var3<Dim>>(expected_on_boundary).get(i) =
          0.5 * ((offset_evolved_vars + 1.0 + i) +
                 (offset_boundary_condition + 1.0 + i));
    }
    return lifted_boundary_correction(
        std::move(expected_on_boundary), mesh, ghost_direction,
        db::get<domain::Tags::DetInvJacobian<Frame::ElementLogical,
                                             Frame::Inertial>>(box),
        get<evolution::dg::Tags::MagnitudeOfNormal>(
            *db::get<evolution::dg::Tags::NormalCovectorAndMagnitude<Dim>>(box)
                 .at(ghost_direction)));
  };

  const auto check_outgoing_and_ghost = [&](const Direction<Dim>&
                                                outgoing_direction) {
    INFO("Ghost");
    CAPTURE(outgoing_direction);
    // Reset the evolved, auxiliary, and time-derivative variables to their
    // baseline values.
    db::mutate<domain::Tags::ExternalBoundaryConditions<Dim>,
               typename System::variables_tag,
               ::Tags::Variables<typename System::auxiliary_variables>,
               dt_variables_tag>(
        [&moving_mesh, &outgoing_direction](
            const auto all_boundary_conditions, const auto vars_ptr,
            const auto aux_vars_ptr, const auto dt_vars_ptr) {
          DirectionMap<Dim, std::unique_ptr<
                                domain::BoundaryConditions::BoundaryCondition>>
              boundary_conditions{};
          boundary_conditions[outgoing_direction.opposite()] =
              std::make_unique<Ghost<System>>(moving_mesh);
          boundary_conditions[outgoing_direction] =
              std::make_unique<DemandOutgoingCharSpeeds<System>>(moving_mesh);
          (*all_boundary_conditions)[0] = std::move(boundary_conditions);

          fill_variables(vars_ptr, offset_evolved_vars);
          fill_variables(aux_vars_ptr, offset_evolved_vars);
          fill_variables(dt_vars_ptr, offset_dt_evolved_vars);
        },
        make_not_null(&box));
    evolution::dg::Actions::detail::
        apply_boundary_conditions_on_all_external_faces<System, Dim,
                                                        IsAuxiliary>(
            make_not_null(&box),
            BndryTerms{moving_mesh, outgoing_direction.opposite().sign()},
            temporaries, volume_fluxes, partial_derivs, primitive_vars_ptr);

    // The physical pass lifts the ghost correction into the time derivatives;
    // the auxiliary pass leaves them untouched.
    auto expected_dt_evolved_vars = dt_evolved_vars;
    if constexpr (not IsAuxiliary) {
      expected_dt_evolved_vars +=
          expected_ghost_dt_correction(outgoing_direction.opposite());
    }
    CHECK_ITERABLE_APPROX(
        get(get<::Tags::dt<Tags::Var1>>(box)),
        get(get<::Tags::dt<Tags::Var1>>(expected_dt_evolved_vars)));
    for (size_t i = 0; i < Dim; ++i) {
      CHECK_ITERABLE_APPROX(
          get<::Tags::dt<Tags::Var2<Dim>>>(box).get(i),
          get<::Tags::dt<Tags::Var2<Dim>>>(expected_dt_evolved_vars).get(i));
    }
    // The auxiliary pass lifts the ghost correction into the auxiliary
    // variables (Var3); the physical pass leaves them untouched.
    auto expected_auxiliary_vars = auxiliary_vars;
    if constexpr (IsAuxiliary) {
      expected_auxiliary_vars +=
          expected_ghost_var3_correction(outgoing_direction.opposite());
    }
    for (size_t i = 0; i < Dim; ++i) {
      CHECK_ITERABLE_APPROX(
          get<Tags::Var3<Dim>>(box).get(i),
          get<Tags::Var3<Dim>>(expected_auxiliary_vars).get(i));
    }
    // The evolved variables are untouched by both passes.
    CHECK_ITERABLE_APPROX(get(get<Tags::Var1>(box)),
                          get(get<Tags::Var1>(evolved_vars)));
    for (size_t i = 0; i < Dim; ++i) {
      CHECK_ITERABLE_APPROX(get<Tags::Var2<Dim>>(box).get(i),
                            get<Tags::Var2<Dim>>(evolved_vars).get(i));
    }
  };
  // DemandOutgoingCharSpeeds +xi, Ghost -xi
  check_outgoing_and_ghost(Direction<Dim>::upper_xi());
  // Ghost +xi, DemandOutgoingCharSpeeds -xi
  check_outgoing_and_ghost(Direction<Dim>::lower_xi());

  const auto expected_time_derivative_dt_correction = [&mesh](
                                                          const auto&
                                                              dt_direction) {
    Variables<tmpl::list<::Tags::dt<Tags::Var1>, ::Tags::dt<Tags::Var2<Dim>>>>
        expected_dt_volume_correction{mesh.number_of_grid_points(), 0.0};
    const Mesh<Dim> mesh_gl{5, Spectral::Basis::Legendre,
                            Spectral::Quadrature::GaussLobatto};
    Variables<
        db::wrap_tags_in<::Tags::dt, typename System::variables_tag::tags_list>>
        dt_correction_gl{mesh_gl.number_of_grid_points(), 0.0};
    const size_t boundary_index =
        dt_direction.side() == Side::Lower
            ? 0
            : mesh_gl.extents(dt_direction.dimension()) - 1;
    get(get<::Tags::dt<Tags::Var1>>(dt_correction_gl))[boundary_index] +=
        offset_boundary_condition;
    for (size_t i = 0; i < Dim; ++i) {
      get<::Tags::dt<Tags::Var2<Dim>>>(dt_correction_gl)
          .get(i)[boundary_index] += offset_boundary_condition + 1.0 + i;
    }
    if (mesh.quadrature(dt_direction.dimension()) ==
        Spectral::Quadrature::GaussLobatto) {
      expected_dt_volume_correction += dt_correction_gl;
    } else {
      // Interpolate to Gauss mesh
      expected_dt_volume_correction +=
          intrp::RegularGrid<Dim>{mesh_gl, mesh}.interpolate(dt_correction_gl);
    }
    return expected_dt_volume_correction;
  };

  // TimeDerivative on one side, DemandOutgoingCharSpeeds on the other.
  // Neither boundary condition `uses_ghost_condition`, so no correction is
  // packaged or lifted in either pass; the physical pass adds only the
  // dg_time_derivative correction to the time derivatives, while the auxiliary
  // pass skips that path and is a complete no-op.
  const auto check_outgoing_and_dt = [&](const Direction<Dim>&
                                             outgoing_direction) {
    INFO("TimeDerivative");
    CAPTURE(outgoing_direction);
    // Reset the evolved, auxiliary, and time-derivative variables to their
    // baseline values.
    db::mutate<domain::Tags::ExternalBoundaryConditions<Dim>,
               typename System::variables_tag,
               ::Tags::Variables<typename System::auxiliary_variables>,
               dt_variables_tag>(
        [&moving_mesh, &outgoing_direction](
            const auto all_boundary_conditions, const auto vars_ptr,
            const auto aux_vars_ptr, const auto dt_vars_ptr) {
          DirectionMap<Dim, std::unique_ptr<
                                domain::BoundaryConditions::BoundaryCondition>>
              boundary_conditions{};
          boundary_conditions[outgoing_direction.opposite()] =
              std::make_unique<TimeDerivative<System>>(moving_mesh,
                                                       offset_dt_evolved_vars);
          boundary_conditions[outgoing_direction] =
              std::make_unique<DemandOutgoingCharSpeeds<System>>(moving_mesh);
          (*all_boundary_conditions)[0] = std::move(boundary_conditions);

          fill_variables(vars_ptr, offset_evolved_vars);
          fill_variables(aux_vars_ptr, offset_evolved_vars);
          fill_variables(dt_vars_ptr, offset_dt_evolved_vars);
        },
        make_not_null(&box));
    evolution::dg::Actions::detail::
        apply_boundary_conditions_on_all_external_faces<System, Dim,
                                                        IsAuxiliary>(
            make_not_null(&box),
            BndryTerms{moving_mesh, outgoing_direction.opposite().sign()},
            temporaries, volume_fluxes, partial_derivs, primitive_vars_ptr);

    auto expected_dt_evolved_vars = dt_evolved_vars;
    if constexpr (not IsAuxiliary) {
      expected_dt_evolved_vars +=
          expected_time_derivative_dt_correction(outgoing_direction.opposite());
    }
    CHECK_ITERABLE_APPROX(
        get(get<::Tags::dt<Tags::Var1>>(box)),
        get(get<::Tags::dt<Tags::Var1>>(expected_dt_evolved_vars)));
    for (size_t i = 0; i < Dim; ++i) {
      CHECK_ITERABLE_APPROX(
          get<::Tags::dt<Tags::Var2<Dim>>>(box).get(i),
          get<::Tags::dt<Tags::Var2<Dim>>>(expected_dt_evolved_vars).get(i));
    }
    // No ghost condition on either side: the auxiliary variables (Var3) are
    // untouched in both passes.
    for (size_t i = 0; i < Dim; ++i) {
      CHECK_ITERABLE_APPROX(get<Tags::Var3<Dim>>(box).get(i),
                            get<Tags::Var3<Dim>>(auxiliary_vars).get(i));
    }
    // The evolved variables are untouched by both passes.
    CHECK_ITERABLE_APPROX(get(get<Tags::Var1>(box)),
                          get(get<Tags::Var1>(evolved_vars)));
    for (size_t i = 0; i < Dim; ++i) {
      CHECK_ITERABLE_APPROX(get<Tags::Var2<Dim>>(box).get(i),
                            get<Tags::Var2<Dim>>(evolved_vars).get(i));
    }
  };
  // DemandOutgoingCharSpeeds +xi, TimeDerivative -xi
  check_outgoing_and_dt(Direction<Dim>::upper_xi());
  // DemandOutgoingCharSpeeds -xi, TimeDerivative +xi
  check_outgoing_and_dt(Direction<Dim>::lower_xi());

  // Ghost on one side, TimeDerivative on the opposite side. The ghost side
  // lifts its correction into the time derivatives (physical pass) or into the
  // auxiliary variables (auxiliary pass); the TimeDerivative side adds its
  // dg_time_derivative correction in the physical pass only.
  const auto check_ghost_and_dt_opposite = [&](const Direction<Dim>&
                                                   ghost_direction) {
    INFO("Ghost and TimeDerivative on opposite sides");
    CAPTURE(ghost_direction);
    auto expected_dt_evolved_vars = dt_evolved_vars;
    if constexpr (not IsAuxiliary) {
      expected_dt_evolved_vars += expected_ghost_dt_correction(ghost_direction);
    }

    // Project to the boundary to figure out what will be the projected
    // dt_var1 passed into the time derivative boundary condition. This is
    // necessary because we apply _and lift_ the Ghost boundary correction
    // before the TimeDerivative correction. This order is determined in the
    // BoundaryCondition base class's `creatable_classes` typelist. In the
    // auxiliary pass no dt correction is lifted.
    Variables<tmpl::list<::Tags::dt<Tags::Var1>, ::Tags::dt<Tags::Var2<Dim>>>>
        expected_dt_on_boundary{mesh.slice_away(ghost_direction.dimension())
                                    .number_of_grid_points()};
    ::dg::project_contiguous_data_to_boundary(
        make_not_null(&expected_dt_on_boundary), expected_dt_evolved_vars, mesh,
        ghost_direction.opposite());

    // Reset the evolved, auxiliary, and time-derivative variables to their
    // baseline values.
    db::mutate<domain::Tags::ExternalBoundaryConditions<Dim>,
               typename System::variables_tag,
               ::Tags::Variables<typename System::auxiliary_variables>,
               dt_variables_tag>(
        [&expected_dt_var1 =
             get<::Tags::dt<Tags::Var1>>(expected_dt_on_boundary),
         &moving_mesh, &ghost_direction](
            const auto all_boundary_conditions, const auto vars_ptr,
            const auto aux_vars_ptr, const auto dt_vars_ptr) {
          DirectionMap<Dim, std::unique_ptr<
                                domain::BoundaryConditions::BoundaryCondition>>
              boundary_conditions{};
          boundary_conditions[ghost_direction.opposite()] =
              std::make_unique<TimeDerivative<System>>(
                  moving_mesh, get(expected_dt_var1)[0]);
          boundary_conditions[ghost_direction] =
              std::make_unique<Ghost<System>>(moving_mesh);
          (*all_boundary_conditions)[0] = std::move(boundary_conditions);

          fill_variables(vars_ptr, offset_evolved_vars);
          fill_variables(aux_vars_ptr, offset_evolved_vars);
          fill_variables(dt_vars_ptr, offset_dt_evolved_vars);
        },
        make_not_null(&box));
    evolution::dg::Actions::detail::
        apply_boundary_conditions_on_all_external_faces<System, Dim,
                                                        IsAuxiliary>(
            make_not_null(&box),
            BndryTerms{moving_mesh, ghost_direction.sign()}, temporaries,
            volume_fluxes, partial_derivs, primitive_vars_ptr);

    if constexpr (not IsAuxiliary) {
      expected_dt_evolved_vars +=
          expected_time_derivative_dt_correction(ghost_direction.opposite());
    }
    CHECK_ITERABLE_APPROX(
        get(get<::Tags::dt<Tags::Var1>>(box)),
        get(get<::Tags::dt<Tags::Var1>>(expected_dt_evolved_vars)));
    for (size_t i = 0; i < Dim; ++i) {
      CHECK_ITERABLE_APPROX(
          get<::Tags::dt<Tags::Var2<Dim>>>(box).get(i),
          get<::Tags::dt<Tags::Var2<Dim>>>(expected_dt_evolved_vars).get(i));
    }
    auto expected_auxiliary_vars = auxiliary_vars;
    if constexpr (IsAuxiliary) {
      expected_auxiliary_vars +=
          expected_ghost_var3_correction(ghost_direction);
    }
    for (size_t i = 0; i < Dim; ++i) {
      CHECK_ITERABLE_APPROX(
          get<Tags::Var3<Dim>>(box).get(i),
          get<Tags::Var3<Dim>>(expected_auxiliary_vars).get(i));
    }
    // The evolved variables are untouched by both passes.
    CHECK_ITERABLE_APPROX(get(get<Tags::Var1>(box)),
                          get(get<Tags::Var1>(evolved_vars)));
    for (size_t i = 0; i < Dim; ++i) {
      CHECK_ITERABLE_APPROX(get<Tags::Var2<Dim>>(box).get(i),
                            get<Tags::Var2<Dim>>(evolved_vars).get(i));
    }
  };
  // Ghost +xi, TimeDerivative -xi
  check_ghost_and_dt_opposite(Direction<Dim>::upper_xi());
  // Ghost -xi, TimeDerivative +xi
  check_ghost_and_dt_opposite(Direction<Dim>::lower_xi());

  // GhostAndTimeDerivative supplies BOTH ghost data and a dg_time_derivative
  // condition on one face. The physical pass lifts the ghost correction AND
  // adds the dt correction to the time derivatives; the auxiliary pass lifts
  // only the ghost part's auxiliary correction into the auxiliary variables
  // (Var3) while the dg_time_derivative path is skipped.
  const auto check_ghost_and_dt_combined_bc = [&](const Direction<Dim>&
                                                      outgoing_direction) {
    INFO("GhostAndTimeDerivative combined on one side");
    CAPTURE(outgoing_direction);
    // Since the Ghost and TimeDerivative are applied in the same direction
    // they both receive the dt_vars _without_ either boundary condition
    // applied, which is different from way Ghost and TimeDerivative are
    // applied in different directions.
    // Reset the evolved, auxiliary, and time-derivative variables to their
    // baseline values.
    db::mutate<domain::Tags::ExternalBoundaryConditions<Dim>,
               typename System::variables_tag,
               ::Tags::Variables<typename System::auxiliary_variables>,
               dt_variables_tag>(
        [&moving_mesh, &outgoing_direction](
            const auto all_boundary_conditions, const auto vars_ptr,
            const auto aux_vars_ptr, const auto dt_vars_ptr) {
          DirectionMap<Dim, std::unique_ptr<
                                domain::BoundaryConditions::BoundaryCondition>>
              boundary_conditions{};
          boundary_conditions[outgoing_direction.opposite()] =
              std::make_unique<GhostAndTimeDerivative<System>>(moving_mesh);
          boundary_conditions[outgoing_direction] =
              std::make_unique<DemandOutgoingCharSpeeds<System>>(moving_mesh);
          (*all_boundary_conditions)[0] = std::move(boundary_conditions);

          fill_variables(vars_ptr, offset_evolved_vars);
          fill_variables(aux_vars_ptr, offset_evolved_vars);
          fill_variables(dt_vars_ptr, offset_dt_evolved_vars);
        },
        make_not_null(&box));
    evolution::dg::Actions::detail::
        apply_boundary_conditions_on_all_external_faces<System, Dim,
                                                        IsAuxiliary>(
            make_not_null(&box),
            BndryTerms{moving_mesh, outgoing_direction.opposite().sign()},
            temporaries, volume_fluxes, partial_derivs, primitive_vars_ptr);

    auto expected_dt_evolved_vars = dt_evolved_vars;
    if constexpr (not IsAuxiliary) {
      expected_dt_evolved_vars +=
          expected_time_derivative_dt_correction(outgoing_direction.opposite());
      expected_dt_evolved_vars +=
          expected_ghost_dt_correction(outgoing_direction.opposite());
    }
    CHECK_ITERABLE_APPROX(
        get(get<::Tags::dt<Tags::Var1>>(box)),
        get(get<::Tags::dt<Tags::Var1>>(expected_dt_evolved_vars)));
    for (size_t i = 0; i < Dim; ++i) {
      CHECK_ITERABLE_APPROX(
          get<::Tags::dt<Tags::Var2<Dim>>>(box).get(i),
          get<::Tags::dt<Tags::Var2<Dim>>>(expected_dt_evolved_vars).get(i));
    }
    auto expected_auxiliary_vars = auxiliary_vars;
    if constexpr (IsAuxiliary) {
      expected_auxiliary_vars +=
          expected_ghost_var3_correction(outgoing_direction.opposite());
    }
    for (size_t i = 0; i < Dim; ++i) {
      CHECK_ITERABLE_APPROX(
          get<Tags::Var3<Dim>>(box).get(i),
          get<Tags::Var3<Dim>>(expected_auxiliary_vars).get(i));
    }
    // The evolved variables are untouched by both passes.
    CHECK_ITERABLE_APPROX(get(get<Tags::Var1>(box)),
                          get(get<Tags::Var1>(evolved_vars)));
    for (size_t i = 0; i < Dim; ++i) {
      CHECK_ITERABLE_APPROX(get<Tags::Var2<Dim>>(box).get(i),
                            get<Tags::Var2<Dim>>(evolved_vars).get(i));
    }
  };
  // DemandOutgoingCharSpeeds +xi, GhostAndTimeDerivative -xi
  check_ghost_and_dt_combined_bc(Direction<Dim>::upper_xi());
  // DemandOutgoingCharSpeeds -xi, GhostAndTimeDerivative +xi
  check_ghost_and_dt_combined_bc(Direction<Dim>::lower_xi());
}

// ============================================================================
// Boundary-evolved-fields facility plumbing
//
// Tests the two facility behaviors added to `BoundaryConditionsImpl.hpp`,
// reusing the standard mocks above: `OptingGhost` is the stock `Ghost`
// extended with the facility opt-in, `DemandOutgoingCharSpeeds` provides the
// non-opting face, and `BoundaryTerms` is the boundary correction.
// (1) On the physical pass the opting boundary condition's
// `boundary_field_time_derivatives` output is written into the per-face
// dt-stash -- verified exactly at the test site as a function of the stored
// boundary value and the projected interior fields -- and the auxiliary pass
// leaves the stash untouched. (2) The stored per-face boundary value is
// spliced into `dg_ghost` as an extra argument in both passes --
// `OptingGhost` checks the delivered value the way the stock mocks check
// their inputs, and the test site verifies the stock lifted corrections,
// which proves the `dg_ghost` path ran (so the input check cannot pass
// vacuously).
// ============================================================================
using BoundaryVar1 = evolution::dg::Tags::BoundaryValue<Tags::Var1>;

// None of the pre-existing boundary conditions in this file opts into the
// boundary-evolved-fields facility, so their field-tag union is empty: every
// other test in this file exercises the boundary-condition framework with the
// facility compiled away entirely, on DataBoxes that carry no facility tag.
// That is the strongest form of the non-opting no-op contract (the spliced
// `dg_ghost` call is identical to the pre-facility one).
static_assert(std::is_same_v<
              evolution::dg::BoundaryEvolvedFields::boundary_evolved_field_tags<
                  standard_boundary_conditions<
                      System<1, SystemType::Nonconservative, false, false>>>,
              tmpl::list<>>);

// The seeded per-face boundary value; distinct from every interior offset so
// its delivery is unambiguous in the expected values.
constexpr double facility_boundary_value = 100.0;
// Coefficients of the mock per-face time derivative
//   dt = facility_dt_stored_coefficient * (stored boundary value)
//        + facility_dt_var1_coefficient * Var1
//        + facility_dt_temporary_coefficient * Var3Squared.
constexpr double facility_dt_stored_coefficient = 2.5;
constexpr double facility_dt_var1_coefficient = -0.5;
constexpr double facility_dt_temporary_coefficient = 0.25;

// The stock `Ghost` extended with the boundary-evolved-fields opt-in. Its
// `dg_ghost` takes the stored boundary value as an extra argument (spliced in
// after the normal covector), checks it, and delegates to the stock `Ghost`
// overload -- which checks every interior input and writes the stock exterior
// data. `boundary_field_time_derivatives` produces the per-face dt from the
// stored value and the projected interior fields (verified exactly at the
// test site); it also receives the `dg_gridless_tags` volume arguments, like
// every boundary-condition method.
template <typename System>
class OptingGhost : public Ghost<System> {
 public:
  OptingGhost() = default;
  OptingGhost(OptingGhost&&) = default;
  OptingGhost& operator=(OptingGhost&&) = default;
  OptingGhost(const OptingGhost&) = default;
  OptingGhost& operator=(const OptingGhost&) = default;
  ~OptingGhost() override = default;

  explicit OptingGhost(CkMigrateMessage* msg) : Ghost<System>(msg) {}

  WRAPPED_PUPable_decl_base_template(
      domain::BoundaryConditions::BoundaryCondition, OptingGhost);

  auto get_clone() const -> std::unique_ptr<
      domain::BoundaryConditions::BoundaryCondition> override {
    return std::make_unique<OptingGhost<System>>(*this);
  }

  // NOLINTNEXTLINE
  void pup(PUP::er& p) override { Ghost<System>::pup(p); }

  using boundary_evolved_variables = tmpl::list<BoundaryVar1>;
  using boundary_field_time_derivatives_evolved_variables_tags =
      tmpl::list<Tags::Var1>;
  using boundary_field_time_derivatives_temporary_tags =
      tmpl::list<Tags::Var3Squared>;

  // Nonconservative system, flat background (the only configuration the
  // facility tests use).
  std::optional<std::string> dg_ghost(
      const gsl::not_null<Scalar<DataVector>*> out_var1,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          out_var2,
      const gsl::not_null<
          tnsr::I<DataVector, System::volume_dim, Frame::Inertial>*>
          out_var3,
      const gsl::not_null<Scalar<DataVector>*> out_var3_squared,
      const std::optional<tnsr::I<DataVector, System::volume_dim,
                                  Frame::Inertial>>& face_mesh_velocity,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
          outward_directed_normal_covector,
      const Scalar<DataVector>& boundary_var1, const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, System::volume_dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& var3_squared, const Scalar<DataVector>& dt_var1,
      const double volume_number) const {
    CHECK_ITERABLE_APPROX(
        get(boundary_var1),
        DataVector(get(boundary_var1).size(), facility_boundary_value));
    return Ghost<System>::dg_ghost(out_var1, out_var2, out_var3,
                                   out_var3_squared, face_mesh_velocity,
                                   outward_directed_normal_covector, var1, var2,
                                   var3_squared, dt_var1, volume_number);
  }

  std::optional<std::string> boundary_field_time_derivatives(
      const gsl::not_null<Scalar<DataVector>*> dt_boundary_var1,
      const std::optional<tnsr::I<DataVector, System::volume_dim,
                                  Frame::Inertial>>& /*face_mesh_velocity*/,
      const tnsr::i<DataVector, System::volume_dim, Frame::Inertial>&
      /*outward_directed_normal_covector*/,
      const Scalar<DataVector>& boundary_var1, const Scalar<DataVector>& var1,
      const Scalar<DataVector>& var3_squared,
      const double volume_number) const {
    CHECK(volume_number == 2.5);
    get(*dt_boundary_var1) =
        facility_dt_stored_coefficient * get(boundary_var1) +
        facility_dt_var1_coefficient * get(var1) +
        facility_dt_temporary_coefficient * get(var3_squared);
    return std::nullopt;
  }
};

template <typename System>
// NOLINTNEXTLINE
PUP::able::PUP_ID OptingGhost<System>::my_PUP_ID = 0;

// The opting condition cannot join `standard_boundary_conditions`: a factory
// list with a non-empty facility union makes the framework read the facility
// storage tags, which the pre-existing test DataBoxes do not (and must not)
// carry.
template <typename System>
using boundary_conditions_with_opting =
    tmpl::list<DemandOutgoingCharSpeeds<System>, OptingGhost<System>>;

static_assert(std::is_same_v<
              evolution::dg::BoundaryEvolvedFields::boundary_evolved_field_tags<
                  boundary_conditions_with_opting<
                      System<1, SystemType::Nonconservative, false, false>>>,
              tmpl::list<BoundaryVar1>>);

template <typename System>
struct MetavariablesWithOptingGhost {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<BoundaryCondition<System>,
                             boundary_conditions_with_opting<System>>>;
  };
};

// One System configuration; the stock corrections depend on the formulation
// and the lift path on the quadrature, so the caller varies both.
template <typename System, size_t Dim = System::volume_dim>
void test_boundary_evolved_fields(const dg::Formulation formulation,
                                  const Spectral::Quadrature quadrature) {
  CAPTURE(formulation);
  CAPTURE(quadrature);
  static_assert(System::volume_dim == 1);
  using field_tags = tmpl::list<BoundaryVar1>;
  using values_tag =
      evolution::dg::Tags::BoundaryEvolvedFieldsValues<Dim, field_tags>;
  using dt_stash_tag =
      evolution::dg::Tags::BoundaryEvolvedFieldsDtStash<Dim, field_tags>;
  using dt_variables_tag =
      db::add_tag_prefix<::Tags::dt, typename System::variables_tag>;

  const Mesh<Dim> mesh{5, Spectral::Basis::Legendre, quadrature};
  const ElementId<Dim> self_id{0, {{{1, 0}}}};
  const Element<Dim> element{self_id, {}};
  const auto opting_direction = Direction<Dim>::lower_xi();
  const auto outgoing_direction = Direction<Dim>::upper_xi();
  const size_t num_face_pts =
      mesh.slice_away(opting_direction.dimension()).number_of_grid_points();

  // The opting condition on one face; the (non-opting) stock
  // DemandOutgoingCharSpeeds on the other, as in `test_1d`.
  std::vector<DirectionMap<
      Dim, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
      external_boundary_conditions{1};
  external_boundary_conditions[0][opting_direction] =
      std::make_unique<OptingGhost<System>>();
  external_boundary_conditions[0][outgoing_direction] =
      std::make_unique<DemandOutgoingCharSpeeds<System>>(false);

  typename values_tag::type boundary_values{};
  Variables<field_tags> stored_value{num_face_pts};
  get(get<BoundaryVar1>(stored_value)) = facility_boundary_value;
  boundary_values[opting_direction] = std::move(stored_value);
  typename dt_stash_tag::type dt_stash{};
  dt_stash[opting_direction] =
      typename dt_stash_tag::type::mapped_type{num_face_pts, 0.0};

  // 2.5 seeds BoundaryConditionVolumeTag (checked inside the mock condition);
  // 4.5 seeds BoundaryCorrectionAuxiliaryVolumeTag (read by the stock
  // auxiliary-correction mock).
  const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>
      no_mesh_velocity{};
  auto box = make_boundary_conditions_box<
      MetavariablesWithOptingGhost<System>, System,
      tmpl::list<Tags::BoundaryConditionVolumeTag,
                 Tags::BoundaryCorrectionAuxiliaryVolumeTag, values_tag,
                 dt_stash_tag>>(mesh, element,
                                std::move(external_boundary_conditions),
                                no_mesh_velocity, formulation, 2.5, 4.5,
                                std::move(boundary_values),
                                std::move(dt_stash));

  Variables<
      typename System::compute_volume_time_derivative_terms::temporary_tags>
      temporaries{mesh.number_of_grid_points()};
  fill_variables(make_not_null(&temporaries), offset_temporaries);
  const Variables<
      db::wrap_tags_in<::Tags::Flux, typename System::flux_variables,
                       tmpl::size_t<Dim>, Frame::Inertial>>
      volume_fluxes{mesh.number_of_grid_points()};
  const Variables<
      db::wrap_tags_in<::Tags::deriv, typename System::gradient_variables,
                       tmpl::size_t<Dim>, Frame::Inertial>>
      partial_derivs{mesh.number_of_grid_points(), 0.0};
  const Variables<tmpl::list<>>* const primitive_vars_ptr = nullptr;

  using BndryTerms = BoundaryTerms<Dim, false, System::system_type,
                                   System::has_inverse_spatial_metric>;
  const BndryTerms boundary_terms{false, opting_direction.sign()};

  const auto& det_inv_jacobian = db::get<
      domain::Tags::DetInvJacobian<Frame::ElementLogical, Frame::Inertial>>(
      box);
  const auto magnitude_of_face_normal =
      [&box](const Direction<Dim>& direction) -> const Scalar<DataVector>& {
    return get<evolution::dg::Tags::MagnitudeOfNormal>(
        *db::get<evolution::dg::Tags::NormalCovectorAndMagnitude<Dim>>(box).at(
            direction));
  };

  {
    INFO("Physical pass: dt-stash write and stored-value feed into dg_ghost");
    evolution::dg::Actions::detail::
        apply_boundary_conditions_on_all_external_faces<System, Dim>(
            make_not_null(&box), boundary_terms, temporaries, volume_fluxes,
            partial_derivs, primitive_vars_ptr);

    // The opting face's dt-stash entry holds the expected per-face time
    // derivative -- a known function of the stored boundary value and the
    // projected interior fields -- sized to the face mesh. The non-opting
    // face has no entry.
    const auto& stash = db::get<dt_stash_tag>(box);
    REQUIRE(stash.contains(opting_direction));
    CHECK(not stash.contains(outgoing_direction));
    const auto& stash_entry = stash.at(opting_direction);
    CHECK(stash_entry.number_of_grid_points() == num_face_pts);
    const double expected_dt =
        facility_dt_stored_coefficient * facility_boundary_value +
        facility_dt_var1_coefficient * offset_evolved_vars +
        facility_dt_temporary_coefficient * offset_temporaries;
    CHECK_ITERABLE_APPROX(get(get<::Tags::dt<BoundaryVar1>>(stash_entry)),
                          DataVector(num_face_pts, expected_dt));

    // The boundary field's dt lives only in the stash: `BoundaryVar1` is not a
    // volume variable, so it cannot appear in the volume time derivative.
    static_assert(
        not tmpl::list_contains_v<typename dt_variables_tag::tags_list,
                                  ::Tags::dt<BoundaryVar1>>);

    // The volume time derivative received exactly the stock ghost correction
    // lifted on the opting face (DemandOutgoingCharSpeeds does not lift);
    // identical formula to `test_1d`'s expected correction. This proves the
    // `dg_ghost` path ran for the opting condition, so its in-mock check of
    // the delivered stored value cannot have passed vacuously.
    using dt_correction_tags = typename dt_variables_tag::tags_list;
    Variables<dt_correction_tags> expected_on_boundary{num_face_pts};
    get(get<::Tags::dt<Tags::Var1>>(expected_on_boundary)) =
        offset_boundary_correction *
        (formulation == dg::Formulation::WeakInertial ? 2.0 : 1.0);
    for (size_t i = 0; i < Dim; ++i) {
      get<::Tags::dt<Tags::Var2<Dim>>>(expected_on_boundary).get(i) =
          offset_boundary_correction + 1.0 + i;
    }
    Variables<dt_correction_tags> expected_dt_vars{
        mesh.number_of_grid_points()};
    fill_variables(make_not_null(&expected_dt_vars), offset_dt_evolved_vars);
    expected_dt_vars += lifted_boundary_correction(
        std::move(expected_on_boundary), mesh, opting_direction,
        det_inv_jacobian, magnitude_of_face_normal(opting_direction));
    CHECK_ITERABLE_APPROX(get(get<::Tags::dt<Tags::Var1>>(box)),
                          get(get<::Tags::dt<Tags::Var1>>(expected_dt_vars)));
    CHECK_ITERABLE_APPROX(
        get<::Tags::dt<Tags::Var2<Dim>>>(box).get(0),
        get<::Tags::dt<Tags::Var2<Dim>>>(expected_dt_vars).get(0));
  }

  {
    INFO("Auxiliary pass: no dt-stash write; stored-value feed into dg_ghost");
    // Reset the volume time derivatives to their seeded values: the stock
    // mocks check the projected dt against the seeds, and the physical pass
    // above lifted a correction into them.
    db::mutate<dt_variables_tag>(
        [](const auto dt_vars_ptr) {
          fill_variables(dt_vars_ptr, offset_dt_evolved_vars);
        },
        make_not_null(&box));
    // Sentinel-seed the stash so a spurious auxiliary-pass write is detected.
    const double sentinel = 999.0;
    db::mutate<dt_stash_tag>(
        [&opting_direction, &sentinel](const auto stash_ptr) {
          get(get<::Tags::dt<BoundaryVar1>>(stash_ptr->at(opting_direction))) =
              sentinel;
        },
        make_not_null(&box));
    const auto auxiliary_vars_before =
        db::get<::Tags::Variables<typename System::auxiliary_variables>>(box);

    evolution::dg::Actions::detail::
        apply_boundary_conditions_on_all_external_faces<
            System, Dim, /*ComputeAuxiliary=*/true>(
            make_not_null(&box), boundary_terms, temporaries, volume_fluxes,
            partial_derivs, primitive_vars_ptr);

    // The auxiliary pass must not produce boundary-field time derivatives: the
    // stash still holds the sentinel.
    CHECK_ITERABLE_APPROX(get(get<::Tags::dt<BoundaryVar1>>(
                              db::get<dt_stash_tag>(box).at(opting_direction))),
                          DataVector(num_face_pts, sentinel));

    // The auxiliary variables received exactly the stock lifted auxiliary
    // correction on the opting face (`0.5 * (int_var2 + ext_var2)`, as in
    // `test_auxiliary_1d`), proving `dg_ghost` -- with the spliced stored
    // value -- ran in this pass too.
    using var3_correction_tags = tmpl::list<Tags::Var3<Dim>>;
    Variables<var3_correction_tags> expected_on_boundary{num_face_pts};
    for (size_t i = 0; i < Dim; ++i) {
      get<Tags::Var3<Dim>>(expected_on_boundary).get(i) =
          0.5 * ((offset_evolved_vars + 1.0 + i) +
                 (offset_boundary_condition + 1.0 + i));
    }
    auto expected_auxiliary_vars = auxiliary_vars_before;
    expected_auxiliary_vars += lifted_boundary_correction(
        std::move(expected_on_boundary), mesh, opting_direction,
        det_inv_jacobian, magnitude_of_face_normal(opting_direction));
    CHECK_ITERABLE_APPROX(get<Tags::Var3<Dim>>(box).get(0),
                          get<Tags::Var3<Dim>>(expected_auxiliary_vars).get(0));
  }
}

#ifdef SPECTRE_DEBUG
// The dt-stash write ASSERTs each entry is sized to the face mesh (the write
// copies per tag with no size negotiation, so a wrong size would otherwise be
// silent); a wrongly-sized entry must trip it.
void test_boundary_evolved_fields_stash_size_assert() {
  using TestSystem = System<1, SystemType::Nonconservative, false, false>;
  constexpr size_t Dim = 1;
  using field_tags = tmpl::list<BoundaryVar1>;
  using values_tag =
      evolution::dg::Tags::BoundaryEvolvedFieldsValues<Dim, field_tags>;
  using dt_stash_tag =
      evolution::dg::Tags::BoundaryEvolvedFieldsDtStash<Dim, field_tags>;

  const Mesh<Dim> mesh{5, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
  const ElementId<Dim> self_id{0, {{{1, 0}}}};
  const Element<Dim> element{self_id, {}};
  const auto opting_direction = Direction<Dim>::lower_xi();
  const size_t num_face_pts =
      mesh.slice_away(opting_direction.dimension()).number_of_grid_points();

  std::vector<DirectionMap<
      Dim, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
      external_boundary_conditions{1};
  external_boundary_conditions[0][opting_direction] =
      std::make_unique<OptingGhost<TestSystem>>();
  external_boundary_conditions[0][Direction<Dim>::upper_xi()] =
      std::make_unique<DemandOutgoingCharSpeeds<TestSystem>>(false);

  typename values_tag::type boundary_values{};
  Variables<field_tags> stored_value{num_face_pts};
  get(get<BoundaryVar1>(stored_value)) = facility_boundary_value;
  boundary_values[opting_direction] = std::move(stored_value);
  // Deliberately size the stash entry wrong (one point too many) so the
  // write-time size assertion fires.
  typename dt_stash_tag::type dt_stash{};
  dt_stash[opting_direction] =
      typename dt_stash_tag::type::mapped_type{num_face_pts + 1, 0.0};

  const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>
      no_mesh_velocity{};
  auto box = make_boundary_conditions_box<
      MetavariablesWithOptingGhost<TestSystem>, TestSystem,
      tmpl::list<Tags::BoundaryConditionVolumeTag,
                 Tags::BoundaryCorrectionAuxiliaryVolumeTag, values_tag,
                 dt_stash_tag>>(mesh, element,
                                std::move(external_boundary_conditions),
                                no_mesh_velocity,
                                dg::Formulation::StrongInertial, 2.5, 4.5,
                                std::move(boundary_values),
                                std::move(dt_stash));
  Variables<
      typename TestSystem::compute_volume_time_derivative_terms::temporary_tags>
      temporaries{mesh.number_of_grid_points()};
  fill_variables(make_not_null(&temporaries), offset_temporaries);
  const Variables<
      db::wrap_tags_in<::Tags::Flux, typename TestSystem::flux_variables,
                       tmpl::size_t<Dim>, Frame::Inertial>>
      volume_fluxes{mesh.number_of_grid_points()};
  const Variables<
      db::wrap_tags_in<::Tags::deriv, typename TestSystem::gradient_variables,
                       tmpl::size_t<Dim>, Frame::Inertial>>
      partial_derivs{mesh.number_of_grid_points(), 0.0};
  const Variables<tmpl::list<>>* const primitive_vars_ptr = nullptr;
  const BoundaryTerms<Dim, false, TestSystem::system_type,
                      TestSystem::has_inverse_spatial_metric>
      boundary_terms{false, opting_direction.sign()};

  CHECK_THROWS_WITH(
      (evolution::dg::Actions::detail::
           apply_boundary_conditions_on_all_external_faces<TestSystem, Dim>(
               make_not_null(&box), boundary_terms, temporaries, volume_fluxes,
               partial_derivs, primitive_vars_ptr)),
      Catch::Matchers::ContainsSubstring("sized to the face mesh"));
}
#endif  // SPECTRE_DEBUG

void test_cartoon_mesh_compatibility() {
  INFO("Test that non-cartoon mesh throws with cartoon boundary conditions");

  // Create a 1D system with cartoon boundary conditions
  using TestSystem = System<1, SystemType::Conservative, false, false>;

  // Create a non-cartoon compatible mesh (regular Legendre basis)
  const Mesh<1> non_cartoon_mesh{3, Spectral::Basis::Legendre,
                                 Spectral::Quadrature::GaussLobatto};

  const ElementId<1> element_id{0};
  const Element<1> element{element_id, {}};
  const double boundary_condition_volume_tag_number{2.5};
  const double boundary_correction_volume_tag_number{3.5};

  // Set up boundary conditions with cartoon BC on one boundary
  std::vector<DirectionMap<
      1, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
      external_boundary_conditions{1};
  external_boundary_conditions[0][Direction<1>::lower_xi()] = std::make_unique<
      domain::BoundaryConditions::Cartoon<BoundaryCondition<TestSystem>>>();
  external_boundary_conditions[0][Direction<1>::upper_xi()] = std::make_unique<
      domain::BoundaryConditions::None<BoundaryCondition<TestSystem>>>();

  auto boundary_correction =
      BoundaryTerms<1, false, SystemType::Conservative, false>{false, 1.0};

  using dt_variables_tag =
      db::add_tag_prefix<::Tags::dt, typename TestSystem::variables_tag>;
  using simple_tags = tmpl::list<
      Parallel::Tags::MetavariablesImpl<MetavariablesWithCartoon<TestSystem>>,
      domain::Tags::ExternalBoundaryConditions<1>, domain::Tags::Mesh<1>,
      domain::Tags::Element<1>, domain::Tags::ElementMap<1, Frame::Grid>,
      domain::CoordinateMaps::Tags::CoordinateMap<1, Frame::Grid,
                                                  Frame::Inertial>,
      ::Tags::Time, domain::Tags::FunctionsOfTime,
      domain::Tags::MeshVelocity<1>,
      domain::Tags::InverseJacobian<1, Frame::ElementLogical, Frame::Inertial>,
      domain::Tags::DetInvJacobian<Frame::ElementLogical, Frame::Inertial>,
      evolution::dg::Tags::NormalCovectorAndMagnitude<1>,
      typename TestSystem::variables_tag,
      ::Tags::Variables<typename TestSystem::auxiliary_variables>,
      dt_variables_tag, Tags::BoundaryConditionVolumeTag,
      Tags::BoundaryCorrectionVolumeTag, ::dg::Tags::Formulation>;
  using compute_tags = tmpl::list<>;

  Variables<typename TestSystem::variables_tag::tags_list> evolved_vars{3, 1.0};
  Variables<typename TestSystem::auxiliary_variables> auxiliary_vars{3, 1.0};
  Variables<db::wrap_tags_in<::Tags::dt,
                             typename TestSystem::variables_tag::tags_list>>
      dt_evolved_vars{3, 0.0};

  auto box = db::create<simple_tags, compute_tags>(
      MetavariablesWithCartoon<TestSystem>{},
      std::move(external_boundary_conditions), non_cartoon_mesh, element,
      ElementMap<1, Frame::Grid>{
          element_id,
          domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(
              domain::CoordinateMaps::Identity<1>{})},
      domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
          domain::CoordinateMaps::Identity<1>{}),
      0.0,
      std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>{},
      std::optional<tnsr::I<DataVector, 1>>{},
      InverseJacobian<DataVector, 1, Frame::ElementLogical, Frame::Inertial>{
          3_st, 1.0},
      Scalar<DataVector>{3_st, 1.0},
      evolution::dg::Tags::NormalCovectorAndMagnitude<1>::type{}, evolved_vars,
      auxiliary_vars, dt_evolved_vars, boundary_condition_volume_tag_number,
      boundary_correction_volume_tag_number, dg::Formulation::StrongInertial);

  // Create minimal temporaries, fluxes, and partial derivatives
  const Variables<tmpl::list<Tags::Var3Squared>> temporaries{3, 1.0};
  const Variables<
      db::wrap_tags_in<::Tags::Flux, tmpl::list<Tags::Var1, Tags::Var2<1>>,
                       tmpl::size_t<1>, Frame::Inertial>>
      volume_fluxes{3, 1.0};
  const Variables<db::wrap_tags_in<::Tags::deriv, tmpl::list<>, tmpl::size_t<1>,
                                   Frame::Inertial>>
      partial_derivs{};

  // This should throw because a cartoon BC is used on lower_xi with
  // an incompatible mesh.
  CHECK_THROWS_WITH(
      (evolution::dg::Actions::detail::
           apply_boundary_conditions_on_all_external_faces<TestSystem, 1>(
               make_not_null(&box), boundary_correction, temporaries,
               volume_fluxes, partial_derivs, nullptr)),
      Catch::Matchers::ContainsSubstring(
          "You might have used a Cartoon boundary condition on an external "
          "boundary condition"));

  // Negative test: cartoon BCs are in the factory and the mesh is incompatible,
  // but no cartoon BC is actually used on any external face. This must not
  // error
  {
    INFO(
        "Test that non-cartoon mesh does not throw when cartoon BC is not "
        "used");
    std::vector<DirectionMap<
        1, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
        no_cartoon_bcs{1};
    no_cartoon_bcs[0][Direction<1>::lower_xi()] = std::make_unique<
        domain::BoundaryConditions::None<BoundaryCondition<TestSystem>>>();
    no_cartoon_bcs[0][Direction<1>::upper_xi()] = std::make_unique<
        domain::BoundaryConditions::None<BoundaryCondition<TestSystem>>>();

    Variables<typename TestSystem::variables_tag::tags_list> evolved_vars2{3,
                                                                           1.0};
    Variables<typename TestSystem::auxiliary_variables> auxiliary_vars2{3, 1.0};
    Variables<db::wrap_tags_in<::Tags::dt,
                               typename TestSystem::variables_tag::tags_list>>
        dt_evolved_vars2{3, 0.0};

    auto box2 = db::create<simple_tags, compute_tags>(
        MetavariablesWithCartoon<TestSystem>{}, std::move(no_cartoon_bcs),
        non_cartoon_mesh, element,
        ElementMap<1, Frame::Grid>{
            element_id,
            domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(
                domain::CoordinateMaps::Identity<1>{})},
        domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
            domain::CoordinateMaps::Identity<1>{}),
        0.0,
        std::unordered_map<
            std::string,
            std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>{},
        std::optional<tnsr::I<DataVector, 1>>{},
        InverseJacobian<DataVector, 1, Frame::ElementLogical, Frame::Inertial>{
            3_st, 1.0},
        Scalar<DataVector>{3_st, 1.0},
        evolution::dg::Tags::NormalCovectorAndMagnitude<1>::type{},
        evolved_vars2, auxiliary_vars2, dt_evolved_vars2,
        boundary_condition_volume_tag_number,
        boundary_correction_volume_tag_number, dg::Formulation::StrongInertial);

    CHECK_NOTHROW(
        (evolution::dg::Actions::detail::
             apply_boundary_conditions_on_all_external_faces<TestSystem, 1>(
                 make_not_null(&box2), boundary_correction, temporaries,
                 volume_fluxes, partial_derivs, nullptr)));
  }
}

SPECTRE_TEST_CASE("Unit.Evolution.DG.ComputeTimeDerivative.BoundaryConditions",
                  "[Unit][Evolution][Actions]") {
  // The test proceeds as follows:
  //
  // 1. prepare all the data in the DataBox
  // 2. call the boundary condition function, which should apply the boundary
  //    condition. We do so switching around the direction the boundary
  //    condition is applied in, checking different ones on each side.
  // 3. inside the boundary conditions we check we have received the expected
  //    values of the different tags
  // 4. we return pre-determined numbers so that we can check the time
  //    derivatives changed in the expected way given the numbers.
  //
  // Notes:
  // - the test is currently only in 1d, but most (if not all) places that need
  //   generalization have a `static_assert(Dim == 1)`. Going to more dimensions
  //   is straightforward but _extremely_ tedious.
  for (const bool moving_mesh : {true, false}) {
    for (const Spectral::Quadrature quadrature :
         {Spectral::Quadrature::Gauss, Spectral::Quadrature::GaussLobatto}) {
      for (const dg::Formulation formulation :
           {dg::Formulation::WeakInertial, dg::Formulation::StrongInertial}) {
        // Second last template parameter on System:
        // - true: has primitive variables
        // - false: no primitive variables

        // last template parameter on System being `false` means flat background
        test_1d<System<1, SystemType::Conservative, false, false>>(
            moving_mesh, formulation, quadrature);
        test_1d<System<1, SystemType::Conservative, true, false>>(
            moving_mesh, formulation, quadrature);

        test_1d<System<1, SystemType::Nonconservative, false, false>>(
            moving_mesh, formulation, quadrature);

        test_1d<System<1, SystemType::Mixed, false, false>>(
            moving_mesh, formulation, quadrature);
        test_1d<System<1, SystemType::Mixed, true, false>>(
            moving_mesh, formulation, quadrature);

        // last template parameter on System being `true` means curved
        // background
        test_1d<System<1, SystemType::Conservative, false, true>>(
            moving_mesh, formulation, quadrature);
        test_1d<System<1, SystemType::Conservative, true, true>>(
            moving_mesh, formulation, quadrature);

        test_1d<System<1, SystemType::Nonconservative, false, true>>(
            moving_mesh, formulation, quadrature);

        test_1d<System<1, SystemType::Mixed, false, true>>(
            moving_mesh, formulation, quadrature);
        test_1d<System<1, SystemType::Mixed, true, true>>(
            moving_mesh, formulation, quadrature);
      }

      // auxiliary-pass analogues across the same System configurations. The
      // LDG auxiliary pass is designed for the strong formulation only, so it
      // is not run with WeakInertial.
      // last template parameter on System being `false` means flat background
      test_1d<System<1, SystemType::Conservative, false, false>, true>(
          moving_mesh, dg::Formulation::StrongInertial, quadrature);
      test_1d<System<1, SystemType::Conservative, true, false>, true>(
          moving_mesh, dg::Formulation::StrongInertial, quadrature);

      test_1d<System<1, SystemType::Nonconservative, false, false>, true>(
          moving_mesh, dg::Formulation::StrongInertial, quadrature);

      test_1d<System<1, SystemType::Mixed, false, false>, true>(
          moving_mesh, dg::Formulation::StrongInertial, quadrature);
      test_1d<System<1, SystemType::Mixed, true, false>, true>(
          moving_mesh, dg::Formulation::StrongInertial, quadrature);

      // last template parameter on System being `true` means curved
      // background
      test_1d<System<1, SystemType::Conservative, false, true>, true>(
          moving_mesh, dg::Formulation::StrongInertial, quadrature);
      test_1d<System<1, SystemType::Conservative, true, true>, true>(
          moving_mesh, dg::Formulation::StrongInertial, quadrature);

      test_1d<System<1, SystemType::Nonconservative, false, true>, true>(
          moving_mesh, dg::Formulation::StrongInertial, quadrature);

      test_1d<System<1, SystemType::Mixed, false, true>, true>(
          moving_mesh, dg::Formulation::StrongInertial, quadrature);
      test_1d<System<1, SystemType::Mixed, true, true>, true>(
          moving_mesh, dg::Formulation::StrongInertial, quadrature);
    }
  }
  // Boundary-evolved-fields facility plumbing (single System configuration,
  // static mesh).
  for (const dg::Formulation formulation :
       {dg::Formulation::WeakInertial, dg::Formulation::StrongInertial}) {
    for (const Spectral::Quadrature quadrature :
         {Spectral::Quadrature::Gauss, Spectral::Quadrature::GaussLobatto}) {
      test_boundary_evolved_fields<
          System<1, SystemType::Nonconservative, false, false>>(formulation,
                                                                quadrature);
    }
  }
#ifdef SPECTRE_DEBUG
  test_boundary_evolved_fields_stash_size_assert();
#endif
  test_cartoon_mesh_compatibility();
}
}  // namespace
