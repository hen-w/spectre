// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>

#include "DataStructures/SimpleSparseMatrix.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Filter.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/ApplyTensorYlmFilter.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/TensorYlm.hpp"
#include "Options/Auto.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

/// \cond
class DataVector;
namespace ylm {
class Spherepack;
}  // namespace ylm
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace Ccz4 {

/// Defines tags and functions used internally in filtering, but
/// tested independently in the unit tests.
namespace filter_detail {

/// The 9 CCZ4 original evolved variable tags, templated on Frame.
/// Scalar tags have no Frame parameter; tensor tags do.
template <typename Frame>
using ccz4_vars_list =
    tmpl::list<Tags::ConformalMetric<DataVector, 3, Frame>,
               Tags::ConformalFactor<DataVector>,
               Tags::ATilde<DataVector, 3, Frame>,
               gr::Tags::TraceExtrinsicCurvature<DataVector>,
               Tags::Theta<DataVector>,
               Tags::GammaHat<DataVector, 3, Frame>,
               gr::Tags::Lapse<DataVector>,
               gr::Tags::Shift<DataVector, 3, Frame>,
               Tags::AuxiliaryShiftB<DataVector, 3, Frame>>;

/*!
 * \brief Transforms spatial tensors between frames using Jacobian
 * contractions.
 *
 * Lower-index tensors (tnsr::ii) are transformed with \p jac_for_lower,
 * while upper-index tensors (tnsr::I) are transformed with
 * \p inv_jac_for_upper. Scalars are copied directly.
 *
 * Takes special care to re-use memory. The Variables arguments must
 * already be allocated to their correct sizes; no memory allocation
 * is done.
 *
 * \tparam SrcFrame Source frame.
 * \tparam DestFrame Destination frame.
 * \param dest A Variables for the destination spatial variables.
 * \param src A Variables containing the source spatial variables.
 * \param jac_for_lower The jacobian dx^src/dx^dest for lower-index tensors.
 * \param inv_jac_for_upper The jacobian dx^dest/dx^src for upper-index tensors.
 */
template <typename SrcFrame, typename DestFrame>
void transform_spatial_tensors_to_different_frame_without_hessians(
    gsl::not_null<Variables<ccz4_vars_list<DestFrame>>*> dest,
    const Variables<ccz4_vars_list<SrcFrame>>& src,
    const InverseJacobian<DataVector, 3, SrcFrame, DestFrame>& jac_for_lower,
    const InverseJacobian<DataVector, 3, DestFrame, SrcFrame>&
        inv_jac_for_upper);

}  // namespace filter_detail

/*!
 * \brief Applies TensorYlm filter in place to CCZ4 variables.
 *
 * When radial_extents is 1, ccz4_vars and temp_storage are assumed to
 * be defined on a spherical slice, with number of grid points
 * corresponding to a spherical-harmonic grid of ell_max, and the
 * filter happens only on that slice.
 *
 * When radial_extents is > 1, ccz4_vars and temp_storage are assumed to
 * be defined on a spherical shell of topology I1 x S2. The filter
 * happens in the entire volume, internally iterating over each
 * spherical slice at a time.
 *
 * For performance reasons, apply_tensor_ylm_filter does not allocate
 * or deallocate memory, but it does take a temp_storage buffer.  The
 * size of temp_storage should at least
 * radial_extents*spectral_size*num_components, where num_components
 * is the total number of independent components in the CCZ4 variable
 * list (i.e. 25), and spectral_size is the size of the S2 Spherepack
 * spectral coefficient array for ell_max, as obtained from the member
 * function ylm::Spherepack::spectral_size().
 *
 * \param ccz4_vars CCZ4 variables at collocation points.
 * \param temp_storage Temporary storage for CCZ4 variables.
 * \param jac_inertial_to_grid Jacobian taking V_x from inertial to grid.
 * \param jac_grid_to_inertial Jacobian taking V_x from grid to inertial.
 * \param filter_matrix_scalar The scalar filter matrix computed by fill_filter.
 * \param filter_matrix_i The Rank-1 matrix computed by fill_filter.
 * \param filter_matrix_ii The Rank-2 symmetric matrix computed by fill_filter.
 * \param ell_max The maximum ylm ell.
 * \param radial_extents The number of radial grid points, can be 1 for slices.
 */
void apply_tensor_ylm_filter(
    gsl::not_null<Variables<filter_detail::ccz4_vars_list<Frame::Inertial>>*>
        ccz4_vars,
    gsl::not_null<Variables<filter_detail::ccz4_vars_list<Frame::Inertial>>*>
        temp_storage,
    const InverseJacobian<DataVector, 3, Frame::Inertial, Frame::Grid>&
        jac_inertial_to_grid,
    const InverseJacobian<DataVector, 3, Frame::Grid, Frame::Inertial>&
        jac_grid_to_inertial,
    const SimpleSparseMatrix& filter_matrix_scalar,
    const SimpleSparseMatrix& filter_matrix_i,
    const SimpleSparseMatrix& filter_matrix_ii, size_t ell_max,
    size_t radial_extents);

/*!
 * \brief DataBox mutator that applies a TensorYlm filter to the CCZ4 variables
 * and caches the filter matrices.
 */
class TensorYlmFilter : public Filters::Filter {
 public:
  struct NumModesToKill {
    using type = size_t;
    static constexpr Options::String help =
        "How many of the top ell modes to set to zero";
  };
  struct HalfPower {
    using type = Options::Auto<size_t, Options::AutoLabel::None>;
    static constexpr Options::String help =
        "The half-power sigma for more complicated filtering. "
        "If None, implements a Heaviside filter.";
  };
  using options = tmpl::list<NumModesToKill, HalfPower>;
  static constexpr Options::String help = {"Tensor Ylm filter."};

  TensorYlmFilter();
  TensorYlmFilter(const TensorYlmFilter& rhs);
  TensorYlmFilter& operator=(const TensorYlmFilter& rhs);
  TensorYlmFilter(TensorYlmFilter&& rhs);
  TensorYlmFilter& operator=(TensorYlmFilter&& rhs);
  ~TensorYlmFilter() override = default;

  WRAPPED_PUPable_decl_template(TensorYlmFilter);  // NOLINT
  explicit TensorYlmFilter(CkMigrateMessage* msg);

  TensorYlmFilter(size_t num_modes_to_kill, std::optional<size_t> half_power);

  std::optional<std::unordered_set<std::string>> blocks_to_filter()
      const override {
    return std::nullopt;
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

 public:  // DataBox-mutator protocol
  using argument_tags = tmpl::list<
      domain::Tags::Mesh<3>,
      domain::Tags::InverseJacobian<3, Frame::Grid, Frame::Inertial>>;

  void operator()(
      gsl::not_null<
          Variables<filter_detail::ccz4_vars_list<Frame::Inertial>>*>
          ccz4_vars,
      const Mesh<3>& mesh,
      const InverseJacobian<DataVector, 3, Frame::Grid, Frame::Inertial>&
          jac_grid_to_inertial) const;

  /// Tuple overload required because filter_all_vars is false for CCZ4
  /// (the 9 filtered tags != the 17+ total system tags). The filter action
  /// passes individual tensors as a tuple when filter_all_vars is false.
  template <typename... TensorTypes>
    requires((not tt::is_a_v<Variables, std::decay_t<TensorTypes>>) and ...)
  void operator()(
      const std::tuple<gsl::not_null<TensorTypes*>...>& tensors,
      const Mesh<3>& mesh,
      const InverseJacobian<DataVector, 3, Frame::Grid, Frame::Inertial>&
          jac_grid_to_inertial) const {
    static_assert(sizeof...(TensorTypes) == 9,
                  "CCZ4 TensorYlmFilter expects exactly 9 tensors");
    const size_t num_points = mesh.number_of_grid_points();
    Variables<filter_detail::ccz4_vars_list<Frame::Inertial>> ccz4_vars(
        num_points);
    // Copy tensors from tuple into Variables
    auto& [conf_metric, conf_factor, a_tilde, trace_k, theta, gamma_hat,
           lapse, shift, aux_shift] = ccz4_vars;
    conf_metric = *std::get<0>(tensors);
    conf_factor = *std::get<1>(tensors);
    a_tilde = *std::get<2>(tensors);
    trace_k = *std::get<3>(tensors);
    theta = *std::get<4>(tensors);
    gamma_hat = *std::get<5>(tensors);
    lapse = *std::get<6>(tensors);
    shift = *std::get<7>(tensors);
    aux_shift = *std::get<8>(tensors);

    (*this)(make_not_null(&ccz4_vars), mesh, jac_grid_to_inertial);

    // Move filtered results back to tuple
    *std::get<0>(tensors) = std::move(conf_metric);
    *std::get<1>(tensors) = std::move(conf_factor);
    *std::get<2>(tensors) = std::move(a_tilde);
    *std::get<3>(tensors) = std::move(trace_k);
    *std::get<4>(tensors) = std::move(theta);
    *std::get<5>(tensors) = std::move(gamma_hat);
    *std::get<6>(tensors) = std::move(lapse);
    *std::get<7>(tensors) = std::move(shift);
    *std::get<8>(tensors) = std::move(aux_shift);
  }

 private:
  friend bool operator==(const TensorYlmFilter& lhs,
                         const TensorYlmFilter& rhs);

  size_t num_modes_to_kill_{0};
  std::optional<size_t> half_power_{std::nullopt};
  // Use Spherepack normalization because the variables are stored as Spherepack
  // modes
  static constexpr ylm::TensorYlm::CoefficientNormalization normalization_ =
      ylm::TensorYlm::CoefficientNormalization::Spherepack;
  // Caches and memory buffers
  // NOLINTNEXTLINE(spectre-mutable)
  mutable size_t cached_l_max_{0};
  // NOLINTNEXTLINE(spectre-mutable)
  mutable SimpleSparseMatrix filter_matrix_scalar_{};
  // NOLINTNEXTLINE(spectre-mutable)
  mutable SimpleSparseMatrix filter_matrix_i_{};
  // NOLINTNEXTLINE(spectre-mutable)
  mutable SimpleSparseMatrix filter_matrix_ii_{};
  // NOLINTNEXTLINE(spectre-mutable)
  mutable Variables<filter_detail::ccz4_vars_list<Frame::Inertial>>
      temp_storage_{};
};

bool operator!=(const TensorYlmFilter& lhs, const TensorYlmFilter& rhs);

}  // namespace Ccz4
