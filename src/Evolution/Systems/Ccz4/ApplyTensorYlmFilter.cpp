// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Ccz4/ApplyTensorYlmFilter.hpp"

#include <cstddef>
#include <cstring>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Structure.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackCache.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/TensorYlmFilter.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/TMPL.hpp"

#include "NumericalAlgorithms/SphericalHarmonics/ApplyTensorYlmFilter.tpp"

namespace Ccz4 {

namespace filter_detail {
template <typename SrcFrame, typename DestFrame>
void transform_spatial_tensors_to_different_frame_without_hessians(
    const gsl::not_null<Variables<ccz4_vars_list<DestFrame>>*> dest,
    const Variables<ccz4_vars_list<SrcFrame>>& src,
    const InverseJacobian<DataVector, 3, SrcFrame, DestFrame>& jac_for_lower,
    const InverseJacobian<DataVector, 3, DestFrame, SrcFrame>&
        inv_jac_for_upper) {
  const auto& [src_conf_metric, src_conf_factor, src_a_tilde, src_trace_k,
               src_theta, src_gamma_hat, src_lapse, src_shift,
               src_aux_shift] = src;
  auto& [dest_conf_metric, dest_conf_factor, dest_a_tilde, dest_trace_k,
         dest_theta, dest_gamma_hat, dest_lapse, dest_shift,
         dest_aux_shift] = *dest;

  // Transform tnsr::ii variables: dest_{ij} = J^k_i * J^l_j * src_{kl}
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      dest_conf_metric.get(i, j) =
          jac_for_lower.get(0, i) *
              (jac_for_lower.get(0, j) * src_conf_metric.get(0, 0) +
               jac_for_lower.get(1, j) * src_conf_metric.get(0, 1) +
               jac_for_lower.get(2, j) * src_conf_metric.get(0, 2)) +
          jac_for_lower.get(1, i) *
              (jac_for_lower.get(0, j) * src_conf_metric.get(1, 0) +
               jac_for_lower.get(1, j) * src_conf_metric.get(1, 1) +
               jac_for_lower.get(2, j) * src_conf_metric.get(1, 2)) +
          jac_for_lower.get(2, i) *
              (jac_for_lower.get(0, j) * src_conf_metric.get(2, 0) +
               jac_for_lower.get(1, j) * src_conf_metric.get(2, 1) +
               jac_for_lower.get(2, j) * src_conf_metric.get(2, 2));
      dest_a_tilde.get(i, j) =
          jac_for_lower.get(0, i) *
              (jac_for_lower.get(0, j) * src_a_tilde.get(0, 0) +
               jac_for_lower.get(1, j) * src_a_tilde.get(0, 1) +
               jac_for_lower.get(2, j) * src_a_tilde.get(0, 2)) +
          jac_for_lower.get(1, i) *
              (jac_for_lower.get(0, j) * src_a_tilde.get(1, 0) +
               jac_for_lower.get(1, j) * src_a_tilde.get(1, 1) +
               jac_for_lower.get(2, j) * src_a_tilde.get(1, 2)) +
          jac_for_lower.get(2, i) *
              (jac_for_lower.get(0, j) * src_a_tilde.get(2, 0) +
               jac_for_lower.get(1, j) * src_a_tilde.get(2, 1) +
               jac_for_lower.get(2, j) * src_a_tilde.get(2, 2));
    }
  }

  // Transform tnsr::I variables: dest^i = (J^{-1})^i_k * src^k
  for (size_t i = 0; i < 3; ++i) {
    dest_gamma_hat.get(i) =
        inv_jac_for_upper.get(i, 0) * src_gamma_hat.get(0) +
        inv_jac_for_upper.get(i, 1) * src_gamma_hat.get(1) +
        inv_jac_for_upper.get(i, 2) * src_gamma_hat.get(2);
    dest_shift.get(i) = inv_jac_for_upper.get(i, 0) * src_shift.get(0) +
                        inv_jac_for_upper.get(i, 1) * src_shift.get(1) +
                        inv_jac_for_upper.get(i, 2) * src_shift.get(2);
    dest_aux_shift.get(i) =
        inv_jac_for_upper.get(i, 0) * src_aux_shift.get(0) +
        inv_jac_for_upper.get(i, 1) * src_aux_shift.get(1) +
        inv_jac_for_upper.get(i, 2) * src_aux_shift.get(2);
  }

  // Copy scalars
  get<>(dest_conf_factor) = get<>(src_conf_factor);
  get<>(dest_trace_k) = get<>(src_trace_k);
  get<>(dest_theta) = get<>(src_theta);
  get<>(dest_lapse) = get<>(src_lapse);
}
}  // namespace filter_detail

void apply_tensor_ylm_filter(
    const gsl::not_null<
        Variables<filter_detail::ccz4_vars_list<Frame::Inertial>>*>
        ccz4_vars,
    const gsl::not_null<
        Variables<filter_detail::ccz4_vars_list<Frame::Inertial>>*>
        temp_storage,
    const InverseJacobian<DataVector, 3, Frame::Inertial, Frame::Grid>&
        jac_inertial_to_grid,
    const InverseJacobian<DataVector, 3, Frame::Grid, Frame::Inertial>&
        jac_grid_to_inertial,
    const SimpleSparseMatrix& filter_matrix_scalar,
    const SimpleSparseMatrix& filter_matrix_i,
    const SimpleSparseMatrix& filter_matrix_ii, const size_t ell_max,
    const size_t radial_extents) {
  const auto& ylm = ylm::get_spherepack_cache(ell_max);
  ASSERT(
      radial_extents * ylm.physical_size() ==
          ccz4_vars->number_of_grid_points(),
      "Mismatch " << radial_extents * ylm.physical_size() << " must equal "
                  << ccz4_vars->number_of_grid_points());
  if (temp_storage->number_of_grid_points() <=
      radial_extents * ylm.spectral_size()) {
    temp_storage->initialize(radial_extents * ylm.spectral_size());
  }

  // Memory aliasing pattern (same as CSW):
  // - ccz4_spectral_vars: aliased into temp_storage (spectral size)
  // - temp_grid_vars: aliased into ccz4_vars (physical size)
  // - temp_ccz4_vars: aliased into temp_storage (physical size, smaller)
  // Do not use any pair simultaneously.
  Variables<filter_detail::ccz4_vars_list<Frame::Grid>> ccz4_spectral_vars(
      temp_storage->data(), temp_storage->size());
  Variables<filter_detail::ccz4_vars_list<Frame::Grid>> temp_grid_vars(
      ccz4_vars->data(), ccz4_vars->size());
  ASSERT(ccz4_vars->size() <= temp_storage->size(),
         "Should have " << ccz4_vars->size() << " <= " << temp_storage->size());
  Variables<filter_detail::ccz4_vars_list<Frame::Grid>> temp_ccz4_vars(
      temp_storage->data(), ccz4_vars->size());

  // 1. Multiply by Jacobians to get into (mostly) grid frame.
  // src: ccz4_vars
  // dest: temp_ccz4_vars
  filter_detail::transform_spatial_tensors_to_different_frame_without_hessians<
      Frame::Inertial, Frame::Grid>(make_not_null(&temp_ccz4_vars), *ccz4_vars,
                                    jac_inertial_to_grid, jac_grid_to_inertial);

  // 1a. Copy
  // src: temp_ccz4_vars
  // dest: temp_grid_vars
  std::memcpy(temp_grid_vars.data(), temp_ccz4_vars.data(),
              temp_ccz4_vars.size() * sizeof(double));

  // 2. Nodal to modal transformation.
  // src: temp_grid_vars
  // dest: ccz4_spectral_vars
  ylm::TensorYlm::filter_detail::nodal_to_modal_ylm(
      make_not_null(&ccz4_spectral_vars), temp_grid_vars, ylm, radial_extents);

  // 3. Filter
  // src: ccz4_spectral_vars
  // dest: ccz4_spectral_vars
  // using temp_grid_vars as temp storage for each tensor
  tmpl::for_each<filter_detail::ccz4_vars_list<Frame::Grid>>(
      [&ccz4_spectral_vars, &temp_grid_vars, radial_extents, &filter_matrix_i,
       &filter_matrix_ii,
       &filter_matrix_scalar]<class Tag>(const tmpl::type_<Tag> /*meta*/) {
        (void)radial_extents;
        constexpr size_t num_independent_components =
            Tag::type::structure::size();
        ASSERT(ccz4_spectral_vars.number_of_grid_points() *
                       num_independent_components <=
                   temp_grid_vars.size(),
               "Insufficient size: must have "
                   << ccz4_spectral_vars.number_of_grid_points() *
                          num_independent_components
                   << " <= " << temp_grid_vars.size());

        Variables<tmpl::list<Tag>> dest_tensor(
            temp_grid_vars.data(),
            ccz4_spectral_vars.number_of_grid_points() *
                num_independent_components);

        // Delta term
        get<Tag>(dest_tensor) = get<Tag>(ccz4_spectral_vars);

        const gsl::span<double> src(
            get<Tag>(ccz4_spectral_vars)[0].data(),
            num_independent_components *
                ccz4_spectral_vars.number_of_grid_points());
        gsl::span<double> dest(
            get<Tag>(dest_tensor)[0].data(),
            num_independent_components * dest_tensor.number_of_grid_points());
        const size_t stride = radial_extents;
        for (size_t offset = 0; offset < stride; ++offset) {
          // Dispatch by tensor symmetry:
          // Symmetry<1> for tnsr::I (and tnsr::i) -> filter_matrix_i
          // Symmetry<1,1> for tnsr::ii -> filter_matrix_ii
          // default (scalars) -> filter_matrix_scalar
          if constexpr (std::is_same_v<typename Tag::type::structure::symmetry,
                                       Symmetry<1>>) {
            filter_matrix_i.increment_multiply_on_right(
                make_not_null(&dest), offset, stride, src, offset, stride);
          } else if constexpr (std::is_same_v<
                                   typename Tag::type::structure::symmetry,
                                   Symmetry<1, 1>>) {
            filter_matrix_ii.increment_multiply_on_right(
                make_not_null(&dest), offset, stride, src, offset, stride);
          } else {
            filter_matrix_scalar.increment_multiply_on_right(
                make_not_null(&dest), offset, stride, src, offset, stride);
          }
        }
        // Copy the result for this tensor back into ccz4_spectral_vars.
        get<Tag>(ccz4_spectral_vars) = get<Tag>(dest_tensor);
      });

  // 4. Modal to nodal transformation.
  // src: ccz4_spectral_vars
  // dest: temp_grid_vars
  ylm::TensorYlm::filter_detail::modal_to_nodal_ylm(
      make_not_null(&temp_grid_vars), ccz4_spectral_vars, ylm, radial_extents);

  // 4a. Copy
  // src: temp_grid_vars
  // dest: temp_ccz4_vars
  std::memcpy(temp_ccz4_vars.data(), temp_grid_vars.data(),
              temp_grid_vars.size() * sizeof(double));

  // 5. Multiply by Jacobians to get back into inertial frame.
  // src: temp_ccz4_vars
  // dest: ccz4_vars
  filter_detail::transform_spatial_tensors_to_different_frame_without_hessians<
      Frame::Grid, Frame::Inertial>(ccz4_vars, temp_ccz4_vars,
                                    jac_grid_to_inertial, jac_inertial_to_grid);
}

TensorYlmFilter::TensorYlmFilter() = default;

TensorYlmFilter::TensorYlmFilter(CkMigrateMessage* msg)
    : Filters::Filter(msg) {}

TensorYlmFilter::TensorYlmFilter(const TensorYlmFilter& rhs)
    : Filters::Filter(rhs),
      num_modes_to_kill_(rhs.num_modes_to_kill_),
      half_power_(rhs.half_power_) {}

TensorYlmFilter& TensorYlmFilter::operator=(const TensorYlmFilter& rhs) {
  if (this != &rhs) {
    num_modes_to_kill_ = rhs.num_modes_to_kill_;
    half_power_ = rhs.half_power_;
  }
  return *this;
}

TensorYlmFilter::TensorYlmFilter(TensorYlmFilter&& rhs)
    : Filters::Filter(std::move(rhs)),
      num_modes_to_kill_(rhs.num_modes_to_kill_),
      half_power_(std::move(rhs.half_power_)) {}

TensorYlmFilter& TensorYlmFilter::operator=(TensorYlmFilter&& rhs) {
  if (this != &rhs) {
    num_modes_to_kill_ = rhs.num_modes_to_kill_;
    half_power_ = std::move(rhs.half_power_);
  }
  return *this;
}

TensorYlmFilter::TensorYlmFilter(const size_t num_modes_to_kill,
                                 std::optional<size_t> half_power)
    : num_modes_to_kill_(num_modes_to_kill), half_power_(half_power) {}

void TensorYlmFilter::pup(PUP::er& p) {
  Filters::Filter::pup(p);
  p | num_modes_to_kill_;
  p | half_power_;
  // The filter matrices and temp storage are lazily initialized,
  // so we don't pup them.
}

void TensorYlmFilter::operator()(
    const gsl::not_null<
        Variables<filter_detail::ccz4_vars_list<Frame::Inertial>>*>
        ccz4_vars,
    const Mesh<3>& mesh,
    const InverseJacobian<DataVector, 3, Frame::Grid, Frame::Inertial>&
        jac_grid_to_inertial) const {
  if (mesh.basis(1) != Spectral::Basis::SphericalHarmonic) {
    return;
  }
  ASSERT(mesh.basis(2) == Spectral::Basis::SphericalHarmonic,
         "TensorYlmFilter requires spherical harmonic basis in both "
         "angular directions.");
  const size_t radial_extents = mesh.extents(0);
  const size_t l_max = mesh.extents(1) - 1;

  // Cache the filter matrices
  if (cached_l_max_ != l_max) {
    ylm::TensorYlm::fill_filter<Scalar<DataVector>::structure>(
        make_not_null(&filter_matrix_scalar_), l_max, num_modes_to_kill_,
        half_power_, normalization_);
    ylm::TensorYlm::fill_filter<tnsr::i<DataVector, 3>::structure>(
        make_not_null(&filter_matrix_i_), l_max, num_modes_to_kill_,
        half_power_, normalization_);
    ylm::TensorYlm::fill_filter<tnsr::ii<DataVector, 3>::structure>(
        make_not_null(&filter_matrix_ii_), l_max, num_modes_to_kill_,
        half_power_, normalization_);
    cached_l_max_ = l_max;
  }

  // Apply the filter
  const auto jac_inertial_to_grid =
      determinant_and_inverse(jac_grid_to_inertial).second;
  apply_tensor_ylm_filter(ccz4_vars, make_not_null(&temp_storage_),
                          jac_inertial_to_grid, jac_grid_to_inertial,
                          filter_matrix_scalar_, filter_matrix_i_,
                          filter_matrix_ii_, l_max, radial_extents);
}

bool operator==(const TensorYlmFilter& lhs, const TensorYlmFilter& rhs) {
  return lhs.num_modes_to_kill_ == rhs.num_modes_to_kill_ and
         lhs.half_power_ == rhs.half_power_;
}

bool operator!=(const TensorYlmFilter& lhs, const TensorYlmFilter& rhs) {
  return not(lhs == rhs);
}

PUP::able::PUP_ID TensorYlmFilter::my_PUP_ID = 0;  // NOLINT

// Explicit instantiations

namespace filter_detail {
template void transform_spatial_tensors_to_different_frame_without_hessians<
    Frame::Grid, Frame::Inertial>(
    gsl::not_null<Variables<ccz4_vars_list<Frame::Inertial>>*> dest,
    const Variables<ccz4_vars_list<Frame::Grid>>& src,
    const InverseJacobian<DataVector, 3, Frame::Grid, Frame::Inertial>&
        jac_for_lower,
    const InverseJacobian<DataVector, 3, Frame::Inertial, Frame::Grid>&
        inv_jac_for_upper);

template void transform_spatial_tensors_to_different_frame_without_hessians<
    Frame::Inertial, Frame::Grid>(
    gsl::not_null<Variables<ccz4_vars_list<Frame::Grid>>*> dest,
    const Variables<ccz4_vars_list<Frame::Inertial>>& src,
    const InverseJacobian<DataVector, 3, Frame::Inertial, Frame::Grid>&
        jac_for_lower,
    const InverseJacobian<DataVector, 3, Frame::Grid, Frame::Inertial>&
        inv_jac_for_upper);
}  // namespace filter_detail
}  // namespace Ccz4

namespace ylm::TensorYlm::filter_detail {
YLM_TENSORYLM_INSTANTIATE_MODAL_NODAL_TRANSFORMS(
    Ccz4::filter_detail::ccz4_vars_list<Frame::Grid>);
YLM_TENSORYLM_INSTANTIATE_MODAL_NODAL_TRANSFORMS(
    Ccz4::filter_detail::ccz4_vars_list<Frame::Inertial>);
}  // namespace ylm::TensorYlm::filter_detail
