// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Ccz4/GradientProjection/BoundaryCorrections/UpwindGradientProjection.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <ostream>
#include <pup.h>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/SimpleSparseMatrix.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/Chart.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/LocalOperators.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/NormalOperator.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/PointwiseFields.hpp"
#include "Evolution/Systems/Ccz4/GradientProjection/PrepareChartFields.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackCache.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/TensorYlmFilter.hpp"
#include "Options/ParseError.hpp"
#include "Options/ParseOptions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"

namespace Ccz4::GradientProjection::BoundaryCorrections {
namespace {
// The native filters of the boundary-correction tensor structures for one L
// and number of removed bands: the scalar, vector and symmetric rank-two
// filters and the trace-free coordinate matrices of the shell velocity filter,
// and the general rank-two and last-pair-symmetric rank-three filters. Each
// holds Delta of I + Delta on the component-major Spherepack storage.
struct BoundaryFilters {
  BoundaryFilters(const size_t l_max, const size_t modes_to_kill)
      : spaces(&shell_filter_angular_spaces(l_max, modes_to_kill)) {
    ylm::TensorYlm::fill_filter<tnsr::ij<DataVector, 3>::structure>(
        make_not_null(&general_rank_two), l_max, modes_to_kill, std::nullopt,
        ylm::TensorYlm::CoefficientNormalization::Spherepack);
    ylm::TensorYlm::fill_filter<tnsr::ijj<DataVector, 3>::structure>(
        make_not_null(&rank_three), l_max, modes_to_kill, std::nullopt,
        ylm::TensorYlm::CoefficientNormalization::Spherepack);
  }

  const ShellFilterAngularSpaces* spaces;
  SimpleSparseMatrix general_rank_two{};
  SimpleSparseMatrix rank_three{};
};

const BoundaryFilters& boundary_filters(const size_t l_max,
                                        const size_t modes_to_kill) {
  // The slot for each L up to the Spherepack bound (the L range of
  // ylm::get_spherepack_cache) publishes the entry most recently requested at
  // that L, so the steady state of a run (one `modes_to_kill` per boundary
  // correction, a fixed L per face) is a single atomic load with no lock.
  // The locked map below owns every entry (the pointers stay valid) and
  // serves slot misses.
  static constexpr size_t maximum_slot_l_max = 150;
  static std::array<std::atomic<const BoundaryFilters*>, maximum_slot_l_max + 1>
      slots{};
  if (l_max <= maximum_slot_l_max) {
    const BoundaryFilters* const entry =
        gsl::at(slots, l_max).load(std::memory_order_acquire);
    if (entry != nullptr and
        entry->spaces->angular_modes_to_kill == modes_to_kill) {
      return *entry;
    }
  }
  static std::mutex mutex{};
  static std::map<std::pair<size_t, size_t>,
                  std::unique_ptr<const BoundaryFilters>>
      cache{};
  const std::lock_guard lock{mutex};
  auto& entry = cache[std::pair{l_max, modes_to_kill}];
  if (entry == nullptr) {
    entry = std::make_unique<const BoundaryFilters>(l_max, modes_to_kill);
  }
  if (l_max <= maximum_slot_l_max) {
    gsl::at(slots, l_max).store(entry.get(), std::memory_order_release);
  }
  return *entry;
}

// The Cartesian storage components of the filtered corrections, transformed
// together: the tnsr::ii storage of a_t, K_t and Theta_t, the four vectors
// GammaHat_t, b_t, (D_i phi)_t and (D_i alpha)_t, the tnsr::ij storage of
// (D_i beta^k)_t and the tnsr::ijj storage of (D_i H)_t.
using SymmetricRankTwo = tnsr::ii<DataVector, 3>::structure;
using GeneralRankTwo = tnsr::ij<DataVector, 3>::structure;
using RankThree = tnsr::ijj<DataVector, 3>::structure;
constexpr size_t symmetric_offset = 0;
constexpr size_t scalar_offset = symmetric_offset + SymmetricRankTwo::size();
constexpr size_t vector_offset = scalar_offset + 2;
constexpr size_t general_offset = vector_offset + 4 * 3;
constexpr size_t rank_three_offset = general_offset + GeneralRankTwo::size();
constexpr size_t number_of_storage_fields =
    rank_three_offset + RankThree::size();

// Applies I + Delta to `groups` tensors of `components` components whose
// Spherepack coefficients are spectral[f + F k], F = number_of_storage_fields,
// for the field f = offset + c + components g of the component c of the
// tensor g, through the component-major layout of `delta`.
void apply_native_filter(const gsl::not_null<std::vector<double>*> spectral,
                         const gsl::not_null<std::vector<double>*> work,
                         const gsl::not_null<std::vector<double>*> filtered,
                         const SimpleSparseMatrix& delta,
                         const size_t spectral_size, const size_t offset,
                         const size_t components, const size_t groups) {
  constexpr size_t stride = number_of_storage_fields;
  for (size_t c = 0; c < components; ++c) {
    for (size_t g = 0; g < groups; ++g) {
      const size_t field = offset + c + components * g;
      for (size_t k = 0; k < spectral_size; ++k) {
        (*work)[g + groups * (k + spectral_size * c)] =
            (*spectral)[field + stride * k];
      }
    }
  }
  const size_t size = groups * spectral_size * components;
  std::copy(work->begin(),
            std::next(work->begin(), static_cast<std::ptrdiff_t>(size)),
            filtered->begin());
  for (size_t g = 0; g < groups; ++g) {
    delta.increment_multiply_on_right(filtered, g, groups, *work, g, groups);
  }
  for (size_t c = 0; c < components; ++c) {
    for (size_t g = 0; g < groups; ++g) {
      const size_t field = offset + c + components * g;
      for (size_t k = 0; k < spectral_size; ++k) {
        (*spectral)[field + stride * k] =
            (*filtered)[g + groups * (k + spectral_size * c)];
      }
    }
  }
}
}  // namespace

double external_upwind_factor(const ExternalCorrection external_correction) {
  switch (external_correction) {
    case ExternalCorrection::Central:
      return 0.0;
    case ExternalCorrection::Upwind:
      return 1.0;
    case ExternalCorrection::DoubledDissipation:
      return 2.0;
    default:
      ERROR("Invalid ExternalCorrection value "
            << static_cast<int>(external_correction));
  }
}

std::ostream& operator<<(std::ostream& os,
                         const ExternalCorrection external_correction) {
  switch (external_correction) {
    case ExternalCorrection::Central:
      return os << "Central";
    case ExternalCorrection::Upwind:
      return os << "Upwind";
    case ExternalCorrection::DoubledDissipation:
      return os << "DoubledDissipation";
    default:
      ERROR("Invalid ExternalCorrection value "
            << static_cast<int>(external_correction));
  }
}

UpwindGradientProjection::UpwindGradientProjection(
    const ExternalCorrection external_correction,
    const double propagating_scalar_penalty,
    const double stationary_gradient_scalar_penalty,
    const double gamma_hat_minus_b_scalar_penalty,
    const std::optional<size_t> shell_boundary_modes_to_kill)
    : external_correction_(external_correction),
      propagating_scalar_penalty_(propagating_scalar_penalty),
      stationary_gradient_scalar_penalty_(stationary_gradient_scalar_penalty),
      gamma_hat_minus_b_scalar_penalty_(gamma_hat_minus_b_scalar_penalty),
      shell_boundary_modes_to_kill_(shell_boundary_modes_to_kill) {
  if (not(propagating_scalar_penalty_ >= 0.0 and
          std::isfinite(propagating_scalar_penalty_) and
          stationary_gradient_scalar_penalty_ >= 0.0 and
          std::isfinite(stationary_gradient_scalar_penalty_) and
          gamma_hat_minus_b_scalar_penalty_ >= 0.0 and
          std::isfinite(gamma_hat_minus_b_scalar_penalty_))) {
    ERROR(
        "The scalar jump penalties must be finite and nonnegative, but are "
        "(propagating, stationary gradient, GammaHat-b) = ("
        << propagating_scalar_penalty_ << ", "
        << stationary_gradient_scalar_penalty_ << ", "
        << gamma_hat_minus_b_scalar_penalty_ << ").");
  }
}

UpwindGradientProjection::UpwindGradientProjection(CkMigrateMessage* msg)
    : BoundaryCorrection(msg) {}

std::unique_ptr<evolution::BoundaryCorrection>
UpwindGradientProjection::get_clone() const {
  return std::make_unique<UpwindGradientProjection>(*this);
}

void UpwindGradientProjection::pup(PUP::er& p) {
  BoundaryCorrection::pup(p);
  p | external_correction_;
  p | propagating_scalar_penalty_;
  p | stationary_gradient_scalar_penalty_;
  p | gamma_hat_minus_b_scalar_penalty_;
  p | shell_boundary_modes_to_kill_;
}

double UpwindGradientProjection::dg_package_data(
    const gsl::not_null<Tags::PackagedVelocities::type*> packaged_w,
    const gsl::not_null<Tags::PackagedGradients::type*> packaged_q,
    const gsl::not_null<Tags::PackagedPotentials::type*> packaged_u,
    const gsl::not_null<tnsr::i<DataVector, 3>*> packaged_normal_covector,
    const tnsr::ii<DataVector, 3>& conformal_metric,
    const Scalar<DataVector>& conformal_factor,
    const tnsr::ii<DataVector, 3>& a_tilde,
    const Scalar<DataVector>& trace_extrinsic_curvature,
    const Scalar<DataVector>& theta, const tnsr::I<DataVector, 3>& gamma_hat,
    const Scalar<DataVector>& lapse, const tnsr::I<DataVector, 3>& shift,
    const tnsr::I<DataVector, 3>& b, const Tags::PotentialGradients::type& q,
    const tnsr::i<DataVector, 3>& normal_covector,
    const std::optional<tnsr::I<DataVector, 3>>& /*mesh_velocity*/,
    const std::optional<Scalar<DataVector>>& /*normal_dot_mesh_velocity*/,
    const Direction<3>& /*face_direction*/) const {
  *packaged_q = q;
  *packaged_normal_covector = normal_covector;
  double largest_speed = 0.0;
  for (size_t s = 0; s < get(lapse).size(); ++s) {
    PhysicalPointState state{};
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        state.conformal_metric(i, j) = conformal_metric.get(i, j)[s];
        state.a_tilde(i, j) = a_tilde.get(i, j)[s];
      }
      gsl::at(state.gamma_hat, i) = gamma_hat.get(i)[s];
      gsl::at(state.shift, i) = shift.get(i)[s];
      gsl::at(state.b, i) = b.get(i)[s];
    }
    state.conformal_factor = get(conformal_factor)[s];
    state.trace_extrinsic_curvature = get(trace_extrinsic_curvature)[s];
    state.theta = get(theta)[s];
    state.lapse = get(lapse)[s];
    PotentialVector u_s{};
    VelocityVector w_s{};
    LogMetricChart chart{};
    checked_chart_from_physical(make_not_null(&u_s), make_not_null(&w_s),
                                make_not_null(&chart), state, "face");
    set_potentials_at(packaged_u, s, u_s);
    set_velocities_at(packaged_w, s, w_s);
    largest_speed =
        std::max(largest_speed,
                 largest_characteristic_speed(
                     chart, u_s,
                     {{get<0>(normal_covector)[s], get<1>(normal_covector)[s],
                       get<2>(normal_covector)[s]}},
                     ::Ccz4::fd::System::f));
  }
  return largest_speed;
}

template <bool ForExternalBoundary>
void UpwindGradientProjection::dg_boundary_terms(
    const gsl::not_null<Tags::VelocityTarget::type*> w_correction,
    const gsl::not_null<Tags::GradientTarget::type*> q_correction,
    const gsl::not_null<Tags::PotentialRateTarget::type*> u_correction,
    const Tags::PackagedVelocities::type& w_int,
    const Tags::PackagedGradients::type& q_int,
    const Tags::PackagedPotentials::type& u_int,
    const tnsr::i<DataVector, 3>& normal_covector_int,
    const Tags::PackagedVelocities::type& w_ext,
    const Tags::PackagedGradients::type& q_ext,
    const Tags::PackagedPotentials::type& u_ext,
    const tnsr::i<DataVector, 3>& /*normal_covector_ext*/,
    const dg::Formulation dg_formulation) const {
  if (dg_formulation != dg::Formulation::StrongInertial) {
    ERROR("UpwindGradientProjection supports only the strong formulation.");
  }
  const double upwind_factor =
      ForExternalBoundary ? external_upwind_factor(external_correction_) : 1.0;
  // An analytic exterior state need not match outgoing fields. Penalizing
  // their jumps would impose extra boundary data beyond ordinary upwinding.
  const double sigma_p =
      ForExternalBoundary ? 0.0 : propagating_scalar_penalty_;
  const double sigma_t =
      ForExternalBoundary ? 0.0 : stationary_gradient_scalar_penalty_;
  const double sigma_g =
      ForExternalBoundary ? 0.0 : gamma_hat_minus_b_scalar_penalty_;
  const bool apply_dissipation = upwind_factor != 0.0 or sigma_p != 0.0 or
                                 sigma_t != 0.0 or sigma_g != 0.0;
  const size_t number_of_points = get<0>(u_int).size();
  for (size_t s = 0; s < number_of_points; ++s) {
    const PotentialVector u_int_s = potentials_at(u_int, s);
    const PotentialVector u_ext_s = potentials_at(u_ext, s);
    const VelocityVector w_int_s = velocities_at(w_int, s);
    const VelocityVector w_ext_s = velocities_at(w_ext, s);
    const GradientVector q_int_s = gradients_at(q_int, s);
    const GradientVector q_ext_s = gradients_at(q_ext, s);
    const std::array<double, 3> normal{{get<0>(normal_covector_int)[s],
                                        get<1>(normal_covector_int)[s],
                                        get<2>(normal_covector_int)[s]}};
    PotentialVector u_average{};
    for (size_t a = 0; a < number_of_potentials; ++a) {
      gsl::at(u_average, a) = 0.5 * (gsl::at(u_int_s, a) + gsl::at(u_ext_s, a));
    }
    VelocityVector jump_w{};
    for (size_t a = 0; a < number_of_velocities; ++a) {
      gsl::at(jump_w, a) = gsl::at(w_ext_s, a) - gsl::at(w_int_s, a);
    }
    GradientVector jump_q{};
    for (size_t i = 0; i < 3; ++i) {
      for (size_t a = 0; a < number_of_potentials; ++a) {
        gsl::at(gsl::at(jump_q, i), a) =
            gsl::at(gsl::at(q_ext_s, i), a) - gsl::at(gsl::at(q_int_s, i), a);
      }
    }
    const LogMetricChart chart{
        stf_decode(&u_average[PotentialIndex::log_metric])};
    const NormalSubspaceVector central = normal_flux(
        chart, u_average, normal, jump_w, jump_q, ::Ccz4::fd::System::f);
    VelocityVector dissipation_w{};
    GradientVector dissipation_q{};
    if (apply_dissipation) {
      const NormalCharacteristics characteristics{chart, u_average, normal,
                                                  ::Ccz4::fd::System::f};
      characteristics.apply_upwind_and_penalties(
          make_not_null(&dissipation_w), make_not_null(&dissipation_q), normal,
          jump_w, jump_q, central, upwind_factor, sigma_p, sigma_t, sigma_g);
    }
    VelocityVector correction_w{};
    for (size_t a = 0; a < number_of_velocities; ++a) {
      gsl::at(correction_w, a) =
          -0.5 * (gsl::at(central, a) + gsl::at(dissipation_w, a));
    }
    GradientVector correction_q{};
    for (size_t i = 0; i < 3; ++i) {
      for (size_t a = 0; a < number_of_potentials; ++a) {
        gsl::at(gsl::at(correction_q, i), a) =
            -0.5 *
            (gsl::at(normal, i) * gsl::at(central, number_of_velocities + a) +
             gsl::at(gsl::at(dissipation_q, i), a));
      }
    }
    PotentialVector correction_u{};
    if constexpr (not ForExternalBoundary) {
      for (size_t a = 0; a < number_of_potentials; ++a) {
        gsl::at(correction_u, a) =
            -0.5 * (gsl::at(u_ext_s, a) - gsl::at(u_int_s, a));
      }
    }
    set_velocities_at(w_correction, s, correction_w);
    set_gradients_at(q_correction, s, correction_q);
    set_potentials_at(u_correction, s, correction_u);
  }
}

void UpwindGradientProjection::dg_filter_boundary_correction(
    const gsl::not_null<Tags::RhsTargets::type*> corrections,
    const Mesh<2>& face_mesh,
    const bool has_nonconforming_block_connection) const {
  if (not shell_boundary_modes_to_kill_.has_value() or
      not has_nonconforming_block_connection or
      face_mesh.basis(0) != Spectral::Basis::SphericalHarmonic or
      face_mesh.basis(1) != Spectral::Basis::SphericalHarmonic) {
    return;
  }
  const size_t modes_to_kill = *shell_boundary_modes_to_kill_;
  const size_t l_max = face_mesh.extents(0) - 1;
  if (modes_to_kill > l_max) {
    ERROR("The number of removed angular bands ShellBoundaryModesToKill = "
          << modes_to_kill
          << " of the shell boundary-correction filter exceeds L = " << l_max
          << " of the face mesh " << face_mesh);
  }
  if (face_mesh.extents(1) != 2 * l_max + 1 or
      face_mesh.quadrature(0) != Spectral::Quadrature::Gauss or
      face_mesh.quadrature(1) != Spectral::Quadrature::Equiangular or
      corrections->number_of_grid_points() !=
          face_mesh.number_of_grid_points()) {
    ERROR(
        "The shell boundary-correction filter requires a spherical-harmonic "
        "face mesh (L + 1, 2 L + 1) with Gauss and Equiangular quadratures "
        "and one correction per face point, but got the face mesh "
        << face_mesh << " and " << corrections->number_of_grid_points()
        << " correction points");
  }
  const ylm::Spherepack& ylm = ylm::get_spherepack_cache(l_max);
  const BoundaryFilters& filters = boundary_filters(l_max, modes_to_kill);
  const ShellFilterAngularSpaces& spaces = *filters.spaces;
  const Matrix& decode = spaces.stf_decode_matrix;
  const Matrix& encode = spaces.stf_encode_matrix;
  const size_t n = ylm.physical_size();
  const size_t spectral_size = ylm.spectral_size();
  constexpr size_t stride = number_of_storage_fields;
  auto& w = get<Tags::VelocityTarget>(*corrections);
  auto& q = get<Tags::GradientTarget>(*corrections);

  std::array<DataVector*, 2> scalars{
      {&w.get(VelocityIndex::trace_extrinsic_curvature),
       &w.get(VelocityIndex::theta)}};
  std::array<std::array<DataVector*, 3>, 4> vectors{};
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(vectors[0], i) = &w.get(VelocityIndex::gamma_hat + i);
    gsl::at(vectors[1], i) = &w.get(VelocityIndex::b + i);
    gsl::at(vectors[2], i) = &q.get(i, PotentialIndex::conformal_factor);
    gsl::at(vectors[3], i) = &q.get(i, PotentialIndex::lapse);
  }
  constexpr auto general_indices = GeneralRankTwo::storage_to_tensor_index();
  constexpr auto rank_three_indices = RankThree::storage_to_tensor_index();
  std::array<size_t, RankThree::size()> symmetric_storage{};
  for (size_t c = 0; c < RankThree::size(); ++c) {
    gsl::at(symmetric_storage, c) = SymmetricRankTwo::get_storage_index(
        rank_three_indices[c][1], rank_three_indices[c][2]);
  }

  // The nodal storage components, nodal[f + F j].
  std::vector<double> nodal(stride * n);
  for (size_t j = 0; j < n; ++j) {
    for (size_t s = 0; s < SymmetricRankTwo::size(); ++s) {
      double value = 0.0;
      for (size_t a = 0; a < 5; ++a) {
        value += decode(s, a) * w.get(VelocityIndex::a + a)[j];
      }
      nodal[symmetric_offset + s + stride * j] = value;
    }
    for (size_t g = 0; g < 2; ++g) {
      nodal[scalar_offset + g + stride * j] = (*gsl::at(scalars, g))[j];
    }
    for (size_t g = 0; g < 4; ++g) {
      for (size_t i = 0; i < 3; ++i) {
        nodal[vector_offset + i + 3 * g + stride * j] =
            (*gsl::at(gsl::at(vectors, g), i))[j];
      }
    }
    for (size_t c = 0; c < GeneralRankTwo::size(); ++c) {
      nodal[general_offset + c + stride * j] =
          q.get(general_indices[c][0],
                PotentialIndex::shift + general_indices[c][1])[j];
    }
    for (size_t c = 0; c < RankThree::size(); ++c) {
      double value = 0.0;
      for (size_t a = 0; a < 5; ++a) {
        value +=
            decode(gsl::at(symmetric_storage, c), a) *
            q.get(rank_three_indices[c][0], PotentialIndex::log_metric + a)[j];
      }
      nodal[rank_three_offset + c + stride * j] = value;
    }
  }

  // One batched analysis, the native filters on the coefficients and one
  // batched synthesis.
  std::vector<double> spectral(stride * spectral_size);
  ylm.phys_to_spec_all_offsets(make_not_null(spectral.data()),
                               make_not_null(nodal.data()), stride);
  std::vector<double> work(RankThree::size() * spectral_size);
  std::vector<double> filtered(work.size());
  const auto filter = [&](const SimpleSparseMatrix& delta, const size_t offset,
                          const size_t components, const size_t groups) {
    apply_native_filter(make_not_null(&spectral), make_not_null(&work),
                        make_not_null(&filtered), delta, spectral_size, offset,
                        components, groups);
  };
  filter(spaces.native_filters[2], symmetric_offset, SymmetricRankTwo::size(),
         1);
  filter(spaces.native_filters[0], scalar_offset, 1, 2);
  filter(spaces.native_filters[1], vector_offset, 3, 4);
  filter(filters.general_rank_two, general_offset, GeneralRankTwo::size(), 1);
  filter(filters.rank_three, rank_three_offset, RankThree::size(), 1);
  ylm.spec_to_phys_all_offsets(make_not_null(nodal.data()),
                               make_not_null(spectral.data()), stride);

  // Back to the targets.
  for (size_t j = 0; j < n; ++j) {
    for (size_t a = 0; a < 5; ++a) {
      double value = 0.0;
      for (size_t s = 0; s < SymmetricRankTwo::size(); ++s) {
        value += encode(a, s) * nodal[symmetric_offset + s + stride * j];
      }
      w.get(VelocityIndex::a + a)[j] = value;
    }
    for (size_t g = 0; g < 2; ++g) {
      (*gsl::at(scalars, g))[j] = nodal[scalar_offset + g + stride * j];
    }
    for (size_t g = 0; g < 4; ++g) {
      for (size_t i = 0; i < 3; ++i) {
        (*gsl::at(gsl::at(vectors, g), i))[j] =
            nodal[vector_offset + i + 3 * g + stride * j];
      }
    }
    for (size_t c = 0; c < GeneralRankTwo::size(); ++c) {
      q.get(general_indices[c][0],
            PotentialIndex::shift + general_indices[c][1])[j] =
          nodal[general_offset + c + stride * j];
    }
    for (size_t i = 0; i < 3; ++i) {
      for (size_t a = 0; a < 5; ++a) {
        q.get(i, PotentialIndex::log_metric + a)[j] = 0.0;
      }
    }
    for (size_t c = 0; c < RankThree::size(); ++c) {
      const size_t i = rank_three_indices[c][0];
      for (size_t a = 0; a < 5; ++a) {
        q.get(i, PotentialIndex::log_metric + a)[j] +=
            encode(a, gsl::at(symmetric_storage, c)) *
            nodal[rank_three_offset + c + stride * j];
      }
    }
  }
}

// NOLINTNEXTLINE
PUP::able::PUP_ID UpwindGradientProjection::my_PUP_ID = 0;

#define INSTANTIATE(EXTERNAL)                                               \
  template void UpwindGradientProjection::dg_boundary_terms<EXTERNAL>(      \
      gsl::not_null<Tags::VelocityTarget::type*>,                           \
      gsl::not_null<Tags::GradientTarget::type*>,                           \
      gsl::not_null<Tags::PotentialRateTarget::type*>,                      \
      const Tags::PackagedVelocities::type&,                                \
      const Tags::PackagedGradients::type&,                                 \
      const Tags::PackagedPotentials::type&, const tnsr::i<DataVector, 3>&, \
      const Tags::PackagedVelocities::type&,                                \
      const Tags::PackagedGradients::type&,                                 \
      const Tags::PackagedPotentials::type&, const tnsr::i<DataVector, 3>&, \
      dg::Formulation) const;
INSTANTIATE(false)
INSTANTIATE(true)
#undef INSTANTIATE
}  // namespace Ccz4::GradientProjection::BoundaryCorrections

template <>
Ccz4::GradientProjection::BoundaryCorrections::ExternalCorrection
Options::create_from_yaml<
    Ccz4::GradientProjection::BoundaryCorrections::ExternalCorrection>::
    create<void>(const Options::Option& options) {
  using ExternalCorrection =
      Ccz4::GradientProjection::BoundaryCorrections::ExternalCorrection;
  const auto value = options.parse_as<std::string>();
  for (const auto external_correction :
       {ExternalCorrection::Central, ExternalCorrection::Upwind,
        ExternalCorrection::DoubledDissipation}) {
    if (value == get_output(external_correction)) {
      return external_correction;
    }
  }
  PARSE_ERROR(options.context(),
              "Failed to convert \""
                  << value
                  << "\" to ExternalCorrection. Must be one of Central, "
                     "Upwind, or DoubledDissipation.");
}
