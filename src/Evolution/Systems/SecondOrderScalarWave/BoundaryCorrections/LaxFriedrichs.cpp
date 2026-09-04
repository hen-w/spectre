// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/SecondOrderScalarWave/BoundaryCorrections/LaxFriedrichs.hpp"

#include <limits>
#include <memory>
#include <optional>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace SecondOrderScalarWave::BoundaryCorrections {

template <size_t Dim>
LaxFriedrichs<Dim>::LaxFriedrichs(CkMigrateMessage* msg)
    : BoundaryCorrection(msg) {}

template <size_t Dim>
LaxFriedrichs<Dim>::LaxFriedrichs(const double tau) : tau_(tau) {}

template <size_t Dim>
std::unique_ptr<evolution::BoundaryCorrection> LaxFriedrichs<Dim>::get_clone()
    const {
  return std::make_unique<LaxFriedrichs>(*this);
}

template <size_t Dim>
void LaxFriedrichs<Dim>::pup(PUP::er& p) {
  BoundaryCorrection::pup(p);
  p | tau_;
}

template <size_t Dim>
double LaxFriedrichs<Dim>::dg_package_data(
    const gsl::not_null<Scalar<DataVector>*> packaged_pi,
    const gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_phi,
    const gsl::not_null<Scalar<DataVector>*> packaged_psi,
    const gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_mesh_velocity,

    const Scalar<DataVector>& psi, const Scalar<DataVector>& pi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& phi,

    const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
    const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
    /*mesh_velocity*/,
    const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity) const {
  get(*packaged_pi) = get(pi);
  get(*packaged_normal_dot_phi) = 0.0;
  for (size_t d = 0; d < Dim; ++d) {
    get(*packaged_normal_dot_phi) += normal_covector.get(d) * phi.get(d);
  }
  get(*packaged_psi) = get(psi);
  if (normal_dot_mesh_velocity.has_value()) {
    get(*packaged_normal_dot_mesh_velocity) = get(*normal_dot_mesh_velocity);
    return 1.0 + max(abs(get(*normal_dot_mesh_velocity)));
  }
  get(*packaged_normal_dot_mesh_velocity) = 0.0;
  return 1.0;
}

template <size_t Dim>
void LaxFriedrichs<Dim>::dg_boundary_terms(
    const gsl::not_null<Scalar<DataVector>*> psi_boundary_correction,
    const gsl::not_null<Scalar<DataVector>*> pi_boundary_correction,

    const Scalar<DataVector>& pi_int,
    const Scalar<DataVector>& normal_dot_phi_int,
    const Scalar<DataVector>& psi_int,
    const Scalar<DataVector>& normal_dot_mesh_velocity_int,

    const Scalar<DataVector>& pi_ext,
    const Scalar<DataVector>& normal_dot_phi_ext,
    const Scalar<DataVector>& psi_ext,
    const Scalar<DataVector>& normal_dot_mesh_velocity_ext,

    dg::Formulation /*dg_formulation*/) const {
  // The advection-consistency terms: v^i contracted with the central-flux
  // correction that upgrades the raw spectral derivative to the
  // LDG-corrected one, applied to BOTH advected fields. For Psi the lift
  // converts the volume v^i d_i Psi term into v^i Phi_i; for Pi it
  // likewise replaces the raw d_i Pi in the volume advection term by its
  // centrally-corrected counterpart. Both terms are zero on a static mesh
  // (both packaged n.v vanish) and on continuous data
  // (n.v_ext = -n.v_int, field_ext = field_int).
  //
  // The penalty coefficient is tau times the largest characteristic-speed
  // magnitude over the modes and both sides of the interface. The grid-frame
  // speeds along a normal n are {-n.v, 1 - n.v, -1 - n.v}, so the largest
  // magnitude on a side is 1 + |n.v|; the mesh velocity is continuous across
  // the interface, so the interior value covers both sides.
  get(*psi_boundary_correction) =
      0.5 * (get(normal_dot_mesh_velocity_int) * get(psi_int) +
             get(normal_dot_mesh_velocity_ext) * get(psi_ext));
  get(*pi_boundary_correction) =
      -0.5 * (get(normal_dot_phi_int) + get(normal_dot_phi_ext)) +
      0.5 * (get(normal_dot_mesh_velocity_int) * get(pi_int) +
             get(normal_dot_mesh_velocity_ext) * get(pi_ext)) -
      tau_ * 0.5 * (1.0 + abs(get(normal_dot_mesh_velocity_int))) *
          (get(pi_ext) - get(pi_int));
}

template <size_t Dim>
double LaxFriedrichs<Dim>::dg_auxiliary_package_data(
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
        psi_times_normal,

    const Scalar<DataVector>& psi, const Scalar<DataVector>& /*pi*/,

    const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
    const std::optional<
        tnsr::I<DataVector, Dim, Frame::Inertial>>& /*mesh_velocity*/,
    const std::optional<Scalar<DataVector>>& /*normal_dot_mesh_velocity*/)
    const {
  for (size_t d = 0; d < Dim; ++d) {
    psi_times_normal->get(d) = get(psi) * normal_covector.get(d);
  }

  return std::numeric_limits<double>::signaling_NaN();
}

template <size_t Dim>
void LaxFriedrichs<Dim>::dg_auxiliary_boundary_terms(
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
        phi_boundary_correction,

    const tnsr::i<DataVector, Dim, Frame::Inertial>& psi_times_normal_int,

    const tnsr::i<DataVector, Dim, Frame::Inertial>& psi_times_normal_ext,

    dg::Formulation /*dg_formulation*/) const {
  for (size_t d = 0; d < Dim; ++d) {
    phi_boundary_correction->get(d) =
        0.5 * (psi_times_normal_int.get(d) + psi_times_normal_ext.get(d));
  }
}

template <size_t LocalDim>
bool operator==(const LaxFriedrichs<LocalDim>& lhs,
                const LaxFriedrichs<LocalDim>& rhs) {
  return lhs.tau_ == rhs.tau_;
}

template <size_t Dim>
bool operator!=(const LaxFriedrichs<Dim>& lhs, const LaxFriedrichs<Dim>& rhs) {
  return not(lhs == rhs);
}

template <size_t Dim>
// NOLINTNEXTLINE
PUP::able::PUP_ID LaxFriedrichs<Dim>::my_PUP_ID = 0;

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(_, data)                                   \
  template class LaxFriedrichs<DIM(data)>;                       \
  template bool operator==(const LaxFriedrichs<DIM(data)>& lhs,  \
                           const LaxFriedrichs<DIM(data)>& rhs); \
  template bool operator!=(const LaxFriedrichs<DIM(data)>& lhs,  \
                           const LaxFriedrichs<DIM(data)>& rhs);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM

}  // namespace SecondOrderScalarWave::BoundaryCorrections
