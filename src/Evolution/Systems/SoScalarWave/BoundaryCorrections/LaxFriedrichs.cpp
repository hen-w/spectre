// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/SoScalarWave/BoundaryCorrections/LaxFriedrichs.hpp"

#include <iostream>
#include <memory>

#include <optional>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace SoScalarWave::BoundaryCorrections {

template <size_t Dim>
LaxFriedrichs<Dim>::LaxFriedrichs(CkMigrateMessage* msg)
    : BoundaryCorrection(msg) {}

template <size_t Dim>
LaxFriedrichs<Dim>::LaxFriedrichs(const double tau1, const double tau2)
    : tau1_(tau1), tau2_(tau2) {}

template <size_t Dim>
std::unique_ptr<evolution::BoundaryCorrection> LaxFriedrichs<Dim>::get_clone()
    const {
  return std::make_unique<LaxFriedrichs>(*this);
}

template <size_t Dim>
void LaxFriedrichs<Dim>::pup(PUP::er& p) {
  BoundaryCorrection::pup(p);
  p | tau1_;
  p | tau2_;
}

template <size_t Dim>
double LaxFriedrichs<Dim>::dg_package_data(
    const gsl::not_null<Scalar<DataVector>*> packaged_pi,
    const gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_phi,

    const Scalar<DataVector>& /*psi*/, const Scalar<DataVector>& pi,
    const Scalar<DataVector>& /*boundary_psi*/,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& phi,

    const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
    const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
    /*mesh_velocity*/,
    const std::optional<Scalar<DataVector>>& /*normal_dot_mesh_velocity*/)
    const {
  get(*packaged_pi) = get(pi);
  get(*packaged_normal_dot_phi) = 0.0;
  for (size_t d = 0; d < Dim; ++d) {
    get(*packaged_normal_dot_phi) += normal_covector.get(d) * phi.get(d);
  }

  // characteristic speeds for CFL; not implemented
  return 0.0;
}

template <size_t Dim>
template <bool ForExternalBoundary>
void LaxFriedrichs<Dim>::dg_boundary_terms(
    const gsl::not_null<Scalar<DataVector>*> psi_boundary_correction,
    const gsl::not_null<Scalar<DataVector>*> pi_boundary_correction,
    const gsl::not_null<Scalar<DataVector>*> boundary_psi_boundary_correction,

    const Scalar<DataVector>& pi_int,
    const Scalar<DataVector>& normal_dot_phi_int,

    const Scalar<DataVector>& pi_ext,
    const Scalar<DataVector>& normal_dot_phi_ext,

    dg::Formulation /*dg_formulation*/) const {
  get(*psi_boundary_correction) = 0.0;
  // Minus sign below to account for minus sign in LiftFlux.hpp
  get(*pi_boundary_correction) =
      -0.5 * (get(normal_dot_phi_int) + get(normal_dot_phi_ext)) -
      tau1_ * 0.5 * (get(pi_ext) - get(pi_int));
  get(*boundary_psi_boundary_correction) = 0.0;
}

template <size_t Dim>
double LaxFriedrichs<Dim>::dg_auxiliary_package_data(
    const gsl::not_null<Scalar<DataVector>*> packaged_psi,
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
        psi_times_normal,

    const Scalar<DataVector>& psi, const Scalar<DataVector>& /*pi*/,
    const Scalar<DataVector>& /*boundary_psi*/,

    const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
    const std::optional<
        tnsr::I<DataVector, Dim, Frame::Inertial>>& /*mesh_velocity*/,
    const std::optional<Scalar<DataVector>>& /*normal_dot_mesh_velocity*/)
    const {
  get(*packaged_psi) = get(psi);
  for (size_t d = 0; d < Dim; ++d) {
    psi_times_normal->get(d) = get(psi) * normal_covector.get(d);
  }
  return 0.0;
}

template <size_t Dim>
void LaxFriedrichs<Dim>::dg_auxiliary_boundary_terms(
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
        phi_boundary_correction,

    const Scalar<DataVector>& psi_int,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& psi_times_normal_int,

    const Scalar<DataVector>& psi_ext,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& psi_times_normal_ext,

    dg::Formulation /*dg_formulation*/) const {
  for (size_t d = 0; d < Dim; ++d) {
    phi_boundary_correction->get(d) =
        0.5 * (psi_times_normal_int.get(d) + psi_times_normal_ext.get(d)) -
        0.5 * tau2_ * (get(psi_ext) - get(psi_int));
  }
}

template <size_t Dim>
// NOLINTNEXTLINE
PUP::able::PUP_ID LaxFriedrichs<Dim>::my_PUP_ID = 0;

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(_, data)                                                \
  template class LaxFriedrichs<DIM(data)>;                                    \
  template void LaxFriedrichs<DIM(data)>::dg_boundary_terms<false>(           \
      gsl::not_null<Scalar<DataVector>*>, gsl::not_null<Scalar<DataVector>*>, \
      gsl::not_null<Scalar<DataVector>*>, const Scalar<DataVector>&,          \
      const Scalar<DataVector>&, const Scalar<DataVector>&,                   \
      const Scalar<DataVector>&, dg::Formulation) const;                      \
  template void LaxFriedrichs<DIM(data)>::dg_boundary_terms<true>(            \
      gsl::not_null<Scalar<DataVector>*>, gsl::not_null<Scalar<DataVector>*>, \
      gsl::not_null<Scalar<DataVector>*>, const Scalar<DataVector>&,          \
      const Scalar<DataVector>&, const Scalar<DataVector>&,                   \
      const Scalar<DataVector>&, dg::Formulation) const;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM

}  // namespace SoScalarWave::BoundaryCorrections
