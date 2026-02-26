// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/SoScalarWave/BoundaryCorrections/CgCollocation.hpp"

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
CgCollocation<Dim>::CgCollocation(CkMigrateMessage* msg)
    : BoundaryCorrection(msg) {}

template <size_t Dim>
std::unique_ptr<evolution::BoundaryCorrection> CgCollocation<Dim>::get_clone()
    const {
  return std::make_unique<CgCollocation>(*this);
}

template <size_t Dim>
void CgCollocation<Dim>::pup(PUP::er& p) {
  BoundaryCorrection::pup(p);
}

template <size_t Dim>
double CgCollocation<Dim>::dg_package_data(
    const gsl::not_null<Scalar<DataVector>*> packaged_dt_psi,
    const gsl::not_null<Scalar<DataVector>*> packaged_dt_pi,
    const gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_d_psi,

    const Scalar<DataVector>& /*psi*/, const Scalar<DataVector>& /*pi*/,

    const Scalar<DataVector>& dt_psi, const Scalar<DataVector>& dt_pi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& d_psi,

    const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
    const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
    /*mesh_velocity*/,
    const std::optional<Scalar<DataVector>>& /*normal_dot_mesh_velocity*/)
    const {
  // Simply package the time derivatives that are already on the face
  // (they were projected from volume via dg_package_data_temporary_tags)
  get(*packaged_dt_psi) = get(dt_psi);
  get(*packaged_dt_pi) = get(dt_pi);
  get(*packaged_normal_dot_d_psi) = 0.0;
  for (size_t d = 0; d < Dim; ++d) {
    // In Euclidean \partial_i\psi is the same as \partial^i\psi
    get(*packaged_normal_dot_d_psi) += normal_covector.get(d) * d_psi.get(d);
  }

  // CG doesn't need characteristic speeds for CFL condition
  return 0.0;
}

template <size_t Dim>
void CgCollocation<Dim>::dg_boundary_terms(
    const gsl::not_null<Scalar<DataVector>*> psi_boundary_correction,
    const gsl::not_null<Scalar<DataVector>*> pi_boundary_correction,

    const Scalar<DataVector>& dt_psi_int, const Scalar<DataVector>& dt_pi_int,
    const Scalar<DataVector>& normal_dot_d_psi_int,

    const Scalar<DataVector>& dt_psi_ext, const Scalar<DataVector>& dt_pi_ext,
    const Scalar<DataVector>& normal_dot_d_psi_ext,

    const dg::Formulation /*dg_formulation*/,
    const bool used_for_external_bc) const {
  // Apply penalty term based on difference in time derivatives
  // D_Psi = 0.5 * (dt_Psi_ext - dt_Psi_int)
  // D_Pi = 0.5 * (dt_Pi_ext - dt_Pi_int)

  if (used_for_external_bc) {
    get(*psi_boundary_correction) = 1.0 * (get(dt_psi_ext) - get(dt_psi_int));
    get(*pi_boundary_correction) = 1.0 * (get(dt_pi_ext) - get(dt_pi_int));
  } else {
    get(*psi_boundary_correction) = 0.0;
    get(*pi_boundary_correction) =
        -0.5 * (get(normal_dot_d_psi_ext) + get(normal_dot_d_psi_int));
  }
}

template <size_t Dim>
// NOLINTNEXTLINE
PUP::able::PUP_ID CgCollocation<Dim>::my_PUP_ID = 0;

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(_, data) template class CgCollocation<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM

}  // namespace SoScalarWave::BoundaryCorrections
