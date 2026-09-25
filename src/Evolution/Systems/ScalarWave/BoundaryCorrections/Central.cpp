// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ScalarWave/BoundaryCorrections/Central.hpp"

#include <cstddef>
#include <memory>
#include <optional>

#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace ScalarWave::BoundaryCorrections {
template <size_t Dim>
Central<Dim>::Central(CkMigrateMessage* msg) : BoundaryCorrection(msg) {}

template <size_t Dim>
std::unique_ptr<evolution::BoundaryCorrection> Central<Dim>::get_clone() const {
  return std::make_unique<Central>(*this);
}

template <size_t Dim>
void Central<Dim>::pup(PUP::er& p) {
  BoundaryCorrection::pup(p);
}

template <size_t Dim>
double Central<Dim>::dg_package_data(
    const gsl::not_null<Scalar<DataVector>*> packaged_pi,
    const gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_phi,
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
        packaged_normal_times_pi,

    const Scalar<DataVector>& /*psi*/, const Scalar<DataVector>& pi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& phi,

    const Scalar<DataVector>& /*constraint_gamma2*/,

    const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
    const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
        mesh_velocity,
    const std::optional<Scalar<DataVector>>& /*normal_dot_mesh_velocity*/,
    const Direction<Dim>& /*face_direction*/) const {
  ASSERT(not mesh_velocity.has_value(),
         "The Central boundary correction for the first-order ScalarWave "
         "system assumes a static mesh with unit wave speed, but a mesh "
         "velocity was supplied.");
  get(*packaged_pi) = get(pi);
  dot_product(packaged_normal_dot_phi, normal_covector, phi);
  for (size_t d = 0; d < Dim; ++d) {
    packaged_normal_times_pi->get(d) = normal_covector.get(d) * get(pi);
  }

  // Maximum characteristic speed for CFL: the scalar wave travels at unit
  // speed and the mesh is static (asserted above).
  return 1.0;
}

template <size_t Dim>
void Central<Dim>::dg_boundary_terms_impl(
    const gsl::not_null<Scalar<DataVector>*> psi_boundary_correction,
    const gsl::not_null<Scalar<DataVector>*> pi_boundary_correction,
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
        phi_boundary_correction,

    const Scalar<DataVector>& /*pi_int*/,
    const Scalar<DataVector>& normal_dot_phi_int,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_times_pi_int,

    const Scalar<DataVector>& /*pi_ext*/,
    const Scalar<DataVector>& normal_dot_phi_ext,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_times_pi_ext,

    const dg::Formulation dg_formulation) const {
  ASSERT(dg_formulation == dg::Formulation::StrongInertial,
         "The Central boundary correction for the first-order ScalarWave "
         "system only supports the StrongInertial DG formulation.");
  get(*psi_boundary_correction) = 0.0;
  // The external normal_dot_phi/normal_times_pi were computed with the
  // exterior outward normal (opposite the interior normal), so the strong-form
  // central flux is the average of the interior and exterior packaged
  // quantities. The overall minus sign accounts for the minus sign in
  // LiftFlux.hpp, mirroring SoScalarWave::BoundaryCorrections::LaxFriedrichs.
  get(*pi_boundary_correction) =
      -0.5 * (get(normal_dot_phi_int) + get(normal_dot_phi_ext));
  for (size_t d = 0; d < Dim; ++d) {
    phi_boundary_correction->get(d) =
        -0.5 * (normal_times_pi_int.get(d) + normal_times_pi_ext.get(d));
  }
}

template <size_t Dim>
// NOLINTNEXTLINE
PUP::able::PUP_ID Central<Dim>::my_PUP_ID = 0;

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(_, data) template class Central<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace ScalarWave::BoundaryCorrections
