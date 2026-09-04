# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def dg_package_data(
    psi,
    pi,
    phi,
    normal_covector,
    mesh_velocity,
    normal_dot_mesh_velocity,
    tau,
):
    if normal_dot_mesh_velocity is None:
        packaged_normal_dot_mesh_velocity = 0.0 * pi
    else:
        packaged_normal_dot_mesh_velocity = normal_dot_mesh_velocity
    return (
        pi,
        np.asarray(np.einsum("i,i", normal_covector, phi)),
        psi,
        np.asarray(packaged_normal_dot_mesh_velocity),
    )


def dg_boundary_terms(
    pi_int,
    normal_dot_phi_int,
    psi_int,
    normal_dot_mesh_velocity_int,
    pi_ext,
    normal_dot_phi_ext,
    psi_ext,
    normal_dot_mesh_velocity_ext,
    use_strong_form,
    tau,
):
    return (
        np.asarray(
            0.5
            * (
                normal_dot_mesh_velocity_int * psi_int
                + normal_dot_mesh_velocity_ext * psi_ext
            )
        ),
        np.asarray(
            -0.5 * (normal_dot_phi_int + normal_dot_phi_ext)
            + 0.5
            * (
                normal_dot_mesh_velocity_int * pi_int
                + normal_dot_mesh_velocity_ext * pi_ext
            )
            - tau
            * 0.5
            * (1.0 + np.abs(normal_dot_mesh_velocity_int))
            * (pi_ext - pi_int)
        ),
    )


def dg_auxiliary_package_data(
    psi,
    pi,
    normal_covector,
    mesh_velocity,
    normal_dot_mesh_velocity,
    tau,
):
    return (np.asarray(psi * normal_covector),)


def dg_auxiliary_boundary_terms(
    psi_times_normal_int,
    psi_times_normal_ext,
    use_strong_form,
    tau,
):
    return (np.asarray(0.5 * (psi_times_normal_int + psi_times_normal_ext)),)
