# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def dg_package_data(
    psi,
    pi,
    dt_psi,
    dt_pi,
    normal_covector,
    mesh_velocity,
    normal_dot_mesh_velocity,
):
    return (np.asarray(dt_psi), np.asarray(dt_pi))


def dg_boundary_terms(
    dt_psi_int, dt_pi_int, dt_psi_ext, dt_pi_ext, use_strong_form
):
    return (
        np.asarray(0.5 * (dt_psi_ext - dt_psi_int)),
        np.asarray(0.5 * (dt_pi_ext - dt_pi_int)),
    )
