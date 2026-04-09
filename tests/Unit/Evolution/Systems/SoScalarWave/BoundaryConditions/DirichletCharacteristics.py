# Distributed under the MIT License.
# See LICENSE.txt for details.

# Tests for DirichletCharacteristics boundary condition.
#
# The ghost state mixes characteristic modes based on their speeds:
#   speed > 0 (outgoing): use interior value
#   speed < 0 (incoming): use analytic value
#   speed == 0:           use analytic if prescribe_zero, else interior
#
# Speeds with mesh velocity v_g:
#   lambda_vpsi  = -v_g . n
#   lambda_vzero = -v_g . n
#   lambda_vplus = +1 - v_g . n
#   lambda_vminus = -1 - v_g . n
#
# The inverse transform from characteristic to evolved fields:
#   psi = v_psi
#   Pi  = (v+ + v-) / 2
#   Phi_j = (v+ - v-) / 2 * n_j + v0_j

import numpy as np


def error(
    face_mesh_velocity,
    outward_directed_normal_covector,
    interior_psi,
    interior_pi,
    interior_phi,
    coords,
    time,
    dim,
):
    return None


# -- Analytic solution helpers (SoPlaneWave with Gaussian profile,
#    same parameters as in DirichletAnalytic.py and the C++ test YAML) --


def _gauss_amplitude():
    return 0.9


def _gauss_width():
    return 0.6


def _gauss_center():
    return 0.0


def _center(dim):
    center = [1.1]
    if dim > 1:
        center.append(0.1)
        if dim > 2:
            center.append(-0.9)
    return np.asarray(center)


def _wave_vector(dim):
    wave_vector = [0.1]
    if dim > 1:
        wave_vector.append(1.1)
        if dim > 2:
            wave_vector.append(2.1)
    return np.asarray(wave_vector)


def _omega(dim):
    wave_vector = _wave_vector(dim)
    return np.sqrt(wave_vector.dot(wave_vector))


def _1d_u(coords, time, dim):
    result = -_omega(dim) * time
    for i in range(dim):
        result += _wave_vector(dim)[i] * (coords[i] - _center(dim)[i])
    return result


def _profile(u):
    return _gauss_amplitude() * np.exp(
        -((u - _gauss_center()) ** 2) / _gauss_width() ** 2
    )


def _first_deriv(u):
    return (
        (-2.0 * _gauss_amplitude() / _gauss_width() ** 2)
        * (u - _gauss_center())
        * np.exp(-((u - _gauss_center()) ** 2) / _gauss_width() ** 2)
    )


def _analytic_psi(coords, time, dim):
    return _profile(_1d_u(coords, time, dim))


def _analytic_pi(coords, time, dim):
    return _omega(dim) * _first_deriv(_1d_u(coords, time, dim))


def _analytic_phi(coords, time, dim):
    result = np.empty([dim])
    du = _first_deriv(_1d_u(coords, time, dim))
    for i in range(dim):
        result[i] = _wave_vector(dim)[i] * du
    return result


# -- Core ghost computation --


def _pick(speed, interior_val, analytic_val, prescribe_zero):
    """Select interior (outgoing) or analytic (incoming) based on speed sign."""
    if speed > 0.0:
        return interior_val
    elif speed < 0.0:
        return analytic_val
    else:
        return analytic_val if prescribe_zero else interior_val


def _compute_ghost(
    face_mesh_velocity,
    normal,
    interior_psi,
    interior_pi,
    interior_phi,
    coords,
    time,
    dim,
    prescribe_zero,
):
    """Compute ghost (psi, pi, phi) by mixing char modes based on speed signs.

    This computes the result via closed-form algebra:
    1. Compute char fields for interior and analytic
    2. Pick each mode based on its speed sign
    3. Invert back to evolved variables:
       psi = v_psi_ext
       pi  = (v+_ext + v-_ext) / 2
       phi_j = (v+_ext - v-_ext) / 2 * n_j + v0_ext_j
    """
    n = normal

    # Analytic solution at the face
    an_psi = _analytic_psi(coords, time, dim)
    an_pi = _analytic_pi(coords, time, dim)
    an_phi = _analytic_phi(coords, time, dim)

    # Characteristic fields: v+ = Pi + n.Phi, v- = Pi - n.Phi,
    #   v_psi = psi, v0 = Phi - n (n.Phi)
    nPhi_i = np.dot(n, interior_phi)
    nPhi_a = np.dot(n, an_phi)

    vplus_int = interior_pi + nPhi_i
    vminus_int = interior_pi - nPhi_i
    vpsi_int = interior_psi
    vzero_int = interior_phi - n * nPhi_i

    vplus_an = an_pi + nPhi_a
    vminus_an = an_pi - nPhi_a
    vpsi_an = an_psi
    vzero_an = an_phi - n * nPhi_a

    # Characteristic speeds
    if face_mesh_velocity is not None:
        vg_dot_n = np.dot(face_mesh_velocity, n)
    else:
        vg_dot_n = 0.0

    speed_vpsi = -vg_dot_n
    speed_vzero = -vg_dot_n
    speed_vplus = 1.0 - vg_dot_n
    speed_vminus = -1.0 - vg_dot_n

    # Mode selection
    vpsi_ext = _pick(speed_vpsi, vpsi_int, vpsi_an, prescribe_zero)
    vplus_ext = _pick(speed_vplus, vplus_int, vplus_an, prescribe_zero)
    vminus_ext = _pick(speed_vminus, vminus_int, vminus_an, prescribe_zero)
    vzero_ext = _pick(speed_vzero, vzero_int, vzero_an, prescribe_zero)

    # Inverse transform
    psi_out = vpsi_ext
    pi_out = 0.5 * (vplus_ext + vminus_ext)
    phi_out = 0.5 * (vplus_ext - vminus_ext) * n + vzero_ext

    return psi_out, pi_out, phi_out


# =====================================================================
# PrescribeZeroSpeedModes = true
# =====================================================================


def psi_prescribe_zero(
    face_mesh_velocity,
    outward_directed_normal_covector,
    interior_psi,
    interior_pi,
    interior_phi,
    coords,
    time,
    dim,
):
    psi_out, _, _ = _compute_ghost(
        face_mesh_velocity,
        outward_directed_normal_covector,
        interior_psi,
        interior_pi,
        interior_phi,
        coords,
        time,
        dim,
        True,
    )
    return psi_out


def pi_prescribe_zero(
    face_mesh_velocity,
    outward_directed_normal_covector,
    interior_psi,
    interior_pi,
    interior_phi,
    coords,
    time,
    dim,
):
    _, pi_out, _ = _compute_ghost(
        face_mesh_velocity,
        outward_directed_normal_covector,
        interior_psi,
        interior_pi,
        interior_phi,
        coords,
        time,
        dim,
        True,
    )
    return pi_out


def phi_prescribe_zero(
    face_mesh_velocity,
    outward_directed_normal_covector,
    interior_psi,
    interior_pi,
    interior_phi,
    coords,
    time,
    dim,
):
    _, _, phi_out = _compute_ghost(
        face_mesh_velocity,
        outward_directed_normal_covector,
        interior_psi,
        interior_pi,
        interior_phi,
        coords,
        time,
        dim,
        True,
    )
    return phi_out


# =====================================================================
# PrescribeZeroSpeedModes = false
# =====================================================================


def psi_keep_zero(
    face_mesh_velocity,
    outward_directed_normal_covector,
    interior_psi,
    interior_pi,
    interior_phi,
    coords,
    time,
    dim,
):
    psi_out, _, _ = _compute_ghost(
        face_mesh_velocity,
        outward_directed_normal_covector,
        interior_psi,
        interior_pi,
        interior_phi,
        coords,
        time,
        dim,
        False,
    )
    return psi_out


def pi_keep_zero(
    face_mesh_velocity,
    outward_directed_normal_covector,
    interior_psi,
    interior_pi,
    interior_phi,
    coords,
    time,
    dim,
):
    _, pi_out, _ = _compute_ghost(
        face_mesh_velocity,
        outward_directed_normal_covector,
        interior_psi,
        interior_pi,
        interior_phi,
        coords,
        time,
        dim,
        False,
    )
    return pi_out


def phi_keep_zero(
    face_mesh_velocity,
    outward_directed_normal_covector,
    interior_psi,
    interior_pi,
    interior_phi,
    coords,
    time,
    dim,
):
    _, _, phi_out = _compute_ghost(
        face_mesh_velocity,
        outward_directed_normal_covector,
        interior_psi,
        interior_pi,
        interior_phi,
        coords,
        time,
        dim,
        False,
    )
    return phi_out
