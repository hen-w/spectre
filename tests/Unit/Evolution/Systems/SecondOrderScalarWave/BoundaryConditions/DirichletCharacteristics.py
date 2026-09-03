# Distributed under the MIT License.
# See LICENSE.txt for details.

# Tests for the SecondOrderScalarWave DirichletCharacteristics boundary
# condition.
#
# The second-order scalar wave system has three characteristic fields (Psi has
# no characteristic field):
#   v^0_i = (delta_i^k - n_i n^k) Phi_k   speed lambda^0
#   v^+   = Pi + n^i Phi_i                speed lambda^+
#   v^-   = Pi - n^i Phi_i                speed lambda^-
#
# The grid-frame characteristic speeds are
#   [lambda^0, lambda^+, lambda^-] = [0, +1, -1]                (static mesh)
#   [lambda^0, lambda^+, lambda^-] = [-ndotv, 1-ndotv, -1-ndotv]  (moving mesh)
# with ndotv = n_i v^i and v^i the face mesh velocity.
#
# The inverse transform reconstructing the evolved (Pi, Phi_i) fields is
#   Pi    = (v^+ + v^-) / 2
#   Phi_i = (v^+ - v^-) / 2 n_i + v^0_i
#
# Ghost-mode selection is per-point and per-mode: a mode with lambda >= 0
# (including exactly 0) is taken from the interior; a mode with lambda < 0 is
# incoming and is prescribed from data (the analytic char field, or 0 with
# ZeroIncomingMode). The lambda^0 mask applies to every component of v^0.
#
# Ghost Psi: the passed boundary_psi_value (the time-integrated
# boundary-evolved value), independent of the mesh velocity.
#
# dt of the boundary-evolved Psi:
#   -Pi_b + (v^i (Phi_b)_i if a mesh velocity is present, else nothing),
# where Pi_b and Phi_b are the SAME ghost (Pi, Phi) reconstructed above.

import numpy as np

# -- Analytic solution helpers --
# SecondOrderWrapper<ScalarWave::Solutions::PlaneWave<Dim>> with a Gaussian
# profile, using the same parameters as the C++ test YAML / construction.
#   u      = k . (x - center) - omega t,  omega = ||k||
#   Psi    = F(u)
#   Pi     = omega F'(u)
#   Phi_i  = k_i F'(u)


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


def _first_deriv(u):
    return (
        (-2.0 * _gauss_amplitude() / _gauss_width() ** 2)
        * (u - _gauss_center())
        * np.exp(-((u - _gauss_center()) ** 2) / _gauss_width() ** 2)
    )


def _analytic_pi(coords, time, dim):
    return _omega(dim) * _first_deriv(_1d_u(coords, time, dim))


def _analytic_phi(coords, time, dim):
    result = np.empty([dim])
    du = _first_deriv(_1d_u(coords, time, dim))
    for i in range(dim):
        result[i] = _wave_vector(dim)[i] * du
    return result


# -- Core ghost computation --


def _char_speeds(normal, face_mesh_velocity):
    """Grid-frame [lambda^0, lambda^+, lambda^-] at this point."""
    if face_mesh_velocity is None:
        return 0.0, 1.0, -1.0
    n_dot_v = np.dot(normal, face_mesh_velocity)
    return -n_dot_v, 1.0 - n_dot_v, -1.0 - n_dot_v


def _selected_modes(
    normal,
    interior_pi,
    interior_phi,
    coords,
    time,
    dim,
    zero_incoming,
    face_mesh_velocity,
):
    """Per-point (v^0, v^+, v^-) after mode selection.

    A mode with lambda >= 0 is taken from the interior; a mode with lambda < 0
    is taken from the data (analytic char field, or 0 with ZeroIncomingMode).
    """
    n = normal
    lambda_zero, lambda_plus, lambda_minus = _char_speeds(n, face_mesh_velocity)

    n_dot_phi_int = np.dot(n, interior_phi)
    vplus_int = interior_pi + n_dot_phi_int
    vminus_int = interior_pi - n_dot_phi_int
    vzero_int = interior_phi - n * n_dot_phi_int

    if zero_incoming:
        vplus_data = 0.0
        vminus_data = 0.0
        vzero_data = np.zeros(dim)
    else:
        an_pi = _analytic_pi(coords, time, dim)
        an_phi = _analytic_phi(coords, time, dim)
        an_n_dot_phi = np.dot(n, an_phi)
        vplus_data = an_pi + an_n_dot_phi
        vminus_data = an_pi - an_n_dot_phi
        vzero_data = an_phi - n * an_n_dot_phi

    vplus_sel = np.where(lambda_plus >= 0.0, vplus_int, vplus_data)
    vminus_sel = np.where(lambda_minus >= 0.0, vminus_int, vminus_data)
    vzero_sel = np.where(lambda_zero >= 0.0, vzero_int, vzero_data)

    return vzero_sel, vplus_sel, vminus_sel


def _ghost_pi_phi(
    normal,
    interior_pi,
    interior_phi,
    coords,
    time,
    dim,
    zero_incoming,
    face_mesh_velocity,
):
    """Reconstruct the ghost (Pi, Phi_i) from the selected char modes."""
    vzero_sel, vplus_sel, vminus_sel = _selected_modes(
        normal,
        interior_pi,
        interior_phi,
        coords,
        time,
        dim,
        zero_incoming,
        face_mesh_velocity,
    )
    pi_out = 0.5 * (vplus_sel + vminus_sel)
    phi_out = 0.5 * (vplus_sel - vminus_sel) * normal + vzero_sel
    return pi_out, phi_out


def _dt_boundary_psi(
    normal,
    interior_pi,
    interior_phi,
    coords,
    time,
    dim,
    zero_incoming,
    face_mesh_velocity,
):
    """dt of the boundary-evolved Psi = -Pi_b + v^i (Phi_b)_i.

    The advection term is present only when a mesh velocity is supplied.
    """
    pi_b, phi_b = _ghost_pi_phi(
        normal,
        interior_pi,
        interior_phi,
        coords,
        time,
        dim,
        zero_incoming,
        face_mesh_velocity,
    )
    result = -pi_b
    if face_mesh_velocity is not None:
        result += np.dot(face_mesh_velocity, phi_b)
    return result


# =====================================================================
# ZeroIncomingMode = false (ghost Psi = boundary-evolved value)
# =====================================================================


def psi_keep_zero(
    face_mesh_velocity,
    outward_directed_normal_covector,
    interior_pi,
    interior_phi,
    boundary_psi_value,
    coords,
    time,
    dim,
):
    return boundary_psi_value


def pi_keep_zero(
    face_mesh_velocity,
    outward_directed_normal_covector,
    interior_pi,
    interior_phi,
    boundary_psi_value,
    coords,
    time,
    dim,
):
    pi_out, _ = _ghost_pi_phi(
        outward_directed_normal_covector,
        interior_pi,
        interior_phi,
        coords,
        time,
        dim,
        False,
        face_mesh_velocity,
    )
    return np.asarray(pi_out)


def phi_keep_zero(
    face_mesh_velocity,
    outward_directed_normal_covector,
    interior_pi,
    interior_phi,
    boundary_psi_value,
    coords,
    time,
    dim,
):
    _, phi_out = _ghost_pi_phi(
        outward_directed_normal_covector,
        interior_pi,
        interior_phi,
        coords,
        time,
        dim,
        False,
        face_mesh_velocity,
    )
    return phi_out


def dt_boundary_psi_keep_zero(
    face_mesh_velocity,
    outward_directed_normal_covector,
    interior_pi,
    interior_phi,
    boundary_psi_value,
    coords,
    time,
    dim,
):
    return np.asarray(
        _dt_boundary_psi(
            outward_directed_normal_covector,
            interior_pi,
            interior_phi,
            coords,
            time,
            dim,
            False,
            face_mesh_velocity,
        )
    )


# =====================================================================
# ZeroIncomingMode = true
# =====================================================================


def psi_zero_incoming(
    face_mesh_velocity,
    outward_directed_normal_covector,
    interior_pi,
    interior_phi,
    boundary_psi_value,
    coords,
    time,
    dim,
):
    return boundary_psi_value


def pi_zero_incoming(
    face_mesh_velocity,
    outward_directed_normal_covector,
    interior_pi,
    interior_phi,
    boundary_psi_value,
    coords,
    time,
    dim,
):
    pi_out, _ = _ghost_pi_phi(
        outward_directed_normal_covector,
        interior_pi,
        interior_phi,
        coords,
        time,
        dim,
        True,
        face_mesh_velocity,
    )
    return np.asarray(pi_out)


def phi_zero_incoming(
    face_mesh_velocity,
    outward_directed_normal_covector,
    interior_pi,
    interior_phi,
    boundary_psi_value,
    coords,
    time,
    dim,
):
    _, phi_out = _ghost_pi_phi(
        outward_directed_normal_covector,
        interior_pi,
        interior_phi,
        coords,
        time,
        dim,
        True,
        face_mesh_velocity,
    )
    return phi_out


def dt_boundary_psi_zero_incoming(
    face_mesh_velocity,
    outward_directed_normal_covector,
    interior_pi,
    interior_phi,
    boundary_psi_value,
    coords,
    time,
    dim,
):
    return np.asarray(
        _dt_boundary_psi(
            outward_directed_normal_covector,
            interior_pi,
            interior_phi,
            coords,
            time,
            dim,
            True,
            face_mesh_velocity,
        )
    )
