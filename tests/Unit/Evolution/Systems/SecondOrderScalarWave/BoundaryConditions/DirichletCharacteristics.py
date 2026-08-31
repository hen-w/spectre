# Distributed under the MIT License.
# See LICENSE.txt for details.

# Tests for the SecondOrderScalarWave DirichletCharacteristics boundary
# condition.
#
# The second-order scalar wave system has three characteristic fields (Psi has
# no characteristic field):
#   v^0_i = (delta_i^k - n_i n^k) Phi_k   speed 0
#   v^+   = Pi + n^i Phi_i                speed +1 (always outgoing)
#   v^-   = Pi - n^i Phi_i                speed -1 (always incoming)
#
# The inverse transform reconstructing the evolved (Pi, Phi_i) fields is
#   Pi    = (v^+ + v^-) / 2
#   Phi_i = (v^+ - v^-) / 2 n_i + v^0_i
#
# Ghost-mode selection (the class supports no moving mesh, so the speeds are
# constant and the branching has no per-point dependence):
#   v^+ ghost = interior v^+   (outgoing, always from the interior)
#   v^- ghost = analytic v^-   (incoming), or 0 if ZeroIncomingMode
#   v^0 ghost = analytic v^0 if PrescribeZeroSpeedModes, else interior v^0
#
# Ghost Psi:
#   CopyPsiFromInterior:     interior Psi
#   PrescribeZeroSpeedModes: analytic Psi
#   otherwise:               the passed boundary_psi_value
#
# dt of the boundary-evolved Psi:
#   CopyPsiFromInterior: 0
#   otherwise:           -0.5 (interior v^+ + (0 if ZeroIncomingMode
#                                              else analytic v^-))

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


def _ghost_char_modes(
    normal,
    interior_pi,
    interior_phi,
    coords,
    time,
    dim,
    prescribe_zero,
    zero_incoming,
):
    """Return the ghost (v^0, v^+, v^-) after mode selection."""
    n = normal

    an_pi = _analytic_pi(coords, time, dim)
    an_phi = _analytic_phi(coords, time, dim)

    n_dot_phi_int = np.dot(n, interior_phi)
    n_dot_phi_an = np.dot(n, an_phi)

    # v^+ ghost: always the interior outgoing mode.
    vplus_ext = interior_pi + n_dot_phi_int
    # v^- ghost: analytic incoming mode, or zero.
    vminus_ext = 0.0 if zero_incoming else (an_pi - n_dot_phi_an)
    # v^0 ghost: analytic if prescribing zero-speed modes, else interior.
    if prescribe_zero:
        vzero_ext = an_phi - n * n_dot_phi_an
    else:
        vzero_ext = interior_phi - n * n_dot_phi_int

    return vzero_ext, vplus_ext, vminus_ext


def _ghost_pi_phi(
    normal,
    interior_pi,
    interior_phi,
    coords,
    time,
    dim,
    prescribe_zero,
    zero_incoming,
):
    """Reconstruct the ghost (Pi, Phi_i) from the selected char modes."""
    vzero_ext, vplus_ext, vminus_ext = _ghost_char_modes(
        normal,
        interior_pi,
        interior_phi,
        coords,
        time,
        dim,
        prescribe_zero,
        zero_incoming,
    )
    pi_out = 0.5 * (vplus_ext + vminus_ext)
    phi_out = 0.5 * (vplus_ext - vminus_ext) * normal + vzero_ext
    return pi_out, phi_out


def _dt_boundary_psi(
    normal, interior_pi, interior_phi, coords, time, dim, zero_incoming
):
    """dt of the boundary-evolved Psi = -0.5 (v^+_int + v^-_analytic)."""
    n = normal
    vplus_int = interior_pi + np.dot(n, interior_phi)
    if zero_incoming:
        vminus = 0.0
    else:
        an_pi = _analytic_pi(coords, time, dim)
        an_phi = _analytic_phi(coords, time, dim)
        vminus = an_pi - np.dot(n, an_phi)
    return -0.5 * (vplus_int + vminus)


# =====================================================================
# PrescribeZeroSpeedModes = true
# =====================================================================


def psi_prescribe_zero(
    face_mesh_velocity,
    outward_directed_normal_covector,
    interior_psi,
    interior_pi,
    interior_phi,
    boundary_psi_value,
    coords,
    time,
    dim,
):
    # Ghost Psi = analytic Psi.
    return np.asarray(_analytic_psi(coords, time, dim))


def pi_prescribe_zero(
    face_mesh_velocity,
    outward_directed_normal_covector,
    interior_psi,
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
        False,
    )
    return np.asarray(pi_out)


def phi_prescribe_zero(
    face_mesh_velocity,
    outward_directed_normal_covector,
    interior_psi,
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
        False,
    )
    return phi_out


def dt_boundary_psi_prescribe_zero(
    face_mesh_velocity,
    outward_directed_normal_covector,
    interior_psi,
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
        )
    )


# =====================================================================
# PrescribeZeroSpeedModes = false (ghost Psi = boundary-evolved value)
# =====================================================================


def psi_keep_zero(
    face_mesh_velocity,
    outward_directed_normal_covector,
    interior_psi,
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
    interior_psi,
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
        False,
    )
    return np.asarray(pi_out)


def phi_keep_zero(
    face_mesh_velocity,
    outward_directed_normal_covector,
    interior_psi,
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
        False,
    )
    return phi_out


def dt_boundary_psi_keep_zero(
    face_mesh_velocity,
    outward_directed_normal_covector,
    interior_psi,
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
        )
    )


# =====================================================================
# CopyPsiFromInterior = true (ghost Psi = interior Psi, dt = 0)
# =====================================================================


def psi_copy_interior(
    face_mesh_velocity,
    outward_directed_normal_covector,
    interior_psi,
    interior_pi,
    interior_phi,
    boundary_psi_value,
    coords,
    time,
    dim,
):
    return interior_psi


def pi_copy_interior(
    face_mesh_velocity,
    outward_directed_normal_covector,
    interior_psi,
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
        False,
    )
    return np.asarray(pi_out)


def phi_copy_interior(
    face_mesh_velocity,
    outward_directed_normal_covector,
    interior_psi,
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
        False,
    )
    return phi_out


def dt_boundary_psi_copy_interior(
    face_mesh_velocity,
    outward_directed_normal_covector,
    interior_psi,
    interior_pi,
    interior_phi,
    boundary_psi_value,
    coords,
    time,
    dim,
):
    return 0.0 * np.asarray(interior_psi)


# =====================================================================
# ZeroIncomingMode = true (PrescribeZeroSpeedModes = false,
#                          CopyPsiFromInterior = false)
# =====================================================================


def psi_zero_incoming(
    face_mesh_velocity,
    outward_directed_normal_covector,
    interior_psi,
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
    interior_psi,
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
        True,
    )
    return np.asarray(pi_out)


def phi_zero_incoming(
    face_mesh_velocity,
    outward_directed_normal_covector,
    interior_psi,
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
        True,
    )
    return phi_out


def dt_boundary_psi_zero_incoming(
    face_mesh_velocity,
    outward_directed_normal_covector,
    interior_psi,
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
        )
    )
