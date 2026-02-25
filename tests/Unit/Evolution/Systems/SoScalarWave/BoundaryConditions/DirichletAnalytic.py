# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def error(
    face_mesh_velocity,
    outward_directed_normal_covector,
    coords,
    time,
    dim,
):
    return None


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


def _second_deriv(u):
    gauss_term = np.exp(-((u - _gauss_center()) ** 2) / _gauss_width() ** 2)
    first_term = (-2.0 * _gauss_amplitude() / _gauss_width() ** 2) * gauss_term
    second_term = (
        (4.0 * _gauss_amplitude() / _gauss_width() ** 4)
        * (u - _gauss_center()) ** 2
        * gauss_term
    )
    return first_term + second_term


def _pi_value(coords, time, dim):
    """Compute the actual Pi value from the analytic solution."""
    return _omega(dim) * _first_deriv(_1d_u(coords, time, dim))


def _psi_value(coords, time, dim):
    """Compute the actual Psi value from the analytic solution."""
    return _profile(_1d_u(coords, time, dim))


def pi(
    face_mesh_velocity,
    outward_directed_normal_covector,
    coords,
    time,
    dim,
):
    # Return 0.0 because Pi is not used by CgCollocation's dg_package_data.
    # Setting to zero helps catch bugs if something unexpectedly uses this.
    return 0.0


def psi(
    face_mesh_velocity,
    outward_directed_normal_covector,
    coords,
    time,
    dim,
):
    # Return 0.0 because Psi is not used by CgCollocation's dg_package_data.
    # Setting to zero helps catch bugs if something unexpectedly uses this.
    return 0.0


def dt_psi(
    face_mesh_velocity,
    outward_directed_normal_covector,
    coords,
    time,
    dim,
):
    # dt<Psi> = -Pi (use the actual value, not NaN)
    return -_pi_value(coords, time, dim)


def dt_pi(
    face_mesh_velocity,
    outward_directed_normal_covector,
    coords,
    time,
    dim,
):
    # dt<Pi> = -∇²Psi = -d²Psi/du² * (du/dx_i)²
    # For the plane wave: du/dx_i = wave_vector[i]
    # So ∇²Psi = d²Psi/du² * sum_i(wave_vector[i]²) = d²Psi/du² * omega²
    u = _1d_u(coords, time, dim)
    return -_omega(dim) ** 2 * _second_deriv(u)
