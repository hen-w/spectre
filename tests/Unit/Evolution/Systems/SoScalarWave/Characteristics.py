# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


# Test functions for characteristic speeds
def char_speed_vpsi(unit_normal):
    return 0.0


def char_speed_vzero(unit_normal):
    return 0.0


def char_speed_vplus(unit_normal):
    return 1.0


def char_speed_vminus(unit_normal):
    return -1.0


# End test functions for characteristic speeds


# Test functions for characteristic speeds with mesh velocity
def char_speed_vpsi_mesh_velocity(unit_normal, mesh_velocity):
    vg_dot_n = np.dot(mesh_velocity, unit_normal)
    return -vg_dot_n


def char_speed_vzero_mesh_velocity(unit_normal, mesh_velocity):
    vg_dot_n = np.dot(mesh_velocity, unit_normal)
    return -vg_dot_n


def char_speed_vplus_mesh_velocity(unit_normal, mesh_velocity):
    vg_dot_n = np.dot(mesh_velocity, unit_normal)
    return 1.0 - vg_dot_n


def char_speed_vminus_mesh_velocity(unit_normal, mesh_velocity):
    vg_dot_n = np.dot(mesh_velocity, unit_normal)
    return -1.0 - vg_dot_n


# End test functions for characteristic speeds with mesh velocity


# Test functions for characteristic fields
def char_field_vpsi(psi, pi, phi, normal_one_form):
    return psi


def char_field_vzero(psi, pi, phi, normal_one_form):
    normal_vector = normal_one_form
    projection_tensor = np.identity(len(normal_vector)) - np.einsum(
        "i,j", normal_one_form, normal_vector
    )
    return np.einsum("ij,j->i", projection_tensor, phi)


def char_field_vplus(psi, pi, phi, normal_one_form):
    normal_vector = normal_one_form
    phi_dot_normal = np.einsum("i,i->", normal_vector, phi)
    return pi + phi_dot_normal


def char_field_vminus(psi, pi, phi, normal_one_form):
    normal_vector = normal_one_form
    phi_dot_normal = np.einsum("i,i->", normal_vector, phi)
    return pi - phi_dot_normal


# End test functions for characteristic fields


# Test functions for evolved fields
def evol_field_psi(vpsi, vzero, vplus, vminus, normal_one_form):
    return vpsi


def evol_field_pi(vpsi, vzero, vplus, vminus, normal_one_form):
    return 0.5 * (vplus + vminus)


def evol_field_phi(vpsi, vzero, vplus, vminus, normal_one_form):
    return 0.5 * (vplus - vminus) * normal_one_form + vzero


# End test functions for evolved fields
