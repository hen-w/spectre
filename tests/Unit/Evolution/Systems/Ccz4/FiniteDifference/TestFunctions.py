# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def constraint_characteristic_speeds(
    lapse, shift, unit_normal_vector, unit_normal_one_form
):
    lapse = np.float64(lapse)
    shift = np.asarray(shift, dtype=np.float64).reshape(3)
    unit_normal_vector = np.asarray(
        unit_normal_vector, dtype=np.float64
    ).reshape(3)
    unit_normal_one_form = np.asarray(
        unit_normal_one_form, dtype=np.float64
    ).reshape(3)

    matrix = np.zeros((4, 4), dtype=np.float64)
    matrix[0, 0] = np.dot(shift, unit_normal_one_form)
    matrix[0, 1:4] = lapse * unit_normal_vector
    matrix[1:4, 0] = lapse * unit_normal_one_form
    matrix[1:, 1:] = matrix[0, 0] * np.eye(3, dtype=np.float64)

    eigenvalues = np.linalg.eigvals(matrix).real
    # We multiply by -1.0 as the matrix above is
    # -1.0 * principal symbol
    eigenvalues = np.sort(-1.0 * eigenvalues)
    # ascending: [smallest, mid, mid, largest]

    # Return one middle value, then largest, then smallest
    return [
        (eigenvalues[1] + eigenvalues[2]) * 0.5,
        eigenvalues[3],
        eigenvalues[0],
    ]
