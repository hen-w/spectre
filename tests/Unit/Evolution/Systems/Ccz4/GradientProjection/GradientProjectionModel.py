# Distributed under the MIT License.
# See LICENSE.txt for details.

"""Linear model of the gradient-projection scheme on a periodic brick.

The semi-discrete operator about Minkowski space (or, for `linearized_rates`,
about a constant state with unit conformal metric, conformal factor phi,
lapse alpha and shift beta) on a periodic chain of identical affine LGL
elements along x (one element along y and z), assembled from the frozen
Cartesian CCZ4 symbol with the Gamma-driver parameter f (beta_t = f b), with
tensor-product central derivatives and jump lifts. Fields are the chart
velocities (a_STF[5], K, Theta, Gamma[3], b[3]) followed by the chart
potentials (H_STF[5], phi, lapse, shift[3]), component-major; the grid index
is x fastest, with the x index running over the elements in order. About
Minkowski space (phi = 1) a perturbation of phi is one of log(phi). The
symbol about a general constant state is the frozen log(phi) symbol of
GradientProjectionReference conjugated by the scaling of the gradients of
phi by phi, independently of the physical-phi formulas.

The operator takes the full upwind first-order target with the scalar jump
penalty `scalar_penalty` on internal faces and the curl ordering term, fits
the potential rates by the mass-weighted gradient least squares with the
element mean from the volume target and the unit jump of the potentials,
corrects the velocity rates with the compatibility defect of the fitted
gradients, and replaces the volume source -eta b of the b rate by
-eta r_beta / f with the complete fitted shift rate r_beta.
"""

import GradientProjectionReference as reference
import numpy as np
from numpy.polynomial import legendre

_EYE = np.eye(3)
_STF = np.zeros((5, 3, 3))
_STF[0] = np.diag([1.0, -1.0, 0.0]) / np.sqrt(2.0)
_STF[1] = np.diag([1.0, 1.0, -2.0]) / np.sqrt(6.0)
for _a, (_i, _j) in enumerate([(0, 1), (0, 2), (1, 2)], 2):
    _STF[_a, _i, _j] = _STF[_a, _j, _i] = 1.0 / np.sqrt(2.0)


def _encode(tensor):
    return np.einsum("aij,ij->a", _STF, tensor)


def _symbol(f):
    """The principal symbol: u_t = D v + E_i q_i and
    v_t = C_i d_i v + B_ij d_i q_j, with the curl ordering term."""
    d = np.zeros((10, 13))
    d[:5, :5] = -np.eye(5)
    d[5, 5] = 1.0 / 3.0
    d[6, 5] = -2.0
    d[6, 6] = 4.0
    d[7:10, 10:13] = f * _EYE
    c = np.zeros((3, 13, 13))
    e = np.zeros((3, 10, 10))
    b = np.zeros((3, 3, 13, 10))
    for i in range(3):
        for j in range(3):
            sym = 0.5 * (
                np.outer(_EYE[i], _EYE[j]) + np.outer(_EYE[j], _EYE[i])
            )
            c[i, :5, 7 + j] = _encode(sym)
            e[i, :5, 7 + j] = _encode(sym)
        c[i, 5, 7 + i] = 1.0
        c[i, 6, 7 + i] = 0.5
        c[i, 7 + i, 5] = c[i, 10 + i, 5] = -4.0 / 3.0
        c[i, 7 + i, 6] = c[i, 10 + i, 6] = 2.0
        e[i, 5, 7 + i] = -1.0 / 3.0
        for j in range(3):
            if i == j:
                b[i, j, :5, :5] = -np.eye(5)
                b[i, j, 5, 5] = 4.0
                b[i, j, 5, 6] = -1.0
                b[i, j, 6, 5] = 2.0
                b[i, j, 7:10, 7:10] = _EYE
                b[i, j, 10:13, 7:10] = _EYE
            sym = 0.5 * (
                np.outer(_EYE[i], _EYE[j]) + np.outer(_EYE[j], _EYE[i])
            )
            b[i, j, :5, 5] = _encode(sym)
            b[i, j, :5, 6] = -_encode(sym)
            for k in range(3):
                for l in range(3):
                    symmetric = (
                        (k == i and l == j) + (k == j and l == i)
                    ) / 6.0
                    # (d_k q_j^j - d_j q_k^j) / 6 in the Gamma and b rates
                    curl = ((k == i and l == j) - (k == j and l == i)) / 6.0
                    b[i, j, 7 + k, 7 + l] += symmetric + curl
                    b[i, j, 10 + k, 7 + l] += symmetric + curl
    return c, b, d, e


def _background_symbol(f, phi, alpha, beta):
    """The frozen symbol about the constant state (H = 0, phi, alpha, beta)
    with w = q = 0, from the frozen log(phi) normal matrices conjugated by the
    scaling S of the gradients of phi."""
    u_log = np.r_[np.zeros(5), np.log(phi), alpha, beta]
    scaling = reference.phi_similarity(phi)
    inverse_scaling = np.linalg.inv(scaling)
    normals = [
        scaling
        @ reference.curved_normal_log_impl(u_log, _EYE[i], f)
        @ inverse_scaling
        for i in range(3)
    ]
    c = np.array([normals[i][:13, :13] for i in range(3)])
    b = np.array(
        [
            [normals[i][:13, 13 + 10 * j : 23 + 10 * j] for j in range(3)]
            for i in range(3)
        ]
    )
    d = normals[0][13:23, :13]
    e = np.array(
        [normals[0][13:23, 13 + 10 * j : 23 + 10 * j] for j in range(3)]
    )
    return c, b, d, e


def _normal_matrix(c, b, d, e, normal):
    """The 43x43 first-order normal matrix acting on (v, q_1, q_2, q_3)."""
    a = np.zeros((43, 43))
    a[:13, :13] = np.einsum("i,iab->ab", normal, c)
    for j in range(3):
        a[:13, 13 + 10 * j : 23 + 10 * j] = np.einsum(
            "i,iab->ab", normal, b[:, j]
        )
    for i in range(3):
        a[13 + 10 * i : 23 + 10 * i, :13] = normal[i] * d
        for j in range(3):
            a[13 + 10 * i : 23 + 10 * i, 13 + 10 * j : 23 + 10 * j] = (
                normal[i] * e[j]
            )
    return a


def _absolute(matrix, f):
    """|A| from the spectral projectors of A^2 with the distinct squared
    speeds 0, 1, 2, f and 4f/3 (a defective A fails the final check)."""
    square = matrix @ matrix
    identity = np.eye(len(matrix))
    result = np.zeros_like(matrix)
    speeds2 = []
    for value in (0.0, 1.0, 2.0, f, 4.0 * f / 3.0):
        if all(abs(value - other) > 1.0e-12 for other in speeds2):
            speeds2.append(value)
    for speed2 in speeds2[1:]:
        projector = identity.copy()
        for other in speeds2:
            if other != speed2:
                projector = (
                    projector @ (square - other * identity) / (speed2 - other)
                )
        result += np.sqrt(speed2) * projector
    error = np.linalg.norm(result @ result - square) / np.linalg.norm(square)
    assert error < 1.0e-12, error
    return result


def _source(kappa, eta, alpha=1.0):
    """Linearized damping terms with kappa_1 = kappa, kappa_2 = 0 about a
    constant state with unit conformal metric and lapse alpha (each damping
    term carries one factor of the lapse), and the volume source -eta b of
    the b rate."""
    result = np.zeros((43, 43))
    result[5, 6] = -3.0 * alpha * kappa
    result[6, 6] = -2.0 * alpha * kappa
    for i in range(3):
        for row in (7 + i, 10 + i):
            result[row, 7 + i] -= alpha * kappa
            for j in range(3):
                result[row, 13 + 10 * j : 18 + 10 * j] += (
                    2.0 * alpha * kappa * _STF[:, i, j]
                )
        result[10 + i, 10 + i] -= eta
    return result


def _operators_1d(points, elements):
    """LGL weights, element-wise differentiation matrix, periodic central
    derivative and jump lift (both including the inverse end weights) and
    element-mean projector on a chain of reference elements [-1, 1]."""
    p = points - 1
    basis = legendre.Legendre.basis(p)
    x = np.r_[-1.0, np.sort(np.real(basis.deriv().roots())), 1.0]
    w = 2.0 / (p * (p + 1) * basis(x) ** 2)
    bary = np.array(
        [1.0 / np.prod(x[j] - np.delete(x, j)) for j in range(points)]
    )
    diff = np.zeros((points, points))
    for j in range(points):
        for k in range(points):
            if j != k:
                diff[j, k] = bary[k] / (bary[j] * (x[j] - x[k]))
        diff[j, j] = -diff[j].sum()
    size = points * elements
    diff = np.kron(np.eye(elements), diff)
    central = diff.copy()
    jump = np.zeros((size, size))
    for element in range(elements):
        first = element * points
        last = first + points - 1
        previous_last = (first - 1) % size
        next_first = (last + 1) % size
        # Lower face (outward normal -1): (u_int - u_ext) / (2 w_0); upper
        # face (outward normal +1): (u_ext - u_int) / (2 w_N).
        central[first, first] += 0.5 / w[0]
        central[first, previous_last] -= 0.5 / w[0]
        central[last, last] -= 0.5 / w[-1]
        central[last, next_first] += 0.5 / w[-1]
        jump[first, first] += 0.5 / w[0]
        jump[first, previous_last] -= 0.5 / w[0]
        jump[last, last] += 0.5 / w[-1]
        jump[last, next_first] -= 0.5 / w[-1]
    mean = np.kron(np.eye(elements), np.outer(np.ones(points), w / w.sum()))
    return np.tile(w, elements), diff, central, jump, mean


def _along(axis, operator, identities):
    """The 3D operator acting along `axis`, x index fastest."""
    ops = list(identities)
    ops[axis] = operator
    return np.kron(ops[2], np.kron(ops[1], ops[0]))


def generator(
    widths,
    kappa,
    eta,
    f,
    scalar_penalty,
    elements_x=1,
    points=3,
    background=None,
):
    """The operator for elements of the given widths, `elements_x` of them
    along x, about Minkowski space or about the constant state
    `background = (phi, alpha, beta)`."""
    counts = (elements_x, 1, 1)
    operators = [_operators_1d(points, count) for count in counts]
    identities = [np.eye(points * count) for count in counts]
    n = np.prod([points * count for count in counts])
    mass = np.kron(
        widths[2] * operators[2][0] / 2.0,
        np.kron(
            widths[1] * operators[1][0] / 2.0, widths[0] * operators[0][0] / 2.0
        ),
    )
    derivatives = [
        _along(i, 2.0 / widths[i] * operators[i][1], identities)
        for i in range(3)
    ]
    centrals = [
        _along(i, 2.0 / widths[i] * operators[i][2], identities)
        for i in range(3)
    ]
    jumps = [
        _along(i, 2.0 / widths[i] * operators[i][3], identities)
        for i in range(3)
    ]
    mean = np.kron(operators[2][4], np.kron(operators[1][4], operators[0][4]))
    if background is None:
        c, b, d, e = _symbol(f)
        lapse = 1.0

        def absolute(matrix):
            return _absolute(matrix, f)

    else:
        phi, lapse, beta = background
        c, b, d, e = _background_symbol(f, phi, lapse, np.asarray(beta))
        absolute = reference.schur3_absolute

    reconstruction = np.zeros((43 * n, 23 * n))
    reconstruction[: 13 * n, : 13 * n] = np.eye(13 * n)
    for i in range(3):
        reconstruction[(13 + 10 * i) * n : (23 + 10 * i) * n, 13 * n :] = (
            np.kron(np.eye(10), derivatives[i])
        )
    first_order = np.kron(_source(kappa, eta, lapse), np.eye(n))
    for i in range(3):
        normal_matrix = _normal_matrix(c, b, d, e, _EYE[i])
        first_order += np.kron(normal_matrix, centrals[i]) - np.kron(
            absolute(normal_matrix) + scalar_penalty * np.eye(43), jumps[i]
        )
    target = first_order @ reconstruction
    volume_potential_target = (
        np.kron(np.column_stack([d, *e]), np.eye(n)) @ reconstruction
    )

    # Element-wise gradient least squares: the pseudo-inverse of the
    # (block-diagonal) broken stiffness, made mean free in each element.
    stiffness = sum(q.T @ (mass[:, None] * q) for q in derivatives)
    values, vectors = np.linalg.eigh(stiffness)
    active = values > 1.0e-11 * values.max()
    assert np.sum(~active) == elements_x
    inverse = (np.eye(n) - mean) @ (
        (vectors[:, active] / values[active]) @ vectors[:, active].T
    )
    fit_rhs = sum(
        np.kron(np.eye(10), derivatives[i].T * mass[None, :])
        @ target[(13 + 10 * i) * n : (23 + 10 * i) * n]
        for i in range(3)
    )
    potential_rates = (
        np.kron(np.eye(10), inverse) @ fit_rhs
        + np.kron(np.eye(10), mean) @ volume_potential_target
    )
    potential_rates[:, 13 * n :] -= np.kron(np.eye(10), mean @ sum(jumps))

    gauge = np.zeros((3, 13))
    gauge[:, 7:10] = _EYE
    gauge[:, 10:13] = -_EYE
    rhs_inverse = np.linalg.inv(np.vstack([d, gauge]))[:, :10]
    defect = sum(
        np.kron(e[i], np.eye(n))
        @ (
            target[(13 + 10 * i) * n : (23 + 10 * i) * n]
            - np.kron(np.eye(10), derivatives[i]) @ potential_rates
        )
        for i in range(3)
    )
    velocity_rates = target[: 13 * n] + np.kron(rhs_inverse, np.eye(n)) @ defect
    # The fitted shift damping replaces the volume source -eta b of the b
    # rate by -eta r_beta / f, with r_beta the complete fitted shift rate.
    for i in range(3):
        rows = slice((10 + i) * n, (11 + i) * n)
        velocity_rates[rows] -= (
            eta / f * potential_rates[(7 + i) * n : (8 + i) * n]
        )
        velocity_rates[rows, (10 + i) * n : (11 + i) * n] += eta * np.eye(n)
    return np.vstack([velocity_rates, potential_rates])


def linearized_rates(
    state, width_x, width_y, width_z, kappa, eta, f, elements_x, scalar_penalty
):
    return (
        generator(
            (width_x, width_y, width_z),
            kappa,
            eta,
            f,
            scalar_penalty,
            int(round(elements_x)),
        )
        @ state
    )


def linearized_rates_about(
    state,
    width_x,
    width_y,
    width_z,
    kappa,
    eta,
    f,
    elements_x,
    background,
    scalar_penalty,
):
    """`linearized_rates` about the constant state
    background = (phi, alpha, beta_x, beta_y, beta_z)."""
    background = np.asarray(background)
    return (
        generator(
            (width_x, width_y, width_z),
            kappa,
            eta,
            f,
            scalar_penalty,
            int(round(elements_x)),
            background=(background[0], background[1], background[2:5]),
        )
        @ state
    )


def _lgl(points):
    p = points - 1
    basis = legendre.Legendre.basis(p)
    x = np.r_[-1.0, np.sort(np.real(basis.deriv().roots())), 1.0]
    w = 2.0 / (p * (p + 1) * basis(x) ** 2)
    bary = np.array(
        [1.0 / np.prod(x[j] - np.delete(x, j)) for j in range(points)]
    )
    diff = np.zeros((points, points))
    for j in range(points):
        for k in range(points):
            if j != k:
                diff[j, k] = bary[k] / (bary[j] * (x[j] - x[k]))
        diff[j, j] = -diff[j].sum()
    return w, diff


def _mesh(geometry, points, elements):
    """Operators of a conforming mesh of curved LGL elements from the nodal
    coordinates and map inverse Jacobians d xi^a / d x^i (x index fastest).
    Internal faces are paired by matching node coordinates; faces without a
    match are external."""
    n = points**3
    data = np.asarray(geometry).reshape(elements, n, 12)
    xyz = data[:, :, :3]
    inverse_jacobian = data[:, :, 3:].reshape(elements, n, 3, 3)
    w, diff = _lgl(points)
    eye = np.eye(points)
    logical = [_along(a, diff, [eye] * 3) for a in range(3)]
    weights = np.kron(w, np.kron(w, w))
    mass = weights[None, :] / np.linalg.det(inverse_jacobian)
    index = np.arange(n).reshape(points, points, points)  # [z, y, x]
    faces = []
    for element in range(elements):
        for axis in range(3):
            for side in (-1, 1):
                ids = np.take(
                    index, 0 if side < 0 else -1, axis=2 - axis
                ).ravel()
                normal = side * inverse_jacobian[element, ids, axis, :]
                magnitude = np.linalg.norm(normal, axis=1)
                faces.append(
                    dict(
                        element=element,
                        ids=ids,
                        normal=normal / magnitude[:, None],
                        lift=magnitude / w[0],
                        center=xyz[element, ids].mean(axis=0),
                    )
                )
    scale = np.max(np.abs(xyz))
    for face in faces:
        face["outside"] = None
        for other in faces:
            if other is face or (
                np.linalg.norm(other["center"] - face["center"])
                > 1.0e-10 * scale
            ):
                continue
            mine = xyz[face["element"], face["ids"]]
            theirs = xyz[other["element"], other["ids"]]
            distance = np.linalg.norm(
                mine[:, None, :] - theirs[None, :, :], axis=2
            )
            match = np.argmin(distance, axis=1)
            assert np.max(distance[np.arange(len(match)), match]) < (
                1.0e-10 * scale
            )
            assert len(set(match)) == len(match)
            face["outside"] = (other["element"], other["ids"][match])
    return logical, inverse_jacobian, mass, faces


def linearized_rates_on_mesh(
    state,
    geometry,
    points,
    elements,
    kappa,
    eta,
    f,
    external_factor,
    scalar_penalty,
):
    """The operator applied to `state` on a conforming curved mesh about
    Minkowski space with analytic (unperturbed) exterior data on external
    faces. Layout: component-major, then element, then node (x fastest)."""
    points = int(round(points))
    elements = int(round(elements))
    n = points**3
    logical, inverse_jacobian, mass, faces = _mesh(geometry, points, elements)
    c, b, d, e = _symbol(f)
    fields = np.asarray(state).reshape(23, elements, n)
    v = fields[:13]
    u = fields[13:]

    def derivative(field, i):
        return sum(
            inverse_jacobian[None, :, :, a, i]
            * np.einsum("st,cet->ces", logical[a], field)
            for a in range(3)
        )

    q = [derivative(u, i) for i in range(3)]
    y = np.concatenate([v, *q])
    target = np.einsum("ab,ben->aen", _source(kappa, eta), y)
    for i in range(3):
        target += np.einsum(
            "ab,ben->aen", _normal_matrix(c, b, d, e, _EYE[i]), derivative(y, i)
        )
    lifted_value_jump = np.zeros((10, elements, n))
    for face in faces:
        element, ids = face["element"], face["ids"]
        interior = y[:, element, ids]
        internal = face["outside"] is not None
        exterior = (
            y[:, face["outside"][0], face["outside"][1]]
            if internal
            else np.zeros_like(interior)
        )
        jump = exterior - interior
        if internal:
            value_jump = (
                u[:, face["outside"][0], face["outside"][1]]
                - u[:, element, ids]
            )
        factor = 1.0 if internal else external_factor
        penalty = scalar_penalty if internal else 0.0
        for k in range(len(ids)):
            a_n = _normal_matrix(c, b, d, e, face["normal"][k])
            correction = a_n @ jump[:, k] + factor * (
                _absolute(a_n, f) @ jump[:, k] + penalty * jump[:, k]
            )
            target[:, element, ids[k]] += 0.5 * face["lift"][k] * correction
            if internal:
                lifted_value_jump[:, element, ids[k]] += (
                    0.5 * face["lift"][k] * value_jump[:, k]
                )
    potential_target = np.einsum("ab,ben->aen", d, v)
    for i in range(3):
        potential_target += np.einsum("ab,ben->aen", e[i], q[i])
    potential_target += lifted_value_jump

    gauge = np.zeros((3, 13))
    gauge[:, 7:10] = _EYE
    gauge[:, 10:13] = -_EYE
    rhs_inverse = np.linalg.inv(np.vstack([d, gauge]))[:, :10]
    potential_rates = np.zeros((10, elements, n))
    velocity_rates = target[:13].copy()
    for element in range(elements):
        derivatives = [
            sum(
                inverse_jacobian[element, :, a, i][:, None] * logical[a]
                for a in range(3)
            )
            for i in range(3)
        ]
        weight = mass[element]
        stiffness = sum(dq.T @ (weight[:, None] * dq) for dq in derivatives)
        values, vectors = np.linalg.eigh(stiffness)
        active = values > 1.0e-11 * values.max()
        assert np.sum(~active) == 1
        mean = np.outer(np.ones(n), weight / weight.sum())
        inverse = (np.eye(n) - mean) @ (
            (vectors[:, active] / values[active]) @ vectors[:, active].T
        )
        fit_rhs = sum(
            derivatives[i].T
            @ (weight[:, None] * target[13 + 10 * i : 23 + 10 * i, element].T)
            for i in range(3)
        )
        rates = (inverse @ fit_rhs + mean @ potential_target[:, element].T).T
        potential_rates[:, element] = rates
        defect = sum(
            e[i]
            @ (
                target[13 + 10 * i : 23 + 10 * i, element]
                - (derivatives[i] @ rates.T).T
            )
            for i in range(3)
        )
        velocity_rates[:, element] += rhs_inverse @ defect
        # The fitted shift damping, see `generator`.
        velocity_rates[10:13, element] -= eta * (
            rates[7:10] / f - v[10:13, element]
        )
    return np.concatenate([velocity_rates, potential_rates]).ravel()
