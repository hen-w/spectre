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

`linearized_rates_on_general_mesh` (and `linearized_rates_on_mesh` on
conforming meshes of LGL elements) assembles the operator about Minkowski
space on curved meshes of LGL elements and spherical-harmonic shells from
the nodal coordinates and inverse Jacobians alone, including the shell
gradient fit and velocity filter, the split scalar penalties, the
wedge-shell transfers and the shell-side boundary-correction filter.
"""

import itertools

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
    """LGL nodes, weights and differentiation matrix on [-1, 1]."""
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
    return x, w, diff


def _lagrange(nodes, targets):
    """The Lagrange interpolation matrix from `nodes` to `targets`."""
    nodes = np.asarray(nodes)
    result = np.ones((len(targets), len(nodes)))
    for j in range(len(nodes)):
        for k in range(len(nodes)):
            if k != j:
                result[:, j] *= (np.asarray(targets) - nodes[k]) / (
                    nodes[j] - nodes[k]
                )
    return result


def _decode_stf():
    """The 9x5 matrix taking the orthonormal STF coordinates to the
    Cartesian components T_ab (index 3a+b); its transpose encodes."""
    return _STF.reshape(5, 9).T


# The rotation generators (E_k)_ab = epsilon_akb of a Cartesian index: the
# infinitesimal rotation about e_k is x -> x + e_k x x = x + E_k x.
_LEVI_CIVITA = np.zeros((3, 3, 3))
for _permutation in itertools.permutations(range(3)):
    _LEVI_CIVITA[_permutation] = np.linalg.det(_EYE[list(_permutation)])
_ROTATION = np.array([_LEVI_CIVITA[:, _k, :] for _k in range(3)])
for _k in range(3):
    assert np.allclose(
        _ROTATION[_k] @ _EYE[(_k + 1) % 3],
        np.cross(_EYE[_k], _EYE[(_k + 1) % 3]),
    )


class _Sphere:
    """The spherical-harmonic angular grid of SphericalHarmonic meshes with
    the given L: L+1 Gauss points in theta (increasing theta, i.e. decreasing
    cos(theta)) and 2L+1 equiangular points phi_j = 2 pi j / (2L+1), theta
    fastest, with the Spherepack integration weights (Gauss weights times
    2 pi / (2L+1)).

    The real scalar harmonics Y_lm (l <= L, 0 <= m <= l, cos and sin) are
    built from orthonormal polynomials in cos(theta) and are W-orthonormal on
    the grid (the quadrature is exact on their products), so the analysis
    onto them is the W-orthogonal projection of nodal data.

    The angular logical derivatives (d_theta and d_phi / sin(theta)) are
    those of the native Spherepack gradient: the analysis followed by the
    exact gradient of the harmonics, except that the native gradient
    annihilates the harmonics with l = L and even m."""

    def __init__(self, ell_max):
        self.L = ell_max
        mu, gauss = legendre.leggauss(ell_max + 1)
        theta = np.arccos(mu[::-1])
        self.theta_nodes = theta
        phi = 2.0 * np.pi * np.arange(2 * ell_max + 1) / (2 * ell_max + 1)
        self.phi_nodes = phi
        self.theta = np.tile(theta, len(phi))
        self.phi = np.repeat(phi, len(theta))
        self.weights = np.tile(gauss[::-1], len(phi)) * (
            2.0 * np.pi / (2 * ell_max + 1)
        )
        self.n = len(self.theta)
        x = np.cos(self.theta)
        s = np.sin(self.theta)
        values, d_theta, d_phi, labels = [], [], [], []
        for m in range(ell_max + 1):
            polynomials = []
            for ell in range(m, ell_max + 1):
                p = np.polynomial.Polynomial(np.r_[np.zeros(ell - m), 1.0])
                for q in polynomials:
                    p = (
                        p
                        - np.sum(gauss * (1 - mu**2) ** m * p(mu) * q(mu)) * q
                    )
                p = p / np.sqrt(np.sum(gauss * (1 - mu**2) ** m * p(mu) ** 2))
                polynomials.append(p)
                legendre_function = s**m * p(x)
                derivative = -(s ** (m + 1)) * p.deriv()(x)
                if m > 0:
                    derivative = derivative + m * s ** (m - 1) * x * p(x)
                for trig in (0, 1) if m > 0 else (0,):
                    f = (
                        np.cos(m * self.phi)
                        if trig == 0
                        else np.sin(m * self.phi)
                    )
                    df = (
                        -m * np.sin(m * self.phi)
                        if trig == 0
                        else m * np.cos(m * self.phi)
                    )
                    values.append(legendre_function * f)
                    d_theta.append(derivative * f)
                    d_phi.append(s ** (m - 1) * p(x) * df if m > 0 else 0.0 * f)
                    labels.append((ell, m))
        values = np.array(values).T
        norms = np.sqrt(self.weights @ values**2)
        self.basis = values / norms
        self.d_theta = np.array(d_theta).T / norms
        self.d_phi = np.array(d_phi).T / norms
        self.labels = labels
        gram = self.basis.T @ (self.weights[:, None] * self.basis)
        assert np.max(np.abs(gram - np.eye(len(labels)))) < 1.0e-12
        self.analysis = self.basis.T * self.weights
        native = np.array(
            [not (ell == ell_max and m % 2 == 0) for ell, m in labels]
        )
        self.gradient = [
            (self.d_theta * native) @ self.analysis,
            (self.d_phi * native) @ self.analysis,
        ]
        self.unit_r = np.array([s * np.cos(self.phi), s * np.sin(self.phi), x])
        self.unit_theta = np.array(
            [x * np.cos(self.phi), x * np.sin(self.phi), -s]
        )
        self.unit_phi = np.array(
            [-np.sin(self.phi), np.cos(self.phi), np.zeros(self.n)]
        )
        self._projectors = {}

    def cardinal(self, directions):
        """The native interpolation of nodal data to the given unit
        directions: Fourier interpolation in phi (odd number of points) and,
        separately for the parts even and odd under phi -> phi + pi, Lagrange
        interpolation in cos(theta) of the even part and of the odd part
        divided by sin(theta)."""
        theta = np.arccos(np.clip(directions[2], -1.0, 1.0))
        phi = np.arctan2(directions[1], directions[0])
        lagrange_theta = _lagrange(np.cos(self.theta_nodes), np.cos(theta))
        factor = 0.5 * np.sin(theta)[:, None] / np.sin(self.theta_nodes)

        def fourier(targets):
            modes = np.arange(1, self.L + 1)

            def trig(points):
                return np.hstack(
                    [
                        np.ones((len(points), 1)),
                        np.cos(np.outer(points, modes)),
                        np.sin(np.outer(points, modes)),
                    ]
                )

            return trig(targets) @ np.linalg.inv(trig(self.phi_nodes))

        same, shifted = fourier(phi), fourier(phi + np.pi)
        result = np.zeros((len(theta), self.n))
        n_theta = self.L + 1
        for j in range(len(self.phi_nodes)):
            result[:, j * n_theta : (j + 1) * n_theta] = lagrange_theta * (
                (0.5 + factor) * same[:, j : j + 1]
                + (0.5 - factor) * shifted[:, j : j + 1]
            )
        return result

    def tensor_projector(self, rank, modes_to_kill):
        """The W-orthogonal nodal projector of rank-`rank` Cartesian tensor
        fields (components of index sum_p a_p 3^(rank-1-p), component-major
        over the grid) onto those whose components are scalar harmonics with
        l <= L and whose total angular momentum is at most L -
        modes_to_kill: the range of the native hard tensor spherical-harmonic
        filter, assembled from the Casimir of the rotation generators (the
        rotation of every index minus the orbital generator x cross grad)."""
        key = (rank, modes_to_kill)
        if key in self._projectors:
            return self._projectors[key]
        assert 0 <= modes_to_kill <= self.L, (modes_to_kill, self.L)
        components = 3**rank
        modes = self.basis.shape[1]
        casimir = np.zeros((components * modes, components * modes))
        for k in range(3):
            orbital = self.analysis @ (
                self.unit_phi[k][:, None] * self.d_theta
                - self.unit_theta[k][:, None] * self.d_phi
            )
            spin = np.zeros((components, components))
            for position in range(rank):
                spin += np.kron(
                    np.eye(3**position),
                    np.kron(_ROTATION[k], np.eye(3 ** (rank - 1 - position))),
                )
            generator = np.kron(spin, np.eye(modes)) - np.kron(
                np.eye(components), orbital
            )
            casimir -= generator @ generator
        values, vectors = np.linalg.eigh(0.5 * (casimir + casimir.T))
        total = 0.5 * (np.sqrt(1.0 + 4.0 * np.maximum(values, 0.0)) - 1.0)
        assert np.max(np.abs(total - np.round(total))) < 1.0e-9
        retained = vectors[:, np.round(total) <= self.L - modes_to_kill]
        synthesis = np.kron(np.eye(components), self.basis)
        result = (synthesis @ retained) @ (
            retained.T @ np.kron(np.eye(components), self.analysis)
        )
        if rank == 1:
            # The radial unit vector has total angular momentum zero; a
            # constant vector has one.
            radial = self.unit_r.ravel()
            assert np.max(np.abs(result @ radial - radial)) < 1.0e-10
            constant = np.repeat(_EYE[0], self.n)
            kept = np.max(np.abs(result @ constant - constant)) < 1.0e-10
            assert kept == (self.L - modes_to_kill >= 1)
        self._projectors[key] = result
        return result

    def structured_projector(self, structure, modes_to_kill):
        """`tensor_projector` in the model's component layouts: "scalar",
        "vector", "stf" (5 orthonormal STF coordinates), "rank3stf" (q_i of
        the five STF components, i major), "rank2" (q_i of a vector, i
        major)."""
        rank, encode = {
            "scalar": (0, np.eye(1)),
            "vector": (1, np.eye(3)),
            "stf": (2, _decode_stf().T),
            "rank2": (2, np.eye(9)),
            "rank3stf": (3, np.kron(np.eye(3), _decode_stf().T)),
        }[structure]
        identity = np.eye(self.n)
        return (
            np.kron(encode, identity)
            @ self.tensor_projector(rank, modes_to_kill)
            @ np.kron(encode.T, identity)
        )


_SPHERES = {}


def _sphere(ell_max):
    if ell_max not in _SPHERES:
        _SPHERES[ell_max] = _Sphere(ell_max)
    return _SPHERES[ell_max]


def _face(element, ids, normal, inverse_weight, plane=None):
    """A face of an element from the unnormalized outward normal covector
    and the inverse end weight of its logical direction. For faces of LGL
    elements `plane` holds the extents and the gradients (points x 2 x 3)
    of the two logical coordinates along the face."""
    magnitude = np.linalg.norm(normal, axis=1)
    return dict(
        element=element,
        ids=ids,
        normal=normal / magnitude[:, None],
        lift=magnitude * inverse_weight,
        plane=plane,
    )


class _LglElement:
    """A curved element with LGL points in every logical direction."""

    shell = False

    def __init__(self, index, xyz, inverse_jacobian, extents):
        self.n = int(np.prod(extents))
        self.xyz = xyz
        nodes = [_lgl(points) for points in extents]
        identities = [np.eye(points) for points in extents]
        logical = [_along(a, nodes[a][2], identities) for a in range(3)]
        self.derivatives = [
            sum(
                inverse_jacobian[:, a, i][:, None] * logical[a]
                for a in range(3)
            )
            for i in range(3)
        ]
        weights = np.kron(nodes[2][1], np.kron(nodes[1][1], nodes[0][1]))
        determinant = np.linalg.det(inverse_jacobian)
        assert np.all(determinant > 0.0)
        self.mass = weights / determinant
        grid = np.arange(self.n).reshape(extents[2], extents[1], extents[0])
        self.faces = []
        for axis in range(3):
            plane_axes = [a for a in range(3) if a != axis]
            for side in (-1, 1):
                ids = np.take(
                    grid, 0 if side < 0 else -1, axis=2 - axis
                ).ravel()
                self.faces.append(
                    _face(
                        index,
                        ids,
                        side * inverse_jacobian[ids, axis, :],
                        1.0 / nodes[axis][1][0],
                        dict(
                            extents=[extents[a] for a in plane_axes],
                            gradients=inverse_jacobian[ids][:, plane_axes, :],
                        ),
                    )
                )

    def fit(self, q_target, provisional):
        """The mass-weighted gradient least squares with the element mean
        of the provisional rates."""
        weight = self.mass
        stiffness = sum(
            dq.T @ (weight[:, None] * dq) for dq in self.derivatives
        )
        values, vectors = np.linalg.eigh(stiffness)
        active = values > 1.0e-11 * values.max()
        assert np.sum(~active) == 1
        mean = np.outer(np.ones(self.n), weight / weight.sum())
        inverse = (np.eye(self.n) - mean) @ (
            (vectors[:, active] / values[active]) @ vectors[:, active].T
        )
        fit_rhs = sum(
            self.derivatives[i].T @ (weight[:, None] * q_target[i].T)
            for i in range(3)
        )
        return (inverse @ fit_rhs + mean @ provisional.T).T


class _ShellElement:
    """A spherical-harmonic shell element: LGL in the first (radial)
    direction, the `_Sphere` grid in the other two (radial index fastest),
    with the angular logical derivatives of `_Sphere.gradient`."""

    shell = True

    def __init__(self, index, xyz, inverse_jacobian, extents):
        self.radial_points = extents[0]
        self.sphere = _sphere(extents[1] - 1)
        assert extents[2] == 2 * self.sphere.L + 1
        self.n = self.radial_points * self.sphere.n
        self.xyz = xyz
        directions = xyz.reshape(self.sphere.n, self.radial_points, 3)
        directions = directions / np.linalg.norm(directions, axis=2)[:, :, None]
        assert np.max(np.abs(directions - self.sphere.unit_r.T[:, None, :])) < (
            1.0e-12
        ), "The shell is not on the standard spherical-harmonic grid."
        _, radial_weights, radial_diff = _lgl(self.radial_points)
        identity = np.eye(self.radial_points)
        logical = [
            np.kron(np.eye(self.sphere.n), radial_diff),
            np.kron(self.sphere.gradient[0], identity),
            np.kron(self.sphere.gradient[1], identity),
        ]
        self.derivatives = [
            sum(
                inverse_jacobian[:, a, i][:, None] * logical[a]
                for a in range(3)
            )
            for i in range(3)
        ]
        determinant = np.linalg.det(inverse_jacobian)
        assert np.all(determinant > 0.0)
        self.mass = np.kron(self.sphere.weights, radial_weights) / determinant
        grid = np.arange(self.n).reshape(self.sphere.n, self.radial_points)
        self.faces = [
            _face(
                index,
                grid[:, 0 if side < 0 else -1],
                side * inverse_jacobian[grid[:, 0 if side < 0 else -1], 0, :],
                1.0 / radial_weights[0],
            )
            for side in (-1, 1)
        ]
        # The fit over the full trial space of radial polynomials times
        # scalar harmonics, M-orthonormalized.
        root = np.sqrt(self.mass)
        trial, _ = np.linalg.qr(
            root[:, None] * np.kron(self.sphere.basis, identity)
        )
        self.trial = trial / root[:, None]
        weighted = np.vstack(
            [root[:, None] * (d @ self.trial) for d in self.derivatives]
        )
        left, singular, right = np.linalg.svd(weighted, full_matrices=False)
        active = singular > 1.0e-11 * singular[0]
        expected = 1 + (
            self.sphere.L + 1 if self.sphere.L % 2 == 0 else self.sphere.L
        )
        assert np.sum(~active) == expected, (np.sum(~active), expected)
        assert singular[active][-1] > 1.0e-6 * singular[0]
        self.pseudo_inverse = (right[active].T / singular[active]) @ left[
            :, active
        ].T
        self.kernel = right[~active].T

    def fit(self, q_target, provisional):
        """The least-squares fit of the gradient targets over the trial
        space, completed in its derivative kernel by the M-projections of the
        provisional rates."""
        root = np.sqrt(self.mass)
        stacked = np.concatenate(
            [root[None, :] * q_target[i] for i in range(3)], 1
        )
        coefficients = (
            stacked @ self.pseudo_inverse.T
            + ((provisional * self.mass) @ self.trial)
            @ self.kernel
            @ self.kernel.T
        )
        return coefficients @ self.trial.T

    def group_filter(self, values, structure, modes_to_kill):
        """The volume-mass-orthogonal projector onto the radial polynomials
        times the retained angular space of the given component structure,
        applied to `values` (components x points). The radial nodal space
        is complete, so it acts on every radial slice separately with that
        slice's mass."""
        projector = self.sphere.structured_projector(structure, modes_to_kill)
        components = values.shape[0]
        root_weights = np.sqrt(np.tile(self.sphere.weights, components))
        symmetric = root_weights[:, None] * projector / root_weights[None, :]
        eigenvalues, eigenvectors = np.linalg.eigh(
            0.5 * (symmetric + symmetric.T)
        )
        assert np.max(
            np.minimum(np.abs(eigenvalues), np.abs(eigenvalues - 1))
        ) < (1.0e-10)
        basis = eigenvectors[:, eigenvalues > 0.5] / root_weights[:, None]
        result = np.empty_like(values)
        for radial in range(self.radial_points):
            ids = np.arange(self.sphere.n) * self.radial_points + radial
            mass = np.tile(self.mass[ids], components)
            slice_values = values[:, ids].ravel()
            coefficients = np.linalg.solve(
                basis.T @ (mass[:, None] * basis),
                basis.T @ (mass * slice_values),
            )
            result[:, ids] = (basis @ coefficients).reshape(components, -1)
        return result


def _boundary_filter(correction, sphere, modes_to_kill):
    """The native hard tensor spherical-harmonic filter (W-orthogonal nodal
    projector) of the 43 w and q corrections on a shell face, blockwise in
    their tensor structures."""
    result = correction.copy()
    blocks = [
        (range(5), "stf"),
        ([5], "scalar"),
        ([6], "scalar"),
        (range(7, 10), "vector"),
        (range(10, 13), "vector"),
        ([13 + 10 * i + a for i in range(3) for a in range(5)], "rank3stf"),
        ([13 + 10 * i + 5 for i in range(3)], "vector"),
        ([13 + 10 * i + 6 for i in range(3)], "vector"),
        ([13 + 10 * i + 7 + j for i in range(3) for j in range(3)], "rank2"),
    ]
    covered = sorted(c for ids, _ in blocks for c in ids)
    assert covered == list(range(43))
    for ids, structure in blocks:
        ids = list(ids)
        projector = sphere.structured_projector(structure, modes_to_kill)
        result[ids] = (projector @ correction[ids].ravel()).reshape(
            len(ids), -1
        )
    return result


def _penalty_projectors(c, b, d, e, normal, f):
    """The spectral projectors of the full first-order normal matrix A_n
    (unit normal n) onto the propagating modes (nonzero speeds), the three
    stationary GammaHat-b modes and the twenty stationary gradient modes.
    With A_n = J F, M = F J (the 23-component normal block) and the
    decomposition jump = J z + (0, t) (t tangential), the GammaHat-b part is
    J Pi_0(M) z, Pi_0(M) the spectral projector of M onto its kernel; this
    fixes the stationary gradient modes to carry no GammaHat-b amplitude."""
    a = _normal_matrix(c, b, d, e, normal)
    speeds2 = []
    for value in (1.0, 2.0, f, 4.0 * f / 3.0):
        if all(abs(value - other) > 1.0e-12 for other in speeds2):
            speeds2.append(value)

    def kernel_projector(matrix):
        identity = np.eye(len(matrix))
        square = matrix @ matrix
        result = identity.copy()
        for speed2 in speeds2:
            result = result @ (square - speed2 * identity) / (-speed2)
        return result

    flux = np.vstack([a[:13], np.hstack([d, *e])])
    embedding = np.zeros((43, 23))
    embedding[:13, :13] = np.eye(13)
    normal_part = np.zeros((23, 43))
    normal_part[:13, :13] = np.eye(13)
    for i in range(3):
        embedding[13 + 10 * i : 23 + 10 * i, 13:] = normal[i] * np.eye(10)
        normal_part[13:, 13 + 10 * i : 23 + 10 * i] = normal[i] * np.eye(10)
    assert np.max(np.abs(embedding @ flux - a)) < 1.0e-13
    propagating = np.eye(43) - kernel_projector(a)
    gamma_hat_minus_b = (
        embedding @ kernel_projector(flux @ embedding) @ normal_part
    )
    stationary_gradient = np.eye(43) - propagating - gamma_hat_minus_b
    for projector, rank in (
        (propagating, 20),
        (gamma_hat_minus_b, 3),
        (stationary_gradient, 20),
    ):
        assert np.max(np.abs(projector @ projector - projector)) < 1.0e-10
        assert abs(np.trace(projector) - rank) < 1.0e-10
    for stationary in (gamma_hat_minus_b, stationary_gradient):
        assert np.max(np.abs(a @ stationary)) < 1.0e-10
        assert np.max(np.abs(stationary @ a)) < 1.0e-10
    return a, propagating, stationary_gradient, gamma_hat_minus_b


def _wedge_face_chart(xyz, plane_gradients, plane_extents):
    """The signed axis permutation, angular map and refinement chart of a
    wedge face (or refined wedge-face patch) on a sphere, from its node
    coordinates `xyz` (3 x points) and the gradients of its two logical
    coordinates (points x 2 x 3, from the inverse Jacobian): after the
    permutation, the face logical coordinates are affine images
    xi = alpha g(X) + beta of g(X), X = x/z, with g the identity
    (equidistant) or (4/pi) arctan (equiangular); alpha = 2^(refinement
    level) and beta the segment offset (alpha = 1, beta = 0 for an
    unrefined face). The node positions fix (alpha, beta) per axis and the
    gradients must match alpha dg(X)/dx, which also distinguishes the two
    maps when the positions alone cannot. Returns a function giving the
    face logical coordinates of directions, infinite for directions on the
    opposite side."""
    nodes = [
        np.tile(_lgl(plane_extents[0])[0], plane_extents[1]),
        np.repeat(_lgl(plane_extents[1])[0], plane_extents[0]),
    ]
    maps = (
        (lambda v: v, lambda v: np.ones_like(v)),
        (
            lambda v: 4.0 / np.pi * np.arctan(v),
            lambda v: 4.0 / np.pi / (1.0 + v**2),
        ),
    )
    scale = np.max(np.abs(plane_gradients))
    found = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            rotated = np.array(signs)[:, None] * xyz[list(permutation)]
            if np.any(rotated[2] <= 0.0):
                continue
            # d(u_k)/dx_j = signs_k delta(j, permutation_k)
            derivatives = np.zeros((3, 3))
            derivatives[np.arange(3), list(permutation)] = signs
            for inverse, inverse_derivative in maps:
                matches = True
                affine = []
                for axis in range(2):
                    ratio = rotated[axis] / rotated[2]
                    u = inverse(ratio)
                    spread = np.max(u) - np.min(u)
                    if spread < 1.0e-10:
                        matches = False
                        break
                    alpha = (np.max(nodes[axis]) - np.min(nodes[axis])) / (
                        spread
                    )
                    beta = np.max(nodes[axis]) - alpha * np.max(u)
                    affine.append((alpha, beta))
                    gradient = (
                        alpha
                        * inverse_derivative(ratio)[:, None]
                        * (
                            derivatives[axis][None, :]
                            - ratio[:, None] * derivatives[2][None, :]
                        )
                        / rotated[2][:, None]
                    )
                    matches = (
                        matches
                        and np.max(np.abs(alpha * u + beta - nodes[axis]))
                        < 1.0e-10
                        and np.max(np.abs(gradient - plane_gradients[:, axis]))
                        < 1.0e-10 * scale
                    )
                if matches:
                    found.append(
                        (permutation, np.array(signs), inverse, affine)
                    )
    assert len(found) == 1, len(found)
    permutation, signs, inverse, affine = found[0]

    def chart(targets):
        rotated = signs[:, None] * targets[list(permutation)]
        with np.errstate(divide="ignore", invalid="ignore"):
            logical = np.array(
                [
                    affine[0][0] * inverse(rotated[0] / rotated[2])
                    + affine[0][1],
                    affine[1][0] * inverse(rotated[1] / rotated[2])
                    + affine[1][1],
                ]
            )
        return np.where(rotated[2] > 0.0, logical, np.inf)

    return chart


def _connect(elements):
    """Pairs the faces of the elements. Conforming faces have matching node
    coordinates. A shell face without a conforming partner connects to all
    unmatched LGL faces on the same sphere (nonconforming wedge faces). All
    other faces are external."""
    faces = [face for element in elements for face in element.faces]
    scale = max(np.max(np.abs(element.xyz)) for element in elements)
    for face in faces:
        face["kind"] = "external"
        mine = elements[face["element"]].xyz[face["ids"]]
        face["radius"] = np.linalg.norm(mine, axis=1)
        for other in faces:
            if other is face or len(other["ids"]) != len(face["ids"]):
                continue
            theirs = elements[other["element"]].xyz[other["ids"]]
            distance = np.linalg.norm(
                mine[:, None, :] - theirs[None, :, :], axis=2
            )
            match = np.argmin(distance, axis=1)
            if np.max(distance[np.arange(len(match)), match]) < 1.0e-10 * scale:
                assert len(set(match)) == len(match)
                face["kind"] = "conforming"
                face["outside"] = (other["element"], other["ids"][match])
    for face in faces:
        shell = elements[face["element"]]
        if face["kind"] != "external" or not shell.shell:
            continue
        radius = face["radius"][0]
        if np.max(np.abs(face["radius"] - radius)) > 1.0e-12 * scale:
            continue
        wedges = [
            other
            for other in faces
            if other["kind"] == "external"
            and not elements[other["element"]].shell
            and np.max(np.abs(other["radius"] - radius)) < 1.0e-12 * scale
        ]
        if not wedges:
            continue
        # The shell receives the wedge traces interpolated (polynomially, in
        # the wedge face logical coordinates) to the shell points each wedge
        # contains, averaged over the contributing wedges.
        shell_directions = shell.xyz[face["ids"]].T / face["radius"]
        counts = np.zeros(len(face["ids"]))
        transfers = []
        for wedge_face in wedges:
            wedge = elements[wedge_face["element"]]
            plane = wedge_face["plane"]["extents"]
            wedge_directions = (
                wedge.xyz[wedge_face["ids"]].T / wedge_face["radius"]
            )
            logical = _wedge_face_chart(
                wedge.xyz[wedge_face["ids"]].T,
                wedge_face["plane"]["gradients"],
                plane,
            )(shell_directions)
            excess = np.max(np.abs(logical), axis=0) - 1.0
            assert not np.any((excess > 1.0e-12) & (excess < 1.0e-9))
            inside = excess <= 1.0e-12
            logical = np.clip(logical[:, inside], -1.0, 1.0)
            xi_matrix = _lagrange(_lgl(plane[0])[0], logical[0])
            eta_matrix = _lagrange(_lgl(plane[1])[0], logical[1])
            transfer = np.zeros((len(face["ids"]), len(wedge_face["ids"])))
            transfer[inside] = (
                eta_matrix[:, :, None] * xi_matrix[:, None, :]
            ).reshape(np.sum(inside), -1)
            counts += inside
            transfers.append((wedge_face, transfer))
            # The wedge receives the shell trace interpolated natively to
            # its face points.
            wedge_face["kind"] = "nonconforming"
            wedge_face["sources"] = [
                (
                    face["element"],
                    face["ids"],
                    shell.sphere.cardinal(wedge_directions),
                )
            ]
        assert np.all(counts > 0)
        face["kind"] = "nonconforming"
        face["filtered"] = True
        face["max_contributors"] = int(counts.max())
        face["sources"] = [
            (
                wedge_face["element"],
                wedge_face["ids"],
                transfer / counts[:, None],
            )
            for wedge_face, transfer in transfers
        ]
    return faces


def linearized_rates_on_general_mesh(
    state,
    geometry,
    layout,
    kappa,
    eta,
    f,
    external_factor,
    penalties,
    shell_angular_modes_to_kill,
    shell_boundary_modes_to_kill,
    require_multiple_contributors=0.0,
):
    """The operator applied to `state` about Minkowski space with analytic
    (unperturbed) exterior data on external faces, on a mesh of curved LGL
    elements and spherical-harmonic shells, conforming or connected
    nonconformingly at wedge-shell interfaces.

    `geometry` holds, for every point, the inertial coordinates and the
    inverse Jacobian d xi^a / d x^i (12 values, index 3 + 3 a + i);
    `layout` holds, for every element, (shell flag, extents); the points of
    the elements follow each other. Layout of `state`: component-major, then
    element, then point.

    `penalties` are the internal scalar penalties of the propagating, the
    stationary gradient and the stationary GammaHat-b modes. On shells the
    velocity filter at unit gain (the volume-mass projector of the
    potential rates and GammaHat-b onto the retained angular space of
    `shell_angular_modes_to_kill` removed bands) acts on the state first. A
    nonnegative `shell_boundary_modes_to_kill` filters the w and q
    corrections received by shells from wedges before the lift."""
    layout = np.asarray(layout).reshape(-1, 4)
    data = np.asarray(geometry).reshape(-1, 12)
    elements = []
    offset = 0
    for index, row in enumerate(layout):
        extents = [int(round(value)) for value in row[1:]]
        n = int(np.prod(extents))
        xyz = data[offset : offset + n, :3]
        inverse_jacobian = data[offset : offset + n, 3:].reshape(n, 3, 3)
        element_type = _ShellElement if row[0] > 0.5 else _LglElement
        elements.append(element_type(index, xyz, inverse_jacobian, extents))
        offset += n
    assert offset == len(data)
    offsets = np.cumsum([0] + [element.n for element in elements])
    faces = _connect(elements)
    if require_multiple_contributors > 0.5:
        # The domain must put shell points on refined wedge-patch
        # boundaries, so the multi-contributor averaging is exercised.
        assert any(
            face.get("max_contributors", 1) > 1 for face in faces
        ), "no shell point has several wedge contributors"
    c, b, d, e = _symbol(f)
    sigma_p, sigma_t, sigma_g = penalties
    gauge = np.zeros((3, 13))
    gauge[:, 7:10] = _EYE
    gauge[:, 10:13] = -_EYE
    recover = np.linalg.inv(np.vstack([d, gauge]))
    rhs_inverse = recover[:, :10]
    shell_angular_modes_to_kill = int(round(shell_angular_modes_to_kill))
    boundary_modes_to_kill = (
        int(round(shell_boundary_modes_to_kill))
        if shell_boundary_modes_to_kill >= 0.0
        else None
    )

    fields = np.asarray(state).reshape(23, -1)
    w = [
        fields[:13, offsets[k] : offsets[k + 1]].copy()
        for k in range(len(elements))
    ]
    u = [fields[13:, offsets[k] : offsets[k + 1]] for k in range(len(elements))]
    q = [
        [u[k] @ element.derivatives[i].T for i in range(3)]
        for k, element in enumerate(elements)
    ]
    for k, element in enumerate(elements):
        if not element.shell:
            continue
        gradient_rates = sum(e[i] @ q[k][i] for i in range(3))
        rates = d @ w[k] + gradient_rates
        filtered = np.vstack(
            [
                element.group_filter(
                    rates[:5], "stf", shell_angular_modes_to_kill
                ),
                element.group_filter(
                    rates[5:6], "scalar", shell_angular_modes_to_kill
                ),
                element.group_filter(
                    rates[6:7], "scalar", shell_angular_modes_to_kill
                ),
                element.group_filter(
                    rates[7:10], "vector", shell_angular_modes_to_kill
                ),
                element.group_filter(
                    gauge @ w[k], "vector", shell_angular_modes_to_kill
                ),
            ]
        )
        filtered[:10] -= gradient_rates
        w[k] = recover @ filtered
    y = [np.concatenate([w[k], *q[k]]) for k in range(len(elements))]
    source = _source(kappa, eta)
    targets = []
    provisional = []
    for k, element in enumerate(elements):
        target = source @ y[k]
        for i in range(3):
            target += _normal_matrix(c, b, d, e, _EYE[i]) @ (
                y[k] @ element.derivatives[i].T
            )
        targets.append(target)
        provisional.append(d @ w[k] + sum(e[i] @ q[k][i] for i in range(3)))

    for face in faces:
        k, ids = face["element"], face["ids"]
        interior_y = y[k][:, ids]
        interior_u = u[k][:, ids]
        internal = face["kind"] != "external"
        if face["kind"] == "conforming":
            other, other_ids = face["outside"]
            exterior_y = y[other][:, other_ids]
            exterior_u = u[other][:, other_ids]
        elif face["kind"] == "nonconforming":
            exterior_y = sum(
                y[other][:, other_ids] @ transfer.T
                for other, other_ids, transfer in face["sources"]
            )
            exterior_u = sum(
                u[other][:, other_ids] @ transfer.T
                for other, other_ids, transfer in face["sources"]
            )
        else:
            exterior_y = np.zeros_like(interior_y)
        jump = exterior_y - interior_y
        correction = np.zeros_like(jump)
        for point in range(len(ids)):
            a, propagating, stationary, gamma_hat_minus_b = _penalty_projectors(
                c, b, d, e, face["normal"][point], f
            )
            dissipation = (1.0 if internal else external_factor) * _absolute(
                a, f
            )
            if internal:
                dissipation = dissipation + (
                    sigma_p * propagating
                    + sigma_t * stationary
                    + sigma_g * gamma_hat_minus_b
                )
            correction[:, point] = 0.5 * (a + dissipation) @ jump[:, point]
        if face.get("filtered", False) and boundary_modes_to_kill is not None:
            correction = _boundary_filter(
                correction, elements[k].sphere, boundary_modes_to_kill
            )
        targets[k][:, ids] += face["lift"] * correction
        if internal:
            provisional[k][:, ids] += (
                0.5 * face["lift"] * (exterior_u - interior_u)
            )

    velocity_rates = []
    potential_rates = []
    for k, element in enumerate(elements):
        q_target = targets[k][13:].reshape(3, 10, element.n)
        rates = element.fit(q_target, provisional[k])
        defect = sum(
            e[i] @ (q_target[i] - rates @ element.derivatives[i].T)
            for i in range(3)
        )
        velocity = targets[k][:13] + rhs_inverse @ defect
        # The fitted shift damping, see `generator`.
        velocity[10:13] -= eta * (rates[7:10] / f - w[k][10:13])
        velocity_rates.append(velocity)
        potential_rates.append(rates)
    return np.concatenate(
        [np.hstack(velocity_rates), np.hstack(potential_rates)]
    ).ravel()


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
    """`linearized_rates_on_general_mesh` on a conforming mesh of curved LGL
    elements with `points` points per dimension, with the scalar penalty
    `scalar_penalty` on all internal modes."""
    points = int(round(points))
    elements = int(round(elements))
    return linearized_rates_on_general_mesh(
        state,
        geometry,
        np.tile([0.0, points, points, points], elements),
        kappa,
        eta,
        f,
        external_factor,
        [scalar_penalty] * 3,
        0.0,
        -1.0,
    )
