# Distributed under the MIT License.
# See LICENSE.txt for details.

# Independent reference implementation of the pointwise algebra of the CCZ4
# gradient-projection scheme, used as a test oracle. The chart maps use SciPy's
# Pade/Schur matrix exponential, logarithm and Frechet derivatives (not an
# eigendecomposition), the normal matrix is assembled column by column from the
# explicit tensor formulas, and the absolute value is computed numerically.
# The argument `f` is the Gamma-driver parameter of beta_t = f b.
# All arrays crossing the C++ interface are flat: 3x3 matrices and 43x43
# matrices row-major, gradients q[i][A] as q[10 * i + A].

from decimal import Decimal, localcontext

import numpy as np
from scipy.linalg import eig, expm, expm_frechet, logm, schur, solve_sylvester

STF = np.zeros((5, 3, 3))
STF[0] = np.diag([1.0, -1.0, 0.0]) / np.sqrt(2.0)
STF[1] = np.diag([1.0, 1.0, -2.0]) / np.sqrt(6.0)
for _a, (_i, _j) in enumerate([(0, 1), (0, 2), (1, 2)], 2):
    STF[_a, _i, _j] = STF[_a, _j, _i] = 1.0 / np.sqrt(2.0)
SYM = np.concatenate([STF, [np.eye(3) / np.sqrt(3.0)]])


def tensor(v):
    return np.einsum("a,aij->ij", v, STF)


def encode(x):
    return np.einsum("aij,ij->a", STF, x)


def stf_basis_matrix(a):
    return STF[int(a)].ravel()


# Chart maps


def tangent_matrix(h, x):
    # D(exp(2H))[X]
    return expm_frechet(2.0 * h, 2.0 * x, compute_expm=False)


def second_derivative_matrix(h, x, y):
    # D^2 exp(2H)[2X, 2Y] from the (1,3) blocks of the exponential of a block
    # upper-bidiagonal matrix, symmetrized over the two directions.
    def block(e, f):
        big = np.zeros((9, 9))
        big[0:3, 0:3] = big[3:6, 3:6] = big[6:9, 6:9] = 2.0 * h
        big[0:3, 3:6] = 2.0 * e
        big[3:6, 6:9] = 2.0 * f
        return expm(big)[0:3, 6:9]

    return block(x, y) + block(y, x)


def log_metric(g):
    return (0.5 * logm(np.reshape(g, (3, 3))).real).ravel()


def metric(h):
    return expm(2.0 * np.reshape(h, (3, 3))).ravel()


def tangent(h, x):
    return tangent_matrix(np.reshape(h, (3, 3)), np.reshape(x, (3, 3))).ravel()


def inverse_tangent(h, x):
    h = np.reshape(h, (3, 3))
    columns = np.array(
        [tangent_matrix(h, s) for s in SYM]
    )  # images of the basis
    matrix = np.einsum("aij,bij->ab", SYM, columns)
    coefficients = np.linalg.solve(
        matrix, np.einsum("aij,ij->a", SYM, np.reshape(x, (3, 3)))
    )
    return np.einsum("a,aij->ij", coefficients, SYM).ravel()


def second_derivative(h, x, y):
    return second_derivative_matrix(
        np.reshape(h, (3, 3)), np.reshape(x, (3, 3)), np.reshape(y, (3, 3))
    ).ravel()


def exp_divided_difference(x, y):
    with localcontext() as context:
        context.prec = 60
        x, y = Decimal(x), Decimal(y)
        if x == y:
            return float(x.exp())
        return float((x.exp() - y.exp()) / (x - y))


def exp_second_divided_difference(x, y, z):
    with localcontext() as context:
        context.prec = 80
        a, b, c = sorted([Decimal(x), Decimal(y), Decimal(z)])

        def first(p, q):
            if p == q:
                return p.exp()
            return (p.exp() - q.exp()) / (p - q)

        if a == c:
            return float(a.exp() / 2)
        return float((first(b, c) - first(a, b)) / (c - a))


# Model functions (adapted from the archived nonlinear algebra reference)


def geometry(h5):
    h = tensor(h5)
    g = expm(2.0 * h)
    columns = np.array(
        [expm_frechet(2.0 * h, 2.0 * x, compute_expm=False) for x in SYM]
    )
    matrix = np.einsum("aij,bij->ab", SYM, columns)

    def forward(x):
        return np.einsum("a,aij->ij", np.einsum("aij,ij->a", SYM, x), columns)

    def inverse(x):
        return np.einsum(
            "a,aij->ij",
            np.linalg.solve(matrix, np.einsum("aij,ij->a", SYM, x)),
            SYM,
        )

    return g, np.linalg.inv(g), forward, inverse


def original_rhs_impl(u, w, q, f):
    g, _, _, inverse = geometry(u[:5])
    alpha = u[6]
    shift = u[7:]
    shift_gradient = q[:, 7:]
    divergence = np.trace(shift_gradient)
    strain = shift_gradient @ g + g @ shift_gradient.T - 2 * g * divergence / 3
    result = np.zeros(10, dtype=np.result_type(u, w, q))
    result[:5] = -alpha * w[:5] + shift @ q[:, :5] + encode(inverse(strain))
    result[5] = shift @ q[:, 5] + (alpha * w[5] - divergence) / 3
    result[6] = shift @ q[:, 6] - 2 * alpha * (w[5] - 2 * w[6])
    result[7:] = f * w[10:]
    return result


def recover_impl(u, q, rhs, difference, f):
    g, _, _, inverse = geometry(u[:5])
    alpha = u[6]
    shift = u[7:]
    bgrad = q[:, 7:]
    divergence = np.trace(bgrad)
    strain = bgrad @ g + g @ bgrad.T - 2 * g * divergence / 3
    w = np.zeros(13)
    w[:5] = (shift @ q[:, :5] + encode(inverse(strain)) - rhs[:5]) / alpha
    w[5] = (3 * (rhs[5] - shift @ q[:, 5]) + divergence) / alpha
    w[6] = (rhs[6] - shift @ q[:, 6]) / (4 * alpha) + w[5] / 2
    w[10:] = rhs[7:] / f
    w[7:10] = difference + w[10:]
    return w


def curved_normal_impl(u, normal, f):
    g, gi, _, inverse = geometry(u[:5])
    alpha, phi, beta = u[6], np.exp(u[5]), u[7:]
    speed = beta @ normal
    raised = gi @ normal

    def tf(x):
        return x - g * np.einsum("ij,ij->", gi, x) / 3

    def apply(y):
        w = y[:13]
        q = y[13:].reshape(3, 10)
        result = np.zeros(43)
        gamma_gradient = np.outer(normal, g @ w[7:10])
        conformal_hessian = np.outer(normal, q[:, 5])
        lapse_hessian = np.outer(normal, q[:, 6])
        metric_source = (
            alpha
            * phi**2
            * tf(
                (
                    gamma_gradient
                    + gamma_gradient.T
                    + conformal_hessian
                    + conformal_hessian.T
                )
                / 2
            )
        )
        metric_source -= phi**2 * tf((lapse_hessian + lapse_hessian.T) / 2)
        result[:5] = (
            speed * w[:5]
            - alpha * phi**2 * (raised @ q[:, :5])
            + 2 * encode(inverse(metric_source))
        )
        result[5] = (
            speed * w[5]
            + alpha * phi**2 * (normal @ w[7:10] + 4 * raised @ q[:, 5])
            - phi**2 * raised @ q[:, 6]
        )
        result[6] = speed * w[6] + alpha * phi**2 * (
            normal @ w[7:10] / 2 + 2 * raised @ q[:, 5]
        )
        result[7:10] = (
            speed * w[7:10]
            + alpha * raised * (2 * w[6] - 4 * w[5] / 3)
            + raised @ q[:, 7:]
            + raised * np.trace(q[:, 7:]) / 3
        )
        result[10:13] = result[7:10]
        result[13:] = np.outer(normal, original_rhs_impl(u, w, q, f)).ravel()
        return result

    return np.column_stack([apply(v) for v in np.eye(43)])


def jacobian(function, value):
    step = 1.0e-25
    return np.column_stack(
        [
            np.imag(function(value.astype(complex) + 1j * step * v)) / step
            for v in np.eye(len(value))
        ]
    )


def eig_absolute(a):
    values, vectors = eig(a)
    result = vectors @ np.diag(np.abs(values.real)) @ np.linalg.inv(vectors)
    return result.real


def schur3_absolute(a, tol=1.0e-8):
    # Ordered real Schur blocks (+, -, 0) and Sylvester couplings.
    scale = np.linalg.norm(a)
    t, q, npos = schur(a, output="real", sort=lambda x, y: x > tol * scale)
    t22, q2, nneg = schur(
        t[npos:, npos:], output="real", sort=lambda x, y: x < -tol * scale
    )
    q[:, npos:] = q[:, npos:] @ q2
    t[:npos, npos:] = t[:npos, npos:] @ q2
    t[npos:, npos:] = t22
    cut = [0, npos, npos + nneg, len(a)]
    blocks = [slice(cut[i], cut[i + 1]) for i in range(3)]
    assert np.linalg.norm(t[blocks[2], blocks[2]]) <= 1.0e-10 * scale
    b = np.zeros_like(t)
    b[blocks[0], blocks[0]] = t[blocks[0], blocks[0]]
    b[blocks[1], blocks[1]] = -t[blocks[1], blocks[1]]
    for i, j in [(0, 1), (1, 2), (0, 2)]:
        bi, bj = blocks[i], blocks[j]
        if cut[i + 1] == cut[i] or cut[j + 1] == cut[j]:
            continue
        rhs = b[bi, bi] @ t[bi, bj] - t[bi, bj] @ b[bj, bj]
        for k in range(i + 1, j):
            bk = blocks[k]
            rhs += b[bi, bk] @ t[bk, bj] - t[bi, bk] @ b[bk, bj]
        b[bi, bj] = solve_sylvester(t[bi, bi], -t[bj, bj], rhs)
    return q @ b @ q.T


# Flat pypp interface


def normal_matrix(u, normal, f):
    return curved_normal_impl(np.array(u), np.array(normal), f).ravel()


def absolute_normal_matrix_eig(u, normal, f):
    return eig_absolute(
        curved_normal_impl(np.array(u), np.array(normal), f)
    ).ravel()


def absolute_normal_matrix_schur(u, normal, f):
    return schur3_absolute(
        curved_normal_impl(np.array(u), np.array(normal), f)
    ).ravel()


def eigenvector_condition(u, normal, f):
    values, vectors = eig(curved_normal_impl(np.array(u), np.array(normal), f))
    return float(np.linalg.cond(vectors))


def sorted_speeds(u, normal, f):
    values = np.linalg.eigvals(
        curved_normal_impl(np.array(u), np.array(normal), f)
    )
    return np.sort(values.real)


def max_imaginary_speed(u, normal, f):
    values = np.linalg.eigvals(
        curved_normal_impl(np.array(u), np.array(normal), f)
    )
    return float(np.max(np.abs(values.imag)))


def original_rhs(u, w, q, f):
    return original_rhs_impl(
        np.array(u), np.array(w), np.reshape(np.array(q), (3, 10)), f
    )


def recover(u, q, rhs, difference, f):
    return recover_impl(
        np.array(u),
        np.reshape(np.array(q), (3, 10)),
        np.array(rhs),
        np.array(difference),
        f,
    )


def rhs_potential_jacobian(u, w, q, f):
    q = np.reshape(np.array(q), (3, 10))
    return jacobian(
        lambda v: original_rhs_impl(v, np.array(w), q, f), np.array(u)
    ).ravel()


# Physical fields and exact derivatives from chart fields with given first
# and second coordinate derivatives at a point.


def metric_and_derivatives(h5, dh5, ddh5):
    # dh5[5 * i + c], ddh5[5 * (3 * i + j) + c]
    h = tensor(np.array(h5))
    dh = [tensor(np.array(dh5)[5 * i : 5 * i + 5]) for i in range(3)]
    ddh = [
        [
            tensor(np.array(ddh5)[5 * (3 * i + j) : 5 * (3 * i + j) + 5])
            for j in range(3)
        ]
        for i in range(3)
    ]
    g = expm(2.0 * h)
    dg = np.array([tangent_matrix(h, dh[i]) for i in range(3)])
    ddg = np.array(
        [
            [
                tangent_matrix(h, ddh[i][j])
                + second_derivative_matrix(h, dh[i], dh[j])
                for j in range(3)
            ]
            for i in range(3)
        ]
    )
    return np.concatenate([g.ravel(), dg.ravel(), ddg.ravel()])


def a_tilde_and_derivative(h5, dh5, a5, da5):
    h = tensor(np.array(h5))
    a = tensor(np.array(a5))
    dh = [tensor(np.array(dh5)[5 * i : 5 * i + 5]) for i in range(3)]
    da = [tensor(np.array(da5)[5 * i : 5 * i + 5]) for i in range(3)]
    a_tilde = 0.5 * tangent_matrix(h, a)
    d_a_tilde = np.array(
        [
            0.5
            * (tangent_matrix(h, da[i]) + second_derivative_matrix(h, dh[i], a))
            for i in range(3)
        ]
    )
    return np.concatenate([a_tilde.ravel(), d_a_tilde.ravel()])


def a_tilde_time_derivative(h5, h_rate5, a5, a_rate5):
    # d/dt of (1/2) D(exp(2H))[2a] / 2 along H_t, a_t by a complex step in t,
    # independent of the block construction of the second derivative.
    h = tensor(np.array(h5))
    h_rate = tensor(np.array(h_rate5))
    a = tensor(np.array(a5))
    a_rate = tensor(np.array(a_rate5))
    step = 1.0e-30
    perturbed = expm_frechet(
        2.0 * (h + 1j * step * h_rate),
        2.0 * (a + 1j * step * a_rate),
        compute_expm=False,
    )
    return (0.5 * np.imag(perturbed) / step).ravel()
