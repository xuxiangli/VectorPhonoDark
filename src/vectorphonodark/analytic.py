"""Analytic results for mcalI matrix calculation.

Dimensionless integrals of functions of q and v_min(q), as functions
of x = (q / q_star)**2, for q_star**2 = 2 * omegaS * mX.

All kernels take the mediator form factor in the general form
    F_DM**2 = (q/q0)**a * (v/v0)**b,    with v0 = c = 1,
where (a, b) is the physics parameter ``fdm``; the common cases are the
heavy mediator (a, b) = (0, 0) and the light mediator (a, b) = (-4, 0).

Functions:
    _t: integral for rectangular volume, [[v1,v2],[q1,q2]]
    _u: integral where v_min(q) sets the lower bound, [v_min(q),v2]
    _b: intermediate function shared by both integrals
    _c, _v, _s: intermediate functions for the non-rectangular integral (_u)

MIRROR: this module and ``analytic_cy.pyx`` are maintained as structural
mirrors -- same functions, same arithmetic, one function per function --
and ``tests/test_analytic_kernels.py`` checks them against each other.
Apply any change to both; an optimization expressible in only one backend
is applied to neither.

The analytic kernels in this module are adapted from vsdm
(https://github.com/blillard/vsdm) by B. Lillard and A. Radick
[arXiv:2502.17547].
"""

import math

import numba
import numpy as np
import scipy.special as spf
from numba import objmode

from . import basis_funcs


@numba.njit
def _b_nk_int(n, k, x):
    # x = q^2 / q_*^2.
    assert k - int(k) == 0, "'k' should be integer valued"
    k = int(k)
    assert x != 0, (
        "Range in 'x' should not include x=0, even if basis function includes q=0."
    )
    total = 0.0
    comb = 1.0
    for j in range(k + 1):
        # comb = math.gamma(k + 1) / (math.gamma(j + 1) * math.gamma(k - j + 1))
        if j > 0:
            comb *= (k - j + 1.0) / j
        ipower = j + (n - k) / 2 + 1
        if ipower == 0:
            summand = math.log(x)
        else:
            summand = (x**ipower) / ipower
        total += comb * summand
    return 0.5 * total


@numba.njit
def _c_alpha_int(alpha, x):
    if alpha % 1 != 0:  # float valued
        with objmode(val="float64"):
            val = spf.hyp2f1(1, alpha, 1 + alpha, -x)
        return x ** (alpha) / alpha**2 * (0.5 - val)
    else:  # integer-valued
        alpha = int(alpha)
    if alpha == -2:
        # return -0.5*math.log( (1+x)/x ) + 0.5/x - 0.125/x**2
        return -0.5 * math.log((1 + x) / x) + (0.5 - 0.125 / x) / x
    elif alpha == -1:
        return math.log((1 + x) / x) - 0.5 / x
    elif alpha == 0:
        with objmode(val="float64"):
            val = spf.spence(1 + x)
        # return (-0.25*(math.log(x))**2 + math.log(x) * math.log(1 + x) + val)
        return -0.25 * (math.log(x)) ** 2 + math.log(x) * math.log1p(x) + val
    elif alpha == 1:
        # return 0.5*x - math.log(1 + x)
        return 0.5 * x - math.log1p(x)
    elif alpha == 2:
        # return 0.125*x**2 - 0.5*x + 0.5*math.log(1 + x)
        return (0.125 * x - 0.5) * x + 0.5 * math.log1p(x)
    elif alpha > 1:  # other positive integers
        # sum = (-1)**alpha * math.log(1+x) / alpha + (1+x)**alpha / (2*alpha**2)
        total = (-1) ** alpha * math.log1p(x) / alpha + (1 + x) ** alpha / (
            2 * alpha**2
        )
        comb_alpha_j = alpha
        for j in range(1, alpha):
            # comb = (
            #     math.gamma(alpha+1) / (math.gamma(j+1) * math.gamma(alpha-j+1))
            #     + math.gamma(alpha) / (math.gamma(j+1) * math.gamma(alpha-j))
            # )
            comb = comb_alpha_j * (2.0 * alpha - j) / j
            comb_alpha_j *= (alpha - j) / (j + 1.0)
            total += (-1) ** (alpha - j) / (2 * alpha * j) * comb * (1 + x) ** j
        return total
    elif alpha < -1:  # other negative integers
        yx = (1 + x) / x
        total = (-1) ** alpha * math.log(yx) / alpha - yx ** (-alpha) / (2 * alpha**2)
        comb_alpha_j = -alpha
        for j in range(1, -alpha):
            # comb = (
            #     math.gamma(-alpha+1) / (math.gamma(j+1) * math.gamma(-alpha-j+1))
            #     + math.gamma(-alpha) / (math.gamma(j+1) * math.gamma(-alpha-j))
            # )
            comb = comb_alpha_j * (-2.0 * alpha - j) / j
            comb_alpha_j *= (-alpha - j) / (j + 1.0)
            total += (-1) ** (alpha + j) / (2 * alpha * j) * comb * yx**j
        return total


@numba.njit
def _v_ab_int(a, b, x):
    sum_2 = 0.0
    comb = 1.0
    for j in range(b + 2 + 1):
        # comb = math.gamma(b+2+1) / (math.gamma(j+1) * math.gamma(b+2-j+1))
        if j > 0:
            comb *= (b + 2 - j + 1.0) / j
        if 2 * j == (b - a):
            sum_2 += comb * math.log(x)
        else:
            sum_2 += comb * x ** (j + (a - b) / 2) / (j + (a - b) / 2)
    return 0.5 * sum_2


@numba.njit
def _s_ab_int(a, b, x):
    logfactor = 0.5 * math.log(x / ((1 + x) ** 2))
    total = logfactor * _v_ab_int(a, b, x)
    comb = 1.0
    for j in range(b + 2 + 1):
        # comb = math.gamma(b+2+1) / (math.gamma(j+1) * math.gamma(b+2-j+1))
        if j > 0:
            comb *= (b + 2 - j + 1.0) / j
        total += 0.5 * comb * _c_alpha_int(j + (a - b) / 2, x)
    return total


@numba.njit
def _t_l_ab_vq_int(ell, a, b, v12_star, q12_star):
    """Rectangular integral T_{l,n}.

    With [v1,v2] in units of v_star = q_star/mX, [q1,q2] in units of
        q_star = sqrt(2*mX*omegaS).

    Always v1 >= 1. Also require q1 > 0.
    """
    assert int(ell) - ell == 0, "'l' must be integer valued"
    ell = int(ell)
    [v1, v2] = v12_star
    [q1, q2] = q12_star
    x1 = q1**2
    x2 = q2**2
    total = 0.0
    k_start = ell % 2
    term_k = (
        math.gamma(0.5 * (k_start + 1 + ell))
        / math.gamma(0.5 * (k_start + 1 - ell))
        * 2.0 ** (ell - k_start)
        / (math.gamma(k_start + 1) * math.gamma(ell - k_start + 1))
    )
    for k in range(ell % 2, ell + 1, 2):
        # only terms with (l-k)%2==0 contribute to the sum:
        # term_k = 2.**(l-k) * math.gamma(0.5*(k+1+l)) / math.gamma(0.5*(k+1-l))
        # term_k /= (math.gamma(k+1)*math.gamma(l-k+1))
        term_q = _b_nk_int(a, k, x2) - _b_nk_int(a, k, x1)
        if k == b + 2:
            term_v = math.log(v2 / v1)
        else:
            term_v = (v2 ** (b + 2 - k) - v1 ** (b + 2 - k)) / (b + 2 - k)
        total += term_v * term_k * term_q
        term_k *= -(ell - k) * (ell + k + 1) / (4.0 * (k + 1) * (k + 2))
    return total


@numba.njit
def _t_l_ab_vq_int_ells(l_max, l_mod, a, b, v12_star, q12_star):
    """Rectangular integrals T_l for every l in range(0, l_max + 1, l_mod).

    Same arithmetic as _t_l_ab_vq_int, term by term; the _b_nk_int endpoint
    differences and the velocity factors depend on k but not on l, so they
    are computed once and shared by all l.
    """
    [v1, v2] = v12_star
    [q1, q2] = q12_star
    x1 = q1**2
    x2 = q2**2

    # k runs with the parity of l, so with l_mod == 2 only even k occur.
    term_q_k = np.empty(l_max + 1)
    term_v_k = np.empty(l_max + 1)
    for k in range(0, l_max + 1, l_mod):
        term_q_k[k] = _b_nk_int(a, k, x2) - _b_nk_int(a, k, x1)
        if k == b + 2:
            term_v_k[k] = math.log(v2 / v1)
        else:
            term_v_k[k] = (v2 ** (b + 2 - k) - v1 ** (b + 2 - k)) / (b + 2 - k)

    out = np.empty(l_max // l_mod + 1)
    for idx_ell in range(l_max // l_mod + 1):
        ell = idx_ell * l_mod
        total = 0.0
        k_start = ell % 2
        term_k = (
            math.gamma(0.5 * (k_start + 1 + ell))
            / math.gamma(0.5 * (k_start + 1 - ell))
            * 2.0 ** (ell - k_start)
            / (math.gamma(k_start + 1) * math.gamma(ell - k_start + 1))
        )
        for k in range(k_start, ell + 1, 2):
            total += term_v_k[k] * term_k * term_q_k[k]
            term_k *= -(ell - k) * (ell + k + 1) / (4.0 * (k + 1) * (k + 2))
        out[idx_ell] = total
    return out


@numba.njit
def _t_l_ab_vq_int_ells_tab(l_max, l_mod, b, v1, v2, pv1, pv2, bq1, bq2, ell_seeds):
    """Rectangular integrals T_l with endpoint values read from tables.

    Same sweep as _t_l_ab_vq_int_ells, term by term, with every endpoint
    evaluation read from the per-call tables (see _fill_endpoint_tables):
    pv1/pv2 hold v**(b + 2 - k) at the two velocity endpoints, bq1/bq2 hold
    _b_nk_int(a, k, q**2) at the two momentum endpoints, and ell_seeds the
    leading term_k of each l. Only valid when no kinematic clamp binds,
    i.e. the (v, q) rectangle lies entirely inside the allowed region.
    """
    # k runs with the parity of l, so with l_mod == 2 only even k occur.
    term_q_k = np.empty(l_max + 1)
    term_v_k = np.empty(l_max + 1)
    for k in range(0, l_max + 1, l_mod):
        term_q_k[k] = bq2[k] - bq1[k]
        if k == b + 2:
            term_v_k[k] = math.log(v2 / v1)
        else:
            term_v_k[k] = (pv2[k] - pv1[k]) / (b + 2 - k)

    out = np.empty(l_max // l_mod + 1)
    for idx_ell in range(l_max // l_mod + 1):
        ell = idx_ell * l_mod
        total = 0.0
        term_k = ell_seeds[idx_ell]
        for k in range(ell % 2, ell + 1, 2):
            total += term_v_k[k] * term_k * term_q_k[k]
            term_k *= -(ell - k) * (ell + k + 1) / (4.0 * (k + 1) * (k + 2))
        out[idx_ell] = total
    return out


@numba.njit
def _u_l_ab_vq_int(ell, a, b, v2_star, q12_star):
    """Non-rectangular integral U_{l,fdm}, with lower bound v1 = v_min(q).

    With v2 in units of v_star = q_star/mX, [q1,q2] in units of
        q_star = sqrt(2*mX*omegaS).
    """
    assert int(ell) - ell == 0, "'l' must be integer valued"
    ell = int(ell)
    v2 = v2_star  # only need v2, v1 is irrelevant
    [q1, q2] = q12_star
    x1 = q1**2
    x2 = q2**2
    total = 0.0
    k_start = ell % 2
    term_k = (
        math.gamma(0.5 * (k_start + 1 + ell))
        / math.gamma(0.5 * (k_start + 1 - ell))
        * 2.0 ** (ell - k_start)
        / (math.gamma(k_start + 1) * math.gamma(ell - k_start + 1))
    )
    for k in range(ell % 2, ell + 1, 2):
        # term_k = (math.gamma(0.5*(k+1+l)) / math.gamma(0.5*(k+1-l))
        #           * 2.**(l-k)/(math.gamma(k+1) * math.gamma(l-k+1)))
        if k == b + 2:
            term_x = (
                math.log(2.0 * v2) * (_b_nk_int(a, k, x2) - _b_nk_int(a, k, x1))
                + _s_ab_int(a, b, x2)
                - _s_ab_int(a, b, x1)
            )
            total += term_k * term_x
        else:
            t_x = v2 ** (b + 2 - k) * (
                _b_nk_int(a, k, x2) - _b_nk_int(a, k, x1)
            ) - 2.0 ** (k - b - 2) * (_b_nk_int(a, b + 2, x2) - _b_nk_int(a, b + 2, x1))
            total += term_k * t_x / (b + 2 - k)
        term_k *= -(ell - k) * (ell + k + 1) / (4.0 * (k + 1) * (k + 2))
    return total


@numba.njit
def _u_l_ab_vq_int_ells(l_max, l_mod, a, b, v2_star, q12_star):
    """Non-rectangular integrals U_l for every l in range(0, l_max + 1, l_mod).

    Same arithmetic as _u_l_ab_vq_int, term by term; the _b_nk_int and
    _s_ab_int endpoint values depend on k but not on l, so they are computed
    once and shared by all l. The two _s_ab_int values are kept separate
    (not pre-differenced) to preserve the exact summation order.
    """
    v2 = v2_star
    [q1, q2] = q12_star
    x1 = q1**2
    x2 = q2**2

    # k runs with the parity of l, so with l_mod == 2 only even k occur.
    b_diff_k = np.empty(l_max + 1)
    for k in range(0, l_max + 1, l_mod):
        b_diff_k[k] = _b_nk_int(a, k, x2) - _b_nk_int(a, k, x1)
    b_diff_b2 = _b_nk_int(a, b + 2, x2) - _b_nk_int(a, b + 2, x1)

    # _s_ab_int is only needed by k == b + 2 terms; evaluate on first use.
    s_x1 = 0.0
    s_x2 = 0.0
    have_s = False

    out = np.empty(l_max // l_mod + 1)
    for idx_ell in range(l_max // l_mod + 1):
        ell = idx_ell * l_mod
        total = 0.0
        k_start = ell % 2
        term_k = (
            math.gamma(0.5 * (k_start + 1 + ell))
            / math.gamma(0.5 * (k_start + 1 - ell))
            * 2.0 ** (ell - k_start)
            / (math.gamma(k_start + 1) * math.gamma(ell - k_start + 1))
        )
        for k in range(k_start, ell + 1, 2):
            if k == b + 2:
                if not have_s:
                    s_x2 = _s_ab_int(a, b, x2)
                    s_x1 = _s_ab_int(a, b, x1)
                    have_s = True
                term_x = math.log(2.0 * v2) * b_diff_k[k] + s_x2 - s_x1
                total += term_k * term_x
            else:
                t_x = v2 ** (b + 2 - k) * b_diff_k[k] - 2.0 ** (k - b - 2) * b_diff_b2
                total += term_k * t_x / (b + 2 - k)
            term_k *= -(ell - k) * (ell + k + 1) / (4.0 * (k + 1) * (k + 2))
        out[idx_ell] = total
    return out


@numba.njit
def mI_star(ell, a, b, v12_star, q12_star):
    """Dimensionless integral related to MathcalI.

    The form factor is F_DM(q, v) = (q/q0)**a * (v/c)**b; ``a`` and ``b`` are
    its exponents (the (a, b) tuple already unpacked by the caller).

    This is $I^{(\\ell)}_\\star$ *without* the (q_star/qBohr)**a * (v_star/c)**b
    prefactor, which the caller applies.

    Integration region v12, q12: given in units of v_star, q_star.

    There are 0, 1, 2 or 3 regions that contribute to mcalI:
        qA < (R1) < qB < (R2) < qC < (R3) < qD.
        R2 is rectangular, bounded by v1 < v < v2. -> _t_l_ab_vq_int
        R1 and R3 are not rectangular: vMin(q) < v < v2. -> _u_l_ab_vq_int
    If vmin(q) > v1 for all q1 < q < q2, then mcalI is given by _u_l_ab_vq_int
    """
    [v1, v2] = v12_star
    [q1, q2] = q12_star
    if v1 == v2 or q1 == q2:
        return 0.0  # No integration volume
    assert q1 < q2, "Need q12 to be ordered"
    assert v1 < v2, "Need v12 to be ordered"
    include_R2 = True
    if v2 < 1:
        # v2 is below the velocity threshold. mcalI=0
        return 0.0
    tilq_m, tilq_p = v2 - math.sqrt(v2**2 - 1.0), v2 + math.sqrt(v2**2 - 1.0)
    if tilq_m > q2 or tilq_p < q1:
        # in this case v2 < vmin(q) for all q in [q1,q2]
        return 0.0
    # Else: there are some q satisfying vmin(q) < v2 in this interval.
    if v1 < 1:
        # There is no v1 = vmin(q) solution for any real q
        include_R2 = False
    # Else: There are two real solutions to v1 = vmin(q)
    else:
        q_m, q_p = v1 - math.sqrt(v1**2 - 1.0), v1 + math.sqrt(v1**2 - 1.0)
        if q_m > q2 or q_p < q1:
            # in this case v1 < vmin(q) for all q in [q1,q2]
            include_R2 = False
    if include_R2 is False:
        q_A = max(q1, tilq_m)
        q_B = min(q2, tilq_p)
        return _u_l_ab_vq_int(ell, a, b, v2, [q_A, q_B])
    # Else: at least part of the integration volume is set by v1 < v.
    q_a = max(q1, tilq_m)
    q_b = max(q1, q_m)  # q_m > tilq_m iff v2 > v1
    q_c = min(q2, q_p)  # q_p < tilq_p iff v2 > v1
    q_d = min(q2, tilq_p)
    include_region = [True, True, True]
    if q_a == q_b:
        include_region[0] = False
    if q_c == q_d:
        include_region[2] = False
    if v1 > 1:
        assert q_b != q_c, "If q_b==q_c then there should be no R2 region..."
    mI_0, mI_1, mI_2 = 0.0, 0.0, 0.0
    if include_region[0]:
        mI_0 = _u_l_ab_vq_int(ell, a, b, v2, [q_a, q_b])
    if include_region[1]:
        mI_1 = _t_l_ab_vq_int(ell, a, b, [v1, v2], [q_b, q_c])
    if include_region[2]:
        mI_2 = _u_l_ab_vq_int(ell, a, b, v2, [q_c, q_d])
    return mI_0 + mI_1 + mI_2


@numba.njit
def mI_star_ells(l_max, l_mod, a, b, v12_star, q12_star):
    """mI_star for every ell in range(0, l_max + 1, l_mod) at once.

    The region decomposition (see mI_star) does not depend on ell, so it is
    done once and each contributing region integral is evaluated for all ell
    together, sharing the ell-independent endpoint values.
    """
    n_ell = l_max // l_mod + 1
    [v1, v2] = v12_star
    [q1, q2] = q12_star
    if v1 == v2 or q1 == q2:
        return np.zeros(n_ell)  # No integration volume
    assert q1 < q2, "Need q12 to be ordered"
    assert v1 < v2, "Need v12 to be ordered"
    include_R2 = True
    if v2 < 1:
        # v2 is below the velocity threshold. mcalI=0
        return np.zeros(n_ell)
    tilq_m, tilq_p = v2 - math.sqrt(v2**2 - 1.0), v2 + math.sqrt(v2**2 - 1.0)
    if tilq_m > q2 or tilq_p < q1:
        # in this case v2 < vmin(q) for all q in [q1,q2]
        return np.zeros(n_ell)
    if v1 < 1:
        include_R2 = False
    else:
        q_m, q_p = v1 - math.sqrt(v1**2 - 1.0), v1 + math.sqrt(v1**2 - 1.0)
        if q_m > q2 or q_p < q1:
            include_R2 = False
    if include_R2 is False:
        q_A = max(q1, tilq_m)
        q_B = min(q2, tilq_p)
        return _u_l_ab_vq_int_ells(l_max, l_mod, a, b, v2, [q_A, q_B])
    q_a = max(q1, tilq_m)
    q_b = max(q1, q_m)
    q_c = min(q2, q_p)
    q_d = min(q2, tilq_p)
    include_region = [True, True, True]
    if q_a == q_b:
        include_region[0] = False
    if q_c == q_d:
        include_region[2] = False
    if v1 > 1:
        assert q_b != q_c, "If q_b==q_c then there should be no R2 region..."
    mI_0 = np.zeros(n_ell)
    mI_1 = np.zeros(n_ell)
    mI_2 = np.zeros(n_ell)
    if include_region[0]:
        mI_0 = _u_l_ab_vq_int_ells(l_max, l_mod, a, b, v2, [q_a, q_b])
    if include_region[1]:
        mI_1 = _t_l_ab_vq_int_ells(l_max, l_mod, a, b, [v1, v2], [q_b, q_c])
    if include_region[2]:
        mI_2 = _u_l_ab_vq_int_ells(l_max, l_mod, a, b, v2, [q_c, q_d])
    return mI_0 + mI_1 + mI_2


@numba.njit
def kin_matrix_lvq(
    ell, nv, nq, v_max, q_max, log_wavelet_q, eps_q, a, b, q_star, v_star, factor
):
    """
    Compute I_l(nv,nq) analytically for given wavelet indices.

    Signature mirrors analytic_cy.kin_matrix_lvq (ell, nv, nq as three ints).
    Error behaviour deliberately differs from that Cython mirror: this
    pure-Python path ``assert``s its preconditions (see _b_nk_int /
    _c_alpha_int), whereas the Cython path is ``noexcept nogil`` and returns
    silently -- it cannot raise from inside the parallel prange loop.

    Parameters
    ----------
    ell : int
        Wavelet angular momentum index (ell = idx_ell * l_mod).
    nv : int
        Velocity wavelet index.
    nq : int
        Momentum wavelet index.
    v_max : float
        Maximum velocity for wavelet basis.
    q_max : float
        Maximum momentum transfer for wavelet basis.
    log_wavelet_q : bool
        Whether momentum wavelets are log-spaced.
    eps_q : float
        Minimum momentum fraction for log-spaced wavelets.
    a : float
    b : float
        Dark matter form factor parameters (a,b) with
        F_DM(q,v) = (q/q0_fdm)**a * (v/v0_fdm)**b
    q_star : float
        Characteristic momentum scale q_star = sqrt(2*mass_dm*energy).
    v_star : float
        Characteristic velocity scale v_star = q_star/mass_dm.
    factor : float
        Overall prefactor for I_l(nv,nq) to get mcalI.

    Returns
    -------
    result : float
        The computed I_l(nv,nq) value.
    """

    n_regions = [1, 1]

    v1, v2, v3 = basis_funcs.haar_support(nv)
    if nv == 0:
        A_v, _ = basis_funcs.haar_value(nv, dim=3)
        n_regions[0] = 1
    else:
        A_v, B_v = basis_funcs.haar_value(nv, dim=3)
        n_regions[0] = 2
    v1, v2, v3 = v1 * v_max, v2 * v_max, v3 * v_max

    if log_wavelet_q:
        q1, q2, q3 = basis_funcs.haar_support_log(nq, eps_q)
        if nq == 0:
            A_q, _ = basis_funcs.haar_value_log(nq, eps_q, p=2)
            n_regions[1] = 1
        else:
            A_q, B_q = basis_funcs.haar_value_log(nq, eps_q, p=2)
            n_regions[1] = 2
    else:
        q1, q2, q3 = basis_funcs.haar_support(nq)
        if nq == 0:
            A_q, _ = basis_funcs.haar_value(nq, dim=3)
            n_regions[1] = 1
        else:
            A_q, B_q = basis_funcs.haar_value(nq, dim=3)
            n_regions[1] = 2
    q1, q2, q3 = q1 * q_max, q2 * q_max, q3 * q_max

    # There is always an A_v A_q term:
    v12_star = [v1 / v_star, v2 / v_star]
    q12_star = [q1 / q_star, q2 / q_star]
    term_AA = A_v * A_q * mI_star(ell, a, b, v12_star, q12_star)

    # There are only B-type contributions if V or Q uses wavelets
    term_AB, term_BA, term_BB = 0.0, 0.0, 0.0
    if n_regions[0] == 2:
        v23_star = [v2 / v_star, v3 / v_star]
        term_BA = B_v * A_q * mI_star(ell, a, b, v23_star, q12_star)
    if n_regions[1] == 2:
        q23_star = [q2 / q_star, q3 / q_star]
        term_AB = A_v * B_q * mI_star(ell, a, b, v12_star, q23_star)
    if n_regions == [2, 2]:
        term_BB = B_v * B_q * mI_star(ell, a, b, v23_star, q23_star)
    result = factor * (term_AA + term_BA + term_AB + term_BB)

    return result


@numba.njit
def _fill_endpoint_tables(
    l_max,
    l_mod,
    nv_max,
    nq_max,
    v_max,
    q_max,
    log_wavelet_q,
    eps_q,
    a,
    b,
    q_star,
    v_star,
):
    """Per-call endpoint tables for the interior fast path.

    The expensive quantities inside _t_l_ab_vq_int_ells depend only on
    (k, endpoint), and for every cell inside the kinematically allowed
    region the endpoints are plain wavelet support points: O(nv + nq)
    distinct values shared by O(nv * nq) cells. This evaluates them once,

        pv_table[n, pt, k] = v ** (b + 2 - k)
        bq_table[n, pt, k] = _b_nk_int(a, k, q**2)

    at the three support points (pt = min, mid, max) of wavelet n, in star
    units built by the same arithmetic as kin_matrix_ells, so table entries
    are bit-identical to direct evaluation; ell_seeds holds the leading
    term_k of the k sweep for each l. k steps in l_mod like the k loops of
    _t_l_ab_vq_int_ells; skipped entries stay zero and are never read.
    Support points at v = 0 or q = 0 (the n = 0 wavelets in a non-log
    basis) are skipped too: those cells start below the kinematic
    threshold, never take the interior path, and so never read their row.
    """
    pv_table = np.zeros((nv_max + 1, 3, l_max + 1))
    bq_table = np.zeros((nq_max + 1, 3, l_max + 1))
    ell_seeds = np.zeros(l_max // l_mod + 1)

    for n in range(nv_max + 1):
        supp_v = basis_funcs.haar_support(n)
        for pt in range(3):
            v_s = supp_v[pt] * v_max / v_star
            if v_s == 0.0:
                continue
            for k in range(0, l_max + 1, l_mod):
                pv_table[n, pt, k] = v_s ** (b + 2 - k)

    for n in range(nq_max + 1):
        if log_wavelet_q:
            supp_q = basis_funcs.haar_support_log(n, eps_q)
        else:
            supp_q = basis_funcs.haar_support(n)
        for pt in range(3):
            q_s = supp_q[pt] * q_max / q_star
            x = q_s**2
            if x == 0.0:
                continue
            for k in range(0, l_max + 1, l_mod):
                bq_table[n, pt, k] = _b_nk_int(a, k, x)

    # Leading term_k of the k sweep for each l, exactly as computed in
    # _t_l_ab_vq_int_ells / _u_l_ab_vq_int_ells.
    for idx_ell in range(l_max // l_mod + 1):
        ell = idx_ell * l_mod
        k_start = ell % 2
        ell_seeds[idx_ell] = (
            math.gamma(0.5 * (k_start + 1 + ell))
            / math.gamma(0.5 * (k_start + 1 - ell))
            * 2.0 ** (ell - k_start)
            / (math.gamma(k_start + 1) * math.gamma(ell - k_start + 1))
        )
    return pv_table, bq_table, ell_seeds


@numba.njit
def _kin_matrix_ells_interior(
    l_max,
    l_mod,
    b,
    n_regions_0,
    n_regions_1,
    v1_s,
    v2_s,
    v3_s,
    A_v,
    B_v,
    A_q,
    B_q,
    factor,
    pv1,
    pv2,
    pv3,
    bq1,
    bq2,
    bq3,
    ell_seeds,
):
    """The four sub-rectangle terms of kin_matrix_ells for an interior cell.

    For a cell that lies entirely inside the kinematically allowed region,
    every mI_star_ells call reduces to its rectangular _t integral with
    unclamped endpoints, so the endpoint tables apply. Terms are combined
    in the same order as kin_matrix_ells.
    """
    n_ell = l_max // l_mod + 1
    term_AA = (
        A_v
        * A_q
        * _t_l_ab_vq_int_ells_tab(
            l_max, l_mod, b, v1_s, v2_s, pv1, pv2, bq1, bq2, ell_seeds
        )
    )
    term_AB = np.zeros(n_ell)
    term_BA = np.zeros(n_ell)
    term_BB = np.zeros(n_ell)
    if n_regions_0 == 2:
        term_BA = (
            B_v
            * A_q
            * _t_l_ab_vq_int_ells_tab(
                l_max, l_mod, b, v2_s, v3_s, pv2, pv3, bq1, bq2, ell_seeds
            )
        )
    if n_regions_1 == 2:
        term_AB = (
            A_v
            * B_q
            * _t_l_ab_vq_int_ells_tab(
                l_max, l_mod, b, v1_s, v2_s, pv1, pv2, bq2, bq3, ell_seeds
            )
        )
    if n_regions_0 == 2 and n_regions_1 == 2:
        term_BB = (
            B_v
            * B_q
            * _t_l_ab_vq_int_ells_tab(
                l_max, l_mod, b, v2_s, v3_s, pv2, pv3, bq2, bq3, ell_seeds
            )
        )
    return factor * (term_AA + term_BA + term_AB + term_BB)


@numba.njit
def kin_matrix_ells(
    l_max,
    l_mod,
    nv,
    nq,
    v_max,
    q_max,
    log_wavelet_q,
    eps_q,
    a,
    b,
    q_star,
    v_star,
    factor,
    pv_table,
    bq_table,
    ell_seeds,
):
    """
    Compute I_l(nv,nq) for every l in range(0, l_max + 1, l_mod) at once.

    Same wavelet decomposition as kin_matrix_lvq (see there for the parameters);
    the integration regions do not depend on l, so the l-independent
    endpoint evaluations inside each region are shared across l via
    mI_star_ells. pv_table/bq_table/ell_seeds are the per-call endpoint
    tables from _fill_endpoint_tables, used by the interior fast path.

    Returns
    -------
    result : np.ndarray
        Array of length l_max // l_mod + 1; entry idx_ell holds
        I_l(nv, nq) for l = idx_ell * l_mod.
    """

    n_regions = [1, 1]

    v1, v2, v3 = basis_funcs.haar_support(nv)
    if nv == 0:
        A_v, _ = basis_funcs.haar_value(nv, dim=3)
        # unused for the scaling wavelet; bound for the interior call
        B_v = 0.0
        n_regions[0] = 1
    else:
        A_v, B_v = basis_funcs.haar_value(nv, dim=3)
        n_regions[0] = 2
    v1, v2, v3 = v1 * v_max, v2 * v_max, v3 * v_max

    if log_wavelet_q:
        q1, q2, q3 = basis_funcs.haar_support_log(nq, eps_q)
        if nq == 0:
            A_q, _ = basis_funcs.haar_value_log(nq, eps_q, p=2)
            # unused for the scaling wavelet; bound for the interior call
            B_q = 0.0
            n_regions[1] = 1
        else:
            A_q, B_q = basis_funcs.haar_value_log(nq, eps_q, p=2)
            n_regions[1] = 2
    else:
        q1, q2, q3 = basis_funcs.haar_support(nq)
        if nq == 0:
            A_q, _ = basis_funcs.haar_value(nq, dim=3)
            # unused for the scaling wavelet; bound for the interior call
            B_q = 0.0
            n_regions[1] = 1
        else:
            A_q, B_q = basis_funcs.haar_value(nq, dim=3)
            n_regions[1] = 2
    q1, q2, q3 = q1 * q_max, q2 * q_max, q3 * q_max

    n_ell = l_max // l_mod + 1

    v1_s, v2_s, v3_s = v1 / v_star, v2 / v_star, v3 / v_star
    q1_s, q2_s, q3_s = q1 / q_star, q2 / q_star, q3 / q_star

    # Interior fast path: if the whole cell lies inside the kinematically
    # allowed region -- the binding corner is the lowest-v edge, since
    # tilq_m falls and tilq_p rises with v (same tilq arithmetic as
    # mI_star_ells, so the classification is exact) -- then no clamp can
    # bind in any of the four mI_star_ells calls below and the endpoint
    # tables apply. Boundary-crossing cells fall through to the general
    # path.
    if v1_s > 1.0:
        sqrt_v1s = math.sqrt(v1_s**2 - 1.0)
        if v1_s - sqrt_v1s <= q1_s and v1_s + sqrt_v1s >= q3_s:
            return _kin_matrix_ells_interior(
                l_max,
                l_mod,
                b,
                n_regions[0],
                n_regions[1],
                v1_s,
                v2_s,
                v3_s,
                A_v,
                B_v,
                A_q,
                B_q,
                factor,
                pv_table[nv, 0],
                pv_table[nv, 1],
                pv_table[nv, 2],
                bq_table[nq, 0],
                bq_table[nq, 1],
                bq_table[nq, 2],
                ell_seeds,
            )

    # There is always an A_v A_q term:
    v12_star = [v1_s, v2_s]
    q12_star = [q1_s, q2_s]
    term_AA = A_v * A_q * mI_star_ells(l_max, l_mod, a, b, v12_star, q12_star)

    # There are only B-type contributions if V or Q uses wavelets
    term_AB = np.zeros(n_ell)
    term_BA = np.zeros(n_ell)
    term_BB = np.zeros(n_ell)
    if n_regions[0] == 2:
        v23_star = [v2_s, v3_s]
        term_BA = B_v * A_q * mI_star_ells(l_max, l_mod, a, b, v23_star, q12_star)
    if n_regions[1] == 2:
        q23_star = [q2_s, q3_s]
        term_AB = A_v * B_q * mI_star_ells(l_max, l_mod, a, b, v12_star, q23_star)
    if n_regions == [2, 2]:
        term_BB = B_v * B_q * mI_star_ells(l_max, l_mod, a, b, v23_star, q23_star)
    result = factor * (term_AA + term_BA + term_AB + term_BB)

    return result


@numba.njit(parallel=True)
def kin_matrix(
    l_max,
    l_mod,
    nv_max,
    nq_max,
    v_max,
    q_max,
    log_wavelet_q,
    eps_q,
    a,
    b,
    q_star,
    v_star,
    factor,
):
    """
    Compute I_l(nv,nq) for all l in [0,l_max], nv in [0,nv_max], nq in [0,nq_max].

    Parameters
    ----------
    l_max : int
        Maximum l value.
    l_mod : int
        Modulo for l values to compute (e.g. l_mod=2 computes only even l).
    nv_max : int
        Maximum nv index.
    nq_max : int
        Maximum nq index.
    v_max : float
        Maximum velocity.
    q_max : float
        Maximum momentum transfer.
    log_wavelet_q : bool
        Whether the q wavelets are log-spaced.
    eps_q : float
        Minimum q/q_max value for log-spaced q wavelets.
    a : float
    b : float
        Dark matter form factor parameters (a,b) with
        F_DM(q,v) = (q/q0_fdm)**a * (v/v0_fdm)**b
    q_star : float
        Characteristic momentum scale q_star = sqrt(2*mass_dm*energy).
    v_star : float
        Characteristic velocity scale v_star = q_star/mass_dm.
    factor : float
        Overall prefactor for I_l(nv,nq) to get mcalI.

    Returns
    -------
    result : np.ndarray
        Array of shape (l_max//l_mod + 1, nv_max + 1, nq_max + 1) containing
        I_l(nv, nq) values.

    Notes
    -----
    The (nv, nq) cells are independent and iterated in parallel; all l values
    of one cell are computed together by kin_matrix_ells, which shares the
    l-independent endpoint evaluations across l. Endpoint values shared
    across cells are precomputed once into per-call tables (see
    _fill_endpoint_tables) and read back by every kinematically interior
    cell; the tables are read-only inside the loop.
    """

    shape = (l_max // l_mod + 1, nv_max + 1, nq_max + 1)
    result = np.zeros(shape, dtype=float)

    pv_table, bq_table, ell_seeds = _fill_endpoint_tables(
        l_max,
        l_mod,
        nv_max,
        nq_max,
        v_max,
        q_max,
        log_wavelet_q,
        eps_q,
        a,
        b,
        q_star,
        v_star,
    )

    n_nq = nq_max + 1
    for idx in numba.prange((nv_max + 1) * n_nq):
        nv = idx // n_nq
        nq = idx % n_nq
        result[:, nv, nq] = kin_matrix_ells(
            l_max,
            l_mod,
            nv,
            nq,
            v_max,
            q_max,
            log_wavelet_q,
            eps_q,
            a,
            b,
            q_star,
            v_star,
            factor,
            pv_table,
            bq_table,
            ell_seeds,
        )
    return result
