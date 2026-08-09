# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

"""Cython mirror of analytic.py: analytic results for mcalI matrix calculation.

The analytic kernels in this module are adapted from vsdm
(https://github.com/blillard/vsdm) by B. Lillard and A. Radick
[arXiv:2502.17547].
"""

import numpy as np
cimport numpy as cnp
cimport cython
from cython.parallel import prange

from libc.math cimport sqrt, log, exp, pow, floor, log2, fabs, log1p

from scipy.special.cython_special cimport hyp2f1, spence, gamma

cdef double PI = 3.14159265358979323846

# Size of the stack buffers in the *_ells helpers (one slot per l or k value,
# so they support l_max <= _MAX_ELLS - 1). Checked in kin_matrix.
# Fixed-size stack arrays keep the helpers `noexcept nogil` and
# thread-private without any heap allocation in the prange loop.
cdef enum:
    _MAX_ELLS = 101

# // ---------------------------------------------------------
# // Haar Wavelet Functions
# //
# // MIRROR of basis_funcs.py (pure-Python / numba). Reimplemented here as
# // `noexcept nogil` C so the parallel integral loop below can call them
# // without the GIL; keep the two in sync.
# // ---------------------------------------------------------

# @cython.nogil
cdef void haar_n_to_lam_mu(int n, int* lam, int* mu) noexcept nogil:
    if n == 0:
        lam[0] = -1
        mu[0] = -1
        return
    lam[0] = <int>floor(log2(<double>n))
    mu[0] = n - (1 << lam[0])

# @cython.nogil
cdef void haar_support(int n, double* x_min, double* x_mid, double* x_max) noexcept nogil:
    if n == 0:
        x_min[0] = 0.0
        x_mid[0] = 1.0
        x_max[0] = 1.0
        return
    
    cdef int lam, mu
    haar_n_to_lam_mu(n, &lam, &mu)
    
    cdef double scale = pow(2.0, -lam)
    x_min[0] = scale * (mu + 0.0)
    x_mid[0] = scale * (mu + 0.5)
    x_max[0] = scale * (mu + 1.0)

# @cython.nogil
cdef void haar_support_log(int n, double eps, double* x_min, double* x_mid, double* x_max) noexcept nogil:
    if n == 0:
        x_min[0] = eps
        x_mid[0] = 1.0
        x_max[0] = 1.0
        return

    cdef int lam, mu
    haar_n_to_lam_mu(n, &lam, &mu)
    
    cdef double length = log(1.0 / eps)
    cdef double scale_exp = pow(2.0, -lam) * length
    
    x_min[0] = eps * exp(scale_exp * (mu + 0.0))
    x_mid[0] = eps * exp(scale_exp * (mu + 0.5))
    x_max[0] = eps * exp(scale_exp * (mu + 1.0))

# @cython.nogil
cdef void haar_value(int n, int dim, double* val_a, double* val_b) noexcept nogil:
    if n == 0:
        val_a[0] = sqrt(<double>dim)
        val_b[0] = -sqrt(<double>dim)
        return
    
    cdef int lam, mu
    haar_n_to_lam_mu(n, &lam, &mu)
    
    cdef double scale = pow(2.0, -lam)
    cdef double x1 = scale * (mu + 0.0)
    cdef double x2 = scale * (mu + 0.5)
    cdef double x3 = scale * (mu + 1.0)
    
    cdef double y1 = pow(x1, dim)
    cdef double y2 = pow(x2, dim)
    cdef double y3 = pow(x3, dim)
    
    val_a[0] = sqrt(dim/(y3 - y1) * (y3 - y2)/(y2 - y1))
    val_b[0] = -sqrt(dim/(y3 - y1) * (y2 - y1)/(y3 - y2))

# @cython.nogil
cdef void haar_value_log(int n, double eps, int p, double* val_a, double* val_b) noexcept nogil:
    if n == 0:
        if p == 0:
            val_a[0] = sqrt(1.0/(1.0 - eps))
        elif p == -1:
            val_a[0] = sqrt(1.0/log(1.0/eps))
        else:
            val_a[0] = sqrt((p + 1.0)/(1.0 - pow(eps, p + 1.0)))
        val_b[0] = -val_a[0]
        return

    cdef int lam, mu
    haar_n_to_lam_mu(n, &lam, &mu)
            
    cdef double length = log(1.0 / eps)
    cdef double scale_exp = pow(2.0, -lam) * length
    cdef double x1 = eps * exp(scale_exp * (mu + 0.0))

    cdef double rho = exp(length * pow(2.0, -(lam + 1)))
    
    cdef double a_n, b_n
    
    if p == 0:
        a_n = sqrt(1.0 / (x1 * (rho - 1.0) * (1.0 + 1.0/rho)))
        b_n = sqrt(1.0 / (x1 * rho * (rho - 1.0) * (1.0 + rho)))
    elif p == -1:
        a_n = sqrt(pow(2.0, lam) / length)
        b_n = a_n
    else:
        b_n = sqrt(
            (p + 1.0) / (
                pow(x1, p + 1.0) * (pow(rho, p + 1.0) - 1.0)
                * (pow(rho, p + 1.0) + 1.0) * pow(rho, p + 1.0)
            )
        )
        a_n = b_n * pow(rho, p + 1.0)
    
    val_a[0] = a_n
    val_b[0] = -b_n


# // ---------------------------------------------------------
# // Test hooks -- expose the cdef Haar primitives to Python so that
# // tests/test_analytic_kernels.py can compare them against basis_funcs.py.
# // Off every hot path; the parallel integral loop calls the cdef versions
# // directly. cpdef holds the GIL, which is fine for calling nogil funcs.
# // ---------------------------------------------------------

cpdef tuple _haar_n_to_lam_mu_test(int n):
    cdef int lam, mu
    haar_n_to_lam_mu(n, &lam, &mu)
    return (lam, mu)

cpdef tuple _haar_support_test(int n):
    cdef double x_min, x_mid, x_max
    haar_support(n, &x_min, &x_mid, &x_max)
    return (x_min, x_mid, x_max)

cpdef tuple _haar_support_log_test(int n, double eps):
    cdef double x_min, x_mid, x_max
    haar_support_log(n, eps, &x_min, &x_mid, &x_max)
    return (x_min, x_mid, x_max)

cpdef tuple _haar_value_test(int n, int dim):
    cdef double val_a, val_b
    haar_value(n, dim, &val_a, &val_b)
    return (val_a, val_b)

cpdef tuple _haar_value_log_test(int n, double eps, int p):
    cdef double val_a, val_b
    haar_value_log(n, eps, p, &val_a, &val_b)
    return (val_a, val_b)


# // ---------------------------------------------------------
# // Analytic Integrals
# // ---------------------------------------------------------

# @cython.nogil
cdef double _b_nk_int(int n, int k, double x) noexcept nogil:
    # Range in 'x' should not include x=0
    if x == 0: return 0.0
    
    cdef double sum_val = 0.0
    cdef int j
    cdef double comb, ipower, summand
    
    comb = 1.0
    for j in range(k + 1):
        # comb = gamma(k + 1.0) / (gamma(j + 1.0) * gamma(k - j + 1.0))
        if j > 0:
            comb *= (k - j + 1.0) / j
        ipower = j + (n - k)/2.0 + 1.0
        
        if ipower == 0:
            summand = log(x)
        else:
            summand = pow(x, ipower) / ipower
        sum_val += comb * summand
        
    return 0.5 * sum_val

# @cython.nogil
cdef double _c_alpha_int(double alpha, double x) noexcept nogil:
    cdef double val, yx, sum_val, comb, comb_alpha_j
    cdef int j
    cdef int alpha_int
    
    # Check if integer (using epsilon for float comparison safety)
    if fabs(alpha - floor(alpha)) > 1e-9:
        val = hyp2f1(1.0, alpha, 1.0 + alpha, -x)
        return pow(x, alpha) / (alpha * alpha) * (0.5 - val)
    
    alpha_int = <int>alpha
    
    if alpha_int == -2:
        return -0.5 * log((1.0 + x) / x) + (0.5 - 0.125 / x) / x
    elif alpha_int == -1:
        return log((1.0 + x) / x) - 0.5 / x
    elif alpha_int == 0:
        val = spence(1.0 + x)
        return (-0.25 * pow(log(x), 2) + log(x) * log1p(x) + val)
    elif alpha_int == 1:
        return 0.5 * x - log1p(x)
    elif alpha_int == 2:
        return (0.125 * x - 0.5) * x + 0.5 * log1p(x)
    elif alpha_int > 1:
        sum_val = pow(-1.0, alpha_int) * log1p(x) / alpha + pow(1.0 + x, alpha_int) / (2.0 * alpha * alpha)
        comb_alpha_j = alpha
        for j in range(1, alpha_int):
            # comb = (
            #     gamma(alpha + 1.0) / (gamma(j + 1.0) * gamma(alpha - j + 1.0))
            #     + gamma(alpha) / (gamma(j + 1.0) * gamma(alpha - j))
            # )
            comb = comb_alpha_j * (2.0 * alpha - j) / j
            comb_alpha_j *= (alpha - j) / (j + 1.0)
            sum_val += pow(-1.0, alpha_int - j) / (2.0 * alpha * j) * comb * pow(1.0 + x, j)
        return sum_val
    elif alpha_int < -1:
        yx = (1.0 + x) / x
        sum_val = pow(-1.0, alpha_int) * log(yx) / alpha - pow(yx, -alpha_int) / (2.0 * alpha * alpha)
        comb_alpha_j = -alpha
        for j in range(1, -alpha_int):
            # comb = (
            #     gamma(-alpha + 1.0) / (gamma(j + 1.0) * gamma(-alpha - j + 1.0))
            #     + gamma(-alpha) / (gamma(j + 1.0) * gamma(-alpha - j))
            # )
            comb = comb_alpha_j * (-2.0 * alpha - j) / j
            comb_alpha_j *= (-alpha - j) / (j + 1.0)
            sum_val += pow(-1.0, alpha_int + j) / (2.0 * alpha * j) * comb * pow(yx, j)
        return sum_val
    return 0.0

# @cython.nogil
cdef double _v_ab_int(int a, int b, double x) noexcept nogil:
    cdef double sum_2 = 0.0
    cdef int j, b_int
    cdef double comb
    b_int = <int>b
    
    comb = 1.0
    for j in range(b_int + 3):
        # comb = gamma(b + 3.0) / (gamma(j + 1.0) * gamma(b + 3.0 - j))
        if j > 0:
            comb *= (b + 3.0 - j) / j
        if 2 * j == (b - a):
            sum_2 += comb * log(x)
        else:
            sum_2 += comb * pow(x, j + (a - b) / 2.0) / (j + (a - b) / 2.0)
    return 0.5 * sum_2

# @cython.nogil
cdef double _s_ab_int(int a, int b, double x) noexcept nogil:
    cdef double logfactor = 0.5 * log(x / pow(1.0 + x, 2))
    cdef double sum_val = logfactor * _v_ab_int(a, b, x)
    cdef int j, b_int
    cdef double comb
    b_int = <int>b
    
    comb = 1.0
    for j in range(b_int + 3):
        # comb = gamma(b + 3.0) / (gamma(j + 1.0) * gamma(b + 3.0 - j))
        if j > 0:
            comb *= (b + 3.0 - j) / j
        sum_val += 0.5 * comb * _c_alpha_int(j + (a - b) / 2.0, x)
    return sum_val

# @cython.nogil
cdef double _t_l_ab_vq_int(int ell, int a, int b, double v1, double v2, double q1, double q2) noexcept nogil:
    cdef double x1 = pow(q1, 2)
    cdef double x2 = pow(q2, 2)
    cdef double sum_val = 0.0
    cdef int k
    cdef double term_k, term_q, term_v
    cdef double k_start
    
    k_start = ell % 2
    term_k = (gamma(0.5 * (k_start + 1 + ell)) / gamma(0.5 * (k_start + 1 - ell))
              * pow(2.0, ell - k_start)
              / (gamma(k_start + 1.0) * gamma(ell - k_start + 1.0)))
    # Range step 2
    for k in range(ell % 2, ell + 1, 2):
        # term_k = pow(2.0, l - k) * gamma(0.5 * (k + 1 + l)) / gamma(0.5 * (k + 1 - l))
        # term_k /= (gamma(k + 1.0) * gamma(l - k + 1.0))
        
        term_q = (_b_nk_int(a, k, x2) - _b_nk_int(a, k, x1))
        
        if k == b + 2:
            term_v = log(v2 / v1)
        else:
            term_v = (pow(v2, b + 2 - k) - pow(v1, b + 2 - k)) / (b + 2.0 - k)
        sum_val += term_v * term_k * term_q

        term_k *= -(ell - k) * (ell + k + 1) / (4.0 * (k + 1) * (k + 2))
    return sum_val

# @cython.nogil
cdef void _t_l_ab_vq_int_ells(int l_max, int l_mod, int a, int b,
                              double v1, double v2, double q1, double q2,
                              double* out) noexcept nogil:
    # MIRROR of analytic.py:_t_l_ab_vq_int_ells. Rectangular integrals T_l for
    # every l in range(0, l_max + 1, l_mod), same arithmetic as
    # _t_l_ab_vq_int term by term; the _b_nk_int endpoint differences and the
    # velocity factors depend on k but not on l, so they are computed once
    # and shared by all l.
    cdef double x1 = pow(q1, 2)
    cdef double x2 = pow(q2, 2)
    cdef double term_q_k[_MAX_ELLS]
    cdef double term_v_k[_MAX_ELLS]
    cdef int k, idx_ell, ell
    cdef double sum_val, term_k
    cdef double k_start

    # k runs with the parity of l, so with l_mod == 2 only even k occur.
    # (while loop: range() with a runtime step cannot compile under nogil)
    k = 0
    while k <= l_max:
        term_q_k[k] = (_b_nk_int(a, k, x2) - _b_nk_int(a, k, x1))
        if k == b + 2:
            term_v_k[k] = log(v2 / v1)
        else:
            term_v_k[k] = (pow(v2, b + 2 - k) - pow(v1, b + 2 - k)) / (b + 2.0 - k)
        k += l_mod

    for idx_ell in range(l_max // l_mod + 1):
        ell = idx_ell * l_mod
        sum_val = 0.0
        k_start = ell % 2
        term_k = (gamma(0.5 * (k_start + 1 + ell)) / gamma(0.5 * (k_start + 1 - ell))
                  * pow(2.0, ell - k_start)
                  / (gamma(k_start + 1.0) * gamma(ell - k_start + 1.0)))
        for k in range(ell % 2, ell + 1, 2):
            sum_val += term_v_k[k] * term_k * term_q_k[k]

            term_k *= -(ell - k) * (ell + k + 1) / (4.0 * (k + 1) * (k + 2))
        out[idx_ell] = sum_val

# @cython.nogil
cdef void _t_l_ab_vq_int_ells_tab(int l_max, int l_mod, int b,
                                  double v1, double v2,
                                  const double* pv1, const double* pv2,
                                  const double* bq1, const double* bq2,
                                  const double* ell_seeds,
                                  double* out) noexcept nogil:
    # MIRROR of analytic.py:_t_l_ab_vq_int_ells_tab. Same sweep as
    # _t_l_ab_vq_int_ells, term by term, with every endpoint evaluation
    # read from the per-call tables (see _fill_endpoint_tables):
    # pv1/pv2 hold pow(v, b + 2 - k) at the two velocity endpoints,
    # bq1/bq2 hold _b_nk_int(a, k, q^2) at the two momentum endpoints,
    # and ell_seeds the leading term_k of each l. Only valid when no
    # kinematic clamp binds, i.e. the (v, q) rectangle lies entirely
    # inside the allowed region.
    cdef double term_q_k[_MAX_ELLS]
    cdef double term_v_k[_MAX_ELLS]
    cdef int k, idx_ell, ell
    cdef double sum_val, term_k

    # k runs with the parity of l, so with l_mod == 2 only even k occur.
    # (while loop: range() with a runtime step cannot compile under nogil)
    k = 0
    while k <= l_max:
        term_q_k[k] = bq2[k] - bq1[k]
        if k == b + 2:
            term_v_k[k] = log(v2 / v1)
        else:
            term_v_k[k] = (pv2[k] - pv1[k]) / (b + 2.0 - k)
        k += l_mod

    for idx_ell in range(l_max // l_mod + 1):
        ell = idx_ell * l_mod
        sum_val = 0.0
        term_k = ell_seeds[idx_ell]
        for k in range(ell % 2, ell + 1, 2):
            sum_val += term_v_k[k] * term_k * term_q_k[k]

            term_k *= -(ell - k) * (ell + k + 1) / (4.0 * (k + 1) * (k + 2))
        out[idx_ell] = sum_val

# @cython.nogil
cdef double _u_l_ab_vq_int(int ell, int a, int b, double v2, double q1, double q2) noexcept nogil:
    cdef double x1 = pow(q1, 2)
    cdef double x2 = pow(q2, 2)
    cdef double sum_val = 0.0
    cdef int k
    cdef double term_k, term_x, t_x
    cdef double k_start
    
    k_start = ell % 2
    term_k = (gamma(0.5 * (k_start + 1 + ell)) / gamma(0.5 * (k_start + 1 - ell))
              * pow(2.0, ell - k_start)
              / (gamma(k_start + 1.0) * gamma(ell - k_start + 1.0)))
    
    for k in range(ell % 2, ell + 1, 2):
        # term_k = (gamma(0.5 * (k + 1 + l)) / gamma(0.5 * (k + 1 - l))
        #           * pow(2.0, l - k) / (gamma(k + 1.0) * gamma(l - k + 1.0)))
        
        if k == b + 2:
            term_x = (log(2.0 * v2) * (_b_nk_int(a, k, x2) - _b_nk_int(a, k, x1))
                      + _s_ab_int(a, b, x2) - _s_ab_int(a, b, x1))
            sum_val += term_k * term_x
        else:
            t_x = (pow(v2, b + 2 - k) * (_b_nk_int(a, k, x2) - _b_nk_int(a, k, x1))
                   - pow(2.0, k - b - 2) * (_b_nk_int(a, b + 2, x2) - _b_nk_int(a, b + 2, x1)))
            sum_val += term_k * t_x / (b + 2.0 - k)

        term_k *= -(ell - k) * (ell + k + 1) / (4.0 * (k + 1) * (k + 2))
    return sum_val

# @cython.nogil
cdef void _u_l_ab_vq_int_ells(int l_max, int l_mod, int a, int b,
                              double v2, double q1, double q2,
                              double* out) noexcept nogil:
    # MIRROR of analytic.py:_u_l_ab_vq_int_ells. Non-rectangular integrals U_l
    # for every l in range(0, l_max + 1, l_mod), same arithmetic as
    # _u_l_ab_vq_int term by term; the _b_nk_int and _s_ab_int endpoint
    # values depend on k but not on l, so they are computed once and shared
    # by all l. The two _s_ab_int values are kept separate (not
    # pre-differenced) to preserve the exact summation order.
    cdef double x1 = pow(q1, 2)
    cdef double x2 = pow(q2, 2)
    cdef double b_diff_k[_MAX_ELLS]
    cdef double b_diff_b2
    cdef int k, idx_ell, ell
    cdef double sum_val, term_k, term_x, t_x
    cdef double k_start

    # k runs with the parity of l, so with l_mod == 2 only even k occur.
    # (while loop: range() with a runtime step cannot compile under nogil)
    k = 0
    while k <= l_max:
        b_diff_k[k] = (_b_nk_int(a, k, x2) - _b_nk_int(a, k, x1))
        k += l_mod
    b_diff_b2 = (_b_nk_int(a, b + 2, x2) - _b_nk_int(a, b + 2, x1))

    # _s_ab_int is only needed by k == b + 2 terms; evaluate on first use.
    cdef double s_x1 = 0.0
    cdef double s_x2 = 0.0
    cdef bint have_s = 0

    for idx_ell in range(l_max // l_mod + 1):
        ell = idx_ell * l_mod
        sum_val = 0.0
        k_start = ell % 2
        term_k = (gamma(0.5 * (k_start + 1 + ell)) / gamma(0.5 * (k_start + 1 - ell))
                  * pow(2.0, ell - k_start)
                  / (gamma(k_start + 1.0) * gamma(ell - k_start + 1.0)))
        for k in range(ell % 2, ell + 1, 2):
            if k == b + 2:
                if not have_s:
                    s_x2 = _s_ab_int(a, b, x2)
                    s_x1 = _s_ab_int(a, b, x1)
                    have_s = 1
                term_x = log(2.0 * v2) * b_diff_k[k] + s_x2 - s_x1
                sum_val += term_k * term_x
            else:
                t_x = (pow(v2, b + 2 - k) * b_diff_k[k]
                       - pow(2.0, k - b - 2) * b_diff_b2)
                sum_val += term_k * t_x / (b + 2.0 - k)

            term_k *= -(ell - k) * (ell + k + 1) / (4.0 * (k + 1) * (k + 2))
        out[idx_ell] = sum_val

# @cython.nogil
cdef double mI_star(int ell, int a, int b, double v1, double v2, double q1, double q2) noexcept nogil:
    if v1 == v2 or q1 == q2:
        return 0.0
    
    cdef int include_R2 = 1
    if v2 < 1.0:
        return 0.0
    
    cdef double sqrt_v2 = sqrt(pow(v2, 2) - 1.0)
    cdef double tilq_m = v2 - sqrt_v2
    cdef double tilq_p = v2 + sqrt_v2
    
    if tilq_m > q2 or tilq_p < q1:
        return 0.0
    
    cdef double q_m = 0.0
    cdef double q_p = 0.0
    
    if v1 < 1.0:
        include_R2 = 0
    else:
        sqrt_v1 = sqrt(pow(v1, 2) - 1.0)
        q_m = v1 - sqrt_v1
        q_p = v1 + sqrt_v1
        if q_m > q2 or q_p < q1:
            include_R2 = 0
            
    cdef double q_A, q_B
    
    if include_R2 == 0:
        q_A = q1 if q1 > tilq_m else tilq_m
        q_B = q2 if q2 < tilq_p else tilq_p
        return _u_l_ab_vq_int(ell, a, b, v2, q_A, q_B)
    
    cdef double q_a = q1 if q1 > tilq_m else tilq_m
    cdef double q_b = q1 if q1 > q_m else q_m
    cdef double q_c = q2 if q2 < q_p else q_p
    cdef double q_d = q2 if q2 < tilq_p else tilq_p
    
    cdef double mI_0 = 0.0
    cdef double mI_1 = 0.0
    cdef double mI_2 = 0.0
    
    if q_a != q_b:
        mI_0 = _u_l_ab_vq_int(ell, a, b, v2, q_a, q_b)
    if v1 != v2 and q_b != q_c: # Check boundaries to avoid calling if range is 0
        mI_1 = _t_l_ab_vq_int(ell, a, b, v1, v2, q_b, q_c)
    if q_c != q_d:
        mI_2 = _u_l_ab_vq_int(ell, a, b, v2, q_c, q_d)

    return mI_0 + mI_1 + mI_2

# @cython.nogil
cdef void mI_star_ells(int l_max, int l_mod, int a, int b,
                       double v1, double v2, double q1, double q2,
                       double* out) noexcept nogil:
    # MIRROR of analytic.py:mI_star_ells. mI_star for every l in
    # range(0, l_max + 1, l_mod) at once: the region decomposition (see
    # mI_star) does not depend on l, so it is done once and each contributing
    # region integral is evaluated for all l together, sharing the
    # l-independent endpoint values.
    cdef int n_ell = l_max // l_mod + 1
    cdef int i
    for i in range(n_ell):
        out[i] = 0.0

    if v1 == v2 or q1 == q2:
        return

    cdef int include_R2 = 1
    if v2 < 1.0:
        return

    cdef double sqrt_v2 = sqrt(pow(v2, 2) - 1.0)
    cdef double tilq_m = v2 - sqrt_v2
    cdef double tilq_p = v2 + sqrt_v2

    if tilq_m > q2 or tilq_p < q1:
        return

    cdef double q_m = 0.0
    cdef double q_p = 0.0
    cdef double sqrt_v1

    if v1 < 1.0:
        include_R2 = 0
    else:
        sqrt_v1 = sqrt(pow(v1, 2) - 1.0)
        q_m = v1 - sqrt_v1
        q_p = v1 + sqrt_v1
        if q_m > q2 or q_p < q1:
            include_R2 = 0

    cdef double q_A, q_B

    if include_R2 == 0:
        q_A = q1 if q1 > tilq_m else tilq_m
        q_B = q2 if q2 < tilq_p else tilq_p
        _u_l_ab_vq_int_ells(l_max, l_mod, a, b, v2, q_A, q_B, out)
        return

    cdef double q_a = q1 if q1 > tilq_m else tilq_m
    cdef double q_b = q1 if q1 > q_m else q_m
    cdef double q_c = q2 if q2 < q_p else q_p
    cdef double q_d = q2 if q2 < tilq_p else tilq_p

    cdef double mI_0[_MAX_ELLS]
    cdef double mI_1[_MAX_ELLS]
    cdef double mI_2[_MAX_ELLS]
    for i in range(n_ell):
        mI_0[i] = 0.0
        mI_1[i] = 0.0
        mI_2[i] = 0.0

    if q_a != q_b:
        _u_l_ab_vq_int_ells(l_max, l_mod, a, b, v2, q_a, q_b, mI_0)
    if v1 != v2 and q_b != q_c: # Check boundaries to avoid calling if range is 0
        _t_l_ab_vq_int_ells(l_max, l_mod, a, b, v1, v2, q_b, q_c, mI_1)
    if q_c != q_d:
        _u_l_ab_vq_int_ells(l_max, l_mod, a, b, v2, q_c, q_d, mI_2)

    for i in range(n_ell):
        out[i] = mI_0[i] + mI_1[i] + mI_2[i]

# @cython.nogil
cdef double kin_matrix_lvq_c(int ell, int nv, int nq,
                            double v_max, double q_max, 
                            int log_wavelet_q, double eps_q, 
                            int a, int b,
                            double q_star, double v_star, double factor) noexcept nogil:
    
    cdef int n_regions_0 = 1
    cdef int n_regions_1 = 1
    
    cdef double v1, v2, v3
    cdef double A_v, B_v
    
    # Haar V
    cdef double supp_v[3]
    haar_support(nv, &supp_v[0], &supp_v[1], &supp_v[2])
    v1 = supp_v[0]
    v2 = supp_v[1]
    v3 = supp_v[2]
    
    cdef double val_v[2]
    haar_value(nv, 3, &val_v[0], &val_v[1])
    A_v = val_v[0]
    
    if nv == 0:
        n_regions_0 = 1
        B_v = 0.0 # Unused
    else:
        n_regions_0 = 2
        B_v = val_v[1] # Negative value
        
    v1 *= v_max
    v2 *= v_max
    v3 *= v_max
    
    # Haar Q
    cdef double q1, q2, q3
    cdef double A_q, B_q
    cdef double supp_q[3]
    cdef double val_q[2]
    
    if log_wavelet_q:
        haar_support_log(nq, eps_q, &supp_q[0], &supp_q[1], &supp_q[2])
        q1 = supp_q[0]
        q2 = supp_q[1]
        q3 = supp_q[2]
        
        haar_value_log(nq, eps_q, 2, &val_q[0], &val_q[1])
        A_q = val_q[0]
        
        if nq == 0:
            n_regions_1 = 1
            B_q = 0.0
        else:
            n_regions_1 = 2
            B_q = val_q[1]
    else:
        haar_support(nq, &supp_q[0], &supp_q[1], &supp_q[2])
        q1 = supp_q[0]
        q2 = supp_q[1]
        q3 = supp_q[2]
        
        haar_value(nq, 3, &val_q[0], &val_q[1])
        A_q = val_q[0]
        
        if nq == 0:
            n_regions_1 = 1
            B_q = 0.0
        else:
            n_regions_1 = 2
            B_q = val_q[1]
            
    q1 *= q_max
    q2 *= q_max
    q3 *= q_max
    
    # Integration
    cdef double term_AA, term_BA, term_AB, term_BB
    term_BA = 0.0
    term_AB = 0.0
    term_BB = 0.0
    
    # Normalize by Star
    cdef double v1_s = v1 / v_star
    cdef double v2_s = v2 / v_star
    cdef double v3_s = v3 / v_star
    cdef double q1_s = q1 / q_star
    cdef double q2_s = q2 / q_star
    cdef double q3_s = q3 / q_star
    
    term_AA = A_v * A_q * mI_star(ell, a, b, v1_s, v2_s, q1_s, q2_s)
    
    if n_regions_0 == 2:
        term_BA = B_v * A_q * mI_star(ell, a, b, v2_s, v3_s, q1_s, q2_s)
        
    if n_regions_1 == 2:
        term_AB = A_v * B_q * mI_star(ell, a, b, v1_s, v2_s, q2_s, q3_s)
        
    if n_regions_0 == 2 and n_regions_1 == 2:
        term_BB = B_v * B_q * mI_star(ell, a, b, v2_s, v3_s, q2_s, q3_s)

    return factor * (term_AA + term_BA + term_AB + term_BB)

# // ---------------------------------------------------------
# // Per-call endpoint tables
# //
# // The expensive quantities inside _t_l_ab_vq_int_ells depend only on
# // (k, endpoint), and for every cell inside the kinematically allowed
# // region the endpoints are plain wavelet support points: O(nv + nq)
# // distinct values shared by O(nv * nq) cells. kin_matrix
# // therefore evaluates them once into tables, and interior cells --
# // the vast majority -- read them back. Cells crossed by the kinematic
# // boundary keep the direct path (mI_star_ells), which stays valid for
# // clamped endpoints.
# // ---------------------------------------------------------

# @cython.nogil
cdef void _fill_endpoint_tables(int l_max, int l_mod, int nv_max, int nq_max,
                                double v_max, double q_max,
                                int log_wavelet_q, double eps_q,
                                int a, int b,
                                double q_star, double v_star,
                                double[:, :, ::1] pv_table,
                                double[:, :, ::1] bq_table,
                                double[::1] ell_seeds) noexcept nogil:
    # MIRROR of analytic.py:_fill_endpoint_tables.
    #     pv_table[n, pt, k] = pow(v, b + 2 - k)
    #     bq_table[n, pt, k] = _b_nk_int(a, k, q^2)
    # at the three support points (pt = min, mid, max) of wavelet n, in
    # star units built by the same arithmetic as kin_matrix_ells_c, so
    # table entries are bit-identical to direct evaluation; ell_seeds
    # holds the leading term_k of the k sweep for each l. k steps in
    # l_mod like the k loops of _t_l_ab_vq_int_ells; skipped entries stay
    # zero and are never read. Support points at v = 0 or q = 0 (the
    # n = 0 wavelets in a non-log basis) are skipped too: those cells
    # start below the kinematic threshold, never take the interior path,
    # and so never read their row.
    cdef int n, pt, k, idx_ell, ell
    cdef double supp[3]
    cdef double v_s, q_s, x
    cdef double k_start

    for n in range(nv_max + 1):
        haar_support(n, &supp[0], &supp[1], &supp[2])
        for pt in range(3):
            v_s = supp[pt] * v_max / v_star
            if v_s == 0.0:
                continue
            k = 0
            while k <= l_max:
                pv_table[n, pt, k] = pow(v_s, b + 2 - k)
                k += l_mod

    for n in range(nq_max + 1):
        if log_wavelet_q:
            haar_support_log(n, eps_q, &supp[0], &supp[1], &supp[2])
        else:
            haar_support(n, &supp[0], &supp[1], &supp[2])
        for pt in range(3):
            q_s = supp[pt] * q_max / q_star
            x = pow(q_s, 2)
            if x == 0.0:
                continue
            k = 0
            while k <= l_max:
                bq_table[n, pt, k] = _b_nk_int(a, k, x)
                k += l_mod

    # Leading term_k of the k sweep for each l, exactly as computed in
    # _t_l_ab_vq_int_ells / _u_l_ab_vq_int_ells.
    for idx_ell in range(l_max // l_mod + 1):
        ell = idx_ell * l_mod
        k_start = ell % 2
        ell_seeds[idx_ell] = (gamma(0.5 * (k_start + 1 + ell)) / gamma(0.5 * (k_start + 1 - ell))
                              * pow(2.0, ell - k_start)
                              / (gamma(k_start + 1.0) * gamma(ell - k_start + 1.0)))

# @cython.nogil
cdef void _kin_matrix_ells_interior_c(int l_max, int l_mod, int b,
                                 int n_regions_0, int n_regions_1,
                                 double v1_s, double v2_s, double v3_s,
                                 double A_v, double B_v, double A_q, double B_q,
                                 double factor,
                                 const double* pv1, const double* pv2, const double* pv3,
                                 const double* bq1, const double* bq2, const double* bq3,
                                 const double* ell_seeds,
                                 double* out, int out_stride) noexcept nogil:
    # MIRROR of analytic.py:_kin_matrix_ells_interior. The four sub-rectangle
    # terms of kin_matrix_ells_c for a cell that lies entirely inside the
    # kinematically allowed region: every mI_star_ells call reduces to
    # its rectangular _t integral with unclamped endpoints, so the
    # endpoint tables apply. Terms are combined in the same order as
    # kin_matrix_ells_c.
    cdef int n_ell = l_max // l_mod + 1
    cdef int i
    cdef double mi[_MAX_ELLS]
    cdef double term_AA[_MAX_ELLS]
    cdef double term_BA[_MAX_ELLS]
    cdef double term_AB[_MAX_ELLS]
    cdef double term_BB[_MAX_ELLS]

    for i in range(n_ell):
        term_BA[i] = 0.0
        term_AB[i] = 0.0
        term_BB[i] = 0.0

    _t_l_ab_vq_int_ells_tab(l_max, l_mod, b, v1_s, v2_s,
                            pv1, pv2, bq1, bq2, ell_seeds, mi)
    for i in range(n_ell):
        term_AA[i] = A_v * A_q * mi[i]

    if n_regions_0 == 2:
        _t_l_ab_vq_int_ells_tab(l_max, l_mod, b, v2_s, v3_s,
                                pv2, pv3, bq1, bq2, ell_seeds, mi)
        for i in range(n_ell):
            term_BA[i] = B_v * A_q * mi[i]

    if n_regions_1 == 2:
        _t_l_ab_vq_int_ells_tab(l_max, l_mod, b, v1_s, v2_s,
                                pv1, pv2, bq2, bq3, ell_seeds, mi)
        for i in range(n_ell):
            term_AB[i] = A_v * B_q * mi[i]

    if n_regions_0 == 2 and n_regions_1 == 2:
        _t_l_ab_vq_int_ells_tab(l_max, l_mod, b, v2_s, v3_s,
                                pv2, pv3, bq2, bq3, ell_seeds, mi)
        for i in range(n_ell):
            term_BB[i] = B_v * B_q * mi[i]

    for i in range(n_ell):
        out[i * out_stride] = factor * (term_AA[i] + term_BA[i] + term_AB[i] + term_BB[i])

# @cython.nogil
cdef void kin_matrix_ells_c(int l_max, int l_mod, int nv, int nq,
                           double v_max, double q_max,
                           int log_wavelet_q, double eps_q,
                           int a, int b,
                           double q_star, double v_star, double factor,
                           const double[:, :, ::1] pv_table,
                           const double[:, :, ::1] bq_table,
                           const double[::1] ell_seeds,
                           double* out, int out_stride) noexcept nogil:
    # MIRROR of analytic.py:kin_matrix_ells. Same wavelet decomposition as
    # kin_matrix_lvq_c, but every l of one (nv, nq) cell is computed together so
    # the l-independent endpoint evaluations are shared. Writes I_l(nv, nq)
    # to out[idx_ell * out_stride] for idx_ell = 0 .. l_max // l_mod.

    cdef int n_regions_0 = 1
    cdef int n_regions_1 = 1

    cdef double v1, v2, v3
    cdef double A_v, B_v

    # Haar V
    cdef double supp_v[3]
    haar_support(nv, &supp_v[0], &supp_v[1], &supp_v[2])
    v1 = supp_v[0]
    v2 = supp_v[1]
    v3 = supp_v[2]

    cdef double val_v[2]
    haar_value(nv, 3, &val_v[0], &val_v[1])
    A_v = val_v[0]

    if nv == 0:
        n_regions_0 = 1
        B_v = 0.0 # Unused
    else:
        n_regions_0 = 2
        B_v = val_v[1] # Negative value

    v1 *= v_max
    v2 *= v_max
    v3 *= v_max

    # Haar Q
    cdef double q1, q2, q3
    cdef double A_q, B_q
    cdef double supp_q[3]
    cdef double val_q[2]

    if log_wavelet_q:
        haar_support_log(nq, eps_q, &supp_q[0], &supp_q[1], &supp_q[2])
        q1 = supp_q[0]
        q2 = supp_q[1]
        q3 = supp_q[2]

        haar_value_log(nq, eps_q, 2, &val_q[0], &val_q[1])
        A_q = val_q[0]

        if nq == 0:
            n_regions_1 = 1
            B_q = 0.0
        else:
            n_regions_1 = 2
            B_q = val_q[1]
    else:
        haar_support(nq, &supp_q[0], &supp_q[1], &supp_q[2])
        q1 = supp_q[0]
        q2 = supp_q[1]
        q3 = supp_q[2]

        haar_value(nq, 3, &val_q[0], &val_q[1])
        A_q = val_q[0]

        if nq == 0:
            n_regions_1 = 1
            B_q = 0.0
        else:
            n_regions_1 = 2
            B_q = val_q[1]

    q1 *= q_max
    q2 *= q_max
    q3 *= q_max

    # Normalize by Star
    cdef double v1_s = v1 / v_star
    cdef double v2_s = v2 / v_star
    cdef double v3_s = v3 / v_star
    cdef double q1_s = q1 / q_star
    cdef double q2_s = q2 / q_star
    cdef double q3_s = q3 / q_star

    # Interior fast path: if the whole cell lies inside the kinematically
    # allowed region -- the binding corner is the lowest-v edge, since
    # tilq_m falls and tilq_p rises with v (same tilq arithmetic as
    # mI_star_ells, so the classification is exact) -- then no clamp can
    # bind in any of the four mI_star_ells calls below and the endpoint
    # tables apply. Boundary-crossing cells fall through to the general
    # path.
    cdef double sqrt_v1s
    if v1_s > 1.0:
        sqrt_v1s = sqrt(pow(v1_s, 2) - 1.0)
        if v1_s - sqrt_v1s <= q1_s and v1_s + sqrt_v1s >= q3_s:
            _kin_matrix_ells_interior_c(l_max, l_mod, b,
                                   n_regions_0, n_regions_1,
                                   v1_s, v2_s, v3_s,
                                   A_v, B_v, A_q, B_q, factor,
                                   &pv_table[nv, 0, 0], &pv_table[nv, 1, 0],
                                   &pv_table[nv, 2, 0],
                                   &bq_table[nq, 0, 0], &bq_table[nq, 1, 0],
                                   &bq_table[nq, 2, 0],
                                   &ell_seeds[0],
                                   out, out_stride)
            return

    # Integration: same four terms as kin_matrix_lvq_c, per l.
    cdef int n_ell = l_max // l_mod + 1
    cdef int i
    cdef double mi[_MAX_ELLS]
    cdef double term_AA[_MAX_ELLS]
    cdef double term_BA[_MAX_ELLS]
    cdef double term_AB[_MAX_ELLS]
    cdef double term_BB[_MAX_ELLS]

    for i in range(n_ell):
        term_BA[i] = 0.0
        term_AB[i] = 0.0
        term_BB[i] = 0.0

    mI_star_ells(l_max, l_mod, a, b, v1_s, v2_s, q1_s, q2_s, mi)
    for i in range(n_ell):
        term_AA[i] = A_v * A_q * mi[i]

    if n_regions_0 == 2:
        mI_star_ells(l_max, l_mod, a, b, v2_s, v3_s, q1_s, q2_s, mi)
        for i in range(n_ell):
            term_BA[i] = B_v * A_q * mi[i]

    if n_regions_1 == 2:
        mI_star_ells(l_max, l_mod, a, b, v1_s, v2_s, q2_s, q3_s, mi)
        for i in range(n_ell):
            term_AB[i] = A_v * B_q * mi[i]

    if n_regions_0 == 2 and n_regions_1 == 2:
        mI_star_ells(l_max, l_mod, a, b, v2_s, v3_s, q2_s, q3_s, mi)
        for i in range(n_ell):
            term_BB[i] = B_v * B_q * mi[i]

    for i in range(n_ell):
        out[i * out_stride] = factor * (term_AA[i] + term_BA[i] + term_AB[i] + term_BB[i])

# // ---------------------------------------------------------
# // Python Wrapper
# //
# // MIRROR of analytic.kin_matrix_lvq; signatures are aligned (ell, nv, nq as
# // three ints). Error behaviour deliberately differs from the Python mirror:
# // the Python path ``assert``s its preconditions, whereas the compiled C
# // integrand (kin_matrix_lvq_c and the cdef Haar primitives) is
# // ``noexcept nogil`` and returns silently -- it cannot raise from inside the
# // parallel prange loop.
# // ---------------------------------------------------------

def kin_matrix_lvq(int ell, 
                  int nv, 
                  int nq, 
                  double v_max, 
                  double q_max, 
                  int log_wavelet_q, 
                  double eps_q, 
                  int a, 
                  int b,
                  double q_star,
                  double v_star,
                  double factor):
    """
    Compute I_l(nv,nq) analytically for given wavelet indices.
    
    Parameters
    ----------
    ell : int
        Wavelet angular momentum index.
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

    cdef double result

    with cython.nogil:
        result = kin_matrix_lvq_c(
            ell, nv, nq,
            v_max, q_max, log_wavelet_q, eps_q,
            a, b, q_star, v_star, factor
        )
    
    return result

def kin_matrix(int l_max,
         int l_mod,
         int nv_max,
         int nq_max,
         double v_max, 
         double q_max, 
         int log_wavelet_q, 
         double eps_q, 
         int a, 
         int b,
         double q_star,
         double v_star,
         double factor):
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
    a : int
    b : int
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
        Array of shape (l_max//l_mod+1, nv_max+1, nq_max+1) containing I_l(nv,nq) values.

    Notes
    -----
    The (nv, nq) cells are independent and iterated in parallel; all l values
    of one cell are computed together by kin_matrix_ells_c, which shares the
    l-independent endpoint evaluations across l. Endpoint values shared
    across cells are precomputed once into per-call tables (see
    _fill_endpoint_tables) and read back by every kinematically interior
    cell; the tables are read-only inside the loop.
    """

    if l_max + 1 > _MAX_ELLS:
        raise ValueError(
            f"kin_matrix: l_max={l_max} exceeds the compiled stack-buffer "
            f"limit (l_max <= {_MAX_ELLS - 1}); raise _MAX_ELLS in "
            f"analytic_cy.pyx to go higher."
        )

    cdef int n_nv = nv_max + 1
    cdef int n_nq = nq_max + 1
    cdef int n_ell = l_max // l_mod + 1

    # Create output numpy array
    cdef cnp.ndarray[double, ndim=3] result = np.zeros((l_max//l_mod + 1, n_nv, n_nq), dtype=np.float64)

    # Get memoryview for fast C access
    cdef double[:, :, ::1] result_view = result

    # Per-call endpoint tables shared by the interior cells (see
    # _fill_endpoint_tables); read-only inside the loop, so safe under
    # concurrent callers.
    cdef double[:, :, ::1] pv_table = np.zeros((n_nv, 3, l_max + 1))
    cdef double[:, :, ::1] bq_table = np.zeros((n_nq, 3, l_max + 1))
    cdef double[::1] ell_seeds = np.zeros(n_ell)

    # Element stride between consecutive l values at fixed (nv, nq) in the
    # C-contiguous (n_ell, n_nv, n_nq) result.
    cdef int ell_stride = n_nv * n_nq
    cdef int total = n_nv * n_nq
    cdef int idx, nv, nq

    with cython.nogil:
        _fill_endpoint_tables(
            l_max, l_mod, nv_max, nq_max,
            v_max, q_max, log_wavelet_q, eps_q,
            a, b, q_star, v_star,
            pv_table, bq_table, ell_seeds
        )
        for idx in prange(total, schedule='dynamic'):
            nv = idx // n_nq
            nq = idx % n_nq

            kin_matrix_ells_c(
                l_max, l_mod, nv, nq,
                v_max, q_max, log_wavelet_q, eps_q,
                a, b, q_star, v_star, factor,
                pv_table, bq_table, ell_seeds,
                &result_view[0, nv, nq], ell_stride
            )

    return result
