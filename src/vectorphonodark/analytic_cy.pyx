# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as cnp
cimport cython
from cython.parallel import prange

from libc.math cimport sqrt, log, exp, pow, floor, log2, fabs

from scipy.special.cython_special cimport hyp2f1, spence, gamma

cdef double PI = 3.14159265358979323846

# // ---------------------------------------------------------
# // Haar Wavelet Functions
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
# // Analytic Integrals
# // ---------------------------------------------------------

# @cython.nogil
cdef double _b_nk_int(int n, int k, double x) noexcept nogil:
    # Range in 'x' should not include x=0
    if x == 0: return 0.0
    
    cdef double sum_val = 0.0
    cdef int j
    cdef double comb, ipower, summand
    
    for j in range(k + 1):
        comb = gamma(k + 1.0) / (gamma(j + 1.0) * gamma(k - j + 1.0))
        ipower = j + (n - k)/2.0 + 1.0
        
        if ipower == 0:
            summand = log(x)
        else:
            summand = pow(x, ipower) / ipower
        sum_val += comb * summand
        
    return 0.5 * sum_val

# @cython.nogil
cdef double _c_alpha_int(double alpha, double x) noexcept nogil:
    cdef double val, yx, sum_val, comb
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
        return (-0.25 * pow(log(x), 2) + log(x) * log(1.0 + x) + val)
    elif alpha_int == 1:
        return 0.5 * x - log(1.0 + x)
    elif alpha_int == 2:
        return (0.125 * x - 0.5) * x + 0.5 * log(1.0 + x)
    elif alpha_int > 1:
        sum_val = pow(-1.0, alpha_int) * log(1.0 + x) / alpha + pow(1.0 + x, alpha_int) / (2.0 * alpha * alpha)
        for j in range(1, alpha_int):
            comb = (
                gamma(alpha + 1.0) / (gamma(j + 1.0) * gamma(alpha - j + 1.0))
                + gamma(alpha) / (gamma(j + 1.0) * gamma(alpha - j))
            )
            sum_val += pow(-1.0, alpha_int - j) / (2.0 * alpha * j) * comb * pow(1.0 + x, j)
        return sum_val
    elif alpha_int < -1:
        yx = (1.0 + x) / x
        sum_val = pow(-1.0, alpha_int) * log(yx) / alpha - pow(yx, -alpha_int) / (2.0 * alpha * alpha)
        for j in range(1, -alpha_int):
            comb = (
                gamma(-alpha + 1.0) / (gamma(j + 1.0) * gamma(-alpha - j + 1.0))
                + gamma(-alpha) / (gamma(j + 1.0) * gamma(-alpha - j))
            )
            sum_val += pow(-1.0, alpha_int + j) / (2.0 * alpha * j) * comb * pow(yx, j)
        return sum_val
    return 0.0

# @cython.nogil
cdef double _v_ab_int(int a, int b, double x) noexcept nogil:
    cdef double sum_2 = 0.0
    cdef int j, b_int
    cdef double comb
    b_int = <int>b
    
    for j in range(b_int + 3):
        comb = gamma(b + 3.0) / (gamma(j + 1.0) * gamma(b + 3.0 - j))
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
    
    for j in range(b_int + 3):
        comb = gamma(b + 3.0) / (gamma(j + 1.0) * gamma(b + 3.0 - j))
        sum_val += 0.5 * comb * _c_alpha_int(j + (a - b) / 2.0, x)
    return sum_val

# @cython.nogil
cdef double _t_l_ab_vq_int(int l, int a, int b, double v1, double v2, double q1, double q2) noexcept nogil:
    cdef double x1 = pow(q1, 2)
    cdef double x2 = pow(q2, 2)
    cdef double sum_val = 0.0
    cdef int k
    cdef double term_k, termQ, termV
    
    # Range step 2
    for k in range(l % 2, l + 1, 2):
        term_k = pow(2.0, l - k) * gamma(0.5 * (k + 1 + l)) / gamma(0.5 * (k + 1 - l))
        term_k /= (gamma(k + 1.0) * gamma(l - k + 1.0))
        
        termQ = (_b_nk_int(a, k, x2) - _b_nk_int(a, k, x1))
        
        if k == b + 2:
            termV = log(v2 / v1)
        else:
            termV = (pow(v2, b + 2 - k) - pow(v1, b + 2 - k)) / (b + 2.0 - k)
        sum_val += termV * term_k * termQ
    return sum_val

# @cython.nogil
cdef double _u_l_ab_vq_int(int l, int a, int b, double v2, double q1, double q2) noexcept nogil:
    cdef double x1 = pow(q1, 2)
    cdef double x2 = pow(q2, 2)
    cdef double sum_val = 0.0
    cdef int k
    cdef double term_k, term_x, t_x
    
    for k in range(l % 2, l + 1, 2):
        term_k = (gamma(0.5 * (k + 1 + l)) / gamma(0.5 * (k + 1 - l))
                  * pow(2.0, l - k) / (gamma(k + 1.0) * gamma(l - k + 1.0)))
        
        if k == b + 2:
            term_x = (log(2.0 * v2) * (_b_nk_int(a, k, x2) - _b_nk_int(a, k, x1))
                      + _s_ab_int(a, b, x2) - _s_ab_int(a, b, x1))
            sum_val += term_k * term_x
        else:
            t_x = (pow(v2, b + 2 - k) * (_b_nk_int(a, k, x2) - _b_nk_int(a, k, x1))
                   - pow(2.0, k - b - 2) * (_b_nk_int(a, b + 2, x2) - _b_nk_int(a, b + 2, x1)))
            sum_val += term_k * t_x / (b + 2.0 - k)
    return sum_val

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
cdef double ilvq_analytic_c(int ell, int nv, int nq, 
                            double v_max, double q_max, 
                            int log_wavelet_q, double eps_q, 
                            int a, int b, double q0_fdm, double v0_fdm,
                            double mass_dm, double mass_sm, double energy) noexcept nogil:
    
    cdef double mass_reduced = (mass_dm * mass_sm) / (mass_dm + mass_sm)
    cdef double qStar = sqrt(2.0 * mass_dm * energy)
    cdef double vStar = qStar / mass_dm
    
    cdef double factor = (
        pow(q_max / v_max, 3) / (2.0 * mass_dm * pow(mass_reduced, 2))
        * pow(2.0 * energy / (q_max * v_max), 2)
        * pow(qStar / q0_fdm, <double>a) * pow(vStar / v0_fdm, <double>b)
    )
    
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
    cdef double v1_s = v1 / vStar
    cdef double v2_s = v2 / vStar
    cdef double v3_s = v3 / vStar
    cdef double q1_s = q1 / qStar
    cdef double q2_s = q2 / qStar
    cdef double q3_s = q3 / qStar
    
    term_AA = A_v * A_q * mI_star(ell, a, b, v1_s, v2_s, q1_s, q2_s)
    
    if n_regions_0 == 2:
        term_BA = B_v * A_q * mI_star(ell, a, b, v2_s, v3_s, q1_s, q2_s)
        
    if n_regions_1 == 2:
        term_AB = A_v * B_q * mI_star(ell, a, b, v1_s, v2_s, q2_s, q3_s)
        
    if n_regions_0 == 2 and n_regions_1 == 2:
        term_BB = B_v * B_q * mI_star(ell, a, b, v2_s, v3_s, q2_s, q3_s)
        
    return factor * (term_AA + term_BA + term_AB + term_BB)

# // ---------------------------------------------------------
# // Python Wrapper
# // ---------------------------------------------------------

def ilvq_analytic(int ell, 
                  int nv, 
                  int nq, 
                  double v_max, 
                  double q_max, 
                  int log_wavelet_q, 
                  double eps_q, 
                  tuple fdm, 
                  double q0_fdm, 
                  double v0_fdm,
                  double mass_dm, 
                  double mass_sm, 
                  double energy, 
                  int verbose=0):
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
    fdm : tuple
        Dark matter form factor parameters (a,b) with
        F_DM(q,v) = (q/q0)**a * (v/v0)**b
    q0_fdm : float
        Reference momentum for form factor in eV.
    v0_fdm : float
        Reference velocity for form factor (dimensionless).
    mass_dm : float
        Dark matter mass in eV.
    mass_sm : float
        Standard model target mass in eV.
    energy : float
        Energy transfer in eV.
    verbose : bool, optional
        Whether to print verbose output. Default is False.  

    Returns
    -------
    Ilvq : float
        The computed I_l(nv,nq) value.
    """

    cdef int a = fdm[0]
    cdef int b = fdm[1]

    cdef double ilvq
    
    with cython.nogil:
        ilvq = ilvq_analytic_c(
            ell, nv, nq,
            v_max, q_max, log_wavelet_q, eps_q,
            a, b, q0_fdm, v0_fdm,
            mass_dm, mass_sm, energy
        )
    
    return ilvq

def ilvq(int l_max, 
         long[:] nv_list, 
         long[:] nq_list, 
         double v_max, 
         double q_max, 
         int log_wavelet_q, 
         double eps_q, 
         tuple fdm, 
         double q0_fdm, 
         double v0_fdm,
         double mass_dm, 
         double mass_sm, 
         double energy, 
         int verbose=0):
    """
    Compute I_l(nv,nq) for all l in [0,l_max], nv in nv_list, nq in nq_list.

    Parameters
    ----------
    l_max : int
        Maximum l value.
    nv_list : list[int]
        List of nv indices.
    nq_list : list[int]
        List of nq indices.
    v_max : float
        Maximum velocity.
    q_max : float
        Maximum momentum transfer.
    log_wavelet_q : bool
        Whether the q wavelets are log-spaced.
    eps_q : float
        Minimum q/q_max value for log-spaced q wavelets.
    fdm : tuple
        Dark matter form factor parameters (a,b) with
        F_DM(q,v) = (q/q0)**a * (v/v0)**b
    q0_fdm : float
        Reference momentum transfer for form factor.
    v0_fdm : float
        Reference velocity for form factor.
    mass_dm : float
        Dark matter mass.
    mass_sm : float
        Standard model target mass.
    energy : float
        Energy transfer.
    verbose : bool, optional
        Whether to print verbose output, by default False.

    Returns
    -------
    ilvq_array : np.ndarray
        Array of shape (l_max+1, len(nv_list), len(nq_list)) containing I_l(nv,nq) values.
    """
    
    cdef int n_nv = nv_list.shape[0]
    cdef int n_nq = nq_list.shape[0]
    cdef int a = fdm[0]
    cdef int b = fdm[1]
    
    # Create output numpy array
    cdef cnp.ndarray[double, ndim=3] ilvq_array = np.zeros((l_max + 1, n_nv, n_nq), dtype=np.float64)
    
    # Get memoryview for fast C access
    cdef double[:, :, ::1] ilvq_view = ilvq_array
    
    cdef int ell, idx_nv, idx_nq
    cdef int nv_val, nq_val
    
    with cython.nogil:
        for ell in prange(l_max + 1, schedule='dynamic'):
            for idx_nv in range(n_nv):
                nv_val = nv_list[idx_nv]
                for idx_nq in range(n_nq):
                    nq_val = nq_list[idx_nq]
                    
                    ilvq_view[ell, idx_nv, idx_nq] = ilvq_analytic_c(
                        ell, nv_val, nq_val,
                        v_max, q_max, log_wavelet_q, eps_q,
                        a, b, q0_fdm, v0_fdm,
                        mass_dm, mass_sm, energy
                    )
    
    return ilvq_array