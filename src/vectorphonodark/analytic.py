"""Analytic results for mcalI matrix calculation.

Dimensionless integrals of functions of q and v_min(q), as functions
of x = (q / q_star)**2, for q_star**2 = 2 * omegaS * mX.

The first set of functions (lower case) assumes F_DM is of the form
    F_DM(q) = (q0/q)**n
e.g. for n=0 (heavy mediator), n=2 (light mediator), so that F_DM**2 is
    F_DM**2 = (q0/q)**(2*n)

The second set of functions (upper case) allows a velocity dependent F_DM,
    F_DM(q,v) = (q/q0)**a * (v/c)**b
for e.g. b=0,2 and a=0,-2,-4.

Functions:
    _t: integral for rectangular volume, [[v1,v2],[q1,q2]]
    _u: integral where v_min(q) sets lower bound, [v_min(q),v2]
    _b: intermediate function for rectangular integral
    _c, _d, _s: intermediate function for non-rectangular integrals (for _u)
"""

# __all__ = ['mI_star', '_t_l_ab_vq_int', '_u_l_ab_vq_int',
        #    '_b_nk_int', '_c_alpha_int', '_s_ab_int', '_v_ab_int']

import math
import numpy as np
import numba
from numba import objmode
import scipy.special as spf

from . import basis_funcs

# import vsdm


@numba.njit
def _b_nk_int(n, k, x):
    # x = q^2 / q_*^2.
    assert k-int(k)==0, "'k' should be integer valued"
    k = int(k)
    assert x!=0, "Range in 'x' should not include x=0, even if basis function includes q=0."
    sum = 0.
    comb = 1.
    for j in range(k+1):
        # comb = math.gamma(k + 1) / (math.gamma(j + 1) * math.gamma(k - j + 1))
        if j > 0:
            comb *= (k - j + 1.) / j
        ipower = j + (n-k)/2 + 1
        if ipower==0:
            summand = math.log(x)
        else:
            summand = (x**ipower)/ipower
        sum += comb*summand
    return 0.5*sum


@numba.njit
def _c_alpha_int(alpha, x):
    if alpha%1 != 0: # float valued
        with objmode(val='float64'):
            val = spf.hyp2f1(1, alpha, 1 + alpha, -x)
        return x**(alpha)/alpha**2 * (0.5 - val)
    else: # integer-valued
        alpha = int(alpha)
    if alpha==-2:
        # return -0.5*math.log( (1+x)/x ) + 0.5/x - 0.125/x**2
        return -0.5*math.log( (1+x)/x ) + (0.5 - 0.125/x)/x
    elif alpha==-1:
        return math.log( (1+x)/x ) - 0.5/x
    elif alpha==0:
        with objmode(val='float64'):
            val = spf.spence(1 + x)
        # return (-0.25*(math.log(x))**2 + math.log(x) * math.log(1 + x) + val)
        return (-0.25*(math.log(x))**2 + math.log(x) * math.log1p(x) + val)
    elif alpha==1:
        # return 0.5*x - math.log(1 + x)
        return 0.5*x - math.log1p(x)
    elif alpha==2:
        # return 0.125*x**2 - 0.5*x + 0.5*math.log(1 + x)
        return (0.125*x - 0.5)*x + 0.5*math.log1p(x)
    elif alpha > 1: # other positive integers
        # sum = (-1)**alpha * math.log(1+x) / alpha + (1+x)**alpha / (2*alpha**2)
        sum = (-1)**alpha * math.log1p(x) / alpha + (1+x)**alpha / (2*alpha**2)
        comb_alpha_j = alpha
        for j in range(1, alpha):
            # comb = (
            #     math.gamma(alpha+1) / (math.gamma(j+1) * math.gamma(alpha-j+1))
            #     + math.gamma(alpha) / (math.gamma(j+1) * math.gamma(alpha-j))
            # )
            comb = comb_alpha_j * (2.*alpha-j) / j
            comb_alpha_j *= (alpha-j) / (j+1.)
            sum += (-1)**(alpha-j)/(2*alpha*j) * comb * (1+x)**j
        return sum
    elif alpha < -1: # other negative integers
        yx = (1+x)/x
        sum = (-1)**alpha * math.log(yx) / alpha - yx**(-alpha) / (2*alpha**2)
        comb_alpha_j = -alpha
        for j in range(1, -alpha):
            # comb = (
            #     math.gamma(-alpha+1) / (math.gamma(j+1) * math.gamma(-alpha-j+1))
            #     + math.gamma(-alpha) / (math.gamma(j+1) * math.gamma(-alpha-j))
            # )
            comb = comb_alpha_j * (-2.*alpha-j) / j
            comb_alpha_j *= (-alpha-j) / (j+1.)
            sum += (-1)**(alpha+j)/(2*alpha*j) * comb * yx**j
        return sum


@numba.njit
def _v_ab_int(a, b, x):
    sum_2 = 0.
    comb = 1.
    for j in range(b+2+1):
        # comb = math.gamma(b+2+1) / (math.gamma(j+1) * math.gamma(b+2-j+1))
        if j > 0:
            comb *= (b+2-j+1.) / j
        if 2*j==(b-a):
            sum_2 += comb * math.log(x)
        else:
            sum_2 += comb * x**(j + (a-b)/2) / (j + (a-b)/2)
    return 0.5 * sum_2


@numba.njit
def _s_ab_int(a, b, x):
    logfactor = 0.5*math.log(x/((1+x)**2))
    sum = logfactor*_v_ab_int(a, b, x)
    comb = 1.
    for j in range(b+2+1):
        # comb = math.gamma(b+2+1) / (math.gamma(j+1) * math.gamma(b+2-j+1))
        if j > 0:
            comb *= (b+2-j+1.) / j
        sum += 0.5*comb*_c_alpha_int(j+(a-b)/2, x)
    return sum


@numba.njit
def _t_l_ab_vq_int(l, a, b, v12_star, q12_star):
    """Rectangular integral T_{l,n}.

    With [v1,v2] in units of v_star = q_star/mX, [q1,q2] in units of
        q_star = sqrt(2*mX*omegaS).

    Always v1 >= 1. Also require q1 > 0.
    """
    assert int(l)-l==0, "'l' must be integer valued"
    l = int(l)
    [v1,v2] = v12_star
    [q1,q2] = q12_star
    x1 = q1**2
    x2 = q2**2
    sum = 0.
    k_start = l%2
    term_k = (math.gamma(0.5*(k_start+1+l)) / math.gamma(0.5*(k_start+1-l))
              * 2.**(l-k_start)/(math.gamma(k_start+1)*math.gamma(l-k_start+1)))
    for k in range(l%2, l+1, 2):
        # only terms with (l-k)%2==0 contribute to the sum:
        # term_k = 2.**(l-k) * math.gamma(0.5*(k+1+l)) / math.gamma(0.5*(k+1-l))
        # term_k /= (math.gamma(k+1)*math.gamma(l-k+1))
        termQ = (_b_nk_int(a, k, x2) - _b_nk_int(a, k, x1))
        if k==b+2:
            termV = math.log(v2/v1)
        else:
            termV = (v2**(b+2-k) - v1**(b+2-k)) / (b+2-k)
        sum += termV*term_k*termQ
        term_k *= -(l-k)*(l+k+1) / (4.*(k+1)*(k+2))
    return sum


@numba.njit
def _u_l_ab_vq_int(l, a, b, v2_star, q12_star):
    """Non-rectangular integral U_{l,fdm}, with lower bound v1 = v_min(q).

    With v2 in units of v_star = q_star/mX, [q1,q2] in units of
        q_star = sqrt(2*mX*omegaS).
    """
    assert int(l)-l==0, "'l' must be integer valued"
    l = int(l)
    v2 = v2_star # only need v2, v1 is irrelevant
    [q1,q2] = q12_star
    x1 = q1**2
    x2 = q2**2
    sum = 0.
    k_start = l%2
    term_k = (math.gamma(0.5*(k_start+1+l)) / math.gamma(0.5*(k_start+1-l))
              * 2.**(l-k_start)/(math.gamma(k_start+1)*math.gamma(l-k_start+1)))
    for k in range(l%2, l+1, 2):
        # term_k = (math.gamma(0.5*(k+1+l)) / math.gamma(0.5*(k+1-l))
        #           * 2.**(l-k)/(math.gamma(k+1) * math.gamma(l-k+1)))
        if k==b+2:
            term_x = (math.log(2.*v2)*(_b_nk_int(a, k, x2) - _b_nk_int(a, k, x1))
                      + _s_ab_int(a, b, x2) - _s_ab_int(a, b, x1))
            sum += term_k*term_x
        else:
            t_x = (v2**(b+2-k)*(_b_nk_int(a, k, x2) - _b_nk_int(a, k, x1))
                   - 2.**(k-b-2)*(_b_nk_int(a, b+2, x2) - _b_nk_int(a, b+2, x1)))
            sum += term_k*t_x/(b+2-k)
        term_k *= -(l-k)*(l+k+1) / (4.*(k+1)*(k+2))
    return sum


@numba.njit
def mI_star(ell, fdm, v12_star, q12_star):
    """Dimensionless integral related to MathcalI.

    fdm: label specifying the form factor type.
    If fdm is an int, float, or tuple of length 1, then:
        FDM2(q) = (q0/q)**(2*n), with n = fdm.
    If fdm is a tuple of length 2, then:
        FDM2(q) = (q/q0)**(a) * (v/c)**b, with (a,b) = fdm.

    This is $I^{(\\ell)}_\\star$ without the prefactors of qStar and vStar:
        if fdm = n: prefactor = (qStar/qBohr)**(-2*n)
        if fdm = (a,b): prefactor = (qStar/qBohr)**a * (vStar/c)**b

    Integration region v12, q12: given in units of vStar, qStar.

    There are 0, 1, 2 or 3 regions that contribute to mcalI:
        qA < (R1) < qB < (R2) < qC < (R3) < qD.
        R2 is rectangular, bounded by v1 < v < v2. -> _t_l_ab_vq_int
        R1 and R3 are not rectangular: vMin(q) < v < v2. -> _u_l_ab_vq_int
    If vmin(q) > v1 for all q1 < q < q2, then mcalI is given by _u_l_ab_vq_int
    """
    (a, b) = fdm
    [v1,v2] = v12_star
    [q1,q2] = q12_star
    if v1==v2 or q1==q2:
        return 0. # No integration volume
    assert q1 < q2, "Need q12 to be ordered"
    assert v1 < v2, "Need v12 to be ordered"
    include_R2 = True
    if v2 < 1:
        # v2 is below the velocity threshold. mcaI=0
        return 0.
    tilq_m, tilq_p = v2 - math.sqrt(v2**2-1.), v2 + math.sqrt(v2**2-1.)
    if tilq_m > q2 or tilq_p < q1:
        # in this case v2 < vmin(q) for all q in [q1,q2]
        return 0.
    # Else: there are some q satisfying vmin(q) < v2 in this interval.
    if v1 < 1:
        # There is no v1 = vmin(q) solution for any real q
        include_R2 = False
    # Else: There are two real solutions to v1 = vmin(q)
    else:
        q_m,q_p = v1 - math.sqrt(v1**2-1.), v1 + math.sqrt(v1**2-1.)
        if q_m > q2 or q_p < q1:
            # in this case v1 < vmin(q) for all q in [q1,q2]
            include_R2 = False
    if include_R2 is False:
        q_A = max(q1, tilq_m)
        q_B = min(q2, tilq_p)
        return _u_l_ab_vq_int(ell, a, b, v2, [q_A, q_B])
    # Else: at least part of the integration volume is set by v1 < v.
    q_a = max(q1, tilq_m)
    q_b = max(q1, q_m) # q_m > tilq_m iff v2 > v1
    q_c = min(q2, q_p) # q_p < tilq_p iff v2 > v1
    q_d = min(q2, tilq_p)
    includeRegion = [True, True, True]
    if q_a==q_b:
        includeRegion[0] = False
    if q_c==q_d:
        includeRegion[2] = False
    if v1>1:
        assert q_b!=q_c, "If q_b==q_c then there should be no R2 region..."
    mI_0,mI_1,mI_2 = 0., 0., 0.
    if includeRegion[0]:
        mI_0 = _u_l_ab_vq_int(ell, a, b, v2, [q_a, q_b])
    if includeRegion[1]:
        mI_1 = _t_l_ab_vq_int(ell, a, b, [v1,v2], [q_b, q_c])
    if includeRegion[2]:
        mI_2 = _u_l_ab_vq_int(ell, a, b, v2, [q_c, q_d])
    return mI_0 + mI_1 + mI_2


@numba.njit
def ilvq_analytic(lnvq, v_max, q_max, log_wavelet_q, eps_q, 
                  fdm, q0_fdm, v0_fdm,
                  mass_dm, mass_sm, energy, verbose=False):
    """
    Compute I_l(nv,nq) analytically for given wavelet indices.
    
    Parameters
    ----------
    lnvq : tuple[int, int, int]
        Tuple of (ell, nv, nq) wavelet indices.
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
    
    (ell, nv, nq) = lnvq
    (a, b) = fdm
    mass_reduced = (mass_dm * mass_sm) / (mass_dm + mass_sm)

    # Integrand is written in terms of dimensionless vStar and qStar:
    qStar = np.sqrt(2*mass_dm*energy)
    vStar = qStar/mass_dm

    factor = (
        (q_max/v_max)**3 / (2*mass_dm*mass_reduced**2) 
        * (2*energy/(q_max*v_max))**2
        * (qStar/q0_fdm)**a * (vStar/v0_fdm)**b
    )
    n_regions = [1,1]

    v1, v2, v3 = basis_funcs.haar_support(nv)
    if nv==0:
        A_v, _ = basis_funcs.haar_value(nv, dim=3)
        n_regions[0] = 1
    else:
        A_v, B_v = basis_funcs.haar_value(nv, dim=3)
        n_regions[0] = 2
    v1, v2, v3 = v1*v_max, v2*v_max, v3*v_max

    if log_wavelet_q:
        q1, q2, q3 = basis_funcs.haar_support_log(nq, eps_q)
        if nq==0:
            A_q, _ = basis_funcs.haar_value_log(nq, eps_q, p=2)
            n_regions[1] = 1
        else:
            A_q, B_q = basis_funcs.haar_value_log(nq, eps_q, p=2)
            n_regions[1] = 2
    else:
        q1, q2, q3 = basis_funcs.haar_support(nq)
        if nq==0:
            A_q, _ = basis_funcs.haar_value(nq, dim=3)
            n_regions[1] = 1
        else:
            A_q, B_q = basis_funcs.haar_value(nq, dim=3)
            n_regions[1] = 2
    q1, q2, q3 = q1*q_max, q2*q_max, q3*q_max

    # There is always an A_v A_q term:
    v12_star = [v1/vStar, v2/vStar]
    q12_star = [q1/qStar, q2/qStar]
    term_AA = A_v*A_q * mI_star(ell, fdm, v12_star, q12_star)

    # There are only B-type contributions if V or Q uses wavelets
    term_AB, term_BA, term_BB = 0., 0., 0.
    if n_regions[0]==2:
        v23_star = [v2/vStar, v3/vStar]
        term_BA = B_v*A_q * mI_star(ell, fdm, v23_star, q12_star)
    if n_regions[1]==2:
        q23_star = [q2/qStar, q3/qStar]
        term_AB = A_v*B_q * mI_star(ell, fdm, v12_star, q23_star)
    if n_regions==[2,2]:
        term_BB = B_v*B_q * mI_star(ell, fdm, v23_star, q23_star)
    Ilvq = factor * (term_AA + term_BA + term_AB + term_BB)

    return Ilvq


# def ilvq_vsdm(lnvq, v_max, q_max, log_wavelet_q, eps_q, 
#               fdm, q0_fdm, v0_fdm,
#               mass_dm, mass_sm, energy, verbose=False):
#     """
#     Compute I_l(nv,nq) using the vsdm package.
#     Only works for linear wavelets.
#     The scaling for q and v are fixed inside vsdm.McalI,
#         with q0_fdm = Q_BOHR and v0_fdm = 1.0.
#     """
    
#     if log_wavelet_q:
#         raise NotImplementedError("vsdm McalI does not support log wavelets yet.")
    
#     (ell, nv, nq) = lnvq

#     basis_v = dict(u0=v_max, type="wavelet", uMax=v_max)
#     basis_q = dict(u0=q_max, type="wavelet", uMax=q_max)
#     dm_model = dict(mX=mass_dm, fdm=fdm, mSM=mass_sm, DeltaE=energy)
#     mI = vsdm.McalI(
#         basis_v, basis_q, dm_model, use_gvar=False, do_mcalI=False
#     )

#     Ilvq = mI.getI_lvq_analytic((ell, nv, nq))

#     return Ilvq


@numba.njit(parallel=True)
def ilvq(l_max, l_mod, nv_max, nq_max, 
         v_max, q_max, log_wavelet_q, eps_q, 
         fdm, q0_fdm, v0_fdm,
         mass_dm, mass_sm, energy, verbose=False):
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
        Array of shape (l_max//l_mod + 1, nv_max + 1, nq_max + 1) containing I_l(nv,nq) values.
    """
    
    shape = (l_max//l_mod + 1, nv_max + 1, nq_max + 1)
    ilvq_array = np.zeros(shape, dtype=float)
    
    for ell in numba.prange(0, l_max + 1, l_mod):
        for nv in range(nv_max + 1):
            for nq in range(nq_max + 1):
                lnvq = (ell, nv, nq)
                ilvq_array[ell//l_mod, nv, nq] = ilvq_analytic(
                    lnvq, v_max, q_max, log_wavelet_q, eps_q,
                    fdm, q0_fdm, v0_fdm,
                    mass_dm, mass_sm, energy, verbose=verbose
                )
    return ilvq_array