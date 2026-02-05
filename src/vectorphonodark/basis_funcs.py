import math
import numba


@numba.njit
def haar_n_to_lam_mu(n: int) -> tuple[int, int]:
    """
    Convert Haar wavelet order n to (lambda, mu) pair.

    Parameters
    ----------
    n : int
        The order of the Haar wavelet. Must be non-negative.

    Returns
    -------
    output : tuple[int, int]
        (lambda, mu) pair representing the scale and index.
    """
    assert n >= 0, "Haar wavelet: Order must be non-negative."
    if n == 0:
        return -1, -1
    lam = math.floor(math.log2(n))
    mu = n - (1 << lam)
    return lam, mu


@numba.njit
def haar_lam_mu_to_n(lam: int, mu: int) -> int:
    """
    Convert (lambda, mu) pair to Haar wavelet order n.

    Parameters
    ----------
    lam : int
        The scale of the Haar wavelet.
    mu : int
        The index of the Haar wavelet.

    Returns
    -------
    output : int
        The order of the Haar wavelet.
    """
    assert lam >= 0 and mu >= 0, "Haar wavelet: Scale and index must be non-negative."
    assert mu < (1 << lam), "Haar wavelet: Index mu must be less than 2^lambda."
    # n = 0 case, for completeness
    if lam == -1 and mu == -1:
        return 0
    n = (1 << lam) + mu
    return n


@numba.njit
def haar_support(n: int) -> list[float]:
    """
    Compute the support of the Haar wavelet function of order n.

    Parameters
    ----------
    n : int
        The order of the Haar wavelet.

    Returns
    -------
    output : list[float]
        The support region [x_min, x_mid, x_max] of the Haar wavelet function.
    """
    if n == 0:
        return [0., 1., 1.]
    
    lam, mu = haar_n_to_lam_mu(n)
    
    x_min = 2.**(-lam) * (mu + 0.)
    x_mid = 2.**(-lam) * (mu + 0.5)
    x_max = 2.**(-lam) * (mu + 1.)
    return [x_min, x_mid, x_max]


@numba.njit
def haar_support_log(n: int, eps: float) -> list[float]:
    """
    Compute the support of the log Haar wavelet function of order n.

    Parameters
    ----------
    n : int
        The order of the Haar wavelet.
    eps : float
        The lower bound of the integral.

    Returns
    -------
    output : list[float]
        The support region [x_min, x_mid, x_max] of the log Haar wavelet function.
    """
    assert eps > 0.0 and eps < 1.0, "Log Haar wavelet: eps must be in (0, 1)."
    
    if n == 0:
        return [eps, 1., 1.]

    lam, mu = haar_n_to_lam_mu(n)
    
    length = math.log(1. / eps)
    
    x_min = eps * math.exp(2.**(-lam) * (mu + 0.) * length)
    x_mid = eps * math.exp(2.**(-lam) * (mu + 0.5) * length)
    x_max = eps * math.exp(2.**(-lam) * (mu + 1.) * length)
    return [x_min, x_mid, x_max]


@numba.njit
def haar_value(n: int, dim: int=3) -> list[float]:
    """
    Compute the values of the Haar wavelet function.

    Parameters
    ----------
    n : int
        The order of the Haar wavelet.
    dim : int
        The dimension of the spherical haar wavelet (default is 3).

    Returns
    -------
    output : list[float]
        The values of the Haar wavelet [a_n, -b_n].
    """
    if n == 0:
        return [math.sqrt(dim)] * 2
    
    lam, mu = haar_n_to_lam_mu(n)
    
    x1 = 2.**(-lam) * (mu + 0.)
    x2 = 2.**(-lam) * (mu + 0.5)
    x3 = 2.**(-lam) * (mu + 1.)
    y1 = x1**dim
    y2 = x2**dim
    y3 = x3**dim
    a_n = math.sqrt(dim/(y3 - y1) * (y3 - y2)/(y2 - y1))
    b_n = math.sqrt(dim/(y3 - y1) * (y2 - y1)/(y3 - y2))
    return [a_n, -b_n]


@numba.njit
def haar_value_log(n: int, eps: float, p: int=2) -> list[float]:
    """
    Compute the values of the log Haar wavelet function.

    Parameters
    ----------
    n : int
        The order of the Haar wavelet.
    eps : float
        The lower bound of the integral.
    p : int
            The weight in orthornormality condition (default is 2).

    Returns
    -------
    output : list[float]
        The values of the log Haar wavelet [a_n, -b_n].
    """
    assert eps > 0.0 and eps < 1.0, "Log Haar wavelet: eps must be in (0, 1)."

    if n == 0:
        match p:
            case 0:
                return [math.sqrt(1./(1 - eps))] * 2
            case -1:
                return [math.sqrt(1./math.log(1./eps))] * 2
            case _:
                return [math.sqrt((p + 1.)/(1. - math.pow(eps, p + 1.)))] * 2

    lam, _ = haar_n_to_lam_mu(n)
            
    length = math.log(1. / eps)
    x1, _, _ = haar_support_log(n, eps)

    rho = math.exp(length * 2.**(-(lam + 1)))
    match p:
        case 0:
            a_n = math.sqrt(1. / (x1 * (rho - 1.) * (1. + 1./rho)))
            b_n = math.sqrt(1. / (x1 * rho * (rho - 1.) * (1. + rho)))
        case -1:
            a_n = b_n = math.sqrt(2.**lam / length)
        case _:
            b_n = math.sqrt(
                (p + 1.) / (
                    math.pow(x1, p + 1.) * (math.pow(rho, p + 1.) - 1.)
                    * (math.pow(rho, p + 1.) + 1.) * math.pow(rho, p + 1.)
                )
            )
            a_n = b_n * math.pow(rho, p + 1.)
    return [a_n, -b_n]


@numba.njit
def haar_func(n: int, x: float, dim: int=3) -> float:
    """
    Evaluate the Haar wavelet basis function at the given point.

    Parameters
    ----------
    n : int
        The order of the Haar wavelet.
    x : float
            The point at which to evaluate the Haar wavelet.
    dim : int
        The dimension of the spherical haar wavelet (default is 3).

    Returns
    -------
    output : float
        The value of the Haar wavelet basis function at point x.
    """

    if n == 0:
        if 0 <= x <= 1:
            return haar_value(n, dim)[0]
        else:
            return 0.0
    
    x_min, x_mid, x_max = haar_support(n)
    a_n, b_n = haar_value(n, dim)
    if x_min < x < x_mid:
        return a_n
    elif x_mid < x < x_max:
        return b_n
    elif x == x_min:
        return a_n if x == 0.0 else 0.5 * a_n
    elif x == x_mid:
        return 0.5 * (a_n + b_n)
    elif x == x_max:
        return b_n if x == 1.0 else 0.5 * b_n
    else:
        return 0.0
    

@numba.njit
def haar_func_log(n: int, x: float, eps: float, p: int=0) -> float:
    """
    Evaluate the log Haar wavelet basis function at the given point.

    Parameters
    ----------
    n : int
        The order of the Haar wavelet.
    x : float
        The point at which to evaluate the log Haar wavelet.
    eps : float
        The lower bound of the integral.
    p : int
        The weight in orthornormality condition (default is 0).

    Returns
    -------
    output : float
        The value of the log Haar wavelet basis function at point x.
    """
    assert eps > 0.0 and eps < 1.0, "Log Haar wavelet: eps must be in (0, 1)."

    if n == 0:
        if eps <= x <= 1.:
            return haar_value_log(n, eps, p)[0]
        else:
            return 0.0
    
    x_min, x_mid, x_max = haar_support_log(n, eps)
    a_n, b_n = haar_value_log(n, eps, p)
    if x_min < x < x_mid:
        return a_n
    elif x_mid < x < x_max:
        return b_n
    elif x == x_min:
        return a_n if x == eps else 0.5 * a_n
    elif x == x_mid:
        return 0.5 * (a_n + b_n)
    elif x == x_max:
        return b_n if x == 1.0 else 0.5 * b_n
    else:
        return 0.0