import math
import numpy as np
import numba


@numba.njit
def haar_support(n) -> list[float]:
    """
    Compute the support of the Haar wavelet basis functions of order n.

    Args:
        n: 
            int: The order of the Haar wavelet.
            tuple[int, int]: (lambda, mu) pair representing the scale and index.

    Returns:
        list[float]: The support indices of the Haar wavelet basis functions.
    """
    assert n >= 0, "Haar wavelet: Order must be non-negative."
    if n == 0:
        lam, mu = 0, -1
    else:
        lam = math.floor(math.log2(n))
        mu = n - (1 << lam)
    
    # n = 0 case
    if mu == -1:
        return [0., 1.]
    
    x_min = 2.**(-lam) * (mu + 0.)
    x_mid = 2.**(-lam) * (mu + 0.5)
    x_max = 2.**(-lam) * (mu + 1.)
    return [x_min, x_mid, x_max]


@numba.njit
def haar_value(n, dim: int=3) -> list[float]:
    """
    Compute the values of the Haar wavelet basis functions at their support points.

    Args:
        n: 
            int: The order of the Haar wavelet.
            tuple[int, int]: (lambda, mu) pair representing the scale and index.
        dim: 
            int: The dimension of the spherical haar wavelet (default is 3).

    Returns:
        list[float]: The values of the Haar wavelet basis functions at their support points.
    """
    assert n >= 0, "Haar wavelet: Order must be non-negative."
    if n == 0:
        lam, mu = 0, -1
    else:
        lam = math.floor(math.log2(n))
        mu = n - (1 << lam)

    # n = 0 case
    if mu == -1:
        return [math.sqrt(dim)]
    
    x1 = 2.**(-lam) * (mu + 0.)
    x2 = 2.**(-lam) * (mu + 0.5)
    x3 = 2.**(-lam) * (mu + 1.)
    y1 = x1**dim
    y2 = x2**dim
    y3 = x3**dim
    a_n = math.sqrt(dim/(y3 - y1) * (y3 - y2)/(y2 - y1))
    b_n = math.sqrt(dim/(y3 - y1) * (y2 - y1)/(y3 - y2))
    return [a_n, -b_n]


def haar_func(n, x: float, dim: int=3) -> float:
    """
    Evaluate the Haar wavelet basis function at a given point.

    Args:
        n: 
            int: The order of the Haar wavelet.
            tuple[int, int]: (lambda, mu) pair representing the scale and index.
        x: 
            float: The point at which to evaluate the Haar wavelet.
        dim: 
            int: The dimension of the spherical haar wavelet (default is 3).

    Returns:
        float: The value of the Haar wavelet basis function at point x.
    """
    if isinstance(n, tuple):
        assert len(n) == 2, "Haar wavelet: Input tuple must be of length 2."
        lam, mu = n
        assert lam >= 0 and mu >= 0, "Haar wavelet: Scale and index must be non-negative."
    elif isinstance(n, int):
        assert n >= 0, "Haar wavelet: Order must be non-negative."
        if n == 0:
            lam, mu = 0, -1
        else:
            lam = math.floor(math.log2(n))
            mu = n - (1 << lam)
    else:
        raise TypeError("Haar wavelet: Input must be an integer or a tuple of two integers.")

    # n = 0 case
    if mu == -1:
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