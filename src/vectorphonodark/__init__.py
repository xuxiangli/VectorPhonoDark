"""VectorPhonoDark: dark matter-phonon scattering rates via the
vector space integration method.

The calculation factorizes into three independent projections -- the DM
velocity distribution (`VDF`), the crystal form factor (`FormFactor`), and
the kinematic kernel (`McalI` / `BinnedMcalI`) -- which are contracted into
a scattering rate by `Rate`. `kin_matrix` exposes the analytic kernel
underneath `McalI` -- the matrix of I_l(nv, nq) integrals -- from the
compiled backend when available, with a numba fallback.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("vectorphonodark")
except PackageNotFoundError:  # running from a checkout without installation
    __version__ = "1.0.1+uninstalled"

from . import constants
from .projection import (
    VDF,
    BinnedMcalI,
    FormFactor,
    McalI,
    kin_matrix,
)
from .rate import Rate, Rotation

__all__ = [
    "BinnedMcalI",
    "FormFactor",
    "McalI",
    "Rate",
    "Rotation",
    "VDF",
    "__version__",
    "constants",
    "kin_matrix",
]
