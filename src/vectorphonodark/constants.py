"""
Physical constants used throughout the rest of the program.

Natural (eV) units are used unless otherwise specified.
"""

import numpy as np
from scipy import special

# electron mass
M_ELEC = 511 * 10**3

# proton mass
M_NUCL = 0.938 * 10**9

# Conversion factors ###

THZ_TO_EV = 6.58212 * 10 ** (-4)
PI = np.pi
ANG_TO_INVEV = 5.06773 * 10 ** (-4)
FM_TO_INVEV = 5.06773 * 10 ** (-9)
AMU_TO_EV = 9.31494 * 10**8
EV_TO_AMU = 1.07354 * 10**-9
G_OVER_CM3_TO_EV_OVER_A3 = 5.6095 * 10**8
KG_TO_EV = 5.6095 * 10**35
SECOND_TO_ONE_OVER_EV = 1.51976 * 10**15
INVEV_TO_CM = 1.973 * 10**-5
INVANG_TO_EV = 1.973 * 10**3
INVCMET_TO_EV = 1.973 * 10 ** (-5)
GEV_TO_EV = 10**9
INVEV_TO_INVGEV = 10**9
KMET_PER_SEC_TO_NONE = 3.33564 * 10 ** (-6)

#####

# fine structure constant
ALPHA_EM = 1.0 / 137.036

# Reference momentum transfer
Q_BOHR = M_ELEC * ALPHA_EM

# dark matter density
RHO_DM = 0.4 * GEV_TO_EV * INVCMET_TO_EV**3  # 0.4

# 1 kg x year exposure
KG_YR = 2.69 * 10**58

# angle (radians) of the North pole relative to the Earth velocity
THETA_E = 42 * (PI / 180)

# Maxwell Boltzmann velocity distribution parameters
V0 = 230 * KMET_PER_SEC_TO_NONE  # 230
VE = 240 * KMET_PER_SEC_TO_NONE  # 240
VESC = 600 * KMET_PER_SEC_TO_NONE  # 600

N0 = (
    PI ** (3 / 2)
    * V0**2
    * (
        V0 * special.erf(VESC / V0)
        - (2 / np.sqrt(PI)) * VESC * np.exp(-(VESC**2) / V0**2)
    )
)
C1 = PI * V0**2 / N0
C2 = np.exp(-((VESC / V0) ** 2))
