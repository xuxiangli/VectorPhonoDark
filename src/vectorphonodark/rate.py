from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt
import quaternionic
import vsdm

from . import constants as const
from .projection import VDF, BinnedMcalI, FormFactor

logger = logging.getLogger(__name__)

# Tolerance for treating a vector as null and a quaternion scalar part as +-1.
_TOL = 1e-12


class Rotation:
    """An active rotation of :math:`\\mathbb{R}^3`, stored as a unit quaternion.

    Build one with :meth:`from_axis_angle` or :meth:`identity`; the axis is in
    the crystal frame.

    All rotations are active and instances are immutable.
    """

    __slots__ = ("_q",)

    def __init__(self, quaternion: npt.ArrayLike) -> None:
        """Wrap a quaternion, scalar part first. Prefer :meth:`from_axis_angle`.

        Accepts anything array-like of length 4; the result is normalized.
        """
        q = quaternionic.array(np.asarray(quaternion, dtype=float))
        norm = float(np.linalg.norm(np.asarray(q)))
        if not np.isfinite(norm) or norm < _TOL:
            raise ValueError(
                f"Rotation requires a non-zero finite quaternion, got norm {norm!r}."
            )
        object.__setattr__(self, "_q", q.normalized)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(
            "Rotation is immutable; compose with @ or build a new one instead."
        )

    def __delattr__(self, name: str) -> None:
        raise AttributeError("Rotation is immutable.")

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def identity(cls) -> "Rotation":
        """The null rotation."""
        return cls([1.0, 0.0, 0.0, 0.0])

    @classmethod
    def from_axis_angle(cls, axis: npt.ArrayLike, angle: float) -> "Rotation":
        """Rotate by ``angle`` radians about ``axis``, right-handed."""
        axis_arr = np.asarray(axis, dtype=float)
        if axis_arr.shape != (3,):
            raise ValueError(f"axis must have shape (3,), got {axis_arr.shape}.")
        norm = float(np.linalg.norm(axis_arr))
        if norm < _TOL:
            raise ValueError("Rotation axis must not be the zero vector.")
        unit = axis_arr / norm
        half = 0.5 * float(angle)
        return cls([np.cos(half), *(unit * np.sin(half))])

    # ------------------------------------------------------------------
    # Algebra
    # ------------------------------------------------------------------

    def __matmul__(self, other: "Rotation") -> "Rotation":
        """Compose. ``(a @ b)`` applies ``b`` first, then ``a``."""
        if not isinstance(other, Rotation):
            return NotImplemented
        return Rotation(self._q * other._q)

    def inverse(self) -> "Rotation":
        """The rotation undoing this one."""
        return Rotation(self._q.conjugate())

    def __repr__(self) -> str:
        axis, angle = self._axis_angle()
        return (
            f"Rotation(axis=[{axis[0]:.6f}, {axis[1]:.6f}, {axis[2]:.6f}], "
            f"angle={angle:.6f} rad)"
        )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def apply(self, vec: npt.ArrayLike) -> np.ndarray:
        """Rotate a vector, or an ``(n, 3)`` stack of vectors."""
        arr = np.asarray(vec, dtype=float)
        if arr.shape[-1] != 3:
            raise ValueError(
                f"vec must have trailing dimension 3, got shape {arr.shape}."
            )
        return np.asarray(self._q.rotate(arr))

    def as_matrix(self) -> np.ndarray:
        """The ``(3, 3)`` rotation matrix ``M`` with ``M @ v == self.apply(v)``."""
        return np.asarray(self._q.to_rotation_matrix, dtype=float)

    def wigner_g(self, l_max: int, l_mod: int = 1) -> np.ndarray:
        """Real Wigner-G coefficients for this rotation.

        Parameters
        ----------
        l_max
            Maximum angular momentum. With ``l_mod == 2`` this must be even.
        l_mod
            1 for all ``ell``, 2 for even ``ell`` only.

        Returns
        -------
        np.ndarray
            1-D array flattened over ``(ell, m_v, m_q)`` in the order
            ``Rate.get_lmvmq_index`` expects.
        """
        if l_mod not in (1, 2):
            raise ValueError("l_mod must be either 1 (all ell) or 2 (even ell only).")
        if l_mod == 2 and l_max % 2 == 1:
            raise ValueError(
                f"l_max={l_max} is odd but l_mod=2 selects even ell only; "
                f"pass l_max={l_max - 1}."
            )
        computed = vsdm.WignerG(l_max, rotations=[self._q], lmod=l_mod)
        return np.asarray(computed.G_array, dtype=float)[0]

    def _axis_angle(self) -> tuple[np.ndarray, float]:
        """``(unit axis, angle in [0, pi])``; ``+z`` for the identity."""
        q = np.asarray(self._q, dtype=float)
        # Fix the sign of the double cover so the angle lands in [0, pi].
        if q[0] < 0.0:
            q = -q
        w = float(np.clip(q[0], -1.0, 1.0))
        sin_half = float(np.sqrt(max(0.0, 1.0 - w * w)))
        if sin_half < _TOL:
            return np.array([0.0, 0.0, 1.0]), 0.0
        return q[1:] / sin_half, 2.0 * float(np.arccos(w))

    @property
    def axis(self) -> np.ndarray:
        """Unit rotation axis; ``+z`` for the identity."""
        return self._axis_angle()[0]

    @property
    def angle(self) -> float:
        """Rotation angle in radians, in ``[0, pi]``."""
        return self._axis_angle()[1]


class Rate:
    """Contract a projected VDF and material form factor into the binned
    scattering rate.

    For each energy bin ``b`` and crystal rotation ``R``, :meth:`binned_rate`
    evaluates

        Gamma_b(R) = (v_max**2 / q_max)
                     * sum_{ell, m_v, m_q} G(ell, m_v, m_q)(R)
                                           * K_b(ell, m_v, m_q)

    where ``G`` are the real Wigner-G coefficients for ``R`` and ``K_b`` is the
    per-bin rate vector assembled by :meth:`_build_binned_mcalK` as
    ``K_b = v_max**3 * V_ell @ I_b_ell @ (F_b_ell).T`` from the VDF coefficients
    ``V``, the kinematic scattering matrix ``I``, and the material form-factor
    coefficients ``F``.

    Normalization
    -------------
    ``Gamma_b`` denotes the rate per target mass, divided by an overall factor
    (sigma_bar * rho_chi). It carries natural units of ``eV**-2``. To turn it
    into an observable at a DM-nucleon reference cross-section ``sigma_bar``
    and exposure ``M * T`` (target mass * time):

        N_events = sigma_bar[eV**-2] * rho_chi[eV**4] * (M * T) * sum_b Gamma_b

    or, solving for the projected reach at a fixed event count:

        sigma_bar[cm**2] = N_events / (KG_YR * RHO_DM * sum_b Gamma_b)
                           * INVEV_TO_CM**2

    Here ``RHO_DM`` is the local density in eV**4, ``KG_YR`` is one kg*yr, and
    ``INVEV_TO_CM`` is the ``hbar c`` factor converting eV**-1 to cm (all in
    :mod:`vectorphonodark.constants`).

    Parameters
    ----------
    physics_params : dict
        - fdm: ``(a, b)`` exponents of the DM form factor
          ``F_DM(q, v) = (q / q0_fdm)**a * (v / v0)**b`` (``v0 = 1``). For a
          spin-independent interaction, use ``a = -2 * f_med``
          (``f_med = 0`` heavy, ``2`` massless) and ``b = 0``.
        - q0_fdm: (optional) reference momentum of the DM form factor
          (default: Bohr momentum ``Q_BOHR``).
        - mass_dm: DM mass ``m_chi`` [eV].
        - mass_sm: SM target-particle mass [eV] (e.g. nucleus).

    numerics_params : dict
        - l_max: (optional) maximum angular momentum for the projection.
        - nv_max: (optional) maximum velocity wavelet index.
        - nq_max: (optional) maximum momentum wavelet index.

    Methods
    -------
    binned_rate(rotations) : dict
        The binned response ``Gamma_b`` for each energy bin at each crystal
        rotation. Takes a list of :class:`Rotation` (build them with
        :meth:`Rotation.from_axis_angle`). Returns a dict keyed by bin index
        whose values are vectors of ``Gamma_b``, one per rotation.
        See Normalization above for the units and the conversion to an event
        count or reach.
    """

    def __init__(
        self,
        physics_params: dict,
        numerics_params: dict,
        vdf: VDF,
        ff: FormFactor,
        mcalI: BinnedMcalI | None = None,
    ):

        if "l_max" not in numerics_params or numerics_params["l_max"] is None:
            self.l_max = min(vdf.l_max, ff.l_max)
        else:
            self.l_max = min(numerics_params["l_max"], vdf.l_max, ff.l_max)

        self.l_mod = max(vdf.l_mod, ff.l_mod)

        if "nv_max" not in numerics_params or numerics_params["nv_max"] is None:
            self.nv_max = vdf.n_max
        else:
            self.nv_max = min(numerics_params["nv_max"], vdf.n_max)

        if "nq_max" not in numerics_params or numerics_params["nq_max"] is None:
            self.nq_max = ff.n_max
        else:
            self.nq_max = min(numerics_params["nq_max"], ff.n_max)

        self.v_max = vdf.v_max
        self.q_max = ff.q_max

        self.fdm = physics_params["fdm"]
        self.q0_fdm = physics_params.get("q0_fdm", const.Q_BOHR)
        self.mass_dm = physics_params["mass_dm"]
        self.mass_sm = physics_params["mass_sm"]

        self.energy_threshold = ff.energy_threshold
        self.energy_bin_width = ff.energy_bin_width
        self.n_bins = ff.n_bins

        self.log_wavelet_q = ff.log_wavelet
        self.eps_q = ff.eps if self.log_wavelet_q else 1.0

        self.binned_mcalK = self._build_binned_mcalK(vdf, ff, mcalI=mcalI)

    def binned_rate(
        self, rotations: "list[Rotation] | Rotation"
    ) -> dict[int, np.ndarray]:
        """Binned rate at each crystal rotation.

        Parameters
        ----------
        rotations : list[Rotation] | Rotation
            The crystal rotations to evaluate at, built with
            :meth:`Rotation.from_axis_angle`. A single :class:`Rotation` is
            accepted as shorthand for a one-element list.

        Returns
        -------
        dict[int, np.ndarray]
            Bin index -> array of length ``len(rotations)``.
        """
        if isinstance(rotations, Rotation):
            rotations = [rotations]
        if not isinstance(rotations, (list, tuple)):
            raise TypeError(
                f"rotations must be a list of Rotation, got "
                f"{type(rotations).__name__}. Build one with "
                "Rotation.from_axis_angle(...)."
            )

        if not rotations:
            raise ValueError("binned_rate needs at least one rotation.")
        bad = next(
            (
                (i, type(r).__name__)
                for i, r in enumerate(rotations)
                if not isinstance(r, Rotation)
            ),
            None,
        )
        if bad is not None:
            raise TypeError(
                f"rotations[{bad[0]}] is a {bad[1]}, not a Rotation. Build "
                "them with Rotation.from_axis_angle(...)."
            )

        l_max = self.l_max
        lmvmq_max = self.get_lmvmq_index(l_max, l_max, l_max)
        G_array = np.stack([rot.wigner_g(l_max, self.l_mod) for rot in rotations])

        logger.debug(
            f"Rate: contracting {len(rotations)} rotation(s) "
            f"over l_max={l_max}, l_mod={self.l_mod}."
        )

        prefactor = self.v_max**2 / self.q_max
        return {
            idx_bin: prefactor
            * (G_array[:, 0 : lmvmq_max + 1] @ mcalK[0 : lmvmq_max + 1])
            for idx_bin, mcalK in self.binned_mcalK.items()
        }

    def get_lmvmq_index(self, ell, mv, mq):
        if self.l_mod == 2 and ell % 2 != 0:
            raise ValueError("l value does not satisfy l_mod=2 condition.")
        if self.l_mod == 2:
            return (
                ell * (4 * ell**2 - 6 * ell - 1) // 6
                + (ell + mv) * (2 * ell + 1)
                + (ell + mq)
            )
        else:
            return (
                ell * (2 * ell - 1) * (2 * ell + 1) // 3
                + (ell + mv) * (2 * ell + 1)
                + (ell + mq)
            )

    def _build_binned_mcalI(self):
        physics_params = {
            "fdm": self.fdm,
            "q0_fdm": self.q0_fdm,
            "energy_threshold": self.energy_threshold,
            "energy_bin_width": self.energy_bin_width,
            "mass_dm": self.mass_dm,
            "mass_sm": self.mass_sm,
        }
        numerics_params = {
            "n_bins": self.n_bins,
            "l_max": self.l_max,
            "l_mod": self.l_mod,
            "nv_max": self.nv_max,
            "nq_max": self.nq_max,
            "v_max": self.v_max,
            "q_max": self.q_max,
            "log_wavelet_q": self.log_wavelet_q,
            "eps_q": self.eps_q,
        }

        binned_mcalI = BinnedMcalI(
            physics_params=physics_params, numerics_params=numerics_params
        )
        binned_mcalI.project()
        return binned_mcalI

    def _check_binned_mcalI(self, mcalI):
        """Fence a precomputed kernel against this Rate's conventions.

        The numerical parameters are fixed by the VDF and material form factor, so the
        supplied BinnedMcalI must have been projected with the same conventions.
        Sizes (l_max, nv_max, nq_max, n_bins) may exceed what this Rate needs --
        the contraction slices them down -- but everything defining the basis
        functions, the energy bins, and the DM model must agree exactly.
        """
        mismatches = []

        for name in ("l_max", "nv_max", "nq_max", "n_bins"):
            if getattr(mcalI, name) < getattr(self, name):
                mismatches.append(
                    f"{name}={getattr(mcalI, name)}, need >= {getattr(self, name)}"
                )

        if mcalI.l_mod != self.l_mod:
            mismatches.append(f"l_mod={mcalI.l_mod}, need {self.l_mod}")
        if mcalI.log_wavelet_q != self.log_wavelet_q:
            mismatches.append(
                f"log_wavelet_q={mcalI.log_wavelet_q}, need {self.log_wavelet_q}"
            )
        if tuple(mcalI.fdm) != tuple(self.fdm):
            mismatches.append(f"fdm={tuple(mcalI.fdm)}, need {tuple(self.fdm)}")

        float_attrs = [
            "v_max",
            "q_max",
            "energy_threshold",
            "energy_bin_width",
            "mass_dm",
            "mass_sm",
            "q0_fdm",
        ]
        if self.log_wavelet_q:
            float_attrs.append("eps_q")
        for name in float_attrs:
            have = getattr(mcalI, name)
            need = getattr(self, name)
            if not np.isclose(have, need, rtol=1e-10, atol=0.0):
                mismatches.append(f"{name}={have}, need {need}")

        if mismatches:
            raise ValueError(
                "Supplied BinnedMcalI is incompatible with the VDF/material "
                "form factor of this Rate: " + "; ".join(mismatches)
            )

        missing = [b for b in range(self.n_bins) if b not in mcalI.mcalIs]
        if missing:
            raise ValueError(
                f"Supplied BinnedMcalI has no projected kernel for bin(s) "
                f"{missing}; call project() or import_hdf5() first."
            )

    def _build_binned_mcalK(self, vdf, ff, mcalI=None):
        """Contract VDF, kernel, and material form factor into rate vector K per bin.

        For each energy bin b and angular momentum l, contract

            K^b_l[m_v, m_q] = v_max^3 * (V_l @ I^b_l @ (F^b_l).T)[m_v, m_q]

        over the radial wavelet indices (n_v, n_q), where V_l are the VDF
        coefficients V_{l m_v n_v}, I^b_l the kernel I_{l n_v n_q} for bin b, and
        F^b_l the form-factor coefficients F_{l m_q n_q}. Each (2l+1)x(2l+1)
        block is flattened into the (l, m_v, m_q) slots of the returned vector
        (see get_lmvmq_index).

        V_l and the form-factor row indices do not depend on the bin, so they
        are precomputed once per l; only I^b and F^b change from bin to bin.

        Returns
        -------
        dict[int, np.ndarray]
            Bin index -> rate vector K^b, indexed by
            get_lmvmq_index(l, m_v, m_q).
        """
        logger.debug(
            f"Rate computation: Using {self.nv_max + 1} velocity wavelets and "
            f"{self.nq_max + 1} momentum wavelets."
        )

        # The kernel I is either supplied precomputed or built here.
        if mcalI is None:
            binned_mcalI = self._build_binned_mcalI()
        else:
            self._check_binned_mcalI(mcalI)
            binned_mcalI = mcalI

        # Bin-independent pieces, cached once per angular momentum l.
        vdf_block_by_ell = {}
        ff_rows_by_ell = {}
        for ell in range(0, self.l_max + 1, self.l_mod):
            vdf_rows = [vdf.get_lm_index(ell, mv) for mv in range(-ell, ell + 1)]
            ff_rows = [ff.fnlms[0].get_lm_index(ell, mq) for mq in range(-ell, ell + 1)]
            vdf_block_by_ell[ell] = vdf.f_lm_n[vdf_rows][:, : self.nv_max + 1]
            ff_rows_by_ell[ell] = ff_rows

        # K^b = v_max^3 * V_l @ I^b_l @ (F^b_l).T, assembled over l for each bin.
        mcalK_shape = (self.get_lmvmq_index(self.l_max, self.l_max, self.l_max) + 1,)
        binned_mcalK = {}
        for idx_bin in range(self.n_bins):
            mcalK = np.zeros(mcalK_shape, dtype=float)

            for ell in range(0, self.l_max + 1, self.l_mod):
                # vdf_block    (2l+1, nv_max+1)      V_{l m_v n_v}
                # kernel_block (nv_max+1, nq_max+1)  I^b_{l n_v n_q}
                # ff_block     (2l+1, nq_max+1)      F^b_{l m_q n_q}
                # mcalK_block  (2l+1, 2l+1)          K^b_{l m_v m_q}
                vdf_block = vdf_block_by_ell[ell]
                kernel_block = binned_mcalI.mcalIs[idx_bin].kernel[
                    ell // self.l_mod, : self.nv_max + 1, : self.nq_max + 1
                ]
                ff_rows = ff_rows_by_ell[ell]
                ff_block = ff.fnlms[idx_bin].f_lm_n[ff_rows][:, : self.nq_max + 1]
                mcalK_block = self.v_max**3 * vdf_block @ kernel_block @ ff_block.T

                target_indices = list(
                    range(
                        self.get_lmvmq_index(ell, -ell, -ell),
                        self.get_lmvmq_index(ell, ell, ell) + 1,
                    )
                )
                mcalK[target_indices] = mcalK_block.flatten()

            binned_mcalK[idx_bin] = mcalK

        return binned_mcalK
