# VectorPhonoDark

A Python package for computing sub-GeV dark matter (DM) detection rates via DM–phonon scattering in crystals, using the vector space integration method.

---

## Physics overview

The differential rate for DM–phonon scattering in a crystal target of $N_T$ primitive cells is (arXiv:1910.08092, 2502.17547)

$$\frac{\mathrm{d}R}{\mathrm{d}\omega} = N_T \frac{\rho_\chi}{m_\chi} \frac{\bar\sigma_0}{4\pi \mu_\chi^2} \int \mathrm{d}^3 \boldsymbol{v} \mathrm{d}^3 \boldsymbol{q} g_\chi(\boldsymbol{v}, t) F_\text{DM}^2(q) f_S^2(\boldsymbol{q}, \omega) \delta\Bigl(\omega + \tfrac{q^2}{2 m_\chi} - \boldsymbol{q}\cdot\boldsymbol{v}\Bigr),$$

where $\rho_\chi$ is the local DM energy density, $g_\chi(\boldsymbol{v}, t)$ the lab-frame DM velocity distribution (VDF), $\bar\sigma_0$ a reference cross section, $F_\text{DM}(q)$ the mediator form factor, $\mu_\chi$ the reduced mass of the DM and the SM particle it couples to, and $f_S^2(\boldsymbol{q}, \omega)$ the material form factor — for single-phonon excitations, a sum over phonon modes built from DFT data.  The energy deposit $\omega$ is divided into bins of width $\Delta\omega$ centered at $\omega_b$, and the binned rate is $R_b = \int_b \mathrm{d}\omega (\mathrm{d}R/\mathrm{d}\omega)$.

The vector space method expands the VDF and the binned material form factor $f_{S,b}^2(\boldsymbol{q}) \equiv \int_b \mathrm{d}\omega f_S^2(\boldsymbol{q}, \omega)$ in a basis of spherical Haar wavelets times real spherical harmonics, $\vert n \ell m \rangle$, on the balls $v \le v_\text{max}$ and $q \le q_\text{max}$.  The binned rate at crystal rotation $\mathcal{R}$ then becomes a matrix contraction,

$$R_b(\mathcal{R}) = N_T \rho_\chi \bar\sigma_0 \frac{v_\text{max}^2}{q_\text{max}} \sum_{\ell, m_v, m_q} G^{(\ell)}_{m_v m_q}(\mathcal{R}) \mathcal{K}^{(\ell)}_{m_v m_q}(\omega_b),$$

with the partial rate matrix

$$\mathcal{K}^{(\ell)}_{m_v m_q}(\omega_b) = v_\text{max}^3 \sum_{n_v, n_q} \langle g_\chi \vert n_v \ell m_v \rangle \mathcal{I}^{(\ell)}_{n_v n_q}(\omega_b) \langle n_q \ell m_q \vert f_{S,b}^2 \rangle,$$

where:

| Symbol | Description |
|--------|-------------|
| $\langle g_\chi \vert n_v \ell m_v \rangle$ | VDF projection — computed once per halo model and stored |
| $\langle n_q \ell m_q \vert f_{S,b}^2 \rangle$ | Binned material form factor projection — computed once per material (phonon data via `phonopy`) and stored |
| $\mathcal{I}^{(\ell)}_{n_v n_q}(\omega_b)$ | Kinematic scattering matrix — analytic form for Haar wavelets (arXiv:2310.01483), evaluated for each DM mass and mediator; involves no material data |
| $G^{(\ell)}_{m_v m_q}(\mathcal{R})$ | Real Wigner G-matrix for the crystal rotation $\mathcal{R}$ |
| $\mathcal{K}^{(\ell)}_{m_v m_q}(\omega_b)$ | Partial rate matrix (arXiv:2310.01480) |

The two projections are computed once and stored. A scan over DM mass, mediator model, target material, and crystal rotation (time of day) costs one analytic kinematic matrix evaluation per mass point and a fast tensor contraction.

---

## Installation

### 1. Create environment

```shell
conda create -n vectorphonodark python=3.14
conda activate vectorphonodark
```

### 2. Install the package

```shell
pip install -e .
```

This also builds the Cython rate kernel (`analytic_cy.pyx`) automatically. A C compiler (e.g. `gcc` or `clang`) is
required for the compiled kernel.

> **Pure Python fallback:** If no C compiler is available, the install still
> succeeds and the package automatically falls back to the pure-Python
> implementation (`analytic.py`). This is slower but produces identical results.

To rebuild the extension in place during development:

```shell
python setup.py build_ext --inplace
```

---

## Project structure

```
VectorPhonoDark/
├── src/vectorphonodark/            # Core package
│   ├── constants.py                # Physical constants in natural units (eV)
│   ├── basis_funcs.py              # Spherical Haar wavelet basis functions
│   ├── mesh.py                     # Integration meshes, wavelet boundaries, projection integrator
│   ├── fnlm.py                     # Fnlm / BinnedFnlm coefficient containers (HDF5 IO)
│   ├── phonopy_funcs.py            # Phonon data loading, Debye-Waller tensor, form-factor evaluation
│   ├── projection.py               # VDF, FormFactor, McalI, BinnedMcalI projection classes
│   ├── rate.py                     # Rotation and Rate: contraction into binned rates
│   ├── analytic.py                 # Kinematic kernel, pure-Python / numba backend
│   └── analytic_cy.pyx             # Kinematic kernel, Cython backend (mirror of analytic.py)
│
├── inputs/
│   ├── material/                   # Per-material phonon data and atomic properties
│   │   ├── Al2O3/                  # Aluminium oxide (sapphire)
│   │   └── GaAs/                   # Gallium arsenide
│   ├── physics_model/              # DM–SM interaction models
│   │   ├── dark_photon.py          # Dark photon (kinetically mixed)
│   │   ├── dark_photon_born.py     # Dark photon in Born approximation
│   │   ├── hadrophilic.py          # Hadrophilic scalar mediator
│   │   ├── scalar_e.py             # Scalar coupled to electrons
│   │   └── U1BmL.py                # U(1)_{B−L} gauge boson
│   └── numerics/
│       └── standard.py             # Default numerical settings (Debye-Waller k-mesh)
│
├── scripts/
│   ├── examples/                   # Example scripts
│   │   ├── quickstart.py               # The whole calculation in memory, coarse and fast
│   │   ├── project_vdf.py              # Project the SHM velocity distribution
│   │   ├── project_material_ff.py      # Project the material form factor
│   │   ├── project_kernel.py           # The kinematic kernel as a standalone object
│   │   └── compute_rate.py             # Contract into rates and a projected reach
│   └── reproduce_paper/            # Regenerate the manuscript figures and tables
│       ├── binwidth/               #   rate convergence vs energy bin width
│       ├── convergence/            #   appendix convergence tables
│       ├── modulation/             #   daily-modulation benchmark vs PhonoDark
│       ├── qmax/                   #   universal q_max = q_cut projection study
│       └── wavelet/                #   linear vs logarithmic basis convergence
│
├── projections/                    # Shipped universal material form factor projections
│
└── output/                         # HDF5 / CSV / figure outputs (created at runtime)
```

The `reproduce_paper/` studies are long-running and print progress to the
terminal. To also keep a log file, run them unbuffered and capture both
streams:

```shell
python -u <script_name>.py 2>&1 | tee log/<log_name>.log
```

---

## Examples

### Quick start

```shell
python scripts/examples/quickstart.py
```

runs the whole pipeline — VDF, material form factor, rate, projected reach — in memory
at deliberately coarse resolution, in under a minute. It is a smoke test of
the installation and a readable map of the workflow, **not** a converged
calculation.

### VDF projection

```shell
python scripts/examples/project_vdf.py
```

Projects the DM velocity distribution $g_{\chi}(\boldsymbol{v})$ (Standard Halo Model) onto the spherical Haar wavelet basis.  The user-defined VDF function is passed via `physics_params["vdf"]`.

### Material form factor projection

```shell
python scripts/examples/project_material_ff.py
```

Loads phonon dispersion data (via `phonopy`) for each crystal and projects the material form factor $\lvert \langle s \vert\mathcal{F}_T(\boldsymbol{q})\rvert i \rangle \rvert^2$ onto the wavelet basis.  The projection extends up to the Debye–Waller cutoff $q_\text{cut}$, above which the material form factor is exponentially suppressed, so the single stored projection serves every DM mass.

The script covers both materials (GaAs, Al₂O₃) and both couplings (hadrophilic, dark photon) — it generates the shipped files in `projections/` (see [Shipped reference projections](#shipped-reference-projections)).

### Rate calculation

```shell
python scripts/examples/compute_rate.py
```

Loads the stored VDF and material form factor projections, builds the Wigner-G coefficients for the chosen crystal rotations, and evaluates the binned rate via an inner product.  Outputs the projected cross-section sensitivity (in cm²) for each DM mass.  See [Normalization and units](#normalization-and-units) for what the returned numbers mean and how the cm² reach is formed.

### Kinematic scattering matrix calculation

```shell
python scripts/examples/project_kernel.py
```

This script demonstrates handling the kinematic kernel as an object of its own — calculating a `BinnedMcalI` and storing it to HDF5 — for reference.

Although not required, `Rate` also supports receiving a stored `BinnedMcalI` as input:

```python
from vectorphonodark import BinnedMcalI, Rate

kernel = BinnedMcalI().import_hdf5("output/mcalI.hdf5", "1.0MeV/(-4, 0)")
rate = Rate(physics_params, numerics_params, vdf=vdf, ff=form_factor, mcalI=kernel)
```

---

## Shipped reference projections

`projections/` ships one universal material form factor projection per
material and coupling — GaAs and Al₂O₃, each with the hadrophilic and dark
photon couplings — with logarithmic wavelets up to the Debye–Waller cutoff
$q_\text{cut}$, $N_q = 512$, $\ell_\text{max} = 5$, a $(2048, 25, 25)$ grid,
and 1 meV energy bins.  Because the projection extends to $q_\text{cut}$,
a single file serves every DM mass:

```python
from vectorphonodark.projection import FormFactor

form_factor = FormFactor().import_hdf5("projections/GaAs_hadrophilic.hdf5", "q_cut")
```

`scripts/examples/project_material_ff.py` regenerates these files from the DFT
inputs in `inputs/material/` (into `output/`; copy them here to refresh the
shipped versions).

---

## Configuring a calculation

### Choosing a material

Edit the `input_params` dictionary in the relevant script to point to the desired material directory:

```python
input_params = {
    "material_input": str(project_root / "inputs" / "material" / "GaAs" / "GaAs.py"),
    ...
}
```

Each material directory contains:
- `POSCAR` — crystal structure (VASP format)
- `FORCE_SETS` — DFT forces for displaced supercells (phonopy format)
- `BORN` — Born effective charges and dielectric tensor
- `<Material>.py` — atomic properties (masses, charge numbers) used in the material form factor

### Choosing a physics model

Edit `physics_model_input` to select the DM–SM interaction:

```python
input_params = {
    ...
    "physics_model_input": str(project_root / "inputs" / "physics_model" / "dark_photon.py"),
}
```

### Defining a custom VDF

Supply a Numba-compiled function via `physics_params["vdf"]`:

```python
import numba
import numpy as np


@numba.njit
def my_vdf(v_xyz, v_0, v_esc) -> float:
    # return the (unnormalized) phase-space density at lab-frame velocity v_xyz
    ...


physics_params = {
    "vdf": my_vdf,
    # keyword arguments forwarded to my_vdf alongside v_xyz
    "vdf_params": {"v_0": ..., "v_esc": ...},
    "model": "my_model",
}
```

### Key numerical parameters

| Parameter | Location | Description |
|-----------|----------|-------------|
| `l_max` | `numerics_params` | Maximum angular momentum $\ell$; higher → better angular resolution |
| `n_max` | `numerics_params` | Maximum radial wavelet index; higher → better radial resolution |
| `n_grid` | `numerics_params` | $(n_r, n_θ, n_φ)$ quadrature grid for the projection (midpoint rule; $n_r$ must be a power of two greater than `n_max`) |
| `q_max` | `numerics_params` | Upper momentum cutoff in eV; set by $2 m_χ (v_\text{esc} + v_E)$ |
| `log_wavelet` | `numerics_params` | Use logarithmic radial basis (recommended when $F_{\text{DM}} \propto q^{-n}$) |

---

## Crystal rotation and daily modulation

### The convention

A `Rotation` is an **active rotation acting on the crystal**, with the wind
held fixed — equivalently, the *inverse* rotation acting on every other vector
(the wind, the spin axis) with the crystal held fixed.  Daily modulation is
then immediate: over `t` hours the Earth, and the crystal bolted to it, rotates
by `+2πt/24` about the spin axis.

### Usage

```python
import numpy as np

from vectorphonodark import Rotation
from vectorphonodark import constants as const

# The Earth's spin axis in the t = 0 crystal frame
spin_axis = np.array([0.0, -np.sin(const.THETA_E), np.cos(const.THETA_E)])

# 24 hourly crystal rotations over one day
rotations = [
    Rotation.from_axis_angle(spin_axis, 2.0 * np.pi * t / 24.0) for t in range(24)
]

binned = rate.binned_rate(rotations)  # {bin index: array of len(rotations)}
```

`binned_rate` accepts a `Rotation` or a list of them (a single one is shorthand
for a one-element list); the Wigner-G coefficients are built from each rotation
internally, at the `Rate`'s own `l_max` and `l_mod` (`rotation.wigner_g(l_max,
l_mod)` exposes one row).

---

## Normalization and units

This section connects the master formula of the [Physics overview](#physics-overview) to the code: what `Rate.binned_rate` actually returns, and how it becomes a cross-section in cm².

`Rate.binned_rate(rotations)` returns a dictionary `{bin_index: array}`, where each array holds the contracted response $\Gamma_b(\mathcal{R})$ for energy bin $b$, one entry per crystal rotation $\mathcal{R}$:

$$\Gamma_b(\mathcal{R}) = \frac{v_\text{max}^2}{q_\text{max}} \sum_{\ell, m_v, m_q} G^{(\ell)}_{m_v m_q}(\mathcal{R}) K^b_{\ell m_v m_q}, \qquad K^b_\ell = v_\text{max}^3 V_\ell \mathcal{I}^b_\ell (F^b_\ell)^{\!\top}.$$

$\Gamma_b$ carries natural units of $\text{eV}^{-2}$. To turn the response into an observable, multiply by the local density $\rho_\chi$, the exposure $M T$, and the reference DM–nucleon cross-section $\bar\sigma_n$:

$$N_\text{events} = \bar\sigma_n [\text{eV}^{-2}] \times \rho_\chi [\text{eV}^4] \times (M T) \times \sum_b \Gamma_b.$$

---

## Symbol reference

A map between the paper's notation (see the [Physics overview](#physics-overview)) and the code, for the quantities that keep their paper symbols as identifiers.

| Code | Paper symbol | Meaning |
|------|--------------|---------|
| `VDF.f_lm_n`, `FormFactor.f_lm_n`, `Fnlm.f_lm_n` | $\langle f \mid n \ell m \rangle$ | wavelet–harmonic projection coefficients of the VDF $g_\chi$ and the material form factor $f_S^2$ |
| `McalI.kernel`, `BinnedMcalI.mcalIs[b]` | $\mathcal{I}^{(\ell)}_{n_v n_q}(\omega_b)$ | kinematic kernel for energy bin $b$ |
| `Rate.binned_mcalK` | $\mathcal{K}^{(\ell)}_{m_v m_q}(\omega_b)$ | partial rate matrix $v_\text{max}^3 V_\ell \mathcal{I}^b_\ell (F^b_\ell)^{\!\top}$ |
| `Rotation.wigner_g` | $G^{(\ell)}_{m_v m_q}(\mathcal{R})$ | real Wigner-G coefficients for rotation $\mathcal{R}$ |

Shared indices: $\ell$ is the angular momentum, $m_v, m_q$ its projections on the velocity/momentum sides, $n_v, n_q$ the radial wavelet indices, and $(\lambda, \mu)$ the Haar level and offset within an $\ell$-channel.

---

## Citation

If you use `VectorPhonoDark` in your work, please cite the accompanying paper:

> Xu-Xiang Li and Zhengkang Zhang,
> *Logarithmic Wavelets for Dark Matter–Phonon Scattering*,
> [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX).

```bibtex
@article{Li:2026xxx,
    author = "Li, Xu-Xiang and Zhang, Zhengkang",
    title = "{Logarithmic Wavelets for Dark Matter--Phonon Scattering}",
    eprint = "XXXX.XXXXX",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    year = "2026"
}
```
