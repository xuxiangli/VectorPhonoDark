# VectorPhonoDark

A Python package for computing sub-GeV dark matter (DM) detection rates via DM–phonon scattering in crystals, using the **vector space / spherical wavelet** method.

The calculation decomposes into three independent projections — the DM velocity distribution function (VDF), the crystal form factor (FF), and the kinematic kernel — each expanded in a spherical Haar wavelet basis.  The scattering rate is then obtained as an algebraic inner product, making it straightforward to scan over DM mass, mediator model, and crystal target without repeating expensive phonon calculations.

---

## Physics overview

The differential DM–phonon scattering rate per unit detector mass can be written schematically as (see arXiv:1910.08092 and 2502.17547)

$$R({\cal R}) \propto \rho_\chi \sum_{s, \ell, m_v, m_q} G^{(\ell)}_{m_v m_q}({\cal R}) \sum_{n_v, n_q} \langle g_{\chi} \vert n_v \ell m_v \rangle \cdot \mathcal{I}^{(\ell)}_{n_v n_q}(F_{\text{DM}}^2 ; \Delta E_s) \cdot \langle n_q \ell m_q \vert f_{g \to s}^2 \rangle$$

where:

| Symbol | Description |
|--------|-------------|
| $\langle g_{\chi} \vert n_v \ell m_v \rangle$ | Projected coefficients of the DM velocity distribution in lab frame $g_{\chi}(\boldsymbol{v})$ |
| $\langle n_q \ell m_q \vert f_{i \to s}^2 \rangle$ | Projected coefficients of the form factor $\lvert \langle s \vert\mathcal{F}_T(\boldsymbol{q})\rvert i \rangle \rvert^2$ |
| $\mathcal{I}^{(\ell)}_{n_v n_q}(F_{\text{DM}}^2 ; \Delta E_s)$ | Kinematic kernel encoding energy–momentum conservation for effective mediator factor $F_{\text{DM}}^2$ and final state $\langle s \vert$ |
| $G^{(\ell)}_{m_v m_q}({\cal R})$ | Wigner D matrix for rotation ${\cal R}$ |

Projected coefficients of DM VDF and crystal form factor are computed once and stored, so the final rate evaluation reduces to analytical evaluation of the kinematic kernelk and a fast tensor contraction.

---

## Installation

### 1. Create environment

```shell
conda create -n vectorphonodark python=3.13
conda activate vectorphonodark
conda install -c conda-forge phonopy=2.44.0
pip install vsdm==0.4.4
```

### 2. Install the package in editable mode

```shell
pip install -e .
```

### 3. Compile Cython extensions (recommended)

The rate kernel (`analytic_cy.pyx`) is compiled with Cython for significant performance gains.  Requires a C compiler (e.g. `gcc` or `clang`) and the `cython` package.

```shell
pip install cython
python setup.py build_ext --inplace
```

> **Pure Python fallback:** If Cython compilation is skipped or fails, the package automatically falls back to the pure-Python implementation (`analytic.py`).  This is slower but produces identical results.

---

## Project structure

```
VectorPhonoDark/
├── src/vectorphonodark/        # Core package
│   ├── constants.py            # Physical constants in natural units (eV)
│   ├── physics.py              # Phonon data loading, form factor evaluation
│   ├── projection.py           # VDF, FormFactor, McalI projection classes
│   ├── rate.py                 # Rate assembly and reach calculation
│   ├── utility.py              # Mesh generation, wavelet projection helpers
│   ├── basis_funcs.py          # Haar wavelet basis functions
│   └── analytic.py / _cy.pyx  # Kinematic kernel (Python / Cython)
│
├── inputs/
│   ├── material/               # Per-material phonon data and atomic properties
│   │   ├── Al2O3/              # Aluminium oxide (sapphire)
│   │   ├── GaAs/               # Gallium arsenide
│   │   └── LiF/                # Lithium fluoride
│   ├── physics_model/          # DM–SM interaction models
│   │   ├── dark_photon.py      # Dark photon (kinetically mixed)
│   │   ├── dark_photon_born.py # Dark photon in Born approximation
│   │   ├── hadrophilic.py      # Hadrophilic scalar mediator
│   │   ├── scalar_e.py         # Scalar coupled to electrons
│   │   └── U1BmL.py            # U(1)_{B−L} gauge boson
│   └── numerics/
│       └── standard.py         # Default numerical precision settings
│
├── scripts/                    # Ready-to-run calculation scripts
│   ├── get_q_cutoff.py         # Query the kinematic q_cutoff for a material
│   ├── proj_vdf.py             # Project VDF (Monte Carlo grid)
│   ├── proj_vdf_quad.py        # Project VDF (Gauss-Legendre quadrature)
│   ├── proj_form_factor.py     # Project crystal form factor
│   ├── proj_delta_func.py      # Project kinematic kernel (BinnedMcalI)
│   └── rate_calc.py            # Compute rate and projected reach
│
├── notebooks/                  # Jupyter notebooks for verification and diagnostics
└── output/                     # HDF5 output files (created at runtime)
```

---

## Workflow

The calculation proceeds in three independent steps.  Each step writes results to an HDF5 file in `output/`, which is consumed by the next step.

### Step 1 — Project the VDF

```shell
python scripts/proj_vdf.py
```

Projects the DM velocity distribution $g_{\chi}(\boldsymbol{v})$ (Standard Halo Model by default) onto the spherical Haar wavelet basis.  The user-defined VDF function is passed via `physics_params["vdf"]`.

For a higher-precision alternative using adaptive Gauss-Legendre quadrature:

```shell
python scripts/proj_vdf_quad.py
```

### Step 2 — Project the form factor

```shell
python scripts/proj_form_factor.py
```

Loads phonon dispersion data (via `phonopy`) for the chosen crystal and projects the DM–phonon form factor $\lvert \langle s \vert\mathcal{F}_T(\boldsymbol{q})\rvert i \rangle \rvert^2$ onto the wavelet basis.  Results are stored as one HDF5 group per $q_\text{max}$ value.

### Step 3 — Compute the rate and projected reach

```shell
python scripts/rate_calc.py
```

Loads the pre-computed VDF and form factor projections, constructs Wigner-D rotation matrices to account for Earth's rotation or crystal's rotation, and evaluates the total rate via an inner product.  Outputs the projected cross-section sensitivity (in cm²) for each DM mass.

### Alternative: kinematic kernel approach

```shell
python scripts/proj_delta_func.py
```

Projects the energy-binned kinematic kernel $\mathcal{I}^{(\ell)}_{n_v n_q}(F_{\text{DM}}^2 ; \Delta E_s)$ directly.  The result can be imported to the Step 3 to reduce computational time.

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
- `FORCE_SETS` — interatomic force constants from DFT
- `BORN` — Born effective charges and dielectric tensor
- `<Material>.py` — atomic properties (masses, charge numbers) used in the form factor

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
def my_vdf(v_xyz, **kwargs) -> float:
    # return the (unnormalized) phase-space density at velocity v_xyz
    ...

physics_params = {
    "vdf": my_vdf,
    "vdf_params": {...},   # keyword arguments forwarded to my_vdf except v_xyz
    "model": "my_model",
}
```

### Key numerical parameters

| Parameter | Location | Description |
|-----------|----------|-------------|
| `l_max` | `numerics_params` | Maximum angular momentum $\ell$; higher → better angular resolution |
| `n_max` | `numerics_params` | Maximum radial wavelet index; higher → better radial resolution |
| `n_grid` | `numerics_params` | $(n_r, n_θ, n_φ)$ Monte Carlo grid for VDF/FF projection |
| `q_max` | `numerics_params` | Upper momentum cutoff in eV; set by $2 m_χ (v_\text{esc} + v_E)$ |
| `log_wavelet` | `numerics_params` | Use logarithmic radial basis (recommended when $F_{\text{DM}} \propto q^{-n}$) |

---

## Output format

All projection results are stored in HDF5 files under `output/`.  Each dataset has the shape `(n_lm, n_max + 1)` and carries metadata attributes (`l_max`, `n_max`, `v_max` / `q_max`, etc.) sufficient to reconstruct the full projection without re-running the calculation.

HDF5 group naming convention:

| File (in recommended names) | Group structure |
|------|----------------|
| `vdf.hdf5` | `<model>/<grid_label>` |
| `<material>_<model>.hdf5` | `linear(or log)/<q_max> eV` |
| `mcalI.hdf5` | `<mass>MeV/<fdm>` |
