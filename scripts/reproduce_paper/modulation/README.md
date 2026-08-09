# Daily modulation benchmark — Al₂O₃, light dark photon

Reproduces the draft's daily-modulation benchmark: the modulation figure
`fig:modulation_al2o3_20meV`, its VectorPhonoDark/PhonoDark comparison table,
and the wall-clock entries of `tab:timing`. The rate is computed with
VectorPhonoDark and, as the reference, with stock
[PhonoDark](https://github.com/tanner-trickle/PhonoDark); see the script
docstrings for what each step computes.

## Pipeline

| Step | Script | Environment | Output (in `output/modulation/`) |
|------|--------|-------------|----------------------------------|
| 1a | `1a_vpd_project.py` | `vectorphonodark` | `vdf.hdf5`, `ff.hdf5` |
| 1b | `1b_vpd_rate.py` | `vectorphonodark` | `vpd_binned.json` |
| 2 | `2_pd.py` | PhonoDark env | `pd_binned.json`, `pd_timing.json`, `pd_runs/` |
| 3 | `3_plot.py` | `vectorphonodark` | `fig_modulation_al2o3_20meV.pdf`, `modulation_comparison_20meV.json` |

Step 1a projects the VDF and material form factor once and stores them to
HDF5; step 1b reads those projections and contracts the rate. Steps {1a, 1b}
and 2 are independent; step 3 needs both 1b and 2. `2_pd.py` locates PhonoDark
through three environment variables: `PD_DIR` (the PhonoDark checkout),
`PD_PYTHON` (the python of its environment), and `PD_MPIRUN` (only needed for
`--nproc > 1`).

## Setting up PhonoDark

PhonoDark is not installable from PyPI; clone it and create its environment:

```shell
conda create -y -n phonodark -c conda-forge python=3.11 "numpy=1.26" \
    scipy "phonopy=2.41.0" h5py mpi4py openmpi sympy matplotlib-base
```

PhonoDark's own `src/phonopy_funcs.py` uses phonopy's pre-2.41.1 getter-style
API (`get_number_of_atoms()` etc.), so any phonopy version up to 2.41.0 works
unmodified.

## Reproducing the timing table

```shell
NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python -u scripts/reproduce_paper/modulation/1a_vpd_project.py
NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python -u scripts/reproduce_paper/modulation/1b_vpd_rate.py
PD_DIR=... PD_PYTHON=... \
    python -u scripts/reproduce_paper/modulation/2_pd.py --nproc 1 --threads 1
```
