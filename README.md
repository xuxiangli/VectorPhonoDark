# VectorPhonoDark
Computes the dark matter-phonon scattering rate using partial rates expansion.

Set up the environment by (anaconda as example)

```shell
conda create -n vectorphonodark python=3.13
conda install -c conda-forge phonopy=2.44.0
pip install vsdm=0.4.4
```

In `scripts/` you can find four scripts for

- projection of form factor function
- projection of velocity distribution function
- projection of the delta function for energy-momentum conservation
- calculation of the reaction rate by combining results from projections above

Before running the scripts, please run the following command in terminal such that python knows the directory of the package:

```shell
pip install -e .
```

Cython is being used for performance. You must have a C compiler and the cython package installed. To compile the Cython code, please run

```shell
python setup.py build_ext --inplace
```

If you want to use pure python mode, please delete the file `analytic_cy.pyx`.