from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        name="vectorphonodark.analytic_cy",
        sources=["src/vectorphonodark/analytic_cy.pyx"],
        include_dirs=[numpy.get_include()],
        # extra_compile_args=["-O3", "-fopenmp"],
        # extra_link_args=["-fopenmp"],
    ),
]

setup(
    name="VectorPhonoDark",
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
    zip_safe=False,
)