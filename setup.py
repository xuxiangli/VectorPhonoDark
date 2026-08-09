"""Build script for the optional Cython extension (analytic_cy).

The package is fully functional without the compiled extension:
projection.py falls back to the pure-Python (numba) kernel in analytic.py.
The extension build is therefore best-effort -- a missing Cython/numpy or a
missing C compiler downgrades the install to pure Python instead of failing.
"""

from setuptools import setup
from setuptools.command.build_ext import build_ext


def get_ext_modules():
    try:
        import numpy
        from Cython.Build import cythonize
        from setuptools import Extension
    except ImportError as exc:
        print(
            f"WARNING: {exc}; skipping the Cython kernel, "
            "the pure-Python fallback (analytic.py) will be used."
        )
        return []

    extensions = [
        Extension(
            name="vectorphonodark.analytic_cy",
            sources=["src/vectorphonodark/analytic_cy.pyx"],
            include_dirs=[numpy.get_include()],
        ),
    ]
    try:
        return cythonize(extensions, compiler_directives={"language_level": "3"})
    except Exception as exc:  # e.g. scipy .pxd headers missing
        print(
            f"WARNING: cythonize failed ({exc}); skipping the Cython kernel, "
            "the pure-Python fallback (analytic.py) will be used."
        )
        return []


class OptionalBuildExt(build_ext):
    """Skip (rather than abort) the install if the C toolchain is unavailable."""

    def run(self):
        try:
            super().run()
        except Exception as exc:
            self._warn(exc)

    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception as exc:
            self._warn(exc)

    @staticmethod
    def _warn(exc):
        print(
            f"WARNING: building the Cython kernel failed ({exc}); "
            "installing without it, the pure-Python fallback (analytic.py) "
            "will be used."
        )


setup(
    ext_modules=get_ext_modules(),
    cmdclass={"build_ext": OptionalBuildExt},
    zip_safe=False,
)
