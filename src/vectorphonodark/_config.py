"""Loading of material / model / numerics input files.

.. warning::
   The input files are *executed as Python* (``import_file`` runs
   ``exec_module`` on the path). Loading one is equivalent to running an
   arbitrary Python script, so only pass paths you trust. This is the
   PhonoDark input convention; a data-only (TOML/YAML) path for the numerics
   block is planned for a later release.
"""

from importlib import util
from types import ModuleType


def import_file(full_name: str, path: str) -> ModuleType:
    """
    Import a module from a given file path.

    Parameters
    ----------
    full_name : str
        The full name to assign to the module.
    path : str
        The file path to the module.

    Returns
    -------
    module
        The imported module.
    """

    spec = util.spec_from_file_location(full_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load a Python module from path {path!r}.")
    mod = util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    return mod
