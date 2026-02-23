"""
Compute the maximum momentum transfer q_cutoff for a given material.

q_cutoff is set by the Brillouin zone boundary scaled by a dimensionless factor,
and serves as the upper kinematic limit for the form-factor projection.

Usage:
    python get_q_cutoff.py
"""

from pathlib import Path

from vectorphonodark.physics import get_q_max


def main():
    project_root = Path(__file__).resolve().parent.parent
    material_input = str(project_root / "inputs" / "material" / "Al2O3" / "Al2O3.py")

    factor = 4.0  # scale factor applied to the Brillouin-zone boundary momentum

    q_cutoff = get_q_max(material_input=material_input, factor=factor)

    print(f"q_cutoff = {q_cutoff} eV")


if __name__ == "__main__":
    main()
