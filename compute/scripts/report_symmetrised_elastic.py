# -*- coding: utf-8 -*-
"""Report the nominal and current fitted operative elastic constants.

The driver builds a Voigt compliance with nu31 = nu13*E3/E1 placed
asymmetrically, scales it to Mandel, inverts, and symmetrises the result
(`mirror._ti_stiffness_local_j`). The engineering constants of that symmetrised
law are therefore not the parameters: this reads them back by inverting the
symmetrised stiffness and reading the compliance, exactly as the appendix
results table reports them. The current SR-09a maps supply the fitted
coordinates; SR-03 supplies the engineering profile intervals separately.

Usage (from compute/):
    python scripts/report_symmetrised_elastic.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SQRT2 = np.sqrt(2.0)


def sym_stiffness(E1, E3, nu12, nu13, G13):
    """The driver's construction, in numpy (mirror._ti_stiffness_local_j)."""
    nu31 = nu13 * E3 / E1
    G12 = E1 / (2.0 * (1.0 + nu12))
    S_v = np.array([
        [1.0 / E1, -nu12 / E1, -nu31 / E3, 0, 0, 0],
        [-nu12 / E1, 1.0 / E1, -nu31 / E3, 0, 0, 0],
        [-nu13 / E3, -nu13 / E3, 1.0 / E3, 0, 0, 0],
        [0, 0, 0, 1.0 / G13, 0, 0],
        [0, 0, 0, 0, 1.0 / G13, 0],
        [0, 0, 0, 0, 0, 1.0 / G12],
    ], dtype=float)
    d = np.array([1.0, 1.0, 1.0, 1.0 / SQRT2, 1.0 / SQRT2, 1.0 / SQRT2])
    S_m = S_v * np.outer(d, d)
    C = np.linalg.inv(S_m)
    return 0.5 * (C + C.T)


def engineering_constants(E1, E3, nu12, nu13, G13):
    """(E_p, E_o, nu_pp, nu_13, nu_31, G_op) of the symmetrised law."""
    C = sym_stiffness(E1, E3, nu12, nu13, G13)
    S_m = np.linalg.inv(C)
    d = np.array([1.0, 1.0, 1.0, 1.0 / SQRT2, 1.0 / SQRT2, 1.0 / SQRT2])
    S = S_m / np.outer(d, d)          # back to Voigt
    Ep = 1.0 / S[0, 0]
    Eo = 1.0 / S[2, 2]
    nu_pp = -S[0, 1] * Ep
    nu_13 = -S[2, 0] * Ep             # strain normal per unit in-plane strain
    nu_31 = -S[0, 2] * Eo             # its converse
    G_op = 1.0 / S[3, 3]
    return Ep, Eo, nu_pp, nu_13, nu_31, G_op


def row(label, E1, E3, nu12, nu13, G13):
    Ep, Eo, nu_pp, nu13s, nu31s, Gop = engineering_constants(E1, E3, nu12, nu13, G13)
    print(f"{label:<26s} parameters      {E1:8.0f} {E3:8.0f} "
          f"{nu12:7.3f} {nu13:7.3f}     ---   (G_op {G13:.0f})")
    print(f"{'':26s} symmetrised law {Ep:8.0f} {Eo:8.0f} "
          f"{nu_pp:7.3f} {nu13s:7.3f} {nu31s:7.3f}   (G_op {Gop:.0f})")


def main() -> int:
    nominal = json.loads(
        (ROOT / "results/data/parameter_sets/opalinus_method_a.json").read_text(
            encoding="utf-8"))

    flat = {e["name"]: e["value"] for e in nominal["params"]}

    row("Nominal",
        flat["E_p"], flat["E_o"], flat["nu_pp"], flat["nu_op"], flat["G_op"])

    for form in (1, 2):
        data = json.loads((ROOT.parent / "selfrev-r2/evidence" /
                           f"SR-09a_map_f{form}.json").read_text(encoding="utf-8"))
        E_p, E_o, G_op, nu_pp, nu_op = data["reference"]["theta"][:5]
        row(f"Form {'I' * form} theta*",
            E_p, E_o, nu_pp, nu_op, G_op)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
