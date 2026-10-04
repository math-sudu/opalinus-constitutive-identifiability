"""Identity of the measured forward calculation used by the current results."""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = (
    "compute/results/data/nordas_curves/digitized_nordas_curves.json",
    "compute/results/data/nordas_curves/digitized_nordas_pore_pressure.json",
    "compute/results/data/noise_budget.json",
)
DEFINITIONS = (
    "compute/src/tfenn_invcalib/curves.py",
    "compute/src/tfenn_invcalib/mirror.py",
    "compute/src/tfenn_invcalib/likelihood.py",
    "compute/src/tfenn_invcalib/theta_space.py",
    "compute/src/tfenn_invcalib/numerics.py",
    "compute/src/tfenn_esubi/data_gen/esubi_driver.py",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(form, steps, producers=()):
    return {"form": form, "n_steps": steps, "effective_stress_path": True,
            "files": {p: digest(ROOT / p) for p in (*DATA, *DEFINITIONS, *producers)}}


def require_current(record):
    for relative, expected in record["files"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"Changed computation input: {relative}")
