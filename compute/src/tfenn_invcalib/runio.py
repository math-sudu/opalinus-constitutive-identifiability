"""Run-output utilities for the tfenn_invcalib inversion stack.

Role: shared plumbing for every E3 entry script and the E2 smoke script --
deterministic seeds, sha256 data fingerprints, environment-version capture,
config echo (all hyperparameters + seed + data fingerprint inside every output
JSON), and numpy-safe JSON serialization. No physics lives here.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DEFAULT_SEED = 20260706

# Project-relative paths of the frozen input artifacts (read-only).
CURVES_REL = "results/data/nordas_curves/digitized_nordas_curves.json"
PORE_PRESSURE_REL = ("results/data/nordas_curves/digitized_nordas_pore_pressure.json")
NOMINAL_REL = "results/data/parameter_sets/opalinus_method_a.json"
NOISE_BUDGET_REL = "results/data/noise_budget.json"
G0A_REPORT_REL = "results/data/gradient_sanity/d4_gradient_sanity_report.json"
CLAIM_CONTRACT_REL = "results/state_refs/p2_claim_contract.json"
MODEL_SCOPE_REL = "results/state_refs/model_scope.json"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def env_versions() -> dict:
    import jax
    import optax
    import scipy

    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "jax": jax.__version__,
        "scipy": scipy.__version__,
        "optax": optax.__version__,
    }


def data_fingerprints(root: str | Path) -> dict:
    """sha256 of every frozen input artifact this run reads."""
    root = Path(root)
    return {
        "digitized_nordas_curves.json": sha256_file(root / CURVES_REL),
        "digitized_nordas_pore_pressure.json": sha256_file(root / PORE_PRESSURE_REL),
        "opalinus_method_a.json": sha256_file(root / NOMINAL_REL),
        "noise_budget.json": sha256_file(root / NOISE_BUDGET_REL),
        "d4_gradient_sanity_report.json": sha256_file(root / G0A_REPORT_REL),
        "p2_claim_contract.json": sha256_file(root / CLAIM_CONTRACT_REL),
        "model_scope.json": sha256_file(root / MODEL_SCOPE_REL),
    }


def config_echo(seed: int, root: str | Path, hyperparams: dict,
                argv: list[str] | None = None) -> dict:
    """Standard config-echo block embedded in every output JSON."""
    return {
        "seed": seed,
        "generated_at": utc_now_iso(),
        "argv": list(sys.argv) if argv is None else argv,
        "env": env_versions(),
        "model_scope": json.loads((Path(root) / MODEL_SCOPE_REL).read_text(encoding="utf-8")),
        "data_fingerprints_sha256": data_fingerprints(root),
        "hyperparams": to_jsonable(hyperparams),
    }


def effort_accounting(wall_clock_seconds: float,
                      n_curves: int,
                      ad_row_iterations: int = 0,
                      derivative_free_nfev: int = 0,
                      extra_curve_forwards: int = 0,
                      note: str | None = None) -> dict:
    """Per-lane effort record: wall clock + a forward-evaluation count.

    ``forward_evaluations`` is the declared headline count -- multi-start
    row-iterations times curves, plus the derivative-free ``nfev`` -- so that
    the cross-lane effort table is discharged from run artifacts instead of
    from prose. The two terms it adds are NOT the same unit, and the record
    keeps the components so a reader can recompute on either footing:

    - one AD row-iteration is one value-and-grad of the whole curve set, which
      costs several forward evaluations, not one;
    - one derivative-free ``nfev`` is one residual-vector evaluation, which is
      already one forward over EVERY curve in the fitted set.

    ``curve_forward_evaluations`` therefore puts both lanes on the curve-
    forward footing (``nfev`` also multiplied by the curve count) and is the
    number to use when comparing the two optimisers' cost directly;
    ``forward_evaluations`` is kept because it is the declared quantity.
    ``extra_curve_forwards`` counts curve forwards outside either loop
    (scoring passes, finite-difference probes, member re-evaluation).
    """
    ad_row_iterations = int(ad_row_iterations)
    derivative_free_nfev = int(derivative_free_nfev)
    n_curves = int(n_curves)
    forward_evaluations = ad_row_iterations * n_curves + derivative_free_nfev
    curve_forwards = (ad_row_iterations * n_curves
                      + derivative_free_nfev * n_curves
                      + int(extra_curve_forwards))
    out = {
        "wall_clock_seconds": round(float(wall_clock_seconds), 3),
        "n_curves_per_evaluation": n_curves,
        "ad_row_iterations": ad_row_iterations,
        "derivative_free_nfev": derivative_free_nfev,
        "extra_curve_forwards": int(extra_curve_forwards),
        "forward_evaluations": forward_evaluations,
        "forward_evaluations_definition": (
            "multi-start row-iterations times curves, plus the derivative-free nfev"),
        "curve_forward_evaluations": curve_forwards,
        "curve_forward_evaluations_definition": (
            "every term on the curve-forward footing: AD row-iterations times "
            "curves, derivative-free nfev times curves, plus extra_curve_forwards"),
        "unit_note": (
            "one AD row-iteration is a value-and-grad over the curve set (several "
            "forward-equivalents); one derivative-free nfev is already one forward "
            "over every fitted curve"),
    }
    if note:
        out["note"] = note
    return out


def to_jsonable(obj):
    """Recursively convert numpy scalars/arrays and paths to plain Python."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json(path: str | Path, obj: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(obj), indent=2), encoding="utf-8")
    return path
