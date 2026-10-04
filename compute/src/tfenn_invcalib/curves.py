"""Measured-curve handling: loading, station evaluation, LOO folds, synthetics.

Role: loads the 5 digitised measured candidate-sites Opalinus triaxial curves
(3 CU + 2 CD, 70 points each, eps_a in [0, 1.5] percent), fixes the frozen
driver's integration grid, builds the leave-one-curve-out folds with
per-fold-type (CU/CD) fields, and generates synthetic-truth curves through
the same frozen forward form for the coverage harness.

Station evaluation (module-level design decision): each digitised panel
carries its own uniform eps_a station grid (per-curve offsets and steps
differ; all lie inside [0, 1.5] percent). The frozen driver is integrated
once per curve on a common uniform axial grid (EPS_A_TOTAL = 0.015 strain,
N_STEPS_DEFAULT = 400 steps, d_eps11 = 3.75e-5 -- finer than the G0a probe's
5e-5), whose axial-strain history is exactly i * d_eps11 by construction of
the frozen CTC integrator. Model q AT the measured stations is then the
piecewise-linear interpolation of that frozen q history (jnp.interp), which
is differentiable in theta: the interpolation abscissae are
theta-independent constants and gradients flow through the q ordinates only.
The frozen integrator itself is untouched -- interpolation only samples its
output graph between integration nodes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

import jax.numpy as jnp

from . import mirror
from .runio import (CURVES_REL, PORE_PRESSURE_REL, sha256_bytes,
                    sha256_file)

EPS_A_TOTAL = 0.015      # strain; covers every measured station (max = 1.5 %)
N_STEPS_DEFAULT = 400    # d_eps11 = 3.75e-5 strain per step


def load_measured(root: str | Path) -> tuple[list[dict], str, str]:
    """Load the digitised measured panels; returns (panels, sha256, cu_note)."""
    path = Path(root) / CURVES_REL
    payload = json.loads(path.read_text(encoding="utf-8"))
    panels = payload["panels_digitized"]
    if len(panels) != 5:
        raise ValueError(f"expected 5 digitised panels, found {len(panels)}")
    return panels, sha256_file(path), payload.get("cu_handling_note", "")


@dataclass
class CurveSet:
    """Stacked constants for a set of curves sharing one integration grid."""
    keys: list[str]
    test_types: list[str]
    is_cu: np.ndarray          # (K,) bool
    theta_deg: np.ndarray      # (K,)
    sigma_r0: np.ndarray       # (K,)
    q_meas: np.ndarray         # (K, N)
    eps_stations: np.ndarray   # (K, N) strain units, per-curve station grids
    M_bond: jnp.ndarray        # (K, 6, 6)
    n_hat: jnp.ndarray         # (K, 3)
    p_conf: jnp.ndarray        # (K,)
    d_eps11: float
    n_steps: int
    eps_a_total: float
    data_sha256: str
    provenance: str
    p_conf_seq: jnp.ndarray | None = None   # (K, n_steps) effective-stress path

    @property
    def n_curves(self) -> int:
        return len(self.keys)

    @property
    def on_effective_stress_path(self) -> bool:
        return self.p_conf_seq is not None


def make_curveset(panels: list[dict], data_sha256: str,
                  n_steps: int = N_STEPS_DEFAULT,
                  provenance: str = "measured") -> CurveSet:
    n_pts = {len(p["eps_a_pct"]) for p in panels}
    if len(n_pts) != 1:
        raise ValueError(f"panels carry differing station counts: {n_pts}")
    eps_stations = np.array([p["eps_a_pct"] for p in panels], dtype=float) / 100.0
    if np.any(np.diff(eps_stations, axis=1) <= 0.0):
        raise ValueError("station grids must be strictly increasing")
    if float(np.max(eps_stations)) > EPS_A_TOTAL + 1e-12:
        raise ValueError("a measured station exceeds the integration span "
                         f"EPS_A_TOTAL={EPS_A_TOTAL}")
    M_list, n_list = [], []
    for p in panels:
        M, n_hat = mirror.path_constants(float(p["theta_deg"]))
        M_list.append(M)
        n_list.append(n_hat)
    return CurveSet(
        keys=[p["key"] for p in panels],
        test_types=[p["test_type"] for p in panels],
        is_cu=np.array([p["test_type"] == "CU" for p in panels]),
        theta_deg=np.array([float(p["theta_deg"]) for p in panels]),
        sigma_r0=np.array([float(p["sigma_r0_MPa"]) for p in panels]),
        q_meas=np.array([p["q_MPa"] for p in panels], dtype=float),
        eps_stations=eps_stations,
        M_bond=jnp.stack(M_list),
        n_hat=jnp.stack(n_list),
        p_conf=jnp.array([float(p["sigma_r0_MPa"]) for p in panels]),
        d_eps11=EPS_A_TOTAL / n_steps,
        n_steps=n_steps,
        eps_a_total=EPS_A_TOTAL,
        data_sha256=data_sha256,
        provenance=provenance,
    )


def load_measured_curveset(root: str | Path,
                           keys: list[str] | None = None,
                           n_steps: int = N_STEPS_DEFAULT,
                           effective_stress_path: bool = False) -> CurveSet:
    """Load the measured curves as a CurveSet (optionally a keyed subset).

    ``effective_stress_path`` attaches the measured undrained effective-stress
    path (see ``attach_effective_stress_path``); the default keeps the frozen
    constant-confinement drive the stored artifacts were produced under.
    """
    panels, sha, _ = load_measured(root)
    if keys is not None:
        by_key = {p["key"]: p for p in panels}
        missing = [k for k in keys if k not in by_key]
        if missing:
            raise KeyError(f"unknown curve keys: {missing}")
        panels = [by_key[k] for k in keys]
    cs = make_curveset(panels, sha, n_steps=n_steps, provenance="measured")
    if effective_stress_path:
        cs = attach_effective_stress_path(cs, root, panels)
    return cs


# ---------------------------------------------------------------------------
# Effective-stress path (undrained curves)
# ---------------------------------------------------------------------------


def load_pore_pressure(root: str | Path) -> tuple[dict, str]:
    """Load the digitised excess pore pressure panels, keyed by specimen id."""
    path = Path(root) / PORE_PRESSURE_REL
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_specimen = {p["specimen_id"]: p for p in payload["panels_digitized"]}
    return by_specimen, sha256_file(path)


def attach_effective_stress_path(cs: CurveSet, root: str | Path,
                                 panels: list[dict]) -> CurveSet:
    """Return a copy of ``cs`` carrying the measured effective-stress path.

    An undrained test is run at constant TOTAL radial stress while its pore
    pressure evolves, so its EFFECTIVE radial stress is
    sigma_r_eff(eps_a) = sigma_r0_eff - p_w(eps_a); the drained curves keep
    their constant confinement. The sequence holds the target at the END of
    each integration step, matching the frozen integrator's uniform axial
    grid. Curves are matched on specimen id, never on panel order.
    """
    pw_by_specimen, pw_sha = load_pore_pressure(root)
    by_key = {p["key"]: p for p in panels}
    eps_step_pct = (np.arange(1, cs.n_steps + 1) * cs.d_eps11) * 100.0
    seqs = []
    for k, is_cu, p0 in zip(cs.keys, cs.is_cu, np.asarray(cs.sigma_r0)):
        if not bool(is_cu):
            seqs.append(np.full(cs.n_steps, float(p0)))
            continue
        specimen = by_key[k].get("specimen_id")
        panel = pw_by_specimen.get(specimen)
        if panel is None:
            raise KeyError(f"no digitised pore pressure for specimen "
                           f"{specimen!r} (curve {k!r})")
        pw = np.interp(eps_step_pct, np.asarray(panel["eps_a_pct"]),
                       np.asarray(panel["pw_MPa"]))
        seqs.append(float(p0) - pw)
    seq = np.asarray(seqs)
    if np.any(seq <= 0.0):
        raise ValueError("the measured path drives an effective radial stress "
                         "to zero or below; the frozen driver has no tensile "
                         "or zero-confinement branch")
    return replace(
        cs,
        p_conf_seq=jnp.asarray(seq),
        data_sha256=f"{cs.data_sha256}+pw:{pw_sha}",
        provenance=f"{cs.provenance}+effective_stress_path",
    )


# ---------------------------------------------------------------------------
# Leave-one-curve-out folds
# ---------------------------------------------------------------------------


def build_loo_folds(panels: list[dict]) -> list[dict]:
    """One fold per curve: held-out curve + 4-curve calibration set.

    Per-fold-type reporting fields: held_out_test_type is the CU/CD label the
    LOO aggregation groups by.
    """
    folds = []
    for i, p in enumerate(panels):
        folds.append({
            "fold_id": i,
            "held_out_key": p["key"],
            "held_out_test_type": p["test_type"],
            "held_out_specimen_id": p.get("specimen_id"),
            "held_out_theta_deg": float(p["theta_deg"]),
            "held_out_sigma_r0_MPa": float(p["sigma_r0_MPa"]),
            "calib_keys": [pp["key"] for j, pp in enumerate(panels) if j != i],
        })
    return folds


# ---------------------------------------------------------------------------
# Synthetic-truth curves (coverage harness)
# ---------------------------------------------------------------------------


def synthetic_curveset(theta_truth: np.ndarray,
                       conditions: list[tuple[float, float]],
                       noise_sigma: float,
                       rng: np.random.Generator,
                       n_stations: int = 70,
                       n_steps: int = N_STEPS_DEFAULT,
                       form: int = 1,
                       p_conf_seqs: np.ndarray | None = None) -> CurveSet:
    """Generate synthetic curves through the same frozen forward form.

    Stations: a uniform 70-point grid over [0, 1.5] percent (the measured
    stations' span). Noise model (declared): additive iid Gaussian
    N(0, noise_sigma^2) on q at every station. Synthetic curves are labeled
    "CD" (no CU systematic-offset term: they are generated by the very same
    fixed-sigma_r forward form, so no CTC-protocol mismatch exists by
    construction). ``form`` selects the generating forward: 1 (default) the
    frozen form-I mirror on a 12-vector truth, 2 the fabric-scaled form-II
    mirror on a 14-vector truth.

    ``p_conf_seqs`` (n_conditions, n_steps) generates each curve along a
    prescribed radial-stress path instead of at a constant confinement, and
    marks the returned set as being on that path. It is how the coverage check
    is posed against the same inference problem the measured inversion poses
    when the undrained curves are driven along their measured effective-stress
    paths: generating at constant confinement there would test a model the
    inversion is not fitting. The curves stay offset-free either way, which is
    what the "CD" label records.
    """
    eps_sta = np.linspace(0.0, EPS_A_TOTAL, n_stations)
    d_eps11 = EPS_A_TOTAL / n_steps
    eps_grid = np.arange(n_steps + 1) * d_eps11
    on_path = p_conf_seqs is not None
    if on_path and len(p_conf_seqs) != len(conditions):
        raise ValueError(f"p_conf_seqs has {len(p_conf_seqs)} rows for "
                         f"{len(conditions)} conditions")
    if form == 1:
        fwd = mirror.forward_jit(n_steps)
        fwd_path = mirror.mirror_forward_path
    elif form == 2:
        fwd = mirror.forward_f2_jit(n_steps)
        fwd_path = mirror.mirror_forward_path_f2
    else:
        raise ValueError(f"unknown form {form!r}; options (1, 2)")
    theta_j = jnp.asarray(np.asarray(theta_truth, dtype=float))
    panels = []
    for i, (bed_deg, p_conf) in enumerate(conditions):
        M, n_hat = mirror.path_constants(float(bed_deg))
        out = (fwd_path(theta_j, M, n_hat, jnp.asarray(p_conf_seqs[i]),
                        d_eps11, n_steps=n_steps) if on_path
               else fwd(theta_j, M, n_hat, float(p_conf), d_eps11))
        q_hist = np.asarray(out["q"])
        if not np.all(np.isfinite(q_hist)):
            raise RuntimeError(f"synthetic truth run non-finite at "
                               f"({bed_deg} deg, {p_conf} MPa)")
        q_clean = np.interp(eps_sta, eps_grid, q_hist)
        q_noisy = q_clean + rng.normal(0.0, noise_sigma, size=n_stations)
        panels.append({
            "key": f"synthCD_{bed_deg:g}deg_{p_conf:g}MPa",
            "test_type": "CD",
            "specimen_id": "synthetic",
            "theta_deg": float(bed_deg),
            "sigma_r0_MPa": float(p_conf),
            "eps_a_pct": (eps_sta * 100.0).tolist(),
            "q_MPa": q_noisy.tolist(),
        })
    blob = json.dumps(panels, sort_keys=True).encode("utf-8")
    form_tag = "" if form == 1 else f", form={form}"
    path_tag = "+effective_stress_path" if on_path else ""
    cs = make_curveset(panels, sha256_bytes(blob), n_steps=n_steps,
                       provenance=f"synthetic(truth-generated, "
                                  f"noise_sigma={noise_sigma}{form_tag})"
                                  f"{path_tag}")
    if on_path:
        cs = replace(cs, p_conf_seq=jnp.asarray(np.asarray(p_conf_seqs)))
    return cs
