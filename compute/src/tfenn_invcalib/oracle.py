"""Best-attainable-fit oracles: all-curve, per-fold four-curve, fixed-delta.

Role: brute-force multi-start fits of the SAME likelihood with no holdout on a
chosen curve set. Two oracles are needed and they answer different questions:

- the ALL-CURVE (five-curve) oracle bounds the fit attainable within the
  frozen constitutive family at an attained optimum. It is a BOUND on the
  attainable fit and is labelled as one: a multi-start that another optimiser
  beats is evidence the bound is not tight, which is reported rather than
  smoothed.
- the FOUR-CURVE oracle per leave-one-curve-out fold sees exactly the data and
  the objective the fold's ensemble saw, so the distance between it and the
  fold's posterior-predictive is an optimiser gap and nothing else. Without it
  the middle share of a gap decomposition silently carries the cost of
  predicting a curve the fit never saw.

Scoring rule (the repaired asymmetry). The oracle's own in-sample per-curve
RMSE is reported under the likelihood's profiled offset, because that is the
predicted mean the likelihood optimised. It is reported a second time under
the held-out rule (offset fixed at the budget midpoint), because that is the
rule the nominal null and the posterior-predictive are scored under, and a
decomposition that mixes the two hands the oracle an offset the other lanes
are denied. Both numbers and their difference are recorded; consumers of the
decomposition take the fixed-delta one.

The G0a failed-start rate is read from the gate report file at run time (never
hardcoded) and echoed, and each oracle re-measures its own multi-start
failed-start rate. It is a statistic about the gradient-sanity experiment, not
a property of this oracle, and is recorded as an input rather than used to
qualify any share of the decomposition.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import ensemble
from .curves import build_loo_folds, load_measured, load_measured_curveset
from .likelihood import (cu_offset_status, load_likelihood_config,
                         make_likelihood, rmse_per_curve)
from .runio import G0A_REPORT_REL, effort_accounting
from .theta_space import load_nominal_theta, space_for_form

PROFILED_SCORING_LABEL = ("in-sample rule: CU offset analytically profiled then "
                          "clipped to the published budget")
FIXED_DELTA_SCORING_LABEL = ("held-out rule: CU offset fixed at the budget "
                             "midpoint mu_d, the rule the nominal null and the "
                             "posterior-predictive are scored under")
NO_OFFSET_SCORING_LABEL = ("one rule: the undrained curves are driven along "
                           "the measured effective-stress path, so there is no "
                           "offset to profile or to hold and every lane is "
                           "fitted and scored the same way")


def read_g0a_failed_start_rate(root: str | Path) -> dict:
    """Read the recorded G0a failed-start rate from the gate report file."""
    path = Path(root) / G0A_REPORT_REL
    report = json.loads(path.read_text(encoding="utf-8"))
    fs = report["failed_start"]
    return {
        "g0a_failed_start_rate": float(fs["rate"]),
        "g0a_failed_start_count": int(fs["count"]),
        "g0a_n_combos_total": int(fs["n_combos_total"]),
        "source": str(path),
        "source_commit": report.get("source_commit"),
    }


def score_both_rules(theta: np.ndarray, like) -> dict:
    """Per-curve RMSE under BOTH offset rules, with their difference.

    The difference is the size of the advantage the profiled rule gives a fit
    over the held-out rule on the same parameter vector; it is zero on every
    CD curve by construction (no offset term) and is the quantity a mixed-rule
    decomposition was absorbing on the CU curves.
    """
    keys = like.curveset.keys
    on_path = like.on_effective_stress_path
    prof = rmse_per_curve(theta, like, profile_delta=True)
    # On the measured effective-stress path there is no offset to profile or
    # hold, so the two rules coincide and their difference is zero by
    # construction on every curve, not just the drained ones.
    fixed = prof if on_path else rmse_per_curve(theta, like,
                                                profile_delta=False)
    return {
        "profiled_MPa": dict(zip(keys, prof.tolist())),
        "fixed_delta_MPa": dict(zip(keys, fixed.tolist())),
        "difference_fixed_minus_profiled_MPa": dict(
            zip(keys, (fixed - prof).tolist())),
        "mean_profiled_MPa": float(np.mean(prof)),
        "mean_fixed_delta_MPa": float(np.mean(fixed)),
        "profiled_rule": (NO_OFFSET_SCORING_LABEL if on_path
                          else PROFILED_SCORING_LABEL),
        "fixed_delta_rule": (NO_OFFSET_SCORING_LABEL if on_path
                             else FIXED_DELTA_SCORING_LABEL),
    }


def _multistart_on(like, space, root, n_starts: int, n_iters: int, seed: int,
                   lr: float, start_scheme: str, fixed_delta_objective: bool):
    """Shared multi-start body: draw admissible starts, run, pick the best."""
    nominal = space.extend_nominal(load_nominal_theta(root))
    rng = np.random.default_rng(seed)
    starts = space.sample_starts(rng, n_starts, start_scheme, nominal)
    objective = like.nll_z_fixed_delta if fixed_delta_objective else like.nll_z
    records = ensemble.run_multistart(objective, starts, n_iters, lr=lr,
                                      form=space.form)
    summary = ensemble.multistart_summary(records)
    ok = [r for r in records if not r.failed]
    if not ok:
        raise RuntimeError("oracle: all starts failed; no attainable-fit bound")
    best = min(ok, key=lambda r: r.final_nll)
    return records, summary, np.asarray(best.theta_best)


def run_oracle(root: str | Path, n_starts: int, n_iters: int, seed: int,
               lr: float = 0.03, start_scheme: str = "box_loguniform",
               form: int = 1, keys: list[str] | None = None,
               fixed_delta_objective: bool = False,
               effective_stress_path: bool = False) -> dict:
    """Multi-start best-attainable-fit oracle; returns records + bound inputs.

    ``form`` selects the constitutive form (1 = frozen form I, the exact
    pre-existing code path; 2 = the 14-parameter fabric-scaled form II);
    curves, likelihood protocol, and bound bookkeeping are form-independent.
    ``keys`` restricts the fitted curve set (default: all 5 measured curves).
    ``fixed_delta_objective`` fits under the held-out offset rule instead of
    the paper's profiled likelihood, which is the sensitivity lane that bounds
    how much of the irreducible share comes from fitting under one offset rule
    and scoring under another; the headline oracle keeps the declared
    likelihood.
    """
    root = Path(root)
    t0 = time.time()
    space = space_for_form(form)
    curveset = load_measured_curveset(
        root, keys=keys, effective_stress_path=effective_stress_path)
    cfg = load_likelihood_config(root)
    like = make_likelihood(curveset, cfg, form=form)
    if effective_stress_path and fixed_delta_objective:
        raise ValueError(
            "--fixed-delta-objective is a sensitivity lane about fitting "
            "under one offset rule and scoring under another; on the measured "
            "effective-stress path there is one rule and the lane has no "
            "meaning")
    records, summary, theta_best = _multistart_on(
        like, space, root, n_starts, n_iters, seed, lr, start_scheme,
        fixed_delta_objective)
    g0a = read_g0a_failed_start_rate(root)
    rmse_in_sample = rmse_per_curve(theta_best, like, profile_delta=True)
    scoring = score_both_rules(theta_best, like)

    return {
        "role": ("best-attainable-fit oracle (no holdout) on the fitted curve "
                 "set; a BOUND on the attainable fit, not a certificate that "
                 "the bound is tight"),
        "objective": ("NLL under the held-out offset rule (delta fixed at mu_d)"
                      if fixed_delta_objective else
                      "same NLL as the ensemble inversion (profiled CU offset)"),
        "fixed_delta_objective": bool(fixed_delta_objective),
        "curve_keys": list(curveset.keys),
        "n_starts": n_starts,
        "n_iters": n_iters,
        "lr": lr,
        "start_scheme": start_scheme,
        "initial_obj": summary["best_initial_nll"],
        "final_obj": summary["best_final_nll"],
        "best_theta": theta_best.tolist(),
        "oracle_rmse_q_per_curve_MPa": dict(zip(curveset.keys, rmse_in_sample.tolist())),
        "oracle_rmse_q_mean_MPa": float(np.mean(rmse_in_sample)),
        "oracle_rmse_scoring_rule": (NO_OFFSET_SCORING_LABEL
                                    if like.on_effective_stress_path
                                    else PROFILED_SCORING_LABEL),
        "scoring_both_rules": scoring,
        "cu_offset_at_best": (None if like.on_effective_stress_path
                              else cu_offset_status(theta_best, like)),
        "oracle_failed_start_rate": summary["failed_start_rate"],
        "n_converged": summary["n_converged"],
        **g0a,
        "effort": effort_accounting(
            time.time() - t0, n_curves=curveset.n_curves,
            ad_row_iterations=n_starts * n_iters,
            extra_curve_forwards=3 * curveset.n_curves),
        "members": [r.as_dict() for r in records],
    }


def run_fold_oracles(root: str | Path, n_starts: int, n_iters: int, seed: int,
                     lr: float = 0.03, start_scheme: str = "box_loguniform",
                     form: int = 1,
                     effective_stress_path: bool = False) -> dict:
    """One four-curve oracle per leave-one-curve-out fold.

    Each fold's oracle is fitted on the fold's four CALIBRATION curves under
    the paper's own likelihood -- the same data and the same objective the
    fold's ensemble had -- and then scored on the fold's HELD-OUT curve. The
    held-out score is recorded under both offset rules; the decomposition
    consumes the fixed-delta one, so the four-curve oracle and the two lanes
    it is differenced against are on one scoring footing.

    The seed is offset per fold exactly as the leave-one-curve-out lane offsets
    it (seed + 100 + fold_id), so the two lanes draw the same starts for the
    same fold and the optimiser gap between them is not a difference of draws.
    """
    root = Path(root)
    t0 = time.time()
    space = space_for_form(form)
    cfg = load_likelihood_config(root)
    panels, _sha, _note = load_measured(root)
    folds = build_loo_folds(panels)
    g0a = read_g0a_failed_start_rate(root)

    fold_results = []
    for fold in folds:
        f0 = time.time()
        cs_cal = load_measured_curveset(
            root, keys=fold["calib_keys"],
            effective_stress_path=effective_stress_path)
        like_cal = make_likelihood(cs_cal, cfg, form=form)
        records, summary, theta_best = _multistart_on(
            like_cal, space, root, n_starts, n_iters,
            seed + 100 + fold["fold_id"], lr, start_scheme, False)

        cs_out = load_measured_curveset(
            root, keys=[fold["held_out_key"]],
            effective_stress_path=effective_stress_path)
        like_out = make_likelihood(cs_out, cfg, form=form)
        held = score_both_rules(theta_best, like_out)
        fold_results.append({
            **fold,
            "n_starts": n_starts,
            "n_iters": n_iters,
            "calib_final_obj": summary["best_final_nll"],
            "calib_initial_obj": summary["best_initial_nll"],
            "oracle4_theta": theta_best.tolist(),
            "oracle4_calib_rmse_q_per_curve_MPa": dict(zip(
                cs_cal.keys,
                rmse_per_curve(theta_best, like_cal, profile_delta=True).tolist())),
            "oracle4_heldout_rmse_MPa": {
                "profiled": held["profiled_MPa"][fold["held_out_key"]],
                "fixed_delta": held["fixed_delta_MPa"][fold["held_out_key"]],
                "difference_fixed_minus_profiled":
                    held["difference_fixed_minus_profiled_MPa"][fold["held_out_key"]],
            },
            "cu_offset_at_oracle4_on_calib": (
                None if like_cal.on_effective_stress_path
                else cu_offset_status(theta_best, like_cal)),
            "cu_offset_at_oracle4_on_heldout": (
                None if like_out.on_effective_stress_path
                else cu_offset_status(theta_best, like_out)),
            "multistart_summary": summary,
            "members": [r.as_dict() for r in records],
            "fold_runtime_seconds": round(time.time() - f0, 1),
        })

    return {
        "role": ("per-fold four-curve best-attainable-fit oracle: the best fit "
                 "attainable on exactly the data the fold's ensemble saw"),
        "objective": ("same NLL as the ensemble inversion "
                      + ("(measured effective-stress path, no offset)"
                         if effective_stress_path else "(profiled CU offset)")),
        "heldout_scoring_rule": (NO_OFFSET_SCORING_LABEL if effective_stress_path
                                 else FIXED_DELTA_SCORING_LABEL),
        "seed_convention": "per fold: seed + 100 + fold_id (matches the LOO lane)",
        "n_starts": n_starts,
        "n_iters": n_iters,
        "lr": lr,
        "start_scheme": start_scheme,
        **g0a,
        "effort": effort_accounting(
            time.time() - t0, n_curves=4,
            ad_row_iterations=len(folds) * n_starts * n_iters,
            extra_curve_forwards=len(folds) * 8,
            note="four calibration curves per fold; held-out scoring counted "
                 "in extra_curve_forwards"),
        "folds": fold_results,
    }
