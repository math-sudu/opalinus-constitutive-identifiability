"""The attained optimum: one definition, shared by every downstream instrument.

theta* for a given form and curve set is the parameter vector attaining the
LOWEST NLL over BOTH optimisers -- the AD multi-start members (ensemble and
oracle runs) and the classical derivative-free staged calibration -- with a
derivative-free polish run from the best AD member, so neither lane is scored
only on its own starts.

Why the definition is shared. The identifiability diagnostics are evaluated at
theta* and the gap decomposition's form-irreducible share is theta*'s held-out
residual. If the two picked their optimum separately they could disagree about
where the model's best fit is while reporting numbers that look like one
analysis. They read the same artifact instead.

What is recorded, and what is NOT claimed. The multi-start oracle is a BOUND on
the attainable fit, and is labelled as one: on both forms the derivative-free
lane attains a lower objective than the AD multi-start does, which is evidence
the bound is not tight, and that is reported rather than smoothed. The
attainment curve -- best-of-N over the AD members -- shows how the bound moved
with the start budget, so a reader can see whether more starts were still
buying improvement when the budget ran out.

Every candidate is RESCORED here on this run's own likelihood before being
compared, so members that arrived from different artifacts are ranked on one
objective and not on whatever objective each artifact happened to record. The
discrepancy between the rescored and stored objective is reported; a large one
means the artifact was produced against different data or a different form.

Candidates ON a box bound are kept, not discarded. The derivative-free lane
optimises in z and reports in theta, so a coordinate driven hard against its
bound saturates the logistic and comes back sitting exactly on it. Such a point
is a perfectly good parameter vector -- the likelihood evaluates there, and it
can be the lowest-NLL point found -- but it is not strictly interior, so it has
no finite z and the box-membership predicate rejects it. Rejecting it would
silently discard the attained optimum and report the lane as having failed, so
the admissibility floor (physical validity) is the only hard filter here, the
CLOSED box is accepted, and every coordinate touching a bound at the optimum is
recorded. That contact is itself a result: a fit whose optimum lies on a prior
bound is telling the reader about the box.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import jax.numpy as jnp

from .baseline import run_staged_baseline
from .curves import load_measured_curveset
from .likelihood import (check_delta_explicit_equivalence, cu_offset_status,
                         load_likelihood_config, make_likelihood, rmse_per_curve)
from .runio import effort_accounting
from .theta_space import space_for_form

# Best-of-N ladder for the attainment curve (truncated to the member count).
ATTAINMENT_LADDER = (1, 2, 4, 8, 16, 32, 64, 128, 256)

# Tie-break tolerance (NLL). "Lowest NLL over both optimisers" does not settle a
# tie, and the derivative-free lane produces near-ties that sit exactly ON a box
# bound, where the gradient need not vanish, z is infinite and the whole
# diagnostic stack is ill-posed. Among candidates within this tolerance of the
# best, a strictly interior one is preferred and the discarded boundary
# alternative is recorded with its objective, so nothing is hidden. The
# tolerance is four orders below the smallest difference the analysis reads (the
# 0.64-4.36 NLL gap between the two optimisers) and eight orders above the
# rescoring drift between artifacts, so it can only ever break a genuine tie.
INTERIOR_PREFERENCE_NLL_TOL = 1e-3


def _box_contact(theta: np.ndarray, space) -> dict:
    """Which coordinates touch or leave their box, on the box-fraction scale.

    u = 0 and u = 1 ARE the box bounds. A coordinate at exactly 0 or 1 is on
    the bound: admissible and scoreable, but with no finite z, so any consumer
    that needs the unconstrained coordinate has to say what it does about it.
    """
    u = np.asarray(space.u_from_theta(np.asarray(theta, dtype=float)),
                   dtype=float)
    names = space.internal_names
    on = [names[i] for i in range(u.size) if u[i] <= 0.0 or u[i] >= 1.0]
    out = [names[i] for i in range(u.size)
           if not np.isfinite(u[i]) or u[i] < 0.0 or u[i] > 1.0]
    return {"u": u, "on_bound": on, "outside_coordinates": out,
            "outside_closed_box": bool(out)}


def _box_proximity(theta: np.ndarray, space,
                   near_fraction: float = 1e-4) -> dict:
    """How close each coordinate sits to its own box bound, in box fractions.

    A coordinate whose optimum sits a millionth of a box width from its floor
    is a boundary solution in everything but name: the profile on that side is
    cut off by the prior, not by the data, and any interval reported for it is
    a statement about the box. The threshold is declared rather than left to a
    reader's eye.
    """
    u = np.asarray(space.u_from_theta(np.asarray(theta, dtype=float)),
                   dtype=float)
    names = space.internal_names
    dist = np.minimum(u, 1.0 - u)
    return {
        "near_bound_fraction_threshold": near_fraction,
        "distance_to_nearest_bound_in_box_fractions": {
            names[i]: float(dist[i]) for i in range(u.size)},
        "coordinates_within_threshold": [
            {"name": names[i], "u": float(u[i]),
             "bound": ("lower" if u[i] <= 1.0 - u[i] else "upper"),
             "distance_box_fractions": float(dist[i])}
            for i in range(u.size) if dist[i] <= near_fraction],
    }


def _candidates_from_artifact(payload: dict, source: str) -> list[dict]:
    """Every parameter vector an artifact offers, tagged with its provenance."""
    out: list[dict] = []
    members = payload.get("members")
    if isinstance(members, list):
        for m in members:
            if m.get("failed"):
                continue
            th = m.get("theta_best")
            if th:
                out.append({"theta": np.asarray(th, dtype=float),
                            "source": source, "lane": "ad_multistart",
                            "index": m.get("start_index"),
                            "stored_obj": m.get("final_nll")})
    runs = payload.get("runs")
    if isinstance(runs, list):
        for i, r in enumerate(runs):
            th = r.get("theta_final")
            if th:
                out.append({"theta": np.asarray(th, dtype=float),
                            "source": source, "lane": "derivative_free",
                            "index": r.get("restart_index", i),
                            "stored_obj": r.get("final_obj")})
    for key in ("best_theta",):
        th = payload.get(key)
        if th and not isinstance(th, dict):
            out.append({"theta": np.asarray(th, dtype=float), "source": source,
                        "lane": payload.get("kind", "artifact_best"),
                        "index": None,
                        "stored_obj": payload.get("final_obj")
                        or payload.get("best_final_obj")})
    bands = payload.get("bands")
    if isinstance(bands, dict) and bands.get("best_theta"):
        out.append({"theta": np.asarray(bands["best_theta"], dtype=float),
                    "source": source, "lane": "ensemble_band_best",
                    "index": None, "stored_obj": bands.get("best_final_nll")})
    # A profile-likelihood artifact is an optimiser too, and a productive one:
    # every grid row is a constrained re-optimisation from a warm start, and a
    # row whose held coordinate happens to sit near a better basin can land
    # BELOW the optimum the ensemble and the derivative-free lane agreed on.
    # Harvesting its rows here is what stops that from being a footnote in one
    # artifact while every downstream diagnostic keeps using the worse point.
    prof = payload.get("profile") or (payload if "per_parameter" in payload else None)
    if isinstance(prof, dict) and isinstance(prof.get("per_parameter"), dict):
        for name, par in prof["per_parameter"].items():
            thetas = par.get("theta_at_grid") or []
            nlls = par.get("profile_nll") or [None] * len(thetas)
            for g, th in enumerate(thetas):
                if not th or nlls[g] is None:
                    continue
                out.append({"theta": np.asarray(th, dtype=float),
                            "source": source, "lane": "profile_likelihood_row",
                            "index": f"{name}[{g}]", "stored_obj": nlls[g]})
    return out


def attain_optimum(root: str | Path, form: int,
                   ad_jsons: list[str], baseline_jsons: list[str],
                   profile_jsons: list[str] | None = None,
                   keys: list[str] | None = None,
                   polish_stages: tuple[str, ...] = ("elastic", "strength", "all"),
                   polish_max_nfev: int = 200,
                   effective_stress_path: bool = False) -> dict:
    """Resolve theta* over every optimiser that produced candidates.

    ``profile_jsons`` admits profile-likelihood artifacts, whose every grid row
    is a constrained re-optimisation and therefore a candidate in its own right.
    """
    root = Path(root)
    t0 = time.time()
    space = space_for_form(form)
    curveset = load_measured_curveset(
        root, keys=keys, effective_stress_path=effective_stress_path)
    cfg = load_likelihood_config(root)
    like = make_likelihood(curveset, cfg, form=form)

    sources = []
    candidates: list[dict] = []
    attainment = []
    for path in list(ad_jsons) + list(baseline_jsons) + list(profile_jsons or []):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        src_form = int(payload.get("config", {}).get("hyperparams", {})
                       .get("form", 1))
        if src_form != form:
            raise SystemExit(f"{path} records form {src_form}, "
                             f"but this run is form {form}")
        got = _candidates_from_artifact(payload, str(path))
        members = payload.get("members")
        sources.append({
            "path": str(path),
            "kind": payload.get("kind"),
            "form": src_form,
            "n_candidates": len(got),
            "n_members_listed": (len(members) if isinstance(members, list)
                                 else len(got)),
            "stored_best_obj": (payload.get("final_obj")
                                or payload.get("best_final_obj")
                                or payload.get("multistart_summary", {})
                                .get("best_final_nll")),
            "data_fingerprints_sha256": payload.get("config", {})
            .get("data_fingerprints_sha256"),
        })
        candidates.extend(got)

    if not candidates:
        raise RuntimeError("no candidate parameter vectors in the given artifacts")

    scored = []
    skipped = []
    for c in candidates:
        th = c["theta"]
        if th.shape != (space.n_theta,):
            skipped.append({**{k: v for k, v in c.items() if k != "theta"},
                            "reason": f"wrong length {th.shape}"})
            continue
        ok, reasons = space.is_admissible(th)
        if not ok:
            skipped.append({**{k: v for k, v in c.items() if k != "theta"},
                            "reason": "; ".join(reasons)})
            continue
        bounds = _box_contact(th, space)
        if bounds["outside_closed_box"]:
            skipped.append({**{k: v for k, v in c.items() if k != "theta"},
                            "reason": "outside the closed box: "
                                      + ", ".join(bounds["outside_coordinates"])})
            continue
        obj = float(like.nll_theta_jit(jnp.asarray(th)))
        scored.append({**c, "obj": obj, "on_box_bound": bounds["on_bound"]})

    if not scored:
        raise RuntimeError("every candidate was inadmissible or outside the box")

    drift = [abs(s["obj"] - s["stored_obj"]) for s in scored
             if isinstance(s["stored_obj"], (int, float))]

    # ---- attainment curve over the AD members, per source -----------------
    for src in sources:
        rows = [s for s in scored
                if s["source"] == src["path"] and s["lane"] == "ad_multistart"]
        if not rows:
            continue
        n_listed = src["n_members_listed"]
        ladder = [n for n in ATTAINMENT_LADDER if n <= n_listed]
        if n_listed not in ladder:
            ladder.append(int(n_listed))
        best_of = {}
        for n in ladder:
            objs = [r["obj"] for r in rows
                    if r["index"] is not None and int(r["index"]) < n]
            best_of[str(n)] = (float(np.min(objs)) if objs else None)
        attainment.append({
            "source": src["path"],
            "n_members_listed": int(n_listed),
            "n_members_scored": len(rows),
            "best_of_n": best_of,
            "note": ("best-of-N over the members whose START INDEX is below N, "
                     "so N counts starts drawn and not starts that survived the "
                     "admissibility filter; a curve still falling at the largest "
                     "N means the bound was not saturated by the start budget"),
        })

    ad = [s for s in scored if s["lane"] == "ad_multistart"
          or s["lane"] == "ensemble_band_best"]
    df = [s for s in scored if s["lane"] == "derivative_free"]
    best_ad = min(ad, key=lambda s: s["obj"]) if ad else None
    best_df = min(df, key=lambda s: s["obj"]) if df else None

    # ---- derivative-free polish from the best AD member -------------------
    polish = None
    if best_ad is not None:
        rec = run_staged_baseline(curveset, cfg, best_ad["theta"],
                                  like.const_total, stages=tuple(polish_stages),
                                  max_nfev_per_stage=polish_max_nfev, form=form)
        th_pol = np.asarray(rec["theta_final"], dtype=float)
        ok, reasons = space.is_admissible(th_pol)
        bounds_pol = _box_contact(th_pol, space)
        usable = bool(ok and not bounds_pol["outside_closed_box"])
        obj_pol = (float(like.nll_theta_jit(jnp.asarray(th_pol)))
                   if usable else float("inf"))
        polish = {
            "role": ("derivative-free staged polish started from the best AD "
                     "member, so neither lane is scored only on its own starts"),
            "stages": list(polish_stages),
            "max_nfev_per_stage": polish_max_nfev,
            "start_obj": best_ad["obj"],
            "final_obj": (obj_pol if usable else None),
            "improvement_NLL": (best_ad["obj"] - obj_pol if usable else None),
            "accepted_as_candidate": usable,
            "rejected_reason": (None if usable else
                                ("admissibility floor: " + "; ".join(reasons)
                                 if not ok else "outside the closed box")),
            "staged_final_obj": rec["final_obj"],
            "staged_obj_note": ("the objective the staged optimiser reached; "
                                "reported even when the endpoint is rejected, "
                                "so a descent to an inadmissible point is "
                                "visible rather than looking like a failure"),
            "admissible": bool(ok),
            "inadmissible_reasons": reasons,
            "on_box_bound": bounds_pol["on_bound"],
            "outside_closed_box": bounds_pol["outside_closed_box"],
            "theta_final": th_pol.tolist(),
            "nfev_total": int(sum(s["nfev"] for s in rec["stages"])),
            "stage_records": rec["stages"],
        }
        if usable:
            scored.append({"theta": th_pol, "source": "polish",
                           "lane": "derivative_free_polish", "index": None,
                           "stored_obj": rec["final_obj"], "obj": obj_pol,
                           "on_box_bound": bounds_pol["on_bound"]})

    best = min(scored, key=lambda s: s["obj"])
    tie_break = None
    if best.get("on_box_bound"):
        interior = [s for s in scored
                    if not s.get("on_box_bound")
                    and s["obj"] <= best["obj"] + INTERIOR_PREFERENCE_NLL_TOL]
        if interior:
            picked = min(interior, key=lambda s: s["obj"])
            tie_break = {
                "applied": True,
                "tolerance_NLL": INTERIOR_PREFERENCE_NLL_TOL,
                "reason": ("the lowest-NLL candidate sat ON a box bound, where "
                           "the gradient need not vanish and z is infinite; a "
                           "strictly interior candidate within the tie "
                           "tolerance was preferred"),
                "boundary_candidate": {
                    "obj": best["obj"], "lane": best["lane"],
                    "source": best["source"],
                    "on_box_bound": best["on_box_bound"]},
                "chosen_candidate": {"obj": picked["obj"], "lane": picked["lane"],
                                     "source": picked["source"]},
                "nll_given_up": picked["obj"] - best["obj"],
            }
            best = picked
    if tie_break is None:
        tie_break = {"applied": False,
                     "tolerance_NLL": INTERIOR_PREFERENCE_NLL_TOL,
                     "reason": "the lowest-NLL candidate is strictly interior"}
    theta_star = np.asarray(best["theta"], dtype=float)

    lane_best = {}
    for lane in sorted({s["lane"] for s in scored}):
        rows = [s for s in scored if s["lane"] == lane]
        b = min(rows, key=lambda s: s["obj"])
        lane_best[lane] = {"obj": b["obj"], "source": b["source"],
                           "n_candidates": len(rows)}

    on_path = like.on_effective_stress_path
    prof = rmse_per_curve(theta_star, like, profile_delta=True)
    # On the measured path the two scoring rules are one rule, because there
    # is no offset to profile or hold.
    fixed = prof if on_path else rmse_per_curve(theta_star, like,
                                                profile_delta=False)
    nfev_polish = polish["nfev_total"] if polish else 0
    return {
        "role": ("the attained optimum: lowest NLL over both optimisers, with "
                 "a derivative-free polish from the best AD member"),
        "form": form,
        "curve_keys": list(curveset.keys),
        "theta_names": space.theta_names,
        "theta_star": theta_star.tolist(),
        "u_star": np.asarray(space.u_from_theta(theta_star)).tolist(),
        "theta_star_on_box_bound": best.get("on_box_bound", []),
        "theta_star_box_proximity": _box_proximity(theta_star, space),
        "interior_tie_break": tie_break,
        "nll_star": best["obj"],
        "attaining_lane": best["lane"],
        "attaining_source": best["source"],
        "lane_best_obj": lane_best,
        "ad_best_obj": (best_ad["obj"] if best_ad else None),
        "derivative_free_best_obj": (best_df["obj"] if best_df else None),
        "ad_minus_derivative_free_NLL": (
            best_ad["obj"] - best_df["obj"]
            if (best_ad and best_df) else None),
        "oracle_is_a_bound_not_a_certificate": (
            "the AD multi-start oracle bounds the attainable fit; where the "
            "derivative-free lane attains a lower objective the bound is not "
            "tight, and the gap is reported"),
        "polish": polish,
        "attainment_curve": attainment,
        "sources": sources,
        "n_candidates_scored": len(scored),
        "n_candidates_skipped": len(skipped),
        "skipped_candidates": skipped[:20],
        "rescoring": {
            "note": ("every candidate rescored on this run's likelihood before "
                     "ranking, so lanes are compared on one objective"),
            "max_abs_drift_vs_stored_obj": (max(drift) if drift else None),
            "n_compared": len(drift),
        },
        "rmse_q_at_theta_star_MPa": {
            "profiled": dict(zip(curveset.keys, prof.tolist())),
            "fixed_delta": dict(zip(curveset.keys, fixed.tolist())),
            "mean_profiled": float(np.mean(prof)),
            "mean_fixed_delta": float(np.mean(fixed)),
        },
        "cu_offset_at_theta_star": (None if on_path else
                                    cu_offset_status(theta_star, like)),
        "delta_explicit_equivalence": (
            None if on_path else
            check_delta_explicit_equivalence(theta_star, like)),
        "cu_treatment": ("measured effective-stress path, no offset nuisance"
                         if on_path else cfg.as_dict()["cu_treatment"]),
        "effort": effort_accounting(
            time.time() - t0, n_curves=curveset.n_curves,
            derivative_free_nfev=nfev_polish,
            extra_curve_forwards=(len(scored) + 4) * curveset.n_curves,
            note="rescoring pass over every stored candidate plus the polish"),
    }
