"""Connected profiles of the current measured geometry and attained fit."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import hashlib
import inspect
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[2]
EVID = ROOT / "selfrev-r2/evidence"
SAVED = EVID / "SR-09a_resolution"
sys.path.insert(0, str(ROOT / "compute/src"))

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares
from tfenn_invcalib import curves, likelihood, numerics, theta_space, mirror
from tfenn_invcalib import current_inputs, bounded_fit
from tfenn_invcalib import profile_likelihood as pl

CUT = 1.92
STEPS = {1: 3200, 2: 1600}
COMPARISON_STEPS = {1: 6400, 2: 3200}
GRID_COORDINATES = ("G_op", "gamma_c", "k_sharpness")


def grid_coordinates(form):
    """Coordinates whose endpoints support the current paired-form map."""
    return GRID_COORDINATES + ("c_j", "phi_j_deg") + (
        ("d_fab", "q_fab") if form == 2 else ())


def recipe(form, steps=None):
    inputs = current_inputs.identity(form, steps or STEPS[form], (
        "compute/src/tfenn_invcalib/bounded_fit.py",))
    # Profile marching shares this file with the forward adapter. A change to
    # its rows must invalidate profiles without discarding valid member fits.
    source = inspect.getsource(Model.__init__) + inspect.getsource(Model.record)
    inputs["model_definition_sha256"] = hashlib.sha256(source.encode()).hexdigest()
    return inputs


def profile_definition():
    source = (inspect.getsource(Model.fit) + inspect.getsource(profile_coordinate)
              + inspect.getsource(interval) + inspect.getsource(pl.solve_connected_crossings)
              + repr(CUT))
    return hashlib.sha256(source.encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    # Windows readers can briefly deny replacement of an open checkpoint.
    deadline = time.monotonic() + 5
    while True:
        try:
            os.replace(tmp, path)
            break
        except PermissionError as exc:
            if getattr(exc, "winerror", None) not in (5, 32) or time.monotonic() >= deadline:
                raise
            time.sleep(0.05)


def output(form):
    return EVID / f"SR-09a_map_f{form}.json"


def interval(space, name, rows, reference):
    rows = sorted(rows, key=lambda row: row["u_fixed"])
    ug = np.array([r["u_fixed"] for r in rows])
    values = np.array([r["nll"] for r in rows])
    block = pl._interval_and_class(ug, values-reference["nll"], np.isfinite(values))
    # A sampled endpoint inside the box cannot establish a prior-edge bound.
    missing = []
    if block["lower_endpoint_at_box_edge"] and ug[0] > 1e-9:
        missing.append("lower crossing")
    if block["upper_endpoint_at_box_edge"] and ug[-1] < 1-1e-9:
        missing.append("upper crossing")
    if missing:
        block["classification"] = "unresolved: " + ", ".join(missing)
        block["interval_u"] = [None, None]
    i = space.internal_names.index(name)
    block["interval_internal_units"] = pl._interval_internal_units(
        space, i, np.asarray(reference["u"]), *block["interval_u"])
    if name == "nu_pp" and block["upper_endpoint_at_box_edge"]:
        block["classification"] = "box-limited"
    # Only quantities consumed by the map are retained; old precision alarms
    # and certification flags are not scientific results.
    return {k: block[k] for k in (
        "interval_u", "interval_internal_units", "classification",
        "lower_endpoint_at_box_edge", "upper_endpoint_at_box_edge")}


def prepare(form):
    previous = read(output(form))
    if previous.get("recipe") == recipe(form):
        if previous.get("profile_definition_sha256") != profile_definition():
            previous.update(profile_definition_sha256=profile_definition(),
                            per_parameter={}, status="partial")
            previous.pop("grid_comparison", None)
        return previous
    warm = [{
        "theta": previous["reference"]["theta"],
        "source": str(output(form).relative_to(ROOT)),
        "source_sha256": current_inputs.digest(output(form)),
        "provenance": ("Previous computed reference; parameter vector only"
                       if previous.get("recipe") else
                       "Pre-SR-01 CU-Z=60 degree fit; parameter vector only"),
        "input_definition": previous.get("recipe"),
    }, *previous.get("historical_warm_starts", [])]
    model = Model(form)
    reference = model.record(model.space.u_from_theta(np.asarray(warm[0]["theta"])))
    return {
        "form": form, "n_steps": STEPS[form], "status": "partial",
        "likelihood_cut": CUT, "reference": reference,
        "recipe": recipe(form), "historical_warm_starts": warm,
        "profile_definition_sha256": profile_definition(),
        "scope": "Connected profile intervals around the attained basin",
        "source": str(output(form).relative_to(ROOT)),
        "per_parameter": {},
    }


class Model:
    def __init__(self, form, steps=None):
        self.space = theta_space.space_for_form(form)
        self.cs = curves.load_measured_curveset(ROOT / "compute", n_steps=steps or STEPS[form],
                                               effective_stress_path=True)
        cfg = likelihood.load_likelihood_config(ROOT / "compute")
        self.constant = self.cs.q_meas.size * math.log(cfg.sigma_q*math.sqrt(2*math.pi))
        cs = self.cs
        forward = mirror.mirror_forward_path if form == 1 else mirror.mirror_forward_path_f2
        def fn(theta, M, n, pressure, increment, stations):
            out = forward(theta, M, n, pressure, increment, n_steps=cs.n_steps, early_exit=True)
            return jnp.interp(stations, jnp.arange(cs.n_steps+1)*increment, out["q"])
        # Sequential curve mapping preserves conditional lateral solves; vmap
        # evaluates both branches of each curve's convergence condition.
        self.r = jax.jit(lambda u: ((jax.lax.map(lambda arg:
            fn(self.space.theta_from_u(u), *arg[:3], cs.d_eps11, arg[3]),
            (cs.M_bond, cs.n_hat, cs.p_conf_seq, jnp.asarray(cs.eps_stations)))
            - jnp.asarray(cs.q_meas))/cfg.sigma_q).ravel())
        self.j = jax.jit(jax.jacfwd(self.r))
        self.lower, self.upper = numerics.solver_bounds_u(self.space)

    def record(self, u):
        theta = np.asarray(self.space.theta_from_u(jnp.asarray(u)))
        ok, reasons = self.space.is_admissible(theta)
        if not ok:
            raise ValueError(f"Inadmissible parameter vector: {reasons}")
        residual = np.asarray(self.r(jnp.asarray(u)))
        if not np.isfinite(residual).all():
            raise ValueError("Invalid integration path")
        return {"u": np.asarray(u).tolist(), "theta": theta.tolist(),
                "nll": float(residual@residual/2 + self.constant)}

    def fit(self, name, value, seed, budget=100, checkpoint=None):
        i = self.space.internal_names.index(name)
        result = bounded_fit.fit(self, np.asarray(self.space.theta_from_u(jnp.asarray(seed["u"]))),
                                 fixed=(i, value), checkpoint=checkpoint, chunk=budget)
        if result["solver_status"] == "failed":
            raise ValueError(f"Unresolved profile fit {name}={value}: {result}")
        return {**result.pop("best"), "u_fixed": value,
                "start_nll": result["initial_nll"], **result}


MODELS = {}


def profile_coordinate(form, name, budget):
    model = MODELS.setdefault(form, None)
    if model is None:
        model = MODELS[form] = Model(form)
    data = read(output(form))
    if data.get("recipe") != recipe(form):
        raise ValueError("Prepare the map with current computation inputs first")
    reference = data["reference"]
    i = model.space.internal_names.index(name)
    u0 = reference["u"][i]
    J = np.asarray(model.j(jnp.asarray(reference["u"])))
    _, singular, right = np.linalg.svd(J, full_matrices=False)
    active = singular > np.finfo(float).eps * max(J.shape) * singular[0]
    variance = np.sum((right[active, i] / singular[active])**2)
    null = np.any(np.abs(right[~active, i]) > np.sqrt(np.finfo(float).eps))
    spacing = 1.0 if null else min(1.0, max(np.sqrt(2*CUT*variance), np.sqrt(np.finfo(float).eps)))
    rows = [{**reference, "u_fixed": u0, "source": data["source"]}]
    work_path = EVID / f"SR-09a_map_f{form}_{name}.json"
    if work_path.exists():
        prior = read(work_path)
        if prior.get("recipe") == recipe(form) and prior["reference"] == reference:
            rows = prior["rows"]
            if (prior["status"] == "complete"
                    and prior.get("profile_definition_sha256") == profile_definition()):
                return form, name, prior["result"]
    def checkpoint(status, **extra):
        save(work_path, {"form": form, "name": name, "n_steps": STEPS[form],
            "budget": budget, "recipe": recipe(form), "reference": reference, "rows": rows,
            "profile_definition_sha256": profile_definition(),
            "status": status, **extra})
    def fit_row(value, seed):
        prior = read(work_path) if work_path.exists() else {}
        pending = prior.get("in_progress")
        if (prior.get("recipe") == recipe(form) and prior.get("reference") == reference
                and prior.get("profile_definition_sha256") == profile_definition()
                and pending and pending["u_fixed"] == value):
            seed = pending["iterate"]
        return model.fit(name, value, seed, budget, checkpoint=lambda best:
            checkpoint("running", in_progress={"u_fixed": value, "iterate": best}))
    if (not work_path.exists() or prior.get("recipe") != recipe(form)
            or prior.get("reference") != reference
            or prior.get("profile_definition_sha256") != profile_definition()):
        checkpoint("running")
    for direction in (-1, 1):
        seed = reference
        offset = spacing
        while True:
            value = float(np.clip(u0 + direction*offset, model.lower[i], model.upper[i]))
            if abs(value-u0) < 1e-12:
                break
            row = next((r for r in rows if r["u_fixed"] == value), None)
            if row is None:
                row = fit_row(value, seed)
                rows.append(row)
                checkpoint("running")
                print(f"form={form} parameter={name} u={value:.6g} delta_nll={row['nll']-reference['nll']:.6g}", flush=True)
            if row["nll"]-reference["nll"] >= CUT or value in (model.lower[i], model.upper[i]):
                break
            seed = row
            offset *= 2
    def evaluate_level(value):
        existing = next((r for r in rows if r["u_fixed"] == value), None)
        if existing is not None:
            return existing["nll"]
        near = min(rows, key=lambda row: abs(row["u_fixed"]-value))
        row = fit_row(value, near)
        rows.append(row)
        checkpoint("running")
        return row["nll"]
    pl.solve_connected_crossings([r["u_fixed"] for r in rows],
        [r["nll"] for r in rows], reference["nll"], evaluate_level)
    result = {**interval(model.space, name, rows, reference), "rows": rows,
              "source": str(work_path.relative_to(ROOT)), "solver_chunk_nfev": budget}
    checkpoint("complete", result=result)
    return form, name, result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-nfev", type=int, default=40)
    ap.add_argument("--only", nargs="+")
    args = ap.parse_args()
    if args.workers < 1 or args.max_nfev < 1:
        ap.error("workers and max-nfev must be positive")
    for form in (1, 2):
        save(output(form), prepare(form))
    if args.prepare:
        for form in (1, 2):
            d = read(output(form))
            print(form, d["reference"])
        return
    jobs = [(f, name) for f in (1, 2)
            for name in theta_space.space_for_form(f).internal_names
            if name not in read(output(f))["per_parameter"]
            and (not args.only or name in args.only)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(profile_coordinate, f, name, args.max_nfev) for f,name in jobs]
        for future in as_completed(futures):
            form, name, result = future.result()
            data = read(output(form))
            data["per_parameter"][name] = result
            data["status"] = ("complete" if len(data["per_parameter"]) ==
                              theta_space.space_for_form(form).n_theta else "partial")
            save(output(form), data)
            print(f"completed form={form} parameter={name}: {result['interval_internal_units']} {result['classification']}", flush=True)


if __name__ == "__main__":
    main()
