"""SR-03: real-data profiles and supplied operative elastic constants.

The profile objective is SR-09a_map.Model, with its measured curves,
effective-stress paths, shear-free controller and Gaussian likelihood.
All constraints are imposed on the compliance of the integrated stiffness.

Run:    python selfrev-r3/evidence/SR-03_operative_elastic.py --run
Replay: python selfrev-r3/evidence/SR-03_operative_elastic.py --verify
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "results/iterative/sr03-operative-elastic"
PROMPT = ROOT / "results/driver_codex_exec/revision-r3-sr03-operative-elastic/prompt.md"
sys.path.insert(0, str(ROOT / "compute/src"))

import psutil
process = psutil.Process()
process.cpu_affinity(process.cpu_affinity()[:8])
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize, brentq, least_squares
from tfenn_invcalib import current_inputs, mirror, numerics, operative_elastic as elastic


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


mapping = module("sr03_mapping", "selfrev-r2/evidence/SR-09a_map.py")
CUT = mapping.CUT
QUANTITIES = ("E_p", "E_o", "nu_pp", "nu_po", "nu_op")
ENDPOINT_METHOD = "exact scalar-coordinate continuation across the final likelihood bracket"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    # Reuse the current producer's Windows-safe atomic replacement. A real
    # reader lock interrupted the first fine-grid E_o checkpoint replacement.
    mapping.save(path, value)


def primary(form):
    data = read(mapping.output(form))
    if data["recipe"] != mapping.recipe(form):
        raise ValueError("Primary map inputs differ from the current measured model")
    return data


class Model(mapping.Model):
    def __init__(self, form, steps=None):
        super().__init__(form, steps)
        self.form = form
        self.sigma_q = mapping.likelihood.load_likelihood_config(ROOT / "compute").sigma_q
        self.e = jax.jit(lambda u: elastic.engineering(self.space.theta_from_u(u)))
        self.ej = jax.jit(jax.jacfwd(self.e))
        def margins(u):
            theta = self.space.theta_from_u(u)
            spd = elastic.positive_definiteness_margin(theta)
            if form == 1:
                return jnp.array([spd])
            d, q = theta[12:14]
            zlo, zhi = jnp.minimum(d, -2*d), jnp.maximum(d, -2*d)
            vertex = jnp.clip(-1/(2*jnp.where(q > 0, q, 1.)), zlo, zhi)
            ends = jnp.minimum(1+zlo+q*zlo*zlo, 1+zhi+q*zhi*zhi)
            fabric = jnp.where(q > 0, jnp.minimum(ends, 1+vertex+q*vertex*vertex), ends)
            return jnp.array([spd, fabric-.05])
        self.margins = jax.jit(margins)
        self.margins_j = jax.jit(jax.jacfwd(margins))

    def record(self, u):
        row = super().record(u)
        row["engineering"] = dict(zip(elastic.NAMES, np.asarray(self.e(jnp.asarray(u))).tolist()))
        row["admissibility_margins"] = np.asarray(self.margins(jnp.asarray(u))).tolist()
        row["rmse_MPa"] = float(self.sigma_q*np.sqrt(2*(row["nll"]-self.constant)/self.cs.q_meas.size))
        row["active_assembly_bounds"] = [
            {"name": self.space.internal_names[i], "side": side}
            for i, value in enumerate(u) for side, edge in (("lower", self.lower[i]), ("upper", self.upper[i]))
            if abs(value-edge) <= 1e-7]
        return row

    def constraints(self, supplied):
        names = tuple(supplied)
        indices = np.array([elastic.NAMES.index(name) for name in names], dtype=int)
        scales = elastic.SCALES[indices]
        targets = np.array([supplied[name] for name in names])
        constraints = [{"type": "ineq", "fun": lambda u: np.asarray(self.margins(jnp.asarray(u))),
                        "jac": lambda u: np.asarray(self.margins_j(jnp.asarray(u)))}]
        if names:
            constraints.append({"type": "eq",
                "fun": lambda u: (np.asarray(self.e(jnp.asarray(u)))[indices]-targets)/scales,
                "jac": lambda u: np.asarray(self.ej(jnp.asarray(u)))[indices]/scales[:, None]})
        return constraints

    def project(self, u, supplied):
        """Find a feasible start; this objective is never a reported likelihood."""
        u = np.clip(np.asarray(u), self.lower, self.upper)
        constraints = self.constraints(supplied)
        if np.min(constraints[0]["fun"](u)) > 0 and (
                not supplied or np.max(abs(constraints[-1]["fun"](u))) < 1e-12):
            return u
        # Complete a partial supply with the seed's OTHER engineering constants,
        # then use the exact inverse. This preserves the elastic axial slopes
        # in Poisson-only starts instead of displacing them in box coordinates.
        constants = dict(zip(elastic.NAMES, np.asarray(self.e(jnp.asarray(u)))))
        constants.update(supplied)
        if "nu_po" in supplied and "nu_op" not in supplied:
            constants["nu_op"] = constants["nu_po"]*constants["E_o"]/constants["E_p"]
        independent = ("E_p", "E_o", "G_op", "nu_pp", "nu_op")
        try:
            candidates = elastic.assembly_candidates(**{key: constants[key] for key in independent})
        except ValueError:
            candidates = []
        for candidate in candidates:
            theta = np.asarray(self.space.theta_from_u(jnp.asarray(u))).copy()
            theta[:5] = candidate
            restored = np.asarray(self.space.u_from_theta(theta))
            if (np.all(restored >= self.lower) and np.all(restored <= self.upper)
                    and np.min(constraints[0]["fun"](restored)) > 0
                    and np.max(abs(constraints[-1]["fun"](restored))) < 1e-12):
                return restored
        sol = minimize(lambda x: (float((x-u)@(x-u)/2), x-u), u, jac=True,
                       bounds=list(zip(self.lower, self.upper)), constraints=constraints,
                       method="SLSQP", options={"ftol": 1e-12, "maxiter": 100})
        while sol.status == 9:
            sol = minimize(lambda x: (float((x-u)@(x-u)/2), x-u), sol.x, jac=True,
                           bounds=list(zip(self.lower, self.upper)), constraints=constraints,
                           method="SLSQP", options={"ftol": 1e-12, "maxiter": 100})
        if not sol.success or (supplied and np.max(abs(constraints[-1]["fun"](sol.x))) > 1e-9):
            raise ValueError(f"Constraint feasibility unresolved: {sol.message}")
        return sol.x

    def fit_supplied(self, supplied, seed, checkpoint=None):
        """Original likelihood with analytic equality and admissibility gradients."""
        started = time.monotonic()
        x = self.project(seed["u"], supplied)
        constraints = self.constraints(supplied)
        count = rejected = 0
        cache = {}
        best = None
        observations = self.cs.q_meas.size
        def objective(u):
            nonlocal count, rejected, best
            key = np.asarray(u).tobytes()
            if key == cache.get("key"):
                return cache["value"]
            count += 1
            theta = np.asarray(self.space.theta_from_u(jnp.asarray(u)))
            valid, _ = self.space.is_admissible(theta)
            r = np.asarray(self.r(jnp.asarray(u))) if valid else None
            if r is None or not np.isfinite(r).all():
                rejected += 1
                # Trial rejection only, never admitted as a likelihood witness.
                margin = np.asarray(self.margins(jnp.asarray(u)))
                violation = np.minimum(margin, 0)
                return 1e12+1e12*float(violation@violation), 2e12*violation@np.asarray(self.margins_j(jnp.asarray(u)))
            J = np.asarray(self.j(jnp.asarray(u)))
            if not np.isfinite(J).all():
                raise ValueError("Nonfinite real-model residual Jacobian")
            cost = float(r@r/2)
            feasible = not supplied or np.max(abs(constraints[-1]["fun"](u))) <= 1e-9
            if feasible and (best is None or cost+self.constant < best["nll"]):
                best = self.record(u)
                if checkpoint:
                    checkpoint(best)
            value = cost/observations, J.T@r/observations
            cache.update(key=key, value=value)
            return value
        passes = []
        initial = self.record(x)
        while True:
            sol = minimize(objective, x, jac=True, method="SLSQP",
                bounds=list(zip(self.lower, self.upper)), constraints=constraints,
                options={"ftol": 1e-8, "maxiter": 100})
            passes.append({"status": int(sol.status), "message": str(sol.message),
                           "nit": int(sol.nit), "nfev": int(sol.nfev)})
            print("fit", self.form, self.cs.n_steps, supplied, "status", sol.status,
                  "cost", sol.fun*observations+self.constant, "calls", count, flush=True)
            x = sol.x
            if sol.status != 9:
                break
        # Remove equality-solver residual before scoring the actual final model.
        corrected = self.project(x, supplied)
        row = self.record(corrected)
        if best is not None and best["nll"] < row["nll"]:
            corrected_best = self.record(self.project(best["u"], supplied))
            if corrected_best["nll"] < row["nll"]:
                row = corrected_best
        return {**row, "supplied": supplied, "initial_nll": initial["nll"],
                "solver_success": bool(sol.success), "solver_status": int(sol.status),
                "solver_message": str(sol.message), "solver_passes": passes,
                "residual_calls": count, "rejected_trials": rejected,
                "constraint_residual": {name: row["engineering"][name]-value for name, value in supplied.items()},
                "seconds": time.monotonic()-started}

    def scalar_chart(self, name, value, base):
        """Eliminate assembly w=theta[4], retaining every other coordinate.

        The scalar map is monotone on the admitted anisotropic branch. Its
        implicit derivative is du4/duj=-d e/duj/(d e/du4). The current box
        bounds on the eliminated coordinate and SPD are enforced explicitly.
        """
        index = elastic.NAMES.index(name)
        base = np.asarray(base)
        free = np.array([i for i in range(self.space.n_theta) if i != 4])
        def original(x):
            u = base.copy()
            u[free] = x
            theta = np.asarray(self.space.theta_from_u(jnp.asarray(u)))
            p, o, _, v, _ = theta[:5]
            alpha = (p+o)**2/(2*p*o)
            if name == "nu_pp":
                denominator = alpha*(1-value)*(1+v)-2*(1+value)*(1-v)
                square = 2*(1-v)*(v-value)/denominator if denominator != 0 else -1.
                w = np.sqrt(square) if square > 0 else np.nan
            elif name == "E_p":
                denominator = 2*p*alpha-value*(2*(1-v)+alpha*(1+v))
                square = 2*(1-v)*(p-value)/denominator if denominator != 0 else -1.
                w = np.sqrt(square) if square > 0 else np.nan
            elif name == "E_o":
                denominator = o*alpha-2*value
                square = (1-v)*(o-value)/denominator if denominator != 0 else -1.
                w = np.sqrt(square) if square > 0 else np.nan
            elif name == "nu_op":
                w = 2*value*p/(p+o)
            elif name == "nu_po":
                def difference(w):
                    delta = 1-v-2*w*w
                    denominator = 2*(1-v)-w*w*(2*(1-v)+alpha*(1+v))
                    return w*(p+o)*delta/(o*denominator)-value
                upper = min(.45, np.sqrt((1-v)/alpha)*(1-4*np.finfo(float).eps))
                if upper <= .01 or difference(.01)*difference(upper) > 0:
                    w = np.nan
                else:
                    w = brentq(difference, .01, upper, xtol=np.finfo(float).eps)
            else:
                raise ValueError(f"No scalar chart for {name}")
            u[4] = (w-.01)/.44
            return u
        def differential(u):
            derivative = np.asarray(self.ej(jnp.asarray(u)))[index]
            if not np.isfinite(derivative).all() or derivative[4] == 0:
                raise ValueError("Singular derived-quantity chart")
            T = np.eye(self.space.n_theta)[:, free]
            T[4] = -derivative[free]/derivative[4]
            return T
        return free, original, differential

    def fit_scalar(self, name, value, seed, checkpoint=None):
        started = time.monotonic()
        start = np.clip(np.asarray(seed["u"]), self.lower, self.upper)
        free0, chart0, _ = self.scalar_chart(name, value, start)
        candidates = [chart0(start[free0]), self.project(start, {name: value})]
        if np.isclose(float(self.e(jnp.asarray(start))[elastic.NAMES.index(name)]), value,
                      rtol=2e-12, atol=2e-12):
            candidates.append(start)
        feasible = []
        for u in candidates:
            if np.isfinite(u).all() and np.all(u >= self.lower) and np.all(u <= self.upper):
                try:
                    feasible.append(self.record(u))
                except ValueError:
                    pass  # A failed starting path is not a likelihood witness.
        if not feasible:
            raise ValueError("No valid constrained start from the supplied seed")
        best = min(feasible, key=lambda row: row["nll"])
        base = np.asarray(best["u"])
        free, original, differential = self.scalar_chart(name, value, base)
        count = rejected = 0
        cache = {}
        def residual(x):
            nonlocal best, count, rejected
            count += 1
            u = original(x)
            th = np.asarray(self.space.theta_from_u(jnp.asarray(u)))
            valid = (np.isfinite(u).all() and np.all(u >= self.lower)
                     and np.all(u <= self.upper) and self.space.is_admissible(th)[0])
            r = np.asarray(self.r(jnp.asarray(u))) if valid else None
            if r is None or not np.isfinite(r).all():
                rejected += 1
                if best is None:
                    raise ValueError("Invalid initial constrained path")
                return np.full(self.cs.q_meas.size, 1e100)
            if best is None or float(r@r/2+self.constant) < best["nll"]:
                best = self.record(u)
                if checkpoint:
                    checkpoint(best)
            cache.update(x=x.copy(), u=u, r=r)
            return r
        def jacobian(x):
            u = cache["u"] if np.array_equal(x, cache.get("x")) else original(x)
            return np.asarray(self.j(jnp.asarray(u)))@differential(u)
        residual(base[free])
        initial_nll = best["nll"]
        passes = []
        x = base[free]
        while True:
            sol = least_squares(residual, x, jac=jacobian, x_scale="jac", tr_solver="exact",
                                bounds=(self.lower[free], self.upper[free]), max_nfev=100)
            passes.append({"status": int(sol.status), "message": sol.message,
                           "nfev": int(sol.nfev), "optimality": float(sol.optimality)})
            print("scalar fit", self.form, self.cs.n_steps, name, value,
                  "status", sol.status, "nll", best["nll"], "calls", count, flush=True)
            if sol.status != 0:
                break
            x = sol.x
        return {**best, "supplied": {name: value}, "initial_nll": initial_nll,
                "solver_success": bool(sol.success), "solver_status": int(sol.status),
                "solver_message": sol.message, "solver_passes": passes,
                "residual_calls": count, "rejected_trials": rejected,
                "constraint_residual": {name: best["engineering"][name]-value},
                "solver": "dense TRF with exact scalar engineering constraint",
                "seconds": time.monotonic()-started}


def continued_fit(model, supplied, seed, checkpoint=None):
    """Self continuation at the existing bounded fitter's relative cost scale."""
    def fit(start):
        if len(supplied) == 1 and next(iter(supplied)) in QUANTITIES:
            name, value = next(iter(supplied.items()))
            return model.fit_scalar(name, value, start, checkpoint)
        return model.fit_supplied(supplied, start, checkpoint)
    first = fit(seed)
    if not first["solver_success"]:
        raise ValueError(f"Failed nuisance optimization: {first['solver_message']}")
    best = first
    passes = []
    while True:
        candidate = fit(best)
        if not candidate["solver_success"]:
            raise ValueError(f"Failed nuisance continuation: {candidate['solver_message']}")
        change = best["nll"]-candidate["nll"]
        passes.append({"start_nll": best["nll"], "attained_nll": candidate["nll"],
                       "solver_status": candidate["solver_status"],
                       "residual_calls": candidate["residual_calls"]})
        if change > 0:
            best = candidate
        if change <= 1e-8*(best["nll"]-model.constant):
            break
    return {**best, "initial_fit": {k: first[k] for k in
            ("initial_nll", "nll", "solver_status", "residual_calls", "seconds")},
            "self_continuation": passes}


def profile_file(form, steps, name):
    return WORK/f"f{form}_{steps}_profile_{name}.json"


def reference_on_grid(form, steps):
    data = primary(form)
    if steps == data["n_steps"]:
        return data["reference"]
    if steps == data["grid_comparison"]["comparison_steps"]:
        current_inputs.require_current(data["grid_comparison"]["recipe"]["inputs"])
        return data["grid_comparison"]["reference"]
    raise ValueError("Use a selected or existing comparison grid")


def bracket(rows, reference, name, direction):
    center = reference["engineering"][name]
    ordered = sorted((r for r in rows if direction*(r["value"]-center) >= 0),
                     key=lambda row: direction*(row["value"]-center))
    inside = ordered[0]
    for row in ordered[1:]:
        if row["nll"]-reference["nll"] >= CUT:
            return inside, row
        inside = row
    return inside, None


def basic_record(row):
    return {key: row[key] for key in ("u", "theta", "nll", "engineering",
        "admissibility_margins", "rmse_MPa", "active_assembly_bounds", "value") if key in row}


def retain_row(rows, candidate):
    incumbent = next((row for row in rows if row["value"] == candidate["value"]), None)
    if incumbent is None:
        rows.append(candidate)
        return candidate
    if candidate["nll"] < incumbent["nll"]:
        rows[rows.index(incumbent)] = candidate
        return candidate
    return incumbent


def audit_profile(model, data, path):
    """Continue each final inside/outside pair before interpreting its signs.

    An optimizer stopping at the same exterior coordinate can retain a higher
    nuisance branch. The adjacent interior solution must be continued across
    the actual final bracket. If it lowers the exterior ordinate below the
    cut, advance the bracket; an old exterior sign never certifies itself.
    """
    if data.get("endpoint_continuation_method") == ENDPOINT_METHOD and data["status"] == "complete":
        return data
    name, reference, rows = data["name"], data["reference"], data["rows"]
    data["status"] = "checking nuisance endpoints"
    for check in data.get("independent_constraint_checks", []):
        row = check["fit"]
        if row["solver_success"]:
            retain_row(rows, {**row, "value": row["supplied"][name]})
    checks = data.setdefault("endpoint_nuisance_checks", [])
    endpoints = {}
    def fit(value, seed):
        def checkpoint(row):
            data["in_progress"] = {"value": value, "record": row}
            save(path, data)
        row = {**continued_fit(model, {name: value}, seed, checkpoint), "value": value}
        retained = retain_row(rows, row)
        data.pop("in_progress", None)
        save(path, data)
        return row, retained
    for direction in (-1, 1):
        while True:
            inside, outside = bracket(rows, reference, name, direction)
            if outside is None:
                ending = data["endpoints"][str(direction)]
                if ending["kind"] == "assembly_domain":
                    endpoints[str(direction)] = ending
                    break
                data["status"] = "running"
                save(path, data)
                return run_profile(data["form"], name, data["n_steps"])
            if inside["value"] != reference["engineering"][name]:
                _, inside = fit(inside["value"], inside)
            previous_outside = basic_record(outside)
            candidate, outside = fit(outside["value"], inside)
            check = {"side": "lower" if direction < 0 else "upper",
                     "inside": basic_record(inside), "previous_outside": previous_outside,
                     "continued_from_inside": basic_record(candidate),
                     "retained_outside": basic_record(outside),
                     "solver_success": candidate["solver_success"],
                     "self_continuation": candidate["self_continuation"],
                     "exterior_sign_changed": outside["nll"]-reference["nll"] < CUT}
            checks.append(check)
            save(path, data)
            print("endpoint continuation", data["form"], data["n_steps"], name,
                  direction, previous_outside["nll"]-reference["nll"],
                  "->", outside["nll"]-reference["nll"], flush=True)
            if check["exterior_sign_changed"]:
                continue
            fraction = (reference["nll"]+CUT-inside["nll"])/(outside["nll"]-inside["nll"])
            endpoint = inside["value"]+fraction*(outside["value"]-inside["value"])
            endpoints[str(direction)] = {"kind": "likelihood_crossing", "value": endpoint,
                "representation": "interpolated estimate with an explicitly continued crossing bracket",
                "bracket": sorted((inside["value"], outside["value"])),
                "inside_delta_nll": inside["nll"]-reference["nll"],
                "outside_delta_nll": outside["nll"]-reference["nll"]}
            break
    data.update(status="complete", endpoints=endpoints,
                interval=[endpoints[str(d)]["value"] for d in (-1, 1)],
                classification=("two-sided likelihood crossings" if all(
                    value["kind"] == "likelihood_crossing" for value in endpoints.values()) else "domain-limited"),
                endpoint_continuation_method=ENDPOINT_METHOD)
    save(path, data)
    return data


def run_profile(form, name, steps=None):
    steps = steps or mapping.STEPS[form]
    model = Model(form, steps)
    reference = model.record(reference_on_grid(form, steps)["u"])
    center = reference["engineering"][name]
    path = profile_file(form, steps, name)
    identity = current_inputs.identity(form, steps)
    data = read(path) if path.exists() else {}
    if data and (data["inputs"] != identity or data["reference"] != reference):
        raise ValueError("Stored profile inputs/reference changed")
    if data.get("status") == "complete":
        return audit_profile(model, data, path)
    rows = data.get("rows", [{**reference, "value": center, "source": "current map reference"}])
    data.update(form=form, n_steps=steps, name=name, inputs=identity,
                reference=reference, rows=rows, status="running")
    coarse_path = profile_file(form, mapping.STEPS[form], name)
    coarse = read(coarse_path) if steps != mapping.STEPS[form] and coarse_path.exists() else None
    def fit_row(value, seed=None, force=False):
        incumbent = next((r for r in rows if r["value"] == value), None)
        if incumbent is not None and not force:
            return incumbent
        if seed is None:
            seed = min(rows, key=lambda r: abs(r["value"]-value))
        if coarse and not force:
            warm = min(coarse["rows"], key=lambda r: abs(r["value"]-value))
            if abs(warm["value"]-value) < abs(seed.get("value", center)-value):
                seed = warm
        pending = data.get("in_progress")
        if pending and pending["value"] == value:
            seed = pending["record"]
        def checkpoint(row):
            data["in_progress"] = {"value": value, "record": row}
            save(path, data)
        row = {**continued_fit(model, {name: value}, seed, checkpoint), "value": value}
        if incumbent is None:
            rows.append(row)
        elif row["nll"] < incumbent["nll"]:
            rows[rows.index(incumbent)] = row
        else:
            row = incumbent
        data.pop("in_progress", None)
        save(path, data)
        print("profile", form, steps, name, value, "delta", row["nll"]-reference["nll"], flush=True)
        return row
    # Finish the retained constrained iterate before another side's checkpoint
    # can replace it. This matters when resuming a partly written crossing.
    if data.get("in_progress"):
        pending = data["in_progress"]
        fit_row(pending["value"], pending["record"], force=True)
    J = np.asarray(model.j(jnp.asarray(reference["u"])))
    active = [i for i, value in enumerate(reference["u"])
              if abs(value-model.lower[i]) <= 1e-7 or abs(value-model.upper[i]) <= 1e-7]
    free = np.array([i for i in range(model.space.n_theta) if i not in active])
    cov = np.linalg.pinv(J[:, free].T@J[:, free])
    derivative = np.asarray(model.ej(jnp.asarray(reference["u"])))[elastic.NAMES.index(name), free]
    spacing = float(np.sqrt(2*CUT*derivative@cov@derivative))
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("No local step available for derived profile")
    data["initial_spacing"] = spacing
    save(path, data)
    endings = {}
    for direction in (-1, 1):
        inside, outside = bracket(rows, reference, name, direction)
        offset = max(spacing, direction*(inside["value"]-center)*2)
        while outside is None:
            value = center+direction*offset
            # Exact engineering range bounds, not assembly-coordinate proxies.
            if name == "nu_pp":
                value = float(np.clip(value, np.nextafter(-1., 0.), .45))
            elif name in ("E_p", "E_o"):
                value = float(np.clip(value, np.nextafter(0., 1.), 5e4 if name == "E_p" else 2e4))
            row = fit_row(value, inside)
            inside, outside = bracket(rows, reference, name, direction)
            if outside is None and row["value"] == inside["value"] and name == "nu_pp" and value == .45:
                endings[str(direction)] = {"kind": "assembly_domain", "value": value}
                break
            offset *= 2
        if outside is None:
            continue
        # The outer row is continued directly from the actual inner branch.
        outside = fit_row(outside["value"], inside, force=True)
        if outside["nll"]-reference["nll"] < CUT:
            return run_profile(form, name, steps)
        a, b = sorted((inside["value"], outside["value"]))
        # Do not locate more coordinate digits than the existing nuisance
        # fitter's relative-cost stopping scale resolves. Explicit likelihood
        # brackets, not extra root digits, carry the inference.
        xtol = (b-a)*1e-8*(reference["nll"]-model.constant)/abs(outside["nll"]-inside["nll"])
        root = brentq(lambda value: fit_row(value)["nll"]-reference["nll"]-CUT,
                      a, b, xtol=max(xtol, np.finfo(float).eps))
        inner, outer = bracket(rows, reference, name, direction)
        endings[str(direction)] = {"kind": "likelihood_crossing", "value": root,
            "bracket": sorted((inner["value"], outer["value"])),
            "inside_delta_nll": inner["nll"]-reference["nll"],
            "outside_delta_nll": outer["nll"]-reference["nll"]}
    data.update(status="complete", endpoints=endings,
                interval=[endings[str(d)]["value"] for d in (-1, 1)],
                classification=("two-sided likelihood crossings" if all(
                    v["kind"] == "likelihood_crossing" for v in endings.values()) else "domain-limited"))
    save(path, data)
    return audit_profile(model, data, path)


def run_supply(form, value, steps=None):
    model = Model(form, steps)
    data = primary(form)
    name = f"f{form}_{model.cs.n_steps}_supplied_nu_pp_{value:g}.json"
    path = WORK/name
    old = read(path) if path.exists() else {}
    reference = model.record(reference_on_grid(form, model.cs.n_steps)["u"])
    identity = current_inputs.identity(form, model.cs.n_steps)
    if old.get("status") == "complete" and old.get("inputs") == identity:
        return old
    seeds = [reference]
    if old:
        seeds.append(old.get("in_progress", old.get("fit")))
    coarse_path = WORK/f"f{form}_{mapping.STEPS[form]}_supplied_nu_pp_{value:g}.json"
    if model.cs.n_steps != mapping.STEPS[form] and coarse_path.exists():
        seeds.append(read(coarse_path)["fit"])
    feasible = [model.record(model.project(seed["u"], {"nu_pp": value})) for seed in seeds]
    seed = min(feasible, key=lambda row: row["nll"])
    fitted = continued_fit(model, {"nu_pp": value}, seed,
        checkpoint=lambda row: save(path, {"form": form, "n_steps": model.cs.n_steps,
            "reference": reference, "inputs": identity, "in_progress": row}))
    result = {"form": form, "n_steps": model.cs.n_steps, "inputs": identity,
              "status": "complete", "reference": reference, "fit": fitted}
    save(path, result)
    print(name, "delta_nll", fitted["nll"]-reference["nll"],
          "status", fitted["solver_status"], "seconds", fitted["seconds"], flush=True)
    return result


def run_partial_demo(form):
    path = WORK/f"f{form}_partial_supply.json"
    if path.exists():
        data = read(path)
        if data.get("status") == "complete":
            current_inputs.require_current(data["inputs"])
            return data
    model = Model(form)
    reference = model.record(primary(form)["reference"]["u"])
    supplied = {"nu_pp": .15, "G_op": reference["engineering"]["G_op"]}
    seed = read(WORK/f"f{form}_{mapping.STEPS[form]}_supplied_nu_pp_0.15.json")["fit"]
    fit = continued_fit(model, supplied, seed, checkpoint=lambda row:
        save(path, {"form": form, "in_progress": row}))
    result = {"form": form, "n_steps": mapping.STEPS[form], "status": "complete",
              "inputs": current_inputs.identity(form, mapping.STEPS[form]),
              "reference": reference, "fit": fit,
              "purpose": "Verify simultaneous partial supply; the shear value is the current reference, not a published corridor endpoint."}
    save(path, result)
    return result


def engineering_from_integrated_stiffness(theta):
    theta = np.asarray(theta)
    C = np.asarray(mirror._ti_stiffness_local_j(theta[0], theta[1], theta[3], theta[4], theta[2]))
    S = np.linalg.inv(C)
    ep, eo = 1/S[0, 0], 1/S[2, 2]
    return np.array([ep, eo, 1/(2*S[3, 3]), -S[0, 1]/S[0, 0],
                     -S[0, 2]/S[0, 0], -S[0, 2]/S[2, 2], 1/(2*S[5, 5]), ep/eo])


def check_mapping():
    records = {f"form_{f}": primary(f)["reference"]["theta"] for f in (1, 2)}
    records["nominal"] = mapping.theta_space.load_nominal_theta(ROOT / "compute").tolist()
    checks = {}
    for label, theta in records.items():
        expected = engineering_from_integrated_stiffness(theta)
        actual = np.asarray(elastic.engineering(jnp.asarray(theta)))
        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
        candidates = elastic.assembly_candidates(**dict(zip(
            ("E_p", "E_o", "G_op", "nu_pp", "nu_op"), expected[[0, 1, 2, 3, 5]])))
        recovered = min(candidates, key=lambda a: np.linalg.norm((a-np.asarray(theta[:5]))/np.asarray(theta[:5])))
        np.testing.assert_allclose(recovered, theta[:5], rtol=2e-12, atol=2e-12)
        checks[label] = {"assembly": theta[:5], "engineering": dict(zip(elastic.NAMES, actual.tolist())),
            "inverse": recovered.tolist(),
            "forward_relative_error": float(np.max(abs((actual-expected)/expected))),
            "inverse_relative_error": float(np.max(abs((recovered-np.asarray(theta[:5]))/np.asarray(theta[:5]))))}
    published = {}
    # Source Table 4, Candidate sites: mean intercept + slope at sigma_r0'=5 MPa.
    examples = {
        "Method_A_at_5_MPa": dict(E_p=3*(2100+180*5), E_o=2100+180*5,
                                  G_op=955+82*5, nu_pp=.15, nu_op=.10),
        "Method_B_at_5_MPa": dict(E_p=2*(4000+250*5), E_o=4000+250*5,
                                  G_op=1429+89*5, nu_pp=.05, nu_op=.40)}
    space = mapping.theta_space.space_for_form(1)
    for label, values in examples.items():
        candidates = elastic.assembly_candidates(**values)
        rows = []
        for candidate in candidates:
            theta = np.asarray(records["form_1"]).copy()
            theta[:5] = candidate
            operative = engineering_from_integrated_stiffness(theta)
            for name, expected in values.items():
                np.testing.assert_allclose(operative[elastic.NAMES.index(name)], expected, rtol=2e-12, atol=2e-12)
            u = np.asarray(space.u_from_theta(theta))
            outside = [space.internal_names[i] for i in range(5) if not 0 <= u[i] <= 1]
            rows.append({"assembly": candidate.tolist(), "outside_assembly_box": outside,
                         "engineering_from_actual_stiffness": dict(zip(elastic.NAMES, operative.tolist())),
                         "physical_spd_margin": float(elastic.positive_definiteness_margin(theta))})
        published[label] = {"target": values, "candidates": rows}
    source = ROOT.parent / "references/papers/R02_Nordas_2024/full.md"
    return {"current_sets": checks, "published_full_supply": published,
            "source": {"path": "../references/papers/R02_Nordas_2024/full.md",
                       "sha256": current_inputs.digest(source),
                       "passages": "Section 6.1, Table 4 Candidate sites, Appendix A equations 8-11; nu_op denotes normal loading and in-plane strain."}}


def derivative_check(model, row, name, residual_check=True):
    u = np.asarray(row["u"])
    value = float(model.e(jnp.asarray(u))[elastic.NAMES.index(name)])
    free, original, differential = model.scalar_chart(name, value, u)
    restored = original(u[free])
    np.testing.assert_allclose(restored, u, rtol=0, atol=2e-12)
    T = differential(restored)
    A = np.asarray(model.ej(jnp.asarray(restored)))[elastic.NAMES.index(name)]
    np.testing.assert_allclose(A@T, 0, rtol=0, atol=2e-10)
    results = []
    J = np.asarray(model.j(jnp.asarray(restored)))@T if residual_check else None
    for h in ((1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11)
              if residual_check else (1e-4, 1e-5, 1e-6)):
        F = numerics.absolute_fd_jacobian(original, u[free], h, model.lower[free], model.upper[free])
        result = {"step": h, "chart_jacobian_relative_error": float(np.linalg.norm(F-T)/np.linalg.norm(T))}
        if residual_check:
            D = numerics.absolute_fd_jacobian(lambda x: model.r(jnp.asarray(original(x))),
                    u[free], h, model.lower[free], model.upper[free])
            result["real_residual_jacobian_relative_error"] = float(np.linalg.norm(D-J)/np.linalg.norm(J))
        results.append(result)
    return {"quantity": name, "value": value, "n_steps": model.cs.n_steps, "record": row,
            "original_constraint_pivot": float(A[4]),
            "maximum_roundtrip_error_u": float(np.max(abs(restored-u))),
            "sweep": results}


def check_form(form):
    path = WORK/f"f{form}_derivatives.json"
    model = Model(form)
    reference = model.record(primary(form)["reference"]["u"])
    checks = {}
    for name in QUANTITIES:
        checks[f"reference:{name}"] = derivative_check(model, reference, name, residual_check=name == "nu_pp")
    profile = read(profile_file(form, mapping.STEPS[form], "nu_pp"))
    for direction in (-1, 1):
        inside, outside = bracket(profile["rows"], profile["reference"], "nu_pp", direction)
        checks[f"nu_pp_crossing:{direction}"] = derivative_check(model, outside, "nu_pp")
    for value in (.05, .15):
        supply = read(WORK/f"f{form}_{mapping.STEPS[form]}_supplied_nu_pp_{value:g}.json")
        checks[f"supplied:{value}"] = derivative_check(model, supply["fit"], "nu_pp")
    result = {"form": form, "inputs": current_inputs.identity(form, mapping.STEPS[form]),
              "checks": checks}
    demo = read(WORK/f"f{form}_partial_supply.json")["fit"]
    equality = model.constraints(demo["supplied"])[-1]
    u = np.asarray(demo["u"])
    A = equality["jac"](u)
    F = numerics.absolute_fd_jacobian(equality["fun"], u, 1e-5, model.lower, model.upper)
    result["simultaneous_partial_constraint"] = {
        "supplied": demo["supplied"], "jacobian_rank": int(np.linalg.matrix_rank(A)),
        "fd_relative_error": float(np.linalg.norm(F-A)/np.linalg.norm(A))}
    if form == 2:
        fine = Model(form, mapping.COMPARISON_STEPS[form])
        block = read(profile_file(form, fine.cs.n_steps, "nu_pp"))
        _, outside = bracket(block["rows"], block["reference"], "nu_pp", 1)
        result["comparison_nu_pp_upper_derivative"] = derivative_check(fine, outside, "nu_pp")
    save(path, result)
    print("derivatives completed", form, flush=True)
    return result


def job(task):
    kind, form, steps, value = task
    if kind == "profile":
        return run_profile(form, value, steps)
    if kind == "supply":
        return run_supply(form, value, steps)
    if kind == "partial":
        return run_partial_demo(form)
    if kind == "derivatives":
        return check_form(form)
    raise ValueError(task)


def run_jobs(tasks, workers):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(job, task): task for task in tasks}
        for future in as_completed(futures):
            future.result()
            print("completed", futures[future], flush=True)


def run_all(workers=4, comparison_only=False):
    if not comparison_only:
        run_jobs([("profile", f, mapping.STEPS[f], q) for f in (1, 2) for q in QUANTITIES], workers)
        run_jobs([("supply", f, mapping.STEPS[f], v) for f in (1, 2) for v in (.05, .15)], workers)
    run_jobs([("profile", f, mapping.COMPARISON_STEPS[f], q) for f in (1, 2) for q in QUANTITIES], workers)
    run_jobs([("supply", f, mapping.COMPARISON_STEPS[f], v) for f in (1, 2) for v in (.05, .15)], workers)
    run_jobs([("partial", f, None, None) for f in (1, 2)], workers)
    run_jobs([("derivatives", f, None, None) for f in (1, 2)], workers)
    save(WORK/"mapping.json", check_mapping())


def relative(path):
    return Path(path).relative_to(ROOT).as_posix()


def profile_summary(data, path):
    boundaries = {}
    for direction in (-1, 1):
        inside, outside = bracket(data["rows"], data["reference"], data["name"], direction)
        witness = outside or inside
        boundaries["lower" if direction < 0 else "upper"] = {
            "kind": data["endpoints"][str(direction)]["kind"],
            "crossing_bracket": data["endpoints"][str(direction)].get("bracket"),
            "inside_delta_nll": inside["nll"]-data["reference"]["nll"],
            "outside_delta_nll": None if outside is None else outside["nll"]-data["reference"]["nll"],
            "active_nuisance_assembly_bounds": witness["active_assembly_bounds"],
            "admissibility_margins": witness["admissibility_margins"]}
    return {"estimate": data["reference"]["engineering"][data["name"]],
            "interval": data["interval"], "classification": data["classification"],
            "endpoint_representation": "interpolated estimates with explicitly continued likelihood brackets",
            "endpoint_continuation_method": data["endpoint_continuation_method"],
            "corrected_exterior_signs": sum(check["exterior_sign_changed"]
                                            for check in data["endpoint_nuisance_checks"]),
            "boundaries": boundaries, "evidence": relative(path)}


def summarize():
    forms = {}
    for form in (1, 2):
        source = primary(form)
        selected, comparison = {}, {}
        for name in QUANTITIES:
            for destination, steps in ((selected, mapping.STEPS[form]), (comparison, mapping.COMPARISON_STEPS[form])):
                path = profile_file(form, steps, name)
                data = read(path)
                if data["status"] != "complete":
                    raise ValueError(f"Incomplete profile: {path}")
                destination[name] = profile_summary(data, path)
        shear = source["per_parameter"]["G_op"]
        shear_fine = source["grid_comparison"]["parameters"]["G_op"]
        supplied = {}
        for value in (.05, .15):
            levels = {}
            for level, steps in (("selected", mapping.STEPS[form]), ("comparison", mapping.COMPARISON_STEPS[form])):
                path = WORK/f"f{form}_{steps}_supplied_nu_pp_{value:g}.json"
                data = read(path)
                row = data["fit"]
                levels[level] = {"nll": row["nll"], "reference_nll": data["reference"]["nll"],
                    "delta_nll": row["nll"]-data["reference"]["nll"],
                    "rmse_MPa": row["rmse_MPa"], "reference_rmse_MPa": data["reference"]["rmse_MPa"],
                    "engineering": row["engineering"], "assembly_elastic": row["theta"][:5],
                    "outside_nominal_profile_cut": row["nll"]-data["reference"]["nll"] > CUT,
                    "constraint_residual": row["constraint_residual"],
                    "evidence": relative(path)}
            supplied[str(value)] = levels
        forms[str(form)] = {"selected_steps": mapping.STEPS[form], "comparison_steps": mapping.COMPARISON_STEPS[form],
            "primary_reference": {"path": relative(mapping.output(form)), "nll": source["reference"]["nll"]},
            "operative_profiles": selected, "comparison_profiles": comparison,
            "supplied_in_plane_poisson": supplied,
            "unchanged_G_op": {"reason_for_reuse": "The integrated Mandel compliance has S44=1/(2*G_op) identically; the old profile fixes exactly this engineering quantity.",
                "estimate_MPa": source["reference"]["theta"][2],
                "selected_interval_MPa": shear["interval_internal_units"],
                "comparison_interval_MPa": shear_fine["interval_internal_units"],
                "selected_classification": shear["classification"], "comparison_classification": shear_fine["classification"],
                "evidence": [relative(mapping.output(form)), "selfrev-r3/evidence/SR-14_numerics.py"]},
            "partial_supply_example": relative(WORK/f"f{form}_partial_supply.json"),
            "new_derivatives": relative(WORK/f"f{form}_derivatives.json")}
    derivative_results = {}
    for form in (1, 2):
        derivative_results[str(form)] = {}
        for key, check in read(WORK/f"f{form}_derivatives.json")["checks"].items():
            available = [row for row in check["sweep"] if "real_residual_jacobian_relative_error" in row]
            if available:
                best = min(available, key=lambda r: r["real_residual_jacobian_relative_error"])
                derivative_results[str(form)][key] = {"step": best["step"],
                    "relative_error": best["real_residual_jacobian_relative_error"]}
        if form == 2:
            check = read(WORK/f"f{form}_derivatives.json")["comparison_nu_pp_upper_derivative"]
            best = min(check["sweep"], key=lambda row: row["real_residual_jacobian_relative_error"])
            derivative_results[str(form)]["comparison:nu_pp_upper"] = {
                "step": best["step"], "relative_error": best["real_residual_jacobian_relative_error"]}
    all_crossings = all(row["classification"] == "two-sided likelihood crossings"
        for form in forms.values() for level in ("operative_profiles", "comparison_profiles")
        for row in form[level].values())
    supplied_excluded = all(row["outside_nominal_profile_cut"]
        for form in forms.values() for levels in form["supplied_in_plane_poisson"].values()
        for row in levels.values())
    cs = mapping.curves.load_measured_curveset(ROOT / "compute", n_steps=mapping.STEPS[1], effective_stress_path=True)
    repaired = read(profile_file(2, mapping.COMPARISON_STEPS[2], "nu_pp")).get("independent_constraint_checks", [])
    return {
        "scope": "Connected derived-quantity likelihood profiles of the five measured axial curves, conditional on the existing assembly box, corrected specimen geometry, shear-free effective-stress controller, and sigma_q=1 MPa independent Gaussian likelihood; cut Delta NLL=1.92.",
        "definitions": {"assembly": "p=theta[0], o=theta[1], g=theta[2], v=theta[3], w=theta[4]",
            "stiffness": "d=1-v-2*w^2; H=C11+C12=p/d; K=C11-C12=p/(1+v); F=C13=w*(p+o)/(2*d); C=C33=(1-v)*o/d; D=H*C-2*F^2.",
            "compliance": "s11=(C/D+1/K)/2; s12=(C/D-1/K)/2; s13=-F/D; s33=H/D.",
            "engineering": "E_p=1/s11, E_o=1/s33, nu_pp=-s12/s11, nu_po=-s13/s11, nu_op=-s13/s33, G_op=g, G_pp=K/2; nu_po/E_p=nu_op/E_o.",
            "directions": "nu_op (nu31) is in-plane strain under bedding-normal loading; nu_po (nu13) is bedding-normal strain under in-plane loading.",
            "physical_admissibility": "Positive moduli, -1<v<1, d>0 and 1-v-w^2*(p+o)^2/(2*p*o)>0 ensure the integrated stiffness is SPD. The original finite assembly box and form-II fabric constraint remain separate constraints."},
        "current_observations": {"curve_keys": cs.keys, "q_shape": list(cs.q_meas.shape),
            "bedding_angles_deg": cs.theta_deg.tolist(), "bedding_normals": np.asarray(cs.n_hat).tolist(),
            "initial_effective_radial_stress_MPa": cs.sigma_r0.tolist(),
            "effective_stress_path": cs.on_effective_stress_path, "likelihood_cut": CUT},
        "engineering_inference": (("E_p, E_o, nu_pp and both directional cross-plane Poisson ratios have two likelihood crossings in both forms on both grids. "
            if all_crossings else "The computed endpoint kinds and classes are listed individually for each engineering quantity and grid. ")
            + "The proper operative nu_pp profile is not the mapped assembly-nu_pp interval. G_op is unchanged and reuses its valid profile. These are domain-conditional results: active nuisance assembly bounds, especially v=0.45 on the upper nu_pp side, are reported for every endpoint. No endpoint is labelled a data-only global material bound."),
        "external_value_consequence": ("Both published in-plane values, 0.15 (Method A) and 0.05 (Method B), incur a refitted likelihood loss exceeding 1.92 under both forms and both grids. Carrying either is an additional, consequential modeling assumption, not a cost-free completion of a quantity left unmeasured by the axial programme."
            if supplied_excluded else "The effect of each supplied in-plane value is reported by form and grid; a supplied value inside the cut is distinguished from one outside it."),
        "full_conversion": {"target": "Build the symmetric normal compliance from supplied (E_p,E_o,nu_pp,nu_op) with s13=-nu_op/E_o; require SPD and invert to H,K,F,C. G_op is copied.",
            "inverse_polynomial": "t=C/H, h=2*F/H. Enumerate real r>0 satisfying ((H+K)*t-2*K*r)*(1+r)^2-2*H*h^2*r=0. Recover v=1-t/r, p=K*(1+v), o=r*p, w=h/(1+r). Keep physically admissible candidates and check the declared assembly domain.",
            "partial_supply": "For any supplied subset A, minimize the unchanged measured NLL over every remaining coordinate subject to engineering_a(theta)=supplied_a for a in A, the original assembly box and admissibility. This does not insert engineering values directly into assembly coordinates. Each of the five profiled quantities is enforced exactly by eliminating w in this connected anisotropic basin; its implicit derivative is du4/duj=-(partial engineering_a/partial uj)/(partial engineering_a/partial u4). G_op retains its unchanged assembly coordinate. Multiple supplied quantities use their simultaneous analytic equality Jacobian.",
            "explicit_in_plane_constraint": "For target n=operative nu_pp and alpha=(p+o)^2/(2*p*o), w^2=2*(1-v)*(v-n)/(alpha*(1-n)*(1+v)-2*(1+n)*(1-v)). Take the positive root and enforce the original w bounds and SPD. This chart is used only where its pivot is nonzero; the equality-constraint formulation remains the definition.",
            "independent_supply_set": "Use subsets of (E_p,E_o,G_op,nu_pp,nu_op). nu_po follows by reciprocity; it can instead be constrained directly, as in its profile. Redundantly supplied directional ratios must satisfy nu_po/E_p=nu_op/E_o.",
            "domain_limitation": "The full published Method-B mean set at 5 MPa is physically SPD but requires assembly w about 0.517, above the retained 0.45 bound. A complete physical set can therefore be outside this assembly domain even when a Poisson-only supply is feasible.",
            "evidence": relative(WORK/"mapping.json")},
        "forms": forms, "real_model_derivative_replays": derivative_results,
        "derivative_interpretation": "Absolute unit-box finite-difference sweeps check the new constraint charts and the real measured-residual Jacobian after the implicit coordinate elimination. Two constrained Form-I fits require steps down to 1e-10; the errors decrease there and rise at 1e-11. The full ladders, including larger-step discrepancies, are retained. Simultaneous partial-supply equality Jacobians are checked separately at both real-data fits.",
        "nuisance_optimization": {"method": ENDPOINT_METHOD,
            "finding": "Self termination at a cached exterior coordinate did not by itself establish the profile crossing. Adjacent interior continuation found lower feasible nuisance solutions, and any changed exterior signs were advanced before reporting a bracket.",
            "form_II_comparison_diagnostic": [{"method": row["method"],
                "target": row["fit"]["supplied"], "nll": row["fit"]["nll"]} for row in repaired],
            "evidence": relative(profile_file(2, mapping.COMPARISON_STEPS[2], "nu_pp"))},
        "parent_coordination": "The historical nominal vector inserts Method-A numbers as assembly coordinates; its actual engineering constants therefore differ from the published physical set. mapping.json gives both that existing nominal law and the exact physical Method-A conversion. Existing baseline predictions were not replaced by this evidence slice.",
        "reference_changes": "No primary reference or existing result dependency was changed. The new engineering inferences supersede only the inference obtained by transferring assembly-coordinate labels to engineering constants."}


def verify_derivative(model, check):
    u = np.asarray(check["record"]["u"])
    free, original, differential = model.scalar_chart(check["quantity"], check["value"], u)
    T = differential(u)
    np.testing.assert_allclose(original(u[free]), u, rtol=0, atol=2e-12)
    A = np.asarray(model.ej(jnp.asarray(u)))[elastic.NAMES.index(check["quantity"])]
    np.testing.assert_allclose(A@T, 0, rtol=0, atol=2e-10)
    F = numerics.absolute_fd_jacobian(original, u[free], 1e-6, model.lower[free], model.upper[free])
    np.testing.assert_allclose(F, T, rtol=1e-5, atol=1e-6)
    samples = [r for r in check["sweep"] if "real_residual_jacobian_relative_error" in r]
    if samples:
        expected = np.asarray(model.j(jnp.asarray(u)))@T
        best = min(samples, key=lambda r: r["real_residual_jacobian_relative_error"])
        D = numerics.absolute_fd_jacobian(lambda x: model.r(jnp.asarray(original(x))),
            u[free], best["step"], model.lower[free], model.upper[free])
        error = float(np.linalg.norm(D-expected)/np.linalg.norm(expected))
        np.testing.assert_allclose(error, best["real_residual_jacobian_relative_error"], rtol=.02, atol=1e-7)


def verify_data(replay_derivatives=True):
    mapped = check_mapping()
    expected_mapping = read(WORK/"mapping.json")
    for label, row in mapped["current_sets"].items():
        for name, value in row["engineering"].items():
            np.testing.assert_allclose(value, expected_mapping["current_sets"][label]["engineering"][name], rtol=2e-12, atol=2e-12)
        for key in ("assembly", "inverse"):
            np.testing.assert_allclose(row[key], expected_mapping["current_sets"][label][key], rtol=2e-12, atol=2e-12)
    for label, row in mapped["published_full_supply"].items():
        stored = expected_mapping["published_full_supply"][label]
        assert row["target"] == stored["target"]
        assert len(row["candidates"]) == len(stored["candidates"])
        for actual, expected in zip(row["candidates"], stored["candidates"]):
            assert actual["outside_assembly_box"] == expected["outside_assembly_box"]
            np.testing.assert_allclose(actual["assembly"], expected["assembly"], rtol=2e-12, atol=2e-12)
            np.testing.assert_allclose(actual["physical_spd_margin"], expected["physical_spd_margin"], rtol=2e-12, atol=2e-12)
            for name, value in actual["engineering_from_actual_stiffness"].items():
                np.testing.assert_allclose(value, expected["engineering_from_actual_stiffness"][name], rtol=2e-12, atol=2e-12)
    assert mapped["source"] == expected_mapping["source"]
    replayed = {}
    count = 0
    for form in (1, 2):
        source = primary(form)
        for steps in (mapping.STEPS[form], mapping.COMPARISON_STEPS[form]):
            model = Model(form, steps)
            reference = model.record(reference_on_grid(form, steps)["u"])
            def replay(row):
                nonlocal count
                key = (form, steps, tuple(row["u"]))
                if key in replayed:
                    actual = replayed[key]
                    np.testing.assert_allclose(actual["nll"], row["nll"], rtol=1e-10, atol=1e-8)
                    return actual
                actual = model.record(row["u"])
                np.testing.assert_allclose(actual["nll"], row["nll"], rtol=1e-10, atol=1e-8)
                np.testing.assert_allclose(actual["theta"], row["theta"], rtol=2e-12, atol=2e-12)
                if "engineering" in row:
                    for name, value in row["engineering"].items():
                        np.testing.assert_allclose(actual["engineering"][name], value, rtol=2e-12, atol=2e-12)
                if "rmse_MPa" in row:
                    np.testing.assert_allclose(actual["rmse_MPa"], row["rmse_MPa"], rtol=1e-10, atol=1e-10)
                integrated = engineering_from_integrated_stiffness(actual["theta"])
                np.testing.assert_allclose(list(actual["engineering"].values()), integrated, rtol=2e-12, atol=2e-12)
                assert np.all(np.asarray(row["u"]) >= model.lower-1e-12)
                assert np.all(np.asarray(row["u"]) <= model.upper+1e-12)
                assert min(actual["admissibility_margins"]) > 0
                replayed[key] = actual
                count += 1
                return actual
            replay(reference_on_grid(form, steps))
            for name in QUANTITIES:
                path = profile_file(form, steps, name)
                block = read(path)
                current_inputs.require_current(block["inputs"])
                assert block["status"] == "complete"
                assert block["endpoint_continuation_method"] == ENDPOINT_METHOD
                np.testing.assert_allclose(block["reference"]["nll"], reference["nll"], rtol=1e-10, atol=1e-8)
                for direction in (-1, 1):
                    inner, outer = bracket(block["rows"], block["reference"], name, direction)
                    inner_actual = replay(inner)
                    if outer is None:
                        assert block["endpoints"][str(direction)]["kind"] != "likelihood_crossing"
                        continue
                    assert outer["solver_success"] and outer["self_continuation"]
                    outer_actual = replay(outer)
                    assert inner_actual["nll"]-reference["nll"] < CUT
                    assert outer_actual["nll"]-reference["nll"] >= CUT
                    for row, actual in ((inner, inner_actual), (outer, outer_actual)):
                        np.testing.assert_allclose(actual["engineering"][name], row["value"], rtol=2e-12, atol=2e-12)
                    bounds = sorted((inner["value"], outer["value"]))
                    assert bounds[0] <= block["interval"][0 if direction < 0 else 1] <= bounds[1]
                    final_check = [check for check in block["endpoint_nuisance_checks"]
                        if check["side"] == ("lower" if direction < 0 else "upper")][-1]
                    assert not final_check["exterior_sign_changed"]
                    assert final_check["inside"]["value"] == inner["value"]
                    assert final_check["retained_outside"]["value"] == outer["value"]
                for check in block["endpoint_nuisance_checks"]:
                    assert check["solver_success"] and check["self_continuation"]
                    for key in ("inside", "previous_outside", "continued_from_inside", "retained_outside"):
                        replay(check[key])
                    assert check["exterior_sign_changed"] == (
                        check["retained_outside"]["nll"]-reference["nll"] < CUT)
                for check in block.get("independent_constraint_checks", []):
                    replay(check["seed"])
                    replay(check["fit"])
                print("verified profile", form, steps, name, block["interval"], flush=True)
            for value in (.05, .15):
                data = read(WORK/f"f{form}_{steps}_supplied_nu_pp_{value:g}.json")
                if "inputs" in data:
                    current_inputs.require_current(data["inputs"])
                actual = replay(data["fit"])
                np.testing.assert_allclose(actual["engineering"]["nu_pp"], value, rtol=2e-12, atol=2e-12)
                assert data["fit"]["solver_success"] and data["fit"]["self_continuation"]
                np.testing.assert_allclose(data["reference"]["nll"], reference["nll"], rtol=1e-10, atol=1e-8)
            shear_block = (source["per_parameter"] if steps == mapping.STEPS[form]
                           else source["grid_comparison"]["parameters"])["G_op"]
            for endpoint in shear_block["interval_u"]:
                row = min(shear_block["rows"], key=lambda r: abs(r["u_fixed"]-endpoint))
                actual = replay(row)
                np.testing.assert_allclose(actual["engineering"]["G_op"], actual["theta"][2], rtol=2e-12)
            if steps == mapping.STEPS[form]:
                demo = read(WORK/f"f{form}_partial_supply.json")
                current_inputs.require_current(demo["inputs"])
                actual = replay(demo["fit"])
                for key, value in demo["fit"]["supplied"].items():
                    np.testing.assert_allclose(actual["engineering"][key], value, rtol=2e-12, atol=2e-12)
                equality = model.constraints(demo["fit"]["supplied"])[-1]
                u = np.asarray(actual["u"])
                A = equality["jac"](u)
                F = numerics.absolute_fd_jacobian(equality["fun"], u, 1e-5, model.lower, model.upper)
                np.testing.assert_allclose(F, A, rtol=1e-5, atol=1e-6)
                assert np.linalg.matrix_rank(A) == len(demo["fit"]["supplied"])
                derivative_data = read(WORK/f"f{form}_derivatives.json")
                partial_check = derivative_data["simultaneous_partial_constraint"]
                assert partial_check["supplied"] == demo["fit"]["supplied"]
                assert partial_check["jacobian_rank"] == np.linalg.matrix_rank(A)
                np.testing.assert_allclose(partial_check["fd_relative_error"], np.linalg.norm(F-A)/np.linalg.norm(A), rtol=.02, atol=1e-9)
                if replay_derivatives:
                    current_inputs.require_current(derivative_data["inputs"])
                    for key, check in derivative_data["checks"].items():
                        replay(check["record"])
                        verify_derivative(model, check)
                    print("verified real-model constrained derivatives", form, flush=True)
            elif form == 2 and replay_derivatives:
                derivative_data = read(WORK/f"f{form}_derivatives.json")
                check = derivative_data["comparison_nu_pp_upper_derivative"]
                replay(check["record"])
                verify_derivative(model, check)
                print("verified comparison-grid upper nu_pp constrained derivative", flush=True)
    return {"replayed_likelihood_witnesses": count, "mapping_and_inverse": "replayed against integrated stiffness",
            "likelihood_crossings": "replayed on both sides of the cut", "new_constraint_derivatives": "replayed" if replay_derivatives else "stored step sweeps"}


def finalize():
    verification = verify_data(replay_derivatives=True)
    findings = summarize()
    result = {"kind": "iterative_exec_result", "schema_version": 1,
        "status": "completed", "stage_id": "revision-r3-sr03-operative-elastic",
        "task": "sr03-operative-elastic", "prompt_sha256": hashlib.sha256(PROMPT.read_bytes()).hexdigest(),
        "scientific_findings": findings, "verification": verification,
        "producer_sources": {p: current_inputs.digest(ROOT/p) for p in (
            "selfrev-r3/evidence/SR-03_operative_elastic.py", "compute/src/tfenn_invcalib/operative_elastic.py")},
        "actual_changed_files": ["selfrev-r3/evidence/SR-03_operative_elastic.py",
            "compute/src/tfenn_invcalib/operative_elastic.py",
            *sorted(relative(p) for p in WORK.glob("*.json") if p.name != "result.json"),
            relative(WORK/"result.json")],
        "remaining_dependencies": [],
        "verify_cmd": "python selfrev-r3/evidence/SR-03_operative_elastic.py --verify"}
    save(WORK/"result.json", result)
    print("completed", relative(WORK/"result.json"), flush=True)


def verify():
    result = read(WORK/"result.json")
    assert result["kind"] == "iterative_exec_result" and result["schema_version"] == 1
    assert result["status"] == "completed"
    assert result["stage_id"] == "revision-r3-sr03-operative-elastic"
    assert result["task"] == "sr03-operative-elastic"
    assert result["prompt_sha256"] == hashlib.sha256(PROMPT.read_bytes()).hexdigest()
    for path, expected in result["producer_sources"].items():
        assert current_inputs.digest(ROOT/path) == expected, f"Changed producer: {path}"
    verification = verify_data()
    assert summarize() == result["scientific_findings"]
    assert result["remaining_dependencies"] == []
    print("PASS SR-03 mappings, current references, constrained real-data likelihood witnesses, numerical derivatives, both-grid inference and supplied constants", verification, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--form", type=int, choices=(1, 2), default=1)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--supply", type=float)
    parser.add_argument("--quantity", choices=QUANTITIES)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--comparison", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--checks", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.supply is not None:
        run_supply(args.form, args.supply, args.steps)
    if args.quantity:
        run_profile(args.form, args.quantity, args.steps)
    if args.run or args.comparison:
        run_all(args.workers, comparison_only=args.comparison)
        finalize()
    if args.checks:
        save(WORK/"mapping.json", check_mapping())
        check_form(args.form)
    if args.finalize:
        finalize()
    if args.verify:
        verify()


if __name__ == "__main__":
    main()
