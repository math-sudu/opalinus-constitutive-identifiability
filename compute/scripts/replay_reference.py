"""Replay the paper's exact numerical states from compute/reference_inputs.json."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "compute/src"))


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


mapping = module("reference_mapping", "selfrev-r2/evidence/SR-09a_map.py")
np, jnp, jax = mapping.np, mapping.jnp, mapping.jax
from tfenn_invcalib import bounded_fit, current_inputs, mirror, operative_elastic
from profile_search import compare_starts, update_reference


def load():
    path = ROOT / "compute/reference_inputs.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    current_inputs.require_current({"files": data["inputs_sha256"]})
    return data


def replay(model, row):
    actual = model.record(np.asarray(row["u"]))
    np.testing.assert_allclose(actual["theta"], row["theta"], rtol=3e-12, atol=3e-10)
    np.testing.assert_allclose(actual["nll"], row["nll"], rtol=1e-10, atol=1e-8)
    return actual


def endpoint(data, name, side):
    i = mapping.theta_space.space_for_form(data["form"]).internal_names.index(name)
    center = data["reference"]["u"][i]
    rows = [row for row in data["coordinate_profiles"][name]["rows"]
            if side * (row["value"] - center) > 0]
    if not rows:
        return {**data["reference"], "value": center}
    return min(rows, key=lambda row: abs(row["nll"]-data["reference"]["nll"]-mapping.CUT))


def summary(data):
    space = mapping.theta_space.space_for_form(data["form"])
    eng = np.asarray(operative_elastic.engineering(jnp.asarray(data["reference"]["theta"])))
    return {"n_steps": data["n_steps"], "reference": data["reference"],
            "physical_parameter_order": list(space.theta_names),
            "internal_coordinate_order": list(space.internal_names),
            "operative_constants": dict(zip(operative_elastic.NAMES, eng.tolist())),
            "coordinate_profiles": {name: {k: block[k] for k in ("interval", "classification")}
                                    for name, block in data["coordinate_profiles"].items()},
            "operative_profiles": {name: {k: block[k] for k in ("interval", "classification")}
                                   for name, block in data["operative_profiles"].items()}}


def check(data):
    model = mapping.Model(data["form"], data["n_steps"])
    np.testing.assert_array_equal(model.cs.theta_deg, [90, 0, 30, 90, 0])
    records = {"reference": data["reference"]}
    for name in ("c_j", "k_sharpness"):
        for side in (-1, 1):
            records[f"{name}_{side}"] = endpoint(data, name, side)
    errors = {name: replay(model, row)["nll"]-row["nll"] for name, row in records.items()}
    residual = np.asarray(model.r(jnp.asarray(data["reference"]["u"])))
    return {"reference_nll": data["reference"]["nll"], "nll_replay_errors": errors,
            "rmse_MPa": float(np.sqrt(np.mean(residual**2))),
            "q_at_reference_MPa": (model.cs.q_meas+residual.reshape(model.cs.q_meas.shape)).tolist()}


def profiles(data, names, operative=False):
    model = mapping.Model(data["form"], data["n_steps"])
    reference = replay(model, data["reference"])
    source = data["operative_profiles" if operative else "coordinate_profiles"]
    result = {}
    for name in names or source:
        block = source[name]
        rows = []
        for row in block["rows"]:
            actual = replay(model, row)
            if operative:
                index = operative_elastic.NAMES.index(name)
                fixed = float(operative_elastic.engineering(jnp.asarray(actual["theta"]))[index])
            else:
                fixed = actual["u"][model.space.internal_names.index(name)]
            np.testing.assert_allclose(fixed, row["value"], rtol=2e-7, atol=2e-8)
            rows.append({"value": row["value"], "delta_nll": actual["nll"]-reference["nll"]})
        result[name] = {"interval": block["interval"], "classification": block["classification"], "rows": rows}
        print(f"Replayed {name}: {len(rows)} nuisance states", file=sys.stderr, flush=True)
    return result


def cross_starts(data):
    model = mapping.Model(data["form"], data["n_steps"])
    result = []
    for attempt in data["cross_start_comparisons"]:
        row = {k: attempt[k] for k in ("case", "coordinate", "value", "status")}
        if attempt["status"] == "terminated":
            actual = replay(model, attempt["fit"])
            index = model.space.internal_names.index(attempt["coordinate"])
            np.testing.assert_allclose(actual["u"][index], attempt["value"], rtol=0, atol=1e-12)
            row["delta_nll"] = actual["nll"]-data["reference"]["nll"]
        else:
            row["error"] = attempt["error"]
        result.append(row)
    return result


def profile_point(data, name, side):
    model = mapping.Model(data["form"], data["n_steps"])
    seed = endpoint(data, name, side)
    seeds = [("retained_nuisance", seed), ("reference", data["reference"])]
    seeds += [(s["label"], s) for s in data["profile_starts"]]
    result = compare_starts(lambda start: model.fit(name, seed["value"], start), seeds)
    reference = update_reference(data["reference"], result["fit"],
        lambda start: bounded_fit.fit(model, start["theta"]))
    changed = reference["nll"] < data["reference"]["nll"]
    return {**result, "coordinate": name, "fixed_u": seed["value"],
            "reference": reference, "reference_changed": changed,
            "profiles_require_rebuild": changed,
            "delta_nll": result["fit"]["nll"]-reference["nll"]}


def responses(data):
    paths = module("reference_paths", "selfrev-r2/evidence/SR-09a_downstream.py")
    model = mapping.Model(data["form"], data["n_steps"])
    theta = np.asarray(data["reference"]["theta"])
    off = theta.copy()
    off[8] = 1e6
    fn = paths.path_function(data["form"], data["n_steps"])
    def run(bed, pressure, stations):
        result = {label: paths.path_row(fn, vector, bed, pressure, stations)
                  for label, vector in (("joint_on", theta), ("joint_off", off))}
        if not all(row["valid"] for row in result.values()):
            raise ValueError(f"Unconverged response at bedding angle {bed}")
        return result
    measured = {key: run(model.cs.theta_deg[i], model.cs.p_conf_seq[i], model.cs.eps_stations[i])
                for i, key in enumerate(model.cs.keys)}
    fn = paths.path_function(data["form"], data["n_steps"], data["orientation_lateral_max_evaluations"])
    sweep = [{"bedding_deg": float(angle), **run(angle, np.full(data["n_steps"], 5.), model.cs.eps_stations[0])}
             for angle in np.arange(0., 91., 5.)]
    inclined = measured["fig2c_CU_Z"]
    on, off_span = inclined["joint_on"]["gamma_final"], inclined["joint_off"]["gamma_final"]
    return {"measured_paths": measured, "drained_5_MPa": sweep,
            "CU_Z_net_final_span_decrease_percent": 100*(on-off_span)/on}


def deformation(data, measurements):
    cs = mapping.curves.load_measured_curveset(ROOT/"compute", n_steps=data["n_steps"], effective_stress_path=True)
    forward = mirror.mirror_forward_path if data["form"] == 1 else mirror.mirror_forward_path_f2
    theta = jnp.asarray(data["reference"]["theta"])
    @jax.jit
    def run(matrix, normal, radial):
        return forward(theta, matrix, normal, radial, cs.d_eps11, n_steps=cs.n_steps, early_exit=True)
    result = {}
    for i, key in enumerate(cs.keys):
        out = {k: np.asarray(v) for k, v in run(cs.M_bond[i], cs.n_hat[i], cs.p_conf_seq[i]).items()}
        if not bool(out["path_converged"]):
            raise ValueError(f"Unconverged deformation path: {key}")
        axial, volume = out["eps"][:, 0]*100, out["eps"][:, :3].sum(axis=1)*100
        row = {"volume_at_0_1_percent_axial_strain_pct": float(np.interp(.1, axial, volume)),
               "volume_at_measured_end_pct": float(np.interp(cs.eps_stations[i][-1]*100, axial, volume))}
        if not cs.is_cu[i]:
            panel = next(p for p in measurements["panels"] if p["q_key"] == key)
            predicted = np.interp(panel["eps_a_pct"], axial, volume)
            mask = np.asarray(panel["comparison_mask"])
            row.update(volume_at_reading_stations_pct=predicted.tolist(),
                measured_window_end_pct=panel["comparison_volume_end_pct"],
                predicted_window_end_pct=float(predicted[mask][-1]),
                RMSE_pct=float(np.sqrt(np.mean((predicted-np.asarray(panel["eps_vol_pct"]))[mask]**2))))
        result[key] = row
    return result


def residual_cases(data):
    model = mapping.Model(data["form"], data["n_steps"])
    result = {}
    for scenario, case in data["residual_cases"].items():
        cov = case["covariance"]
        whiteners = []
        for x in model.cs.eps_stations:
            C = cov["nugget_sd_MPa"]**2*np.eye(len(x))
            if cov["discrepancy_variance_MPa2"]:
                z = cov["efold_multiplier"]*abs(x[:, None]-x[None, :])/cov["efold_strain"]
                C += cov["discrepancy_variance_MPa2"]*(1+z)*np.exp(-z)
            whiteners.append(np.linalg.inv(np.linalg.cholesky(C)))
        def evaluate(row):
            r = np.asarray(model.r(jnp.asarray(row["u"]))).reshape(model.cs.q_meas.shape)
            w = np.einsum("kij,kj->ki", whiteners, r)
            actual = float((w*w).sum()/2+cov["nll_constant"])
            np.testing.assert_allclose(actual, row["nll"], rtol=1e-10, atol=1e-8)
            return actual
        ref = evaluate(case["reference"])
        result[scenario] = {name: evaluate(row)-ref for name, row in case["probes"].items()}
    # Rescaled intervals keep the primary minimizer and sigma=1 arithmetic.
    scaled = {}
    for name, profile in data["scale_profiles"].items():
        ref = replay(model, profile["reference"])["nll"]
        scaled[name] = {}
        i = model.space.internal_names.index(name)
        for key, end in profile["endpoints"].items():
            sigma = float(key.split(":")[0])
            values = {role: (replay(model, end[role])["nll"]-ref)/sigma**2
                      for role in ("inside", "outside") if role in end}
            u = np.asarray(profile["reference"]["u"]).copy()
            u[i] = end["value"]
            scaled[name][key] = {"kind": end["kind"],
                                "endpoint": float(model.space.theta_from_u(jnp.asarray(u))[i]),
                                "bracket_delta_nll": values}
    result["scale_intervals"] = scaled
    return result


def expanded_profiles(data, names):
    producer = module("reference_expanded", "selfrev-r3/evidence/SR-06_bound_effects.py")
    models, result = {}, {}
    for key, profile in data["expanded_profiles"].items():
        if names and profile["name"] not in names:
            continue
        steps = profile["n_steps"]
        if steps not in models:
            models[steps] = producer.Model(data["form"], steps)
        model = models[steps]
        reference = replay(model, profile["reference"])
        ends = {}
        for side, end in profile["endpoints"].items():
            delta = {}
            for role in ("inside", "outside", "witness"):
                if role not in end:
                    continue
                actual = replay(model, end[role])
                fixed = (actual["engineering"]["nu_pp"] if profile["name"] == "operative_nu_pp"
                         else actual["theta"][model.names.index(profile["name"])])
                np.testing.assert_allclose(fixed, end[role]["value"], rtol=2e-7, atol=2e-8)
                delta[role] = actual["nll"]-reference["nll"]
            ends[side] = {"value": end["value"], "delta_nll": delta}
        result[key] = {"n_steps": steps, "reference_nll": reference["nll"],
                       "interval": profile["interval"], "endpoints": ends}
        print(f"Replayed expanded {key}, {steps} steps", file=sys.stderr, flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("summary", "check", "profiles", "cross-starts", "profile-point",
                                      "responses", "deformation", "residuals", "expanded"))
    ap.add_argument("--form", type=int, choices=(1, 2), default=1)
    ap.add_argument("--names", nargs="+")
    ap.add_argument("--operative", action="store_true")
    ap.add_argument("--coordinate", default="c_j")
    ap.add_argument("--side", choices=("lower", "upper"), default="lower")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    if args.output and args.output.exists():
        ap.error(f"Output already exists: {args.output}")
    inputs = load()
    data = {**inputs["forms"][str(args.form)], "form": args.form}
    if args.command == "profiles":
        result = profiles(data, args.names, args.operative)
    elif args.command == "profile-point":
        result = profile_point(data, args.coordinate, -1 if args.side == "lower" else 1)
    elif args.command == "deformation":
        result = deformation(data, inputs["volume_measurements"])
    elif args.command == "expanded":
        result = expanded_profiles(data, args.names)
    else:
        result = {"summary": summary, "check": check, "cross-starts": cross_starts,
                  "responses": responses, "residuals": residual_cases}[args.command](data)
    text = json.dumps(result, indent=2, allow_nan=False)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
