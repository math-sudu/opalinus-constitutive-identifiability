"""Recalculate the paired constitutive analysis from the measured inputs."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

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


mapping = module("input_reproduction_map", "selfrev-r2/evidence/SR-09a_map.py")
np, jnp = mapping.np, mapping.jnp
from tfenn_invcalib import bounded_fit, curves, ensemble, likelihood, operative_elastic
from profile_search import ReferenceImproved, compare_starts, update_reference

# Fresh searches own their output tree and can coexist with supplied references.
mapping.EVID = ROOT / "results/from-inputs/profiles"
original_profile_definition = mapping.profile_definition
mapping.profile_definition = lambda: original_profile_definition() + ":" + ":".join(
    mapping.current_inputs.digest(path) for path in
    (Path(__file__), Path(__file__).with_name("profile_search.py")))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fit_file(form):
    return ROOT / f"results/from-inputs/fit_f{form}.json"


def adopt_reference(form, data, fitted, reference):
    """Invalidate the generated consumers of the previous fitted reference."""
    data.update(reference=reference, per_parameter={}, status="partial")
    data.pop("grid_comparison", None)
    mapping.save(mapping.output(form), data)
    fitted["reference"] = reference
    mapping.save(fit_file(form), fitted)
    steps = mapping.STEPS[form]
    targets = [ROOT / f"results/from-inputs/responses_f{form}.json"]
    targets += [ROOT / f"results/from-inputs/elastic/f{form}_{steps}_profile_{name}.json"
                for name in ("E_p", "E_o", "nu_pp", "nu_po", "nu_op")]
    targets += [mapping.EVID / f"SR-09a_map_f{form}_{name}.json"
                for name in mapping.theta_space.space_for_form(form).internal_names]
    generated = (ROOT / "results/from-inputs").resolve()
    for target in targets:
        if not target.resolve().is_relative_to(generated):
            raise ValueError(f"Derived output is outside the generated directory: {target}")
        target.unlink(missing_ok=True)
    print(f"Updated form {form} reference NLL: {reference['nll']}", flush=True)


def fit(args):
    path = fit_file(args.form)
    space = mapping.theta_space.space_for_form(args.form)
    if path.exists():
        data = read(path)
        if (data["seed"] != args.seed or data["n_starts"] != args.starts
                or data["adam_iters"] != args.adam_iters):
            raise ValueError("Existing fit uses different starts; choose a fresh checkout or restore its settings")
        mapping.current_inputs.require_current(data["inputs"])
    else:
        nominal = space.extend_nominal(mapping.theta_space.load_nominal_theta(ROOT / "compute"))
        starts = space.sample_starts(np.random.default_rng(args.seed), args.starts,
                                     "box_loguniform", nominal)
        cs = curves.load_measured_curveset(ROOT / "compute", n_steps=400,
                                          effective_stress_path=True)
        objective = likelihood.make_likelihood(cs, likelihood.load_likelihood_config(ROOT / "compute"),
                                                form=args.form)
        seeds = ensemble.run_multistart(objective.nll_z, starts, args.adam_iters,
                                        lr=.03, form=args.form)
        # Only successful initial paths are passed to the bounded continuation.
        data = {"form": args.form, "seed": args.seed, "n_starts": args.starts,
                "adam_iters": args.adam_iters,
                "inputs": mapping.recipe(args.form),
                "initial": [{"index": r.start_index, "failed": r.failed,
                             "theta": None if r.failed else r.theta_best} for r in seeds],
                "members": []}
        mapping.save(path, data)
    if args.initial_only:
        return {"initial_states": str(path), "valid_starts": sum(not r["failed"] for r in data["initial"])}
    model = mapping.Model(args.form)
    done = {r["index"] for r in data["members"]}
    for seed in data["initial"]:
        if seed["index"] in done or seed["failed"]:
            continue
        pending = data.get("in_progress")
        theta = pending["fit"]["theta"] if pending and pending["index"] == seed["index"] else seed["theta"]
        def checkpoint(row):
            data["in_progress"] = {"index": seed["index"], "fit": row}
            mapping.save(path, data)
        result = bounded_fit.fit(model, theta, checkpoint=checkpoint)
        data["members"].append({"index": seed["index"], "fit": result})
        data.pop("in_progress", None)
        mapping.save(path, data)
        print(f"Form {args.form}, member {seed['index']}: {result['solver_status']}", flush=True)
    valid = [r["fit"]["best"] for r in data["members"] if r["fit"]["solver_status"] != "failed"]
    if "reference" in data:
        valid.append(data["reference"])
    if not valid:
        raise ValueError("No valid calibrated state was obtained")
    reference = min(valid, key=lambda r: r["nll"])
    data["reference"] = reference
    mapping.save(path, data)
    destination = mapping.output(args.form)
    if destination.exists() and read(destination)["reference"] != reference:
        raise ValueError("A different reference already owns the profiles; use a fresh checkout for this fit")
    if not destination.exists():
        mapping.save(destination, {"form": args.form, "n_steps": mapping.STEPS[args.form],
            "recipe": mapping.recipe(args.form), "reference": reference,
            "source": str(path.relative_to(ROOT)), "likelihood_cut": mapping.CUT,
            "profile_definition_sha256": mapping.profile_definition(),
            "per_parameter": {}, "status": "partial"})
    return {"reference": reference, "output": str(path)}


def profiles(args):
    data = read(mapping.output(args.form))
    names = args.names or mapping.theta_space.space_for_form(args.form).internal_names
    fitted = read(fit_file(args.form))
    seeds = [(f"member_{row['index']}", row["fit"]["best"]) for row in fitted["members"]
             if row["fit"]["solver_status"] != "failed"]
    class ProfileModel(mapping.Model):
        def fit(self, name, value, seed, budget=100, checkpoint=None):
            fit_one = super().fit
            compared = compare_starts(lambda start: fit_one(name, value, start, budget, checkpoint),
                [("adjacent", seed), ("reference", data["reference"]), *seeds])
            reference = update_reference(data["reference"], compared["fit"],
                lambda start: bounded_fit.fit(self, start["theta"]))
            if reference["nll"] < data["reference"]["nll"]:
                raise ReferenceImproved({k: reference[k] for k in ("u", "theta", "nll")})
            return {**compared["fit"], "start_comparisons": compared["attempts"]}
    mapping.MODELS[args.form] = ProfileModel(args.form)
    while True:
        try:
            for name in names:
                _, _, result = mapping.profile_coordinate(args.form, name, 100)
                data["per_parameter"][name] = result
                mapping.save(mapping.output(args.form), data)
            break
        except ReferenceImproved as improvement:
            adopt_reference(args.form, data, fitted, improvement.reference)
    data["status"] = "complete" if len(data["per_parameter"]) == mapping.MODELS[args.form].space.n_theta else "partial"
    mapping.save(mapping.output(args.form), data)
    return {name: data["per_parameter"][name]["classification"] for name in names}


def elastic(args):
    producer = module("input_reproduction_elastic", "selfrev-r3/evidence/SR-03_operative_elastic.py")
    producer.mapping.EVID = mapping.EVID
    producer.WORK = ROOT / "results/from-inputs/elastic"
    data = read(mapping.output(args.form))
    fitted = read(fit_file(args.form))
    seeds = [(f"member_{row['index']}", row["fit"]["best"]) for row in fitted["members"]
             if row["fit"]["solver_status"] != "failed"]
    single_fit = producer.continued_fit
    def fit_multiple(model, supplied, seed, checkpoint=None):
        compared = compare_starts(lambda start: single_fit(model, supplied, start, checkpoint),
            [("adjacent", seed), ("reference", data["reference"]), *seeds])
        reference = update_reference(data["reference"], compared["fit"],
            lambda start: bounded_fit.fit(model, start["theta"]))
        if reference["nll"] < data["reference"]["nll"]:
            raise ReferenceImproved({k: reference[k] for k in ("u", "theta", "nll")})
        return {**compared["fit"], "start_comparisons": compared["attempts"]}
    producer.continued_fit = fit_multiple
    while True:
        try:
            return {name: producer.run_profile(args.form, name)["interval"]
                    for name in (args.names or producer.QUANTITIES)}
        except ReferenceImproved as improvement:
            prior_names = list(data["per_parameter"])
            adopt_reference(args.form, data, fitted, improvement.reference)
            if prior_names:
                profiles(SimpleNamespace(form=args.form, names=prior_names))
                data = read(mapping.output(args.form))
                fitted = read(fit_file(args.form))


def responses(args):
    producer = module("input_reproduction_paths", "selfrev-r2/evidence/SR-09a_downstream.py")
    model = mapping.Model(args.form)
    theta = np.asarray(read(mapping.output(args.form))["reference"]["theta"])
    off = theta.copy()
    off[8] = 1e6
    fn = producer.path_function(args.form, model.cs.n_steps)
    paths = []
    for i, key in enumerate(model.cs.keys):
        drive = (model.cs.theta_deg[i], model.cs.p_conf_seq[i], model.cs.eps_stations[i])
        on = producer.path_row(fn, theta, *drive)
        disabled = producer.path_row(fn, off, *drive)
        if not on["valid"] or not disabled["valid"]:
            raise ValueError(f"Invalid constitutive path: {key}")
        paths.append({"key": key, "joint_on": on, "joint_off": disabled})
    sweep = []
    for angle in np.arange(0., 91., 5.):
        drive = (angle, np.full(model.cs.n_steps, 5.), model.cs.eps_stations[0])
        on = producer.path_row(fn, theta, *drive)
        disabled = producer.path_row(fn, off, *drive)
        if not on["valid"] or not disabled["valid"]:
            raise ValueError(f"Invalid drained path at {angle} degrees")
        sweep.append({"bedding_deg": angle, "joint_on": on, "joint_off": disabled})
    result = {"form": args.form, "measured_paths": paths, "drained_5_MPa": sweep}
    path = ROOT / f"results/from-inputs/responses_f{args.form}.json"
    mapping.save(path, result)
    return {"output": str(path), "measured_paths": len(paths), "orientations": len(sweep)}


def check(args):
    model = mapping.Model(args.form)
    nominal = model.space.extend_nominal(mapping.theta_space.load_nominal_theta(ROOT / "compute"))
    record = model.record(model.space.u_from_theta(nominal))
    np.testing.assert_array_equal(model.cs.theta_deg, [90, 0, 30, 90, 0])
    assert model.cs.q_meas.shape == (5, 70)
    engineering = np.asarray(operative_elastic.engineering(jnp.asarray(nominal)))
    assert np.isfinite(engineering).all()
    return {"form": args.form, "n_steps": model.cs.n_steps,
            "observations": model.cs.q_meas.size, "nominal_nll": record["nll"],
            "operative_constants": dict(zip(operative_elastic.NAMES, engineering.tolist()))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("check", "fit", "profiles", "elastic", "responses"))
    ap.add_argument("--form", type=int, choices=(1, 2), default=1)
    ap.add_argument("--seed", type=int, default=20260706)
    ap.add_argument("--starts", type=int, default=64)
    ap.add_argument("--adam-iters", type=int, default=2000)
    ap.add_argument("--initial-only", action="store_true")
    ap.add_argument("--names", nargs="+")
    args = ap.parse_args()
    if args.starts < 1 or args.adam_iters < 1:
        ap.error("starts and adam-iters must be positive")
    result = {"check": check, "fit": fit, "profiles": profiles,
              "elastic": elastic, "responses": responses}[args.command](args)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
