"""Precompute SR-09a dependent results on the selected fixed grids.

Continue the 64 saved full-data members and the 64 saved oracle members of
each four-curve fold. The first 32 fold members originated in the saved LOO
ensemble; all 64 now supply the predictive pool. Held-out observations never enter a fold
objective or its starting vectors. These are warm-start least-squares results,
not a rerun of the former Adam experiment or evidence about its efficiency.

Fits checkpoint valid iterates and continue until solver termination. The fold
without the inclined specimen preserves its structurally unobserved shear and
joint coordinates at their own fold-specific historical starting values.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import importlib.util
import hashlib
import inspect
import multiprocessing
import os
from pathlib import Path
import sys
import time
import threading

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("sr09a_map", HERE / "SR-09a_map.py")
mapping = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mapping
spec.loader.exec_module(mapping)
ROOT = mapping.ROOT
WORK = ROOT / "results/iterative/sr09a-resolution"
RUNS = ROOT / "compute/results/runs"
sys.path.insert(0, str(ROOT / "compute/scripts"))

import jax
import jax.numpy as jnp
import numpy as np
import psutil
from scipy.optimize import least_squares
from tfenn_invcalib import ensemble, mirror, theta_space
from tfenn_invcalib import bounded_fit, current_inputs
from report_symmetrised_elastic import engineering_constants

read, save = mapping.read, mapping.save


def now():
    return datetime.now(timezone.utc).isoformat()


def group_path(form, fold):
    return WORK / f"downstream_f{form}_{'full' if fold is None else f'fold{fold}'}.json"


def source_members(form, fold):
    suffix = "_f2" if form == 2 else ""
    if fold is None:
        path = RUNS / f"e5_ensemble_esp{suffix}/ensemble_inversion.json"
        data = read(path)
        return path, data["members"], None
    path = RUNS / f"e5_oracle4_esp{suffix}/oracle_folds.json"
    oracle = read(path)["folds"][fold]
    loo = read(RUNS / f"e5_loo_esp{suffix}/loo_result.json")["folds"][fold]
    assert oracle["held_out_key"] == loo["held_out_key"]
    assert len(oracle["members"]) == 64 and len(loo["members"]) == 32
    for a, b in zip(oracle["members"], loo["members"]):
        np.testing.assert_array_equal(a["theta_best"], b["theta_best"])
    return path, oracle["members"], {
        k: oracle[k] for k in ("fold_id", "held_out_key", "calib_keys")}


class InvalidPath(ValueError):
    pass


def fit_member(model, train, theta, budget=100, checkpoint=None):
    return bounded_fit.fit(model, theta, train=train, chunk=budget, checkpoint=checkpoint)


def predict(model, theta, *, u=None):
    if u is None:
        u = model.space.u_from_theta(np.asarray(theta))
    # r is in units of sigma_q, which the model's constant also uses.
    r = np.asarray(model.r(jnp.asarray(u))).reshape(model.cs.q_meas.shape)
    q = model.cs.q_meas + r * mapping.likelihood.load_likelihood_config(
        ROOT / "compute").sigma_q
    return [row.tolist() if np.isfinite(row).all() else None for row in q]


def prediction_definition():
    return hashlib.sha256(inspect.getsource(predict).encode()).hexdigest()


def selected_members(rows, count=None, delta_nll=ensemble.BAND_DELTA_NLL):
    candidates = [r for r in rows if r["fit"]["best"] is not None
                  and (count is None or r["start_index"] < count)]
    if not candidates:
        raise ValueError("No valid fitted members")
    cutoff = ensemble.band_threshold(min(r["fit"]["best"]["nll"] for r in candidates), delta_nll)
    return [r for r in candidates if r["fit"]["best"]["nll"] <= cutoff]


def path_function(form, steps, n_newton=mirror.N_NEWTON):
    forward = mirror.mirror_forward_path if form == 1 else mirror.mirror_forward_path_f2

    @jax.jit
    def evaluate(theta, M, n, pressure, stations):
        out = forward(theta, M, n, pressure, mapping.curves.EPS_A_TOTAL/steps,
                      n_steps=steps, n_newton=n_newton, early_exit=True)
        x = jnp.arange(steps+1) * mapping.curves.EPS_A_TOTAL/steps
        return {"valid": out["path_converged"], "q_peak_MPa": jnp.max(out["q"]),
                "peak_strain": x[jnp.argmax(out["q"])], "gamma_final": out["gamma"][-1],
                "q_stations_MPa": jnp.interp(stations, x, out["q"]),
                "gamma_stations": jnp.interp(stations, x, out["gamma"])}
    return evaluate


def path_row(fn, theta, bed, pressure, stations):
    M, n = mirror.path_constants(float(bed))
    out = fn(jnp.asarray(theta), M, n, jnp.asarray(pressure), jnp.asarray(stations))
    if not bool(out["valid"]):
        return {"valid": False}
    return {k: np.asarray(v).tolist() for k, v in out.items()}


def derived_definition():
    source = "".join(inspect.getsource(fn) for fn in
                     (derive_full, path_function, path_row, selected_members, predict))
    source += current_inputs.digest(ROOT / "compute/scripts/report_symmetrised_elastic.py")
    source += current_inputs.digest(ensemble.__file__)
    return hashlib.sha256(source.encode()).hexdigest()


def derive_full(data, model, path):
    current_map = read(mapping.output(data["form"]))
    if (current_map.get("recipe") != mapping.recipe(data["form"])
            or data["recipe"].get("inputs") != current_map["recipe"]):
        raise ValueError("Stale inputs in the map or full-data continuation")
    if current_map["status"] != "complete":
        raise ValueError("Complete and reconcile the parameter map before deriving its consumers")
    reference = current_map["reference"]
    valid = [r["fit"]["best"] for r in data["members"] if r["fit"]["best"]]
    if any(candidate["nll"] < reference["nll"]-1e-8 for candidate in valid):
        raise ValueError("Reconcile the lower full-data member with the profile reference first")
    data["map_reference"] = best = reference
    if (data.get("derived", {}).get("reference") != best
            or data.get("derived", {}).get("definition_sha256") != derived_definition()):
        data.pop("derived", None)
    if "derived" not in data:
        u = jnp.asarray(best["u"])
        J = np.asarray(model.j(u))
        if not np.isfinite(J).all():
            raise InvalidPath("Invalid sensitivity at attained full-data fit")
        transform = np.asarray(jax.jacfwd(model.space.theta_from_u)(u))
        J_physical = np.linalg.solve(transform.T, J.T).T
        _, singular, right = np.linalg.svd(J, full_matrices=False)
        th = best["theta"]
        data["derived"] = {
            "definition_sha256": derived_definition(),
            "reference": best, "nll_minus_map_reference": best["nll"]-reference["nll"],
            "q_at_reference_MPa": predict(model, th, u=best["u"]),
            "fisher_box_coordinates": (J.T@J).tolist(),
            "fisher_physical_coordinates": (J_physical.T@J_physical).tolist(),
            "residual_jacobian_box_coordinates": J.tolist(),
            "singular_values": singular.tolist(), "right_singular_vectors": right.tolist(),
            "operative_constants_names": ["E_p", "E_o", "nu_pp", "nu_13", "nu_31", "G_op"],
            "operative_constants": list(engineering_constants(th[0], th[1], th[3], th[4], th[2])),
            "measured_paths": [], "orientation_sweep": [],
            "orientation_lateral_max_evaluations": mirror.N_NEWTON,
            "joint_counterfactual": "c_j=1e6 MPa, all other coordinates and drives unchanged"}
        save(path, data)
    derived = data["derived"]
    fn = path_function(data["form"], data["n_steps"])
    off = np.array(best["theta"])
    off[8] = 1e6
    for i, key in enumerate(model.cs.keys):
        if any(r["key"] == key for r in derived["measured_paths"]):
            continue
        args = (model.cs.theta_deg[i], model.cs.p_conf_seq[i], model.cs.eps_stations[i])
        on = path_row(fn, best["theta"], *args)
        no_joint = path_row(fn, off, *args)
        row = {"key": key, "attained": on, "joint_disabled": no_joint}
        if on["valid"] and no_joint["valid"] and on["gamma_final"] > 0:
            row["joint_share_of_gamma"] = 1-no_joint["gamma_final"]/on["gamma_final"]
        derived["measured_paths"].append(row)
        save(path, data)
    members = selected_members(data["members"])
    sweep_fn = path_function(data["form"], data["n_steps"],
                             derived["orientation_lateral_max_evaluations"])
    for confinement in (5.0, 13.0):
        pressure = np.full(data["n_steps"], confinement)
        for bed in np.arange(0.0, 90.1, 5.0):
            if any(r["bed_deg"] == bed and r["confinement_MPa"] == confinement
                   for r in derived["orientation_sweep"]):
                continue
            args = (bed, pressure, model.cs.eps_stations[0])
            peaks = []
            for r in members:
                out = path_row(sweep_fn, r["fit"]["best"]["theta"], *args)
                peaks.append(out.get("q_peak_MPa"))
            derived["orientation_sweep"].append({"bed_deg": float(bed),
                "confinement_MPa": confinement,
                "attained": path_row(sweep_fn, best["theta"], *args),
                "joint_disabled": path_row(sweep_fn, off, *args),
                "member_start_indices": [r["start_index"] for r in members],
                "member_peaks_MPa": peaks})
            save(path, data)
            print(f"sweep form={data['form']} p={confinement} bed={bed}", flush=True)


def continue_members(data, model, train, seeds, path, budget, workers):
    """Independent member fits share compiled functions, not fitting state."""
    pending = data.pop("in_progress", {})
    checkpoints = data.setdefault("in_progress_members", {})
    if isinstance(pending.get("start_index"), int):
        checkpoints[str(pending["start_index"])] = {"iterate": pending["iterate"]}
    lock = threading.Lock()
    save(path, data)

    def run_member(i, seed):
        theta = data.get("warm_start_vectors", {}).get(str(i), {}).get("theta", seed["theta_best"])
        if str(i) in checkpoints:
            theta = checkpoints[str(i)]["iterate"]["theta"]
        theta = np.array(theta)
        for index in model.unobserved_coordinates:
            theta[index] = seed["theta_best"][index]

        def checkpoint(best):
            with lock:
                checkpoints[str(i)] = {"iterate": best}
                save(path, data)

        fit = fit_member(model, train, theta, budget, checkpoint)
        row = {"start_index": i, "fit": fit}
        if fit["best"]:
            row["q_prediction_MPa"] = predict(model, fit["best"]["theta"], u=fit["best"]["u"])
        with lock:
            checkpoints.pop(str(i), None)
            data["members"].append(row)
            save(path, data)
        fold = data["recipe"]["fold"]
        label = None if fold is None else fold["fold_id"]
        print(f"fit form={data['form']} fold={label} member={i+1}/{len(seeds)} "
              f"nll={fit['best']['nll'] if fit['best'] else None}", flush=True)

    completed = {r["start_index"] for r in data["members"]}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_member, i, seed) for i, seed in enumerate(seeds) if i not in completed]
        for future in as_completed(futures):
            future.result()
    data["members"].sort(key=lambda row: row["start_index"])
    data.pop("in_progress_members", None)
    save(path, data)


def run_group(form, fold, budget=100, derive=True, member_workers=8):
    path = group_path(form, fold)
    source, seeds, info = source_members(form, fold)
    recipe = {"form": form, "fold": info, "n_steps": mapping.STEPS[form],
              "inputs": mapping.recipe(form),
              "source_sha256": current_inputs.digest(source),
              "source_geometry": "Historical CU-Z=60 degree fits; only theta_best is reused",
              "source": str(source.relative_to(ROOT)),
              "seed_thetas": [r["theta_best"] for r in seeds]}
    data = None
    warm = {}
    if path.exists():
        previous = read(path)
        previous_recipe = dict(previous.get("recipe", {}))
        # The forward adapter and bounded solver have their own identities.
        # This former whole-file stamp also covered predictions and caused
        # readout-only fixes to discard unchanged calibration results.
        previous_recipe.pop("producer_sha256", None)
        if previous_recipe == recipe:
            data = previous
            data["recipe"] = recipe
        elif previous.get("recipe", {}).get("fold") == info:
            # A changed definition invalidates objectives, predictions and
            # termination, but vectors from this SAME training fold may seed
            # fresh model evaluations. Never transfer a full-data fit to a fold.
            provenance = {"source": str(path.relative_to(ROOT)),
                          "source_sha256": current_inputs.digest(path),
                          "input_definition": previous.get("recipe", {}).get("inputs")}
            for row in previous.get("members", []):
                if row["fit"]["best"] is not None:
                    warm[str(row["start_index"])] = {
                        "theta": row["fit"]["best"]["theta"], **provenance}
            pending = previous.get("in_progress", {})
            if isinstance(pending.get("start_index"), int):
                warm[str(pending["start_index"])] = {
                    "theta": pending["iterate"]["theta"], **provenance}
            for index, pending in previous.get("in_progress_members", {}).items():
                warm[index] = {"theta": pending["iterate"]["theta"], **provenance}
        if (data and data["status"] in ("complete", "fits_complete")
                and data.get("prediction_definition_sha256") == prediction_definition()):
            return str(path)
    if data is None:
        data = {"recipe": recipe, "form": form, "n_steps": mapping.STEPS[form],
                "status": "running", "started_at": now(), "members": [],
                "warm_start_vectors": warm}
    save(path, data)
    model = mapping.Model(form)
    if data.get("prediction_definition_sha256") != prediction_definition():
        for row in data["members"]:
            if row["fit"]["best"]:
                best = row["fit"]["best"]
                row["q_prediction_MPa"] = predict(model, best["theta"], u=best["u"])
        data.pop("derived", None)
        data.pop("prediction_summary", None)
        data["prediction_definition_sha256"] = prediction_definition()
    data["curve_keys"] = model.cs.keys
    data["q_measured_MPa"] = model.cs.q_meas.tolist()
    data["strain_stations"] = model.cs.eps_stations.tolist()
    data["theta_deg"] = model.cs.theta_deg.tolist()
    data["bedding_normals"] = np.asarray(model.cs.n_hat).tolist()
    included = np.array([info is None or k in info["calib_keys"] for k in model.cs.keys])
    train = np.flatnonzero(np.repeat(included, model.cs.q_meas.shape[1]))
    unobserved = ("G_op", "c_j", "phi_j_deg") if np.isin(model.cs.theta_deg[included], (0., 90.)).all() else ()
    model.unobserved_coordinates = tuple(model.space.internal_names.index(name) for name in unobserved)
    data["structurally_unobserved_coordinates"] = list(unobserved)
    if fold is None and "reference_fit" not in data:
        theta = read(mapping.output(form))["reference"]["theta"]
        if data.get("in_progress", {}).get("start_index") == "reference":
            theta = data["in_progress"]["iterate"]["theta"]
        def reference_checkpoint(best):
            data["in_progress"] = {"start_index": "reference", "iterate": best}
            save(path, data)
        data["reference_fit"] = fit_member(model, train, theta, budget, reference_checkpoint)
        data.pop("in_progress", None)
        save(path, data)
    continue_members(data, model, train, seeds, path, budget, member_workers)
    if fold is None:
        candidates = [r["fit"]["best"] for r in data["members"] if r["fit"]["best"]]
        if data["reference_fit"]["best"]:
            candidates.append(data["reference_fit"]["best"])
        data["map_reference"] = min(candidates, key=lambda x: x["nll"])
        if derive:
            derive_full(data, model, path)
    data["status"] = "complete" if derive else "fits_complete"
    data["finished_at"] = now()
    save(path, data)
    return str(path)


def source_baseline(form):
    data = read(RUNS / "e5_null_srcpp_esp/null_baselines.json")["forms"][str(form)]
    current_inputs.require_current(data["inputs"])
    return data


def aggregate_predictions():
    from run_null_baselines import build, OUTPUT
    save(OUTPUT, build())
    for form in (1, 2):
        full = read(group_path(form, None))
        baseline = source_baseline(form)
        for fold in range(5):
            path = group_path(form, fold)
            data = read(path)
            index = data["curve_keys"].index(data["recipe"]["fold"]["held_out_key"])
            measured = np.array(data["q_measured_MPa"][index])
            ensemble_rows = selected_members(data["members"])
            valid = [r for r in data["members"] if r["fit"]["best"]]
            oracle = min(valid, key=lambda r: r["fit"]["best"]["nll"])
            predictions = [r["q_prediction_MPa"][index] for r in ensemble_rows]
            lanes = {"source_baseline": baseline["curves"][index]["variants"]["softening"]["q_MPa"],
                     "four_curve_best": oracle["q_prediction_MPa"][index],
                     "five_curve_best": full["derived"]["q_at_reference_MPa"][index]}
            if any(p is None for p in predictions) or any(p is None for p in lanes.values()):
                data["prediction_summary"] = {"status": "invalid_heldout_path"}
            else:
                lanes["ensemble_median"] = np.median(predictions, axis=0).tolist()
                rmses = {k: float(np.sqrt(np.mean((np.array(v)-measured)**2)))
                         for k, v in lanes.items()}
                data["prediction_summary"] = {"status": "complete", "rmse_MPa": rmses,
                    "selection": {"delta_nll": ensemble.BAND_DELTA_NLL,
                                  "pool_size": len(data["members"]),
                                  "reference_nll": oracle["fit"]["best"]["nll"]},
                    "ensemble_start_indices": [r["start_index"] for r in ensemble_rows],
                    "ensemble_median_q_MPa": lanes["ensemble_median"]}
            data.pop("nominal_prediction_MPa", None)
            save(path, data)


def pin_worker(slots, cpus, per_worker):
    slot = slots.get()
    psutil.Process().cpu_affinity(cpus[slot*per_worker:(slot+1)*per_worker])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--cpus-per-worker", type=int, default=4)
    ap.add_argument("--max-nfev", type=int, default=40)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    jobs = [(f, fold) for fold in (None, 0, 1, 2, 3, 4) for f in (1, 2)]
    for form, fold in jobs:
        _, members, _ = source_members(form, fold)
        print(f"group form={form} fold={fold} starts={len(members)}", flush=True)
    if args.dry_run:
        return
    cpus = psutil.Process().cpu_affinity()
    if min(args.workers, args.cpus_per_worker, args.max_nfev) < 1 or args.workers*args.cpus_per_worker > len(cpus):
        ap.error("positive budgets and enough allowed CPUs for each worker are required")
    status_path = WORK / "numerical_resume.json"
    status = {"kind": "sr09a_fixed_grid_downstream", "status": "running", "started_at": now(),
        "pid": os.getpid(), "create_time": psutil.Process().create_time(),
        "producer": str(Path(__file__).resolve().relative_to(ROOT)),
        "command": "python -B selfrev-r2/evidence/SR-09a_downstream.py "
                   f"--workers {args.workers} --cpus-per-worker {args.cpus_per_worker} --max-nfev {args.max_nfev}",
        "workers": args.workers, "integration_steps": mapping.STEPS,
        "fit_count": 768, "completed_groups": [],
        "stdout": "results/iterative/sr09a-resolution/downstream.stdout.log",
        "stderr": "results/iterative/sr09a-resolution/downstream.stderr.log",
        "results": [str(group_path(f, k).relative_to(ROOT)) for f, k in jobs],
        "continuation": "Read current outputs, reconcile any lower full-data basin with the map, then propagate retained results. Warm-start fits do not reproduce old Adam efficiency comparisons. Never restart the old refinement or budget-doubling loop."}
    save(status_path, status)
    slots = multiprocessing.Queue()
    for i in range(args.workers):
        slots.put(i)
    try:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=pin_worker,
                                 initargs=(slots, cpus, args.cpus_per_worker)) as pool:
            futures = {pool.submit(run_group, f, k, args.max_nfev, False): (f, k) for f, k in jobs}
            errors = []
            for future in as_completed(futures):
                try:
                    result = future.result()
                    status["completed_groups"].append(result)
                except Exception as exc:
                    form, fold = futures[future]
                    errors.append({"form": form, "fold": fold, "error": repr(exc)})
                    status["group_errors"] = errors
                save(status_path, status)
            if errors:
                raise RuntimeError(f"Unfinished groups: {errors}")
        status.update(status="fits_complete", finished_at=now(),
                      continuation="Run selfrev-r3/evidence/SR-01_geometry_recompute.py to reconcile maps and derive consumers.")
    except Exception as exc:
        status.update(status="failed", error=repr(exc), finished_at=now())
        save(status_path, status)
        raise
    save(status_path, status)


if __name__ == "__main__":
    main()
