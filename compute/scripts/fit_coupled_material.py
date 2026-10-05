"""Constrain the coupled material with measured q, CU pressure and CD volume.

Run from the project checkout. The saved fit is the continuation state; a new
fit resumes its material parameters unless --fresh is supplied. Pressure is
predicted with conserved fluid content, never imposed from the observations.
"""

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "compute" / "src"))

from tfenn_invcalib.energetic_coupling import VOLUME
from tfenn_invcalib.poromechanical_coupling import TIElasticity, PorousEnergyClosure
from tfenn_invcalib.coupled_friction_damage import CoupledFrictionDamage, FrictionDamageLaw

OUTPUT = ROOT / "results" / "constitutive-method" / "material_fit.json"
PATH_OUTPUT = OUTPUT.with_name("material_paths.csv")
NAMES = ("E_parallel", "E_normal", "nu_parallel", "nu_pn", "G_normal",
         "matrix_cohesion", "matrix_friction", "dilation_fraction", "parallel_weight",
         "plane_friction", "plane_compliance", "initial_density", "damage_resistance",
         "damage_peak_density", "damage_residual_fraction", "compaction_onset", "compaction_hardening",
         "damage_shape", "dilation_residual_fraction", "parallel_residual_weight",
         "dilation_density_ratio", "matrix_contraction", "stiffness_pressure_fraction",
         "friction_residual_fraction")
# The fresh start uses the approximately isotropic late-strength envelope,
# c + mu*p', and a cap above the maximum measured CU-P mean effective stress.
# These are initial values, not imposed material constraints.
DEFAULT = np.array([15000., 4500., .25, .38, 3500., 3.9, .826, .6, 1.,
                    .25, .6, .05, .012, .6, .03, 30., 5000., 1., 0., 1., 1., 0., 0., 1.])
# The prepared S-path start needs a parallel weight below the former .2
# search bound to recover CD-P strength. Positive definiteness requires w>0.
LOWER = np.array([6000., 2000., 0., .02, 800., .001, .1, 0., .05,
                  .02, .02, .005, .0005, .04, 0., 5., 20., .5, 0., .2, .05, 0., 0., 0.])
# The former 1.2 friction bound excluded the retained-energy peak-flow
# requirement 1.37727. Permit that peak range while fitting its residual ratio.
UPPER = np.array([30000., 12000., .48, .65, 15000., 12., 2., 1., 2.,
                  .7, 8., 1., .1, 5., .5, 40., 100000., 1.8, 1., 2., 20., 5., 1., 1.])
LOG = np.array([True, True, False, False, True, True, False, False, True,
                False, True, True, True, True, False, True, True, False, False, True, True, False, False, False])
# Positive scales use logarithmic coordinates; fractions and Poisson ratios
# stay linear. Translate the lower bounds to one. The continuation state
# always stores physical parameters.
COORDINATE_ORIGIN = LOWER.copy()
COORDINATE_ORIGIN[LOG] = np.log(COORDINATE_ORIGIN[LOG])
COORDINATE_ORIGIN -= 1.
# Reference constituents are explicit assumptions, not specimen measurements.
# alpha=1 and ell_n=ell_t=1 fix the density/localization rescaling freedom.
FIXED = dict(grain_bulk=50000., fluid_bulk=2200., porosity=.15, matrix_slope=1.,
             plane_normal_factor=1., plane_shear_factor=1.)
REFERENCE_CONFINEMENT = 13.


def encode(values):
    result = np.array(values, dtype=float)
    result[LOG] = np.log(result[LOG])
    return result-COORDINATE_ORIGIN


def decode(coordinates):
    result = np.asarray(coordinates)+COORDINATE_ORIGIN
    result[LOG] = np.exp(result[LOG])
    return dict(zip(NAMES, result))


def observations():
    folder = ROOT / "compute/results/data/nordas_curves"
    q = json.loads((folder / "digitized_nordas_curves.json").read_text(encoding="utf-8"))["panels_digitized"]
    u = json.loads((folder / "digitized_nordas_pore_pressure.json").read_text(encoding="utf-8"))["panels_digitized"]
    v = json.loads((ROOT / "selfrev-r3/evidence/SR-10_measured_volume.json").read_text(encoding="utf-8"))["panels"]
    pressure = {p["specimen_id"]: p for p in u}
    volume = {p["specimen_id"]: p for p in v}
    paths = []
    for panel in q:
        channels = {"q_MPa": (np.array(panel["eps_a_pct"])/100, np.array(panel["q_MPa"]), 1.)}
        if panel["specimen_id"] in pressure:
            p = pressure[panel["specimen_id"]]
            channels["pressure_MPa"] = (np.array(p["eps_a_pct"])/100, np.array(p["pw_MPa"]), 1.)
        if panel["specimen_id"] in volume:
            p = volume[panel["specimen_id"]]
            mask = np.array(p["comparison_mask"], dtype=bool)
            channels["volume"] = (np.array(p["eps_a_pct"])[mask]/100,
                                  np.array(p["eps_vol_pct"])[mask]/100,
                                  p["median_stroke_halfwidth_pct"]/100)
        paths.append(dict(key=panel["key"], angle=panel["theta_deg"],
                          confinement=panel["sigma_r0_MPa"], drainage=panel["test_type"],
                          specimen_id=panel["specimen_id"], channels=channels))
    return paths


def material(parameters, angle, confinement, fixed_storage=False):
    p = parameters
    theta = np.deg2rad(angle)
    normal = [np.sin(theta), 0., np.cos(theta)]
    # Historical fits had one reference stiffness for both preparation groups.
    # Their exact limit is pressure_fraction=0, including --revision diagnoses.
    elastic = TIElasticity(*(p[name] for name in NAMES[:5])).at_confinement(
        confinement, REFERENCE_CONFINEMENT, p.get("stiffness_pressure_fraction", 0.))
    energy = PorousEnergyClosure(elastic, normal, **FIXED,
        compaction_onset=p["compaction_onset"], compaction_hardening=p["compaction_hardening"],
        plane_reference_damage=p["initial_density"] if fixed_storage else None)
    law = FrictionDamageLaw(
        matrix_cohesion=p["matrix_cohesion"], matrix_friction=p["matrix_friction"],
        matrix_dilatancy=p["matrix_friction"]*p["dilation_fraction"],
        parallel_weight=p["parallel_weight"], shear_weight=1.,
        plane_friction=p["plane_friction"], damage_resistance=p["damage_resistance"],
        damage_peak_density=p["damage_peak_density"],
        damage_residual_fraction=p["damage_residual_fraction"], damage_shape=p["damage_shape"],
        dilation_residual_fraction=p["dilation_residual_fraction"],
        parallel_residual_weight=p["parallel_residual_weight"],
        dilation_density_ratio=p["dilation_density_ratio"],
        dilation_initial_density=p["initial_density"],
        # Historical fits predate shear contraction; zero is their exact law.
        matrix_contraction=p.get("matrix_contraction", 0.),
        friction_residual_fraction=p.get("friction_residual_fraction", 1.))
    return CoupledFrictionDamage(energy, law, p["plane_compliance"])


def path_response(parameters, path, steps, fixed_storage=False):
    p0 = path["confinement"]
    model = material(parameters, path["angle"], p0, fixed_storage)
    initial = previous = model.consolidated_state(p0, parameters["initial_density"])
    boundary = ({"pressure": 0.} if path["drainage"] == "CD" else
                {"fluid_content": initial["fluid_content"]})
    end = max(channel[0][-1] for channel in path["channels"].values())
    # Solve at every consumed observation strain. Interpolation across a
    # softening jump would invent stresses/volumes between distinct branches.
    stations = np.unique(np.concatenate([np.linspace(0., end, steps+1)]
                         + [channel[0] for channel in path["channels"].values()]))
    rows = []
    for axial in stations:
        state = model.step(initial["strain"][0]+axial, [p0, p0, 0., 0., 0.], previous, **boundary)
        f = state["response"]
        rows.append(dict(axial_strain=axial, q_MPa=f["total_stress"][0]-p0,
            pressure_MPa=f["pressure"], volume=VOLUME @ (state["strain"]-initial["strain"]),
            matrix_axial=state["matrix_strain"][0], matrix_volume=VOLUME @ state["matrix_strain"],
            plane_axial=model.energy.B[0] @ state["plane_modes"],
            plane_slip=np.linalg.norm(state["plane_modes"][1:]),
            compaction=state["compaction"]-initial["compaction"],
            density=state["d"], resistance_MPa=state["damage_resistance"],
            matrix_weight=model.law.parallel_metric_weight(state["d"])[0],
            matrix_friction=model.law.friction(state["d"])[0],
            dilatancy=model.law.dilatancy(state["d"])[0],
            matrix_equivalent_force_MPa=state["matrix_equivalent_force"],
            matrix_mean_force_MPa=state["matrix_mean_force"],
            cap_slack_MPa=model.energy.compaction_onset-f["compaction_force"],
            matrix_release_MPa=f["matrix_self_release"], plane_release_MPa=f["plane_to_matrix_release"]))
        previous = state
    return rows


class MaterialResidual:
    def __init__(self, steps, mapper=map):
        self.steps = steps
        self.paths = observations()
        self.mapper = mapper

    def __call__(self, coordinates):
        parameters = decode(coordinates)
        residuals = []
        responses = self.mapper(path_response, repeat(parameters), self.paths, repeat(self.steps))
        for path, rows in zip(self.paths, responses):
            axial = np.array([row["axial_strain"] for row in rows])
            for name, (stations, values, scale) in path["channels"].items():
                predicted = np.array([row[name] for row in rows])[np.searchsorted(axial, stations)]
                # Each measured path/channel has equal total weight. Raster
                # sampling density is not the number of independent specimens.
                residuals.extend((predicted-values)/(scale*np.sqrt(len(values))))
                if name == "q_MPa":
                    # Peak resistance is a target observable in its own right;
                    # its short strain interval must not be diluted by the long
                    # post-peak branch in the uniformly sampled curve mean.
                    # A value at the measured peak strain can lie on the
                    # computed descending branch and hide an earlier overshoot.
                    calculated_peak = max(row[name] for row in rows)
                    residuals.append((calculated_peak-np.max(values))/scale)
        return np.array(residuals)


def evaluate(parameters, steps):
    all_rows, summary = [], {}
    for path in observations():
        rows = path_response(parameters, path, steps)
        axial = np.array([row["axial_strain"] for row in rows])
        errors, observations_at_end = {}, {}
        for name, (stations, values, scale) in path["channels"].items():
            predicted = np.array([row[name] for row in rows])[np.searchsorted(axial, stations)]
            errors[name] = dict(rmse=float(np.sqrt(np.mean((predicted-values)**2))),
                                scale=scale, count=len(values))
            observations_at_end[name] = dict(axial_strain=float(stations[-1]),
                                             measured=float(values[-1]), predicted=float(predicted[-1]))
        peak = max(rows, key=lambda row: row["q_MPa"])
        upeak = max(rows, key=lambda row: row["pressure_MPa"])
        q_stations, q_values, _ = path["channels"]["q_MPa"]
        observed_peak_index = int(np.argmax(q_values))
        summary[path["key"]] = dict(channels=errors, channel_endpoints=observations_at_end,
            q_peak=dict(axial_strain=peak["axial_strain"], q_MPa=peak["q_MPa"]),
            measured_q_peak=dict(axial_strain=float(q_stations[observed_peak_index]),
                                 q_MPa=float(q_values[observed_peak_index])),
            pressure_peak=dict(axial_strain=upeak["axial_strain"], pressure_MPa=upeak["pressure_MPa"]),
            final=rows[-1])
        all_rows.extend(dict(path=path["key"], **row) for row in rows)
    with PATH_OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("fit", "evaluate"))
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--initial-radius", type=float, default=.05)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--parameters", type=Path,
                        help="Use the material parameters in this JSON instead of a saved fit.")
    parser.add_argument("--start", nargs=2, action="append", default=[],
                        metavar=("PARAMETER", "VALUE"),
                        help="Change an optimizer starting coordinate while preserving the saved incumbent.")
    args = parser.parse_args()
    if args.fresh and args.parameters:
        parser.error("--fresh and --parameters select different starting states.")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if args.parameters:
        previous = json.loads(args.parameters.read_text(encoding="utf-8"))
    else:
        previous = json.loads(OUTPUT.read_text(encoding="utf-8")) if OUTPUT.exists() and not args.fresh else {}
    if previous:
        parameters = dict(previous["parameters"])
        parameters.setdefault("stiffness_pressure_fraction", 0.)
        parameters.setdefault("friction_residual_fraction", 1.)
        incumbent = [parameters[name] for name in NAMES]
    else:
        incumbent = DEFAULT.copy()
    start = np.array(incumbent)
    for name, value in args.start:
        if args.action != "fit" or name not in NAMES:
            parser.error("--start requires action fit and a named material parameter.")
        index = NAMES.index(name)
        start[index] = float(value)
        if not LOWER[index] <= start[index] <= UPPER[index]:
            parser.error(f"--start {name} must lie within [{LOWER[index]}, {UPPER[index]}].")
    record = dict(interpretation="Joint material-response development; unresolved discrepancies remain subject to physical diagnosis.",
        parameters=dict(zip(NAMES, incumbent)), fixed_parameters=FIXED, steps=args.steps,
        reference_elasticity=dict(confinement_MPa=REFERENCE_CONFINEMENT,
            scale="1 + stiffness_pressure_fraction * (initial_effective_confinement / 13 MPa - 1)",
            interpretation="Scale all three TI moduli at preparation, preserving Poisson ratios; rebuild matrix/plane storage, Biot tensor and fluid modulus from this fixed reference energy."),
        friction_law="matrix_friction through damage_peak_density; above it, matrix_friction * [friction_residual_fraction + (1 - friction_residual_fraction) * damage_peak_density / density].",
        dilation_law="Signed matrix flow: positive damage-mobilized dilation scales with instantaneous friction, minus decaying shear contraction; the onset reference equals initial_density.",
        integration="Uniform base grid plus every consumed observation strain; no observation is interpolated across a softening jump.",
        bounds={name: [lo, hi] for name, lo, hi in zip(NAMES, LOWER, UPPER)},
        objective="Equal weight per path/channel: q/1 MPa, pressure/1 MPa, CD volume/its digitization half-width; each calculated maximum q is additionally compared with the measured maximum at unit weight. This is a scaled least-squares development objective, not a likelihood or uncertainty estimate.",
        observations="Measured q and pressure in compute/results/data/nordas_curves and retained measured CD volume fields in selfrev-r3/evidence/SR-10_measured_volume.json.")

    def save():
        pending = OUTPUT.with_suffix(".tmp")
        pending.write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
        pending.replace(OUTPUT)

    if args.action == "fit":
        # Stress drops cross observation strains as parameters move. Fit the
        # unchanged measured-channel objective using function values instead
        # of a finite-difference residual Jacobian across those transitions.
        options = dict(initial_tr_radius=args.initial_radius, final_tr_radius=.001)
        nfev, best_cost = 0, np.inf
        try:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                residual = MaterialResidual(args.steps, pool.map if args.workers > 1 else map)

                def objective(coordinates):
                    nonlocal nfev, best_cost
                    values = residual(coordinates)
                    cost = float(.5*values @ values)
                    nfev += 1
                    if cost < best_cost:
                        best_cost = cost
                        record["parameters"] = decode(coordinates)
                        record["optimization"] = dict(status="running", method="COBYQA",
                            nfev=nfev, cost=cost, steps=args.steps, options=options)
                        save()
                        print(f"nfev={nfev} cost={cost:.7g}", flush=True)
                    elif nfev % 20 == 0:
                        print(f"nfev={nfev} best_cost={best_cost:.7g}", flush=True)
                    return cost

                # COBYQA can move a bound-adjacent x0 when constructing its
                # interpolation set. Keep the actual incumbent even when all
                # points evaluated by the restarted optimizer are worse.
                objective(encode(incumbent))
                result = minimize(objective, encode(start), method="COBYQA",
                    bounds=list(zip(encode(LOWER), encode(UPPER))), options=options)
        except Exception as exc:
            record.setdefault("optimization", {})["status"] = "failed"
            record["optimization"]["failure"] = str(exc)
            save()
            raise
        record["optimization"] = dict(status="finished", method="COBYQA", message=result.message,
            solver_success=bool(result.success), nfev=nfev, cost=best_cost,
            steps=args.steps, options=options)
        save()
    record["paths"] = evaluate(record["parameters"], args.steps)
    if args.action == "evaluate" and "optimization" in previous:
        record["optimization"] = previous["optimization"]
    save()
    for name, path in record["paths"].items():
        print(name, "q peak", path["q_peak"], "channels", path["channels"], flush=True)


if __name__ == "__main__":
    # Constitutive solves are small; parallelism belongs to independent measured
    # paths, not threaded BLAS within each 6x6 matrix product.
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    main()
