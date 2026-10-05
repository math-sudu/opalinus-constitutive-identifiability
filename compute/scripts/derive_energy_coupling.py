"""Produce the unfitted two-mode constitutive calculation accompanying the derivation."""

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "compute" / "src"))
from tfenn_invcalib.energetic_coupling import EnergyClosure, MonotonicShear


def run():
    output = ROOT / "results" / "constitutive-method"
    output.mkdir(parents=True, exist_ok=True)
    parameters = dict(E0=25000., nu0=.25, bulk_slope=2., shear_slope=2.)
    evolution = dict(d0=.1, D=.2, overlap=.8, matrix_friction=60.,
                     plane_friction=2., resistance=.01)
    elastic = EnergyClosure(**parameters)
    coupled = MonotonicShear(elastic, **evolution)
    onset = coupled.plane_driven_onset()
    # Include analytical onset and a short interval on either side in the history.
    strain = np.unique(np.r_[np.linspace(0, .012, 401),
                             onset["strain"] * np.array([.99, 1., 1.01])])
    rows, summaries = [], {}
    for name, evolving in [("evolving_localization", True), ("fixed_localization", False)]:
        model = MonotonicShear(elastic, **evolution, evolving_localization=evolving)
        state = np.array([0., 0., model.d0])
        events = {}
        for e in strain:
            state = model.step(e, state)
            response = model.response(e, state)
            row = {"closure": name, "applied_mandel_strain": float(e),
                   "matrix_mode": float(state[0]), "plane_mode": float(state[1]),
                   "matrix_crack_density": float(state[2]), **response}
            rows.append(row)
            for key, active in [("plane_sliding", state[1] > 1e-12),
                                ("matrix_sliding", state[0] > 1e-12),
                                ("matrix_damage", state[2] > model.d0 + 1e-10)]:
                if active and key not in events:
                    events[key] = float(e)
        summaries[name] = {"first_sampled_events": events, "final": row}
        if evolving:
            m, j, d = state
            k, rho = model.shear_stiffness, model.overlap
            tm = row["stress"] - model.matrix_friction
            tp = k * (rho * (strain[-1] - m) - j) - model.plane_friction
            _, sp = elastic.plane_compliance(d)
            summaries[name]["active_branch_strength_interaction"] = {
                "matrix_excess_stress_MPa": float(tm),
                "plane_excess_stress_MPa": float(tp),
                "matrix_term": float(tm**2 / (2*model.matrix_storage*model.resistance)),
                "plane_term": float(model.D*sp[1]*tp**2 / (2*model.resistance)),
            }
    with (output / "energy_coupling_path.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    document = {
        "interpretation": "Unfitted constitutive calculation, not Opalinus Clay measurements or predictions.",
        "units": "MPa for stress, stored energy and damage resistance; dimensionless Mandel strain and crack density.",
        "energy_parameters": parameters, "evolution_parameters": evolution,
        "analytical_plane_driven_damage_onset": onset, "paths": summaries,
    }
    (output / "energy_coupling_summary.json").write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(document, indent=2))


if __name__ == "__main__":
    run()
