"""Reproduce the fixed-storage intervention at the saved material parameters."""
import csv
import json
from pathlib import Path

from fit_coupled_material import observations, path_response

ROOT = Path(__file__).resolve().parents[2]


def main():
    folder = ROOT / "results/constitutive-method"
    fit = json.loads((folder / "material_fit.json").read_text(encoding="utf-8"))
    rows = []
    for path in observations():
        if path["key"] not in ("fig2c_CU_Z", "fig3b_CD_P"):
            continue
        response = path_response(fit["parameters"], path, fit["steps"], fixed_storage=True)
        rows.extend(dict(path=path["key"], **row) for row in response)
        peak = max(response, key=lambda row: row["q_MPa"])
        print(path["key"], "peak q", peak["q_MPa"], "final u", response[-1]["pressure_MPa"], flush=True)
    with (folder / "storage_intervention.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
