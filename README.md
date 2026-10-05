# Coupling weak-plane storage and matrix damage in anisotropic porous rock

Constitutive source and digitized inputs for the manuscript by Pengcheng Zhu
and Tielin Chen. Plane storage depends on the evolving matrix compliance;
its energy derivative supplies plane-driven matrix damage and reciprocal
plane-backstress relaxation. The saturated transversely isotropic extension
predicts stress, undrained pore pressure and drained volume.

## Install and reproduce

Run from this repository's root with Python 3.13:

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r compute/requirements.txt
.venv/Scripts/python compute/scripts/derive_energy_coupling.py
.venv/Scripts/python compute/scripts/fit_coupled_material.py evaluate --parameters compute/material_parameters.json
.venv/Scripts/python compute/scripts/derive_storage_intervention.py
```

On POSIX use `.venv/bin/python`. Below, `python` denotes that interpreter.
The package uses NumPy and SciPy with double precision; it needs no GPU.

The first scientific command reproduces the analytical shear example and both
storage variants. The second evaluates the full-precision parameter input on
all five measured paths, using 80 uniform intervals plus every observation
strain. The third freezes plane storage at its initial value on CU-Z and
CD-P with the same parameters. It follows the material evaluation because
it reads the generated material record.

Outputs are written under `results/constitutive-method/`:
`energy_coupling_path.csv`, `energy_coupling_summary.json`,
`material_fit.json`, `material_paths.csv`, and `storage_intervention.csv`.
These generated files are not versioned.

The material input gives stress peaks of approximately 24.35, 45.73, 21.57,
25.67 and 34.42 MPa for CU-S, CU-P, CU-Z, CD-S and CD-P. CU-Z ends at
0.763 MPa excess pore pressure; freezing plane storage gives a 36.11 MPa peak
and −1.989 MPa terminal pressure. Both CD volume windows are included in the
fit. The manuscript discusses remaining peak-strain and CU-P effective-stress
differences.

## Continue or start a calibration

```powershell
python compute/scripts/fit_coupled_material.py fit --workers 5
python compute/scripts/fit_coupled_material.py fit --parameters compute/material_parameters.json --workers 5
python compute/scripts/fit_coupled_material.py fit --fresh --workers 5
```

The first command continues the saved fit. The second starts from the
published material input. The third starts from the initializer in the
source. COBYQA minimizes channel-normalized squared residuals and five
peak-stress residuals. New fits can enter different branches and need not
recover the published attained state.

Compression is positive, computational strain is dimensionless and stress
is in MPa. CU tests conserve fluid content at constant total radial stress
after preparation at 13 MPa; their measured pressure is a calibration channel.
CD tests hold pore pressure at zero after preparation at 5 MPa. Each channel
has equal total weight, with scales of 1 MPa for stress and pressure and the
digitized line half-width for each CD volume trace.

## Source and measurements

- `energetic_coupling.py`: reciprocal storage, conjugate forces and the convex
  monotonic shear slice.
- `poromechanical_coupling.py`: reciprocal TI elasticity, porous energy,
  contact and compaction.
- `coupled_friction_damage.py`: simultaneous matrix flow, Coulomb plane slip,
  damage and material-point branch continuation.
- `fit_coupled_material.py`: loading, observations, evolution parameters,
  scaled residuals, calibration and response export.

Measured stress and pressure traces are in
`compute/results/data/nordas_curves/`; measured CD volume windows and
line-width bands are in `selfrev-r3/evidence/SR-10_measured_volume.json`.
The historical folder name is retained by the common input loader.
All were digitized from Nordas et al. (2024), *Rock Mechanics and Rock
Engineering*, [DOI 10.1007/s00603-023-03509-7](https://doi.org/10.1007/s00603-023-03509-7),
Figs. 2a–f and 3a–d. They are published-curve readings, not the source authors'
original laboratory data. `compute/material_parameters.json` contains the
numerical input for replay; fixed grain/fluid properties and evolution-law
conventions are explicit in the producer.

Run the mechanical identity, contact, evolution and branch tests with:

```powershell
python -m pytest compute/tests -q
```

Earlier commits retain the preceding paired-calibration study. The current
files implement the constitutive method described above.
