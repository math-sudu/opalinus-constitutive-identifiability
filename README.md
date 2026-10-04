# Opalinus Clay constitutive parameter identifiability

Source code and measured inputs for *Parameter identifiability in two
anisotropic constitutive models for Opalinus Clay*, by Pengcheng Zhu and
Tielin Chen. Form I combines transverse elastic isotropy, matrix and bedding
returns, and cohesion softening. Form II adds fabric-dependent matrix strength.

## Installation and input check

The research calculations used CPython 3.13.7 on Windows, CPU JAX/JAXLIB
0.10.0, NumPy 2.4.4 and SciPy 1.16.2. Direct dependencies are pinned below.
JAX uses double precision. Run from the repository root:

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r compute/requirements.txt
.venv/Scripts/python compute/scripts/run_from_inputs.py check --form 1
.venv/Scripts/python compute/scripts/run_from_inputs.py check --form 2
```

On POSIX, use `.venv/bin/python`. In the following commands, `python` means
the environment interpreter. `check` integrates the five paths with nominal
parameters, evaluates all 350 measurement stations and the operative elastic
constants, and checks specimen orientation and finite model output.

## Recalculate the analysis

```powershell
python compute/scripts/run_from_inputs.py fit --form 1
python compute/scripts/run_from_inputs.py fit --form 2
python compute/scripts/run_from_inputs.py profiles --form 1
python compute/scripts/run_from_inputs.py profiles --form 2
python compute/scripts/run_from_inputs.py elastic --form 1
python compute/scripts/run_from_inputs.py elastic --form 2
python compute/scripts/run_from_inputs.py responses --form 1
python compute/scripts/run_from_inputs.py responses --form 2
```

`fit` draws 64 admissible starts with seed 20260706, runs the original 2000-step
Adam initialization on 400 integration steps, then continues each valid state
with bounded least squares on 3200 steps for form I or 1600 for form II.
It records failed starts explicitly and selects the smallest attained NLL.
Interrupted bounded fits resume from their saved iterate. Optimization and
profiles can take substantial CPU time; the first use also compiles JAX code.
The paper used continued fits across integration refinements. A fresh run
recalculates the analysis from random starts, so optimizer basins and their
associated local profile intervals can differ from the published attained fit.

`profiles` follows the connected coordinate likelihood interval at
delta NLL = 1.92, including nuisance refits and domain edges. Use
`--names c_j k_sharpness` to select coordinates. `elastic` applies exact
constraints to the assembled compliance; selectable names are `E_p`, `E_o`,
`nu_pp`, `nu_po`, and `nu_op`. `G_op` already has a coordinate profile.
`responses` computes joint-on/off histories on the measured paths and at
19 bedding angles under constant 5 MPa effective confinement. The net joint
effect is `100*(gamma_on-gamma_off)/gamma_on` at the final step.

Generated fit states and response files are written to `results/from-inputs/`.
Connected coordinate and operative profiles use the paths declared in their
producer modules. These generated files are excluded from version control.
The `selfrev-r2/evidence/` and `selfrev-r3/evidence/` paths retain the original
numerical producer locations; the repository contains their source code only.

## Measurements and conventions

Measured inputs were digitized by the authors from Nordas et al.,
*Rock Mechanics and Rock Engineering*,
[DOI 10.1007/s00603-023-03509-7](https://doi.org/10.1007/s00603-023-03509-7).
Deviatoric stresses come from Figs. 2a–c and 3a,b; excess pore pressures
come from Figs. 2d–f. The reading rules and station arrays are in
`compute/results/data/nordas_curves/`. These are digitized published curves,
not the source authors' original laboratory records.

The five specimens are CU-S, CU-P, CU-Z, CD-S and CD-P, with bedding-plane
angles 90, 0, 30, 90 and 0 degrees to the loading axis. Compression is positive,
stress is in MPa, and computational strain is dimensionless. The CU paths
follow measured excess pore pressures from 13 MPa initial effective
confinement; CD paths remain at 5 MPa. The independent Gaussian residual
standard deviation is 1 MPa. The shared likelihood loader retains legacy
constant-confinement offset fields, which are inactive on these measured
effective-stress paths.

To repeat image reading, obtain the source article through its DOI and run:

```powershell
python compute/scripts/digitize_source_curves.py --pdf path/to/source.pdf --out-dir results/reading
```

The script reads native embedded images; scanned or recompressed copies may
have different image identifiers. The source PDF is not included.

Parameter order, units and admissibility are defined in
`compute/src/tfenn_invcalib/theta_space.py`; the differentiable constitutive
implementation is `mirror.py`, and operative elastic mapping is
`operative_elastic.py`. The release includes no generated results,
checkpoints, manuscript files, figures or research history.
