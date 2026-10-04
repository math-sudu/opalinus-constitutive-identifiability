# Constitutive dependence of calibrated properties in anisotropic Opalinus Clay

Code, measured inputs and full-precision replay states for the paper by Pengcheng Zhu and
Tielin Chen. Form I combines transverse elastic isotropy, matrix and bedding
returns, and cohesion softening. Form II adds fabric-dependent matrix strength.

## Installation

The research calculations used CPython 3.13.7 on Windows, CPU JAX/JAXLIB
0.10.0, NumPy 2.4.4 and SciPy 1.16.2. Direct dependencies are pinned below.
JAX uses double precision. Run from the repository root:

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r compute/requirements.txt
.venv/Scripts/python compute/scripts/replay_reference.py check --form 1
.venv/Scripts/python compute/scripts/replay_reference.py check --form 2
```

On POSIX, use `.venv/bin/python`. In the following commands, `python` means
the environment interpreter. `check` replays the published reference and four
constrained cohesion/sharpness states per form. It checks their parameter
vectors and NLLs, then outputs all 350 predicted stress ordinates and the RMSE.
The reference NLLs are 2827.773495797381 and 2240.5364175963173, on 3200 and
1600 integration steps for forms I and II, respectively.

## Replay the reported instance

Run each command with `--form 1` or `--form 2`:

```powershell
python compute/scripts/replay_reference.py summary --form 1
python compute/scripts/replay_reference.py profiles --form 1
python compute/scripts/replay_reference.py profiles --operative --form 1
python compute/scripts/replay_reference.py cross-starts --form 1
python compute/scripts/replay_reference.py responses --form 1
python compute/scripts/replay_reference.py deformation --form 1
python compute/scripts/replay_reference.py residuals --form 1
python compute/scripts/replay_reference.py expanded --form 1
python compute/scripts/replay_reference.py profile-point --form 1 --coordinate c_j --side lower
```

`compute/reference_inputs.json` contains the complete physical `theta` and
internal `u` vectors, reference objectives, retained nuisance vectors, fixed
coordinates and endpoint brackets. It also supplies the independent CD-volume
readings and the covariance definitions. Its input bindings identify the
exported model, data and driver bytes; `.gitattributes` preserves their LF
line endings across Git checkouts. `summary` prints parameter ordering and
the retained intervals; `profiles` evaluates every supplied row with the model.
Use `--names c_j k_sharpness` to select coordinate profiles, or
`--operative --names nu_pp` for the operative Poisson profile.

`cross-starts` replays the completed comparisons between three attained
starting regions. `profile-point` performs new nuisance fits from the retained
row, reference and those starts, reports failures, and chooses the lowest NLL.
If it improves the reference, an unconstrained continuation supplies the new
reference and the output identifies the need to rebuild the profiles.
The supplied states are preserved.

`responses` regenerates measured-path and drained 5 MPa joint-on/off histories
at 19 bedding angles. `deformation` evaluates CU volume at 0.1% axial strain
and at the measured endpoints, and CD volume on the digitized comparison
windows. `residuals` replays the sigma=2/4 cohesion, sharpness and cross-plane
Poisson brackets,
the short-range correlated nuisance fits and the matched-variance shear
comparison. `expanded` replays the retained cohesion, operative Poisson and
fabric-profile brackets on both integration grids, including 51200/204800
steps for form II. Use `--names c_j` to select one expanded quantity.

Results print as JSON; `--output results/example.json` writes a new file.
Existing output files are not overwritten. The first numerical call compiles
JAX code; complete profile and fine-grid replay takes longer than `check`.

## Start a new calibration

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
delta NLL = 1.92, compares the successful fitted-member starts at each nuisance
fit, and restarts the profiles when a lower unconstrained reference is found. Use
`--names c_j k_sharpness` to select coordinates. `elastic` applies exact
constraints to the assembled compliance with the same comparison and reference
update rule; selectable names are `E_p`, `E_o`,
`nu_pp`, `nu_po`, and `nu_op`. `G_op` already has a coordinate profile.
`responses` computes joint-on/off histories on the measured paths and at
19 bedding angles under constant 5 MPa effective confinement. The net joint
effect is `100*(gamma_on-gamma_off)/gamma_on` at the final step.

Generated fit states, responses and profiles are written to `results/from-inputs/`.
These generated files are excluded from version control.
The `selfrev-r2/evidence/` and `selfrev-r3/evidence/` paths retain the original
numerical producer locations; the repository contains their source code only.

## Measurements and conventions

Measured inputs were digitized by the authors from Nordas et al.,
*Rock Mechanics and Rock Engineering*,
[DOI 10.1007/s00603-023-03509-7](https://doi.org/10.1007/s00603-023-03509-7).
Deviatoric stresses come from Figs. 2a–c and 3a,b; excess pore pressures
come from Figs. 2d–f. The reading rules and station arrays are in
`compute/results/data/nordas_curves/`. CD-volume readings from Figs. 3c,d,
their pixel ranges and comparison masks are in `compute/reference_inputs.json`.
These are digitized published curves,
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
`operative_elastic.py`. Full-precision states are numerical inputs to the
reported-instance replay. Stress/strain arrays and other model outputs are
regenerated by the commands above.
