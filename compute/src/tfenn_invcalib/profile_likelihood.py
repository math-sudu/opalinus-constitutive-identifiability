"""Profile likelihood over every coordinate of the prior box.

Role: for each parameter in turn, hold it at a grid of values across its FULL
prior box and re-optimise every other parameter, so the reported width is the
likelihood's own statement about that coordinate rather than the scatter of a
finite set of optimiser endpoints. Nothing here depends on ensemble size or on
an admission rule.

Grid and measure. Grid points are evenly spaced in the BOX FRACTION u -- the
transform-native measure, log for the log-sampled coordinates and linear
otherwise -- so a grid is uniform in the same measure the prior box is drawn in
and in which the ensemble band ratio R is defined. The endpoints are excluded:
u = 0 and u = 1 are the box bounds, where the z-transform is singular and no
finite z holds the coordinate.

Re-optimisation. At each grid point the held coordinate is frozen and every
other coordinate is re-optimised by the SAME adam-in-z optimiser the inversion
uses (``ensemble.run_adam_z``), not a second optimiser that could disagree with
it. Freezing is done by masking: the row's objective sees
``mask * z + frozen``, so the held coordinate contributes a constant, its
gradient is identically zero, and adam leaves it alone. All (coordinate, grid
point) rows advance as one batch.

Continuation, and why it marches instead of sweeping. A profile is only a
lower envelope if every row actually reached its own optimum, and a row started
from the global optimum can be left in the wrong basin when the held value is
far from it. The cure is to walk a converged basin along the grid, one step at
a time:

  - each coordinate's ANCHOR is the grid point nearest u*, seeded from theta*
    itself, which is where the profile is known to touch zero;
  - from the anchor the profile marches OUTWARD in both directions, each row
    seeded from its already-converged immediate neighbour, so every
    re-optimisation is a one-grid-step perturbation of a solved problem;
  - then an ESCALATING polish ladder, each pass at a larger budget than the
    last, re-seeded from the better neighbour and run only on rows not yet
    certified. A row is certified when a pass at a larger budget buys it no
    more than the tolerance -- a measurement of its convergence rather than an
    assumption about it. Rows still improving when the ladder runs out are
    reported as uncertified.

Two earlier designs failed here, and both failures are on the record because
they shape the settings above. The first ran the continuation as simultaneous
sweeps over all rows: that propagates a good basin only ONE grid point per
sweep, so reaching a box edge needs about n_grid/2 of them. The second kept a
single modest budget for every row. Measured convergence
(evidence/SR-04_profile_budget_probe.py) shows why that cannot work: at 400
iterations most rows are within 0.02 NLL of their 8000-iteration value while a
minority sit 18 NLL high -- ten times the cut this file reads intervals at --
and the seed barely matters. A uniform small budget therefore corrupts exactly
the rows whose shape decides a classification, and a uniform large one spends
almost all of its work on rows that converged in the first few hundred
iterations.

Rows whose objective goes non-finite are recorded as FAILED and excluded from
the profile's minimum, its maximum and its interval, rather than entering the
curve as an enormous finite number that would read as a sharply-determined
coordinate.

Batching. Each march step advances one row per coordinate, so the batch is
n_theta rows wide and the schedule compiles once per step shape. That also
bounds peak memory: a reverse-mode gradient through a 400-step integration
keeps a tape proportional to the batch size, and a full (coordinate x grid)
batch is several times the multi-start ensemble's.

Reported per coordinate: the profile NLL curve, the 95 per cent
profile-likelihood interval at Delta NLL = 1.92 (the chi-square(1) 0.95 quantile
halved), the maximum Delta NLL over the box, and whether each interval endpoint
is interior or AT A BOX EDGE. The edge flag is the prior-edge test: an interval
that runs into its own box bound is a statement about the box, not about the
data, and a coordinate whose identified band is pressed against a bound is
bounded on one side rather than constrained.

Classification, replacing the band-ratio cuts:
  identified                    both interval endpoints interior to the box
  bounded from above / below    one endpoint interior, the other at a box edge
  non-identifiable at box scale max Delta NLL over the box < 1.92

The band-ratio classification is recomputed beside it from the ensemble
artifact and the two are compared parameter by parameter. Whether the
constrained/equifinal split reproduces is a measured outcome and is reported
either way.

Touch rule (design doc D14 decision 1). A profile classification is only
defined where the curve TOUCHES the reference. If the minimum Delta NLL over
the usable grid is >= the 1.92 cut, the whole curve sits at or above the level
the interval would be read at, any "crossing" the walk could return is an
artifact of adjacent grid points rather than a measurement (the defect that
gave form-II nu_op a 0.071-wide "identified" interval on a curve whose minimum
was +2.685), and the coordinate is reported as ``undetermined (profile does
not touch the reference)`` with no interval. If the minimum sits between 0.1
and 1.92 the classification stands but the artifact carries the touch deficit
as a warning field -- 0.1 clears the measured optimisation residuals
(budget-probe 4000-vs-8000 residual 0.065, form-I worst touch 0.038). The
touch value (min Delta NLL over the usable grid) is stated per coordinate in
every mode.

Per-row certification (D14 decision 4). Every artifact records, per
(coordinate, grid point) row, whether the row is certified and the NLL
improvement bought by the last polish pass that ran it, alongside the existing
counts -- "which rows are uncertified" is read from the artifact, never
inferred from counts.

Coordinate subsets and continuation. ``run_profile_likelihood`` accepts a
subset of internal coordinate names and runs the march/ladder/readout on those
coordinates only. ``run_profile_continuation`` restores each selected
coordinate's rows (grid, NLLs, theta vectors) from a prior artifact, skips the
march, and re-optimises every row from its OWN restored point -- never from a
neighbour, because the question a continuation answers is whether the row's
own claimed value is tight. Its artifact is a complete replacement of the
prior readout: non-selected coordinates' stored rows are carried unchanged,
and every coordinate is re-read against the continuation run's own reference
(the lower of the optimum's NLL and every row NLL, selected or carried).
"""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import brentq

import jax
import jax.numpy as jnp

from . import ensemble

# chi-square(1) 0.95 quantile halved: the 95 per cent profile-likelihood cut.
DELTA_NLL_95 = 1.92
# A polish pass improving no row by more than this (NLL) counts as converged.
SWEEP_IMPROVEMENT_TOL = 1e-3
# Smallest box fraction the warm start z* is allowed to take. An attained
# optimum can sit ON a box bound (the derivative-free lane optimises in z and
# saturates the logistic), where z is infinite and no warm start exists; the
# warm start is pulled this far inside instead.
#
# The margin is chosen to keep the start MOVABLE, not merely representable.
# Measured at the form-I optimum: four coordinates (nu_pp, nu_op, r_res,
# phi_j) sit at box fractions 1.5e-08 to 3.3e-07, where the transform Jacobian
# du/dz is 1e-08 to 3e-07 and their gradient components are ~5e-06, three
# thousandths of the gradient norm. At 1e-04 the Jacobian is 1e-04 and the
# whole move costs +0.073 NLL -- far below the 1.92 cut this file reads
# intervals at -- while 1e-03 already costs +0.86 and 1e-02 costs +19.4. The
# cost of the move is measured and reported per run, so it stays a
# representation detail on the record rather than a silent change of optimum.
Z_START_EPS = 1e-4
# Local-grid spacing where a coordinate has no finite Cramer-Rao bound: an
# exactly flat direction has no interval to resolve, so the local points only
# need to sit near u*, and a declared fraction of the box does that.
LOCAL_GRID_FALLBACK_U = 0.02
# Band-ratio cuts recomputed for comparison (the classification being replaced).
BAND_RATIO_CUTS = (0.15, 0.50)
# Touch rule (D14 decision 1): a classification is only defined where the
# curve touches the reference. min Delta NLL >= DELTA_NLL_95 over the usable
# grid means no crossing is a measurement and the coordinate is undetermined;
# a minimum between TOUCH_WARNING_DNLL and the cut keeps the classification
# but carries the touch deficit as a warning field. 0.1 clears the measured
# optimisation residuals (budget probe 4000-vs-8000: 0.065; form-I worst
# touch: 0.038).
TOUCH_WARNING_DNLL = 0.1

CLASS_IDENTIFIED = "identified"
CLASS_BOUNDED_ABOVE = "bounded from above"
CLASS_BOUNDED_BELOW = "bounded from below"
CLASS_NON_IDENTIFIABLE = "non-identifiable at box scale"
CLASS_UNDETERMINED_NO_TOUCH = "undetermined (profile does not touch the reference)"

# Artifact strings shared by the from-scratch and continuation modes, kept as
# single sources so the two artifacts cannot drift apart.
_ROLE_NOTE = ("profile likelihood over the full prior box, one coordinate at "
              "a time; independent of ensemble size, band width and "
              "admission rule")
_OPTIMISER_NOTE = ("ensemble.run_adam_z -- the same batched adam-in-z the "
                   "inversion uses; the held coordinate is frozen by masking "
                   "so its gradient is identically zero")
_FAILED_ROW_NOTE = ("a row whose objective went non-finite is recorded "
                    "as failed and excluded from the profile's minimum, "
                    "maximum and interval; it is not a large Delta NLL")
_CERTIFICATION_READING = ("a row is certified when a pass at a larger budget "
                          "buys it no more than the tolerance; uncertified "
                          "rows are still descending and their profile value "
                          "is an upper bound, so any interval or class "
                          "resting on them is not established")


def grid_fractions(n_grid: int) -> np.ndarray:
    """``n_grid`` interior box fractions, evenly spaced, endpoints excluded."""
    return np.linspace(0.0, 1.0, n_grid + 2)[1:-1]


def _logit(u: np.ndarray) -> np.ndarray:
    return np.log(u / (1.0 - u))


def _dedupe_and_pad(pts: np.ndarray, n_out: int,
                    tol: float = 1e-9) -> np.ndarray:
    """Sorted, duplicate-free grid of exactly ``n_out`` points.

    Merging a coarse box scan with a Cramer-Rao-spaced local grid produces
    duplicates whenever the local offsets run off the box: at an optimum
    sitting on a bound, every negative offset clips to the same point, and a
    coordinate with no finite bound spreads its local points onto the two
    bounds. Those collapsed points are re-optimised rows that measure a value
    already known, so they are removed and the freed rows are placed at the
    midpoints of the widest remaining gaps, where the curve is least resolved.
    """
    keep = np.sort(np.asarray(pts, dtype=float))
    keep = keep[np.concatenate([[True], np.diff(keep) > tol])]
    while keep.size < n_out:
        gaps = np.diff(keep)
        j = int(np.argmax(gaps))
        keep = np.insert(keep, j + 1, 0.5 * (keep[j] + keep[j + 1]))
    return keep[:n_out]


def _optimise_rows(like, n_theta: int, coords: np.ndarray, z_fix: np.ndarray,
                   z_seed: np.ndarray, iters: int, lr: float):
    """Re-optimise a batch of rows, each with one coordinate held fixed.

    ``coords[r]`` is the frozen coordinate of row r and ``z_fix[r]`` the
    z-value it is held at. Freezing is by masking, so the held coordinate
    contributes a constant and its gradient is identically zero.
    """
    n_rows = coords.size
    mask = np.ones((n_rows, n_theta))
    frozen = np.zeros((n_rows, n_theta))
    mask[np.arange(n_rows), coords] = 0.0
    frozen[np.arange(n_rows), coords] = z_fix
    MASK = jnp.asarray(mask)
    FROZEN = jnp.asarray(frozen)

    def _row(z_row, mask_row, frozen_row):
        return like.nll_z(mask_row * z_row + frozen_row)

    vg_raw = jax.vmap(jax.value_and_grad(_row, argnums=0), in_axes=(0, 0, 0))
    nll_raw = jax.jit(jax.vmap(_row, in_axes=(0, 0, 0)))
    run = ensemble.run_adam_z(lambda z: vg_raw(z, MASK, FROZEN),
                              lambda z: nll_raw(z, MASK, FROZEN),
                              z_seed, iters, lr=lr)
    nll = np.asarray(run.best_nll)
    z = np.asarray(run.best_z)
    masked = mask * z + frozen
    return nll, z, masked


def _march(like, n_theta: int, n_grid: int, z_fix_grid: np.ndarray,
           anchors: np.ndarray, z_star: np.ndarray,
           anchor_iters: int, step_iters: int, lr: float,
           sel: np.ndarray):
    """Outward continuation from each selected coordinate's anchor grid point.

    ``sel`` holds the GLOBAL theta indices of the coordinates being profiled;
    ``z_fix_grid`` and ``anchors`` are position-aligned with it (row p of
    ``z_fix_grid`` is the grid of coordinate ``sel[p]``). Returns per-row best
    NLL, best z and masked z (row index p*n_grid + g), plus a per-step record.
    Every row is seeded from its already-converged immediate neighbour, so no
    re-optimisation is ever more than one grid step away from a solved
    problem.
    """
    n_sel = sel.size
    n_rows = n_sel * n_grid
    best_nll = np.full(n_rows, np.inf)
    best_z = np.tile(z_star, (n_rows, 1))
    masked_z = np.zeros((n_rows, n_theta))
    steps = []
    pos_all = np.arange(n_sel)

    def _row_index(p, g):
        return p * n_grid + g

    # --- the anchors: seeded from theta* itself ---------------------------
    rows = np.array([_row_index(p, anchors[p]) for p in pos_all])
    nll, z, mk = _optimise_rows(like, n_theta, sel[pos_all],
                                z_fix_grid[pos_all, anchors],
                                np.tile(z_star, (n_sel, 1)),
                                anchor_iters, lr)
    best_nll[rows], best_z[rows], masked_z[rows] = nll, z, mk
    steps.append({"stage": "anchor", "offset": 0, "n_rows": int(rows.size),
                  "n_iters": anchor_iters,
                  "n_rows_nonfinite": int(np.sum(~np.isfinite(nll))),
                  "best_row_nll": float(np.min(nll))})

    # --- march outward, one grid step at a time, in both directions -------
    for direction in (+1, -1):
        for k in range(1, n_grid):
            g = anchors + direction * k
            live = pos_all[(g >= 0) & (g < n_grid)]
            if live.size == 0:
                break
            g_live = g[live]
            prev = np.array([_row_index(p, gi - direction)
                             for p, gi in zip(live, g_live)])
            rows = np.array([_row_index(p, gi) for p, gi in zip(live, g_live)])
            seed = best_z[prev]
            nll, z, mk = _optimise_rows(like, n_theta, sel[live],
                                        z_fix_grid[live, g_live],
                                        seed, step_iters, lr)
            better = nll < best_nll[rows]
            best_nll[rows[better]] = nll[better]
            best_z[rows[better]] = z[better]
            masked_z[rows[better]] = mk[better]
            steps.append({"stage": "march", "direction": int(direction),
                          "offset": int(direction * k), "n_rows": int(live.size),
                          "n_iters": step_iters,
                          "n_rows_nonfinite": int(np.sum(~np.isfinite(nll))),
                          "best_row_nll": float(np.min(nll))})
    return best_nll, best_z, masked_z, steps


def _polish_ladder(like, n_theta: int, n_grid: int, z_fix_grid: np.ndarray,
                   best_nll: np.ndarray, best_z: np.ndarray,
                   masked_z: np.ndarray, ladder: tuple[int, ...], lr: float,
                   tol: float, sel: np.ndarray):
    """Escalating re-optimisation passes, run only on rows not yet certified.

    Why a ladder rather than one budget for everyone. Measured on
    representative rows (evidence/SR-04_profile_budget_probe.py): most rows are
    within 0.02 NLL of their 8000-iteration value after 400 iterations, while a
    minority sit 18 NLL high there -- ten times the 1.92 cut this file reads
    intervals at -- and need a few thousand. Giving every row the largest
    budget wastes almost all of it; giving every row the small one silently
    corrupts the minority, and those are exactly the rows whose profile shape
    decides a classification.

    So each pass runs at a larger budget than the last, re-seeded from the
    better neighbour, and only on rows that are not yet certified. A row is
    CERTIFIED when a pass at a larger budget buys it no more than ``tol``: that
    is a measurement of its convergence, not an assumption about it. Rows still
    improving when the ladder runs out are reported as uncertified, and the
    profile says so rather than presenting them as converged.

    ``sel`` holds the GLOBAL theta indices of the profiled coordinates;
    ``z_fix_grid`` and the row layout (p*n_grid + g) are position-aligned with
    it. Besides the summary and the per-row certified flags, returns per-row
    ``last_improvement``: the NLL improvement bought by the LAST pass that ran
    each row (0.0 when that pass bought nothing; NaN for a row no pass ran,
    which cannot happen here because the first pass runs every row, but the
    continuation shares the convention).
    """
    n_sel = sel.size
    pos_all = np.repeat(np.arange(n_sel), n_grid)
    coords_global = sel[pos_all]
    grids_all = np.tile(np.arange(n_grid), n_sel)
    n_rows = pos_all.size
    certified = np.zeros(n_rows, dtype=bool)
    last_improvement = np.full(n_rows, np.nan)
    records = []

    for iters in ladder:
        live = np.flatnonzero(~certified)
        if live.size == 0:
            break
        seed = np.array(best_z[live], dtype=float)
        for a, r in enumerate(live):
            g = grids_all[r]
            cands = [r + d for d in (-1, +1) if 0 <= g + d < n_grid]
            if cands:
                seed[a] = best_z[min(cands, key=lambda c: best_nll[c])]
        nll, z, mk = _optimise_rows(
            like, n_theta, coords_global[live],
            z_fix_grid[pos_all[live], grids_all[live]], seed, iters, lr)
        improved = np.isfinite(nll) & (nll < best_nll[live])
        gain = np.where(improved, best_nll[live] - nll, 0.0)
        rows_imp = live[improved]
        best_nll[rows_imp] = nll[improved]
        best_z[rows_imp] = z[improved]
        masked_z[rows_imp] = mk[improved]
        # certified: this pass, at a larger budget, bought no more than tol
        certified[live[gain <= tol]] = True
        last_improvement[live] = gain
        records.append({
            "stage": "polish", "n_iters": iters, "n_rows_run": int(live.size),
            "n_rows_improved": int(np.sum(improved)),
            "max_improvement_NLL": float(np.max(gain)) if gain.size else 0.0,
            "mean_improvement_NLL": (float(np.mean(gain[improved]))
                                     if np.any(improved) else 0.0),
            "n_rows_certified_after": int(np.sum(certified)),
            "n_rows_nonfinite": int(np.sum(~np.isfinite(nll))),
        })

    summary = {
        "ladder": list(ladder),
        "passes": records,
        "convergence_tolerance_NLL": tol,
        "n_rows_certified": int(np.sum(certified)),
        "n_rows_uncertified": int(np.sum(~certified)),
        "uncertified_rows": [
            {"coordinate": int(coords_global[r]), "grid_index": int(grids_all[r])}
            for r in np.flatnonzero(~certified)],
        "converged": bool(np.all(certified)),
        "reading": _CERTIFICATION_READING,
    }
    return best_nll, best_z, masked_z, summary, certified, last_improvement


def solve_connected_crossings(u_grid, nll, reference_nll, evaluate):
    """Solve the statistical level on each side of the connected minimum.

    ``evaluate(u)`` performs the constrained nuisance fit and retains that
    actual row. A single linear interpolation can put an endpoint far below
    the likelihood cut when a broad initial bracket crosses a curved profile.
    Brent's bracketed solve uses the existing 1.92 level directly.
    """
    order = np.argsort(u_grid)
    u = np.asarray(u_grid)[order]
    delta = np.asarray(nll)[order] - reference_nll - DELTA_NLL_95
    if not np.isfinite(delta).all():
        raise ValueError("Cannot locate a profile level through invalid rows")
    center = int(np.argmin(delta))
    if delta[center] >= 0:
        raise ValueError("Profile does not reach the reference likelihood component")
    for direction in (-1, 1):
        inside = center
        outside = inside + direction
        while 0 <= outside < len(u) and delta[outside] < 0:
            inside, outside = outside, outside + direction
        if 0 <= outside < len(u):
            a, b = sorted((float(u[inside]), float(u[outside])))
            brentq(lambda value: evaluate(value) - reference_nll - DELTA_NLL_95, a, b)


def _interval_and_class(u_grid: np.ndarray, dnll: np.ndarray,
                        valid: np.ndarray) -> dict:
    """95 per cent interval, edge flags and class from one profile curve.

    ``valid`` marks the grid points whose re-optimisation returned a finite
    objective. A failed point is not a very large Delta NLL -- it is no
    measurement at all -- so the walk treats it as the end of the usable grid
    and the run records how many were lost. Reading a failed row as a huge
    finite value would turn a broken optimisation into a sharply-determined
    coordinate, which is the one misreading this diagnostic cannot afford.
    """
    ok = np.flatnonzero(valid)
    if ok.size == 0:
        return {"interval_u": [None, None], "interval_width_u": None,
                "lower_endpoint_at_box_edge": None,
                "upper_endpoint_at_box_edge": None,
                "lower_walk_stopped_by_failed_point": None,
                "upper_walk_stopped_by_failed_point": None,
                "max_delta_nll_over_box": None,
                "min_delta_nll_over_grid": None,
                "touch_deficit_warning_NLL": None,
                "argmin_grid_index": None,
                "argmin_u": None, "n_failed_grid_points": int(valid.size),
                "interval_resolution_u": float(u_grid[1] - u_grid[0]),
                "interval_width_at_grid_resolution": None,
                "classification": "undetermined (every grid point failed)"}
    g_min = int(ok[np.argmin(dnll[ok])])
    max_dnll = float(np.max(dnll[ok]))
    min_dnll = float(dnll[g_min])

    # Touch rule (D14 decision 1): the classification is only defined where
    # the curve touches the reference. A curve whose minimum already sits at
    # or above the cut has no crossing to measure -- the outward walk would
    # only return adjacent grid points of a curve that never comes down, which
    # is how a wholly-superseded form-II curve once read out as a 0.071-wide
    # "identified" interval. No interval is reported.
    if min_dnll >= DELTA_NLL_95:
        return {"interval_u": [None, None], "interval_width_u": None,
                "lower_endpoint_at_box_edge": None,
                "upper_endpoint_at_box_edge": None,
                "lower_walk_stopped_by_failed_point": None,
                "upper_walk_stopped_by_failed_point": None,
                "max_delta_nll_over_box": max_dnll,
                "min_delta_nll_over_grid": min_dnll,
                "touch_deficit_warning_NLL": None,
                "argmin_grid_index": g_min,
                "argmin_u": float(u_grid[g_min]),
                "n_failed_grid_points": int(valid.size - ok.size),
                "interval_resolution_u": float(u_grid[1] - u_grid[0]),
                "interval_width_at_grid_resolution": None,
                "classification": CLASS_UNDETERMINED_NO_TOUCH}

    def _cross(direction: int) -> tuple[float, bool, bool]:
        """Walk outward from the minimum to the first crossing of the cut.

        Returns (u at the crossing, endpoint sits at a box edge, the walk was
        stopped by a failed grid point rather than by the box). The walk only
        runs when the touch rule above has established dnll[g_min] < the cut,
        so every point it stands on satisfies dnll[g] < DELTA_NLL_95 (it only
        advances past points below the cut) and the first point at or above
        the cut is always a genuine bracket.
        """
        g = g_min
        while True:
            g_next = g + direction
            if g_next < 0 or g_next >= u_grid.size:
                # never crossed inside the grid: the interval runs to the bound
                return (0.0 if direction < 0 else 1.0), True, False
            if not valid[g_next]:
                # the usable grid ends here; the endpoint is not established
                return float(u_grid[g]), False, True
            if dnll[g_next] >= DELTA_NLL_95 > dnll[g]:
                lo_u, hi_u = u_grid[g], u_grid[g_next]
                lo_d, hi_d = dnll[g], dnll[g_next]
                frac = ((DELTA_NLL_95 - lo_d) / (hi_d - lo_d)
                        if hi_d > lo_d else 0.0)
                return float(lo_u + frac * (hi_u - lo_u)), False, False
            g = g_next

    u_lo, lo_edge, lo_truncated = _cross(-1)
    u_hi, hi_edge, hi_truncated = _cross(+1)
    # A minimum sitting ON the first or last usable grid point has no interior
    # side to walk down: the profile is cut off by the box, not by the data.
    if g_min == int(ok[0]) and not lo_truncated:
        lo_edge = True
    if g_min == int(ok[-1]) and not hi_truncated:
        hi_edge = True
    if max_dnll < DELTA_NLL_95:
        cls = CLASS_NON_IDENTIFIABLE
    elif lo_edge and hi_edge:
        cls = CLASS_NON_IDENTIFIABLE
    elif lo_edge:
        cls = CLASS_BOUNDED_ABOVE
    elif hi_edge:
        cls = CLASS_BOUNDED_BELOW
    else:
        cls = CLASS_IDENTIFIED
    return {
        "interval_u": [u_lo, u_hi],
        "interval_width_u": float(u_hi - u_lo),
        "lower_endpoint_at_box_edge": bool(lo_edge),
        "upper_endpoint_at_box_edge": bool(hi_edge),
        "lower_walk_stopped_by_failed_point": bool(lo_truncated),
        "upper_walk_stopped_by_failed_point": bool(hi_truncated),
        "max_delta_nll_over_box": max_dnll,
        "min_delta_nll_over_grid": min_dnll,
        # 0.1 < min < 1.92: the classification stands, but the curve's
        # minimum clears the measured optimisation residuals, so the deficit
        # is carried as a warning rather than absorbed.
        "touch_deficit_warning_NLL": (min_dnll
                                      if min_dnll > TOUCH_WARNING_DNLL
                                      else None),
        "argmin_grid_index": g_min,
        "argmin_u": float(u_grid[g_min]),
        "n_failed_grid_points": int(valid.size - ok.size),
        "interval_resolution_u": float(u_grid[1] - u_grid[0]),
        "interval_width_at_grid_resolution": bool(
            (u_hi - u_lo) <= 2.0 * (u_grid[1] - u_grid[0]) + 1e-12),
        "classification": cls,
    }


def _optimum_context(like, space, theta_star: np.ndarray):
    """Optimum-side context shared by every mode of the lane.

    Returns (theta_star, nll_star, u_star_raw, z_star, warm_start_record):
    the NLL at theta*, its raw box fractions, the pulled-inside warm start in
    z (see Z_START_EPS) and the record documenting the cost of the pull.
    """
    n_theta = space.n_theta
    internal_names = space.internal_names
    theta_star = np.asarray(theta_star, dtype=float)
    nll_star = float(like.nll_theta_jit(jnp.asarray(theta_star)))
    u_star_raw = np.asarray(space.u_from_theta(theta_star), dtype=float)
    u_star = np.clip(u_star_raw, Z_START_EPS, 1.0 - Z_START_EPS)
    on_bound = [internal_names[i] for i in range(n_theta)
                if u_star_raw[i] != u_star[i]]
    z_star = _logit(u_star)
    theta_start = np.asarray(space.theta_from_u(jnp.asarray(u_star)))
    nll_start = float(like.nll_theta_jit(jnp.asarray(theta_start)))
    du_dz = u_star * (1.0 - u_star)
    warm_start = {
        "eps": Z_START_EPS,
        "coordinates_pulled_inside": on_bound,
        "nll_at_theta_star": nll_star,
        "nll_at_warm_start": nll_start,
        "nll_cost_of_pulling_inside": nll_start - nll_star,
        "min_transform_jacobian_du_dz": float(np.min(du_dz)),
        "note": ("an optimum sitting on a box bound has no finite z; the warm "
                 "start is pulled inside to a margin that keeps every "
                 "coordinate MOVABLE rather than merely representable, and "
                 "the NLL cost of that move is reported, not absorbed"),
    }
    return theta_star, nll_star, u_star_raw, z_star, warm_start


def _resolve_subset(space, coordinates) -> np.ndarray:
    """Global theta indices for a coordinate subset, in space order.

    ``None`` selects every coordinate. Unknown names raise with the list of
    valid internal names; an empty selection is refused.
    """
    internal_names = list(space.internal_names)
    if coordinates is None:
        return np.arange(space.n_theta)
    unknown = [nm for nm in coordinates if nm not in internal_names]
    if unknown:
        raise ValueError(f"unknown coordinate names {unknown}; valid internal "
                         f"names: {internal_names}")
    if not coordinates:
        raise ValueError(f"empty coordinate subset; valid internal names: "
                         f"{internal_names}")
    return np.array(sorted({internal_names.index(nm) for nm in coordinates}),
                    dtype=int)


def _touch_rule_record() -> dict:
    """The touch rule as an artifact block, so every run states it."""
    return {
        "undetermined_at_min_delta_nll": DELTA_NLL_95,
        "warning_above_min_delta_nll": TOUCH_WARNING_DNLL,
        "reading": ("a profile classification is only defined where the curve "
                    "touches the reference: min Delta NLL over the usable grid "
                    ">= 1.92 yields 'undetermined (profile does not touch the "
                    "reference)' with no interval; a minimum between 0.1 and "
                    "1.92 keeps the classification and carries the deficit in "
                    "touch_deficit_warning_NLL (0.1 clears the measured "
                    "optimisation residuals); min_delta_nll_over_grid states "
                    "the touch value per coordinate"),
    }


PER_ROW_CERTIFICATION_NOTE = (
    "per grid point, in grid order: grid_point_certified is true when the "
    "last polish pass that ran the row bought it no more than the improvement "
    "tolerance; grid_point_last_improvement_NLL is the NLL improvement bought "
    "by the LAST pass that ran the row in THIS process (0.0 when that pass "
    "bought nothing), null where no pass ran the row in this process (a row "
    "carried from a prior artifact, or an improvement measured against no "
    "finite stored value); a failed row (profile_nll null) is failed "
    "regardless of its flag and never enters the curve's minimum, maximum or "
    "interval")


def _readout_block(space, i: int, u_star_raw: np.ndarray,
                   u_grid_i: np.ndarray, nll_i: np.ndarray,
                   valid_i: np.ndarray, dnll_i: np.ndarray, theta_rows_i,
                   certified_i: np.ndarray, last_imp_i: np.ndarray,
                   grid_meta: dict) -> dict:
    """One coordinate's per_param block; shared by every mode of the lane.

    ``grid_meta`` carries the grid-construction fields (anchor grid index,
    local flags, local-grid scale and source) -- computed fresh in a
    from-scratch run, carried from the prior artifact in a continuation.
    ``theta_rows_i`` holds one full theta vector per grid point, as numpy rows
    or as stored lists carried verbatim.
    """
    internal_names = space.internal_names
    summary = _interval_and_class(u_grid_i, dnll_i, valid_i)
    u_lo, u_hi = summary["interval_u"]

    return {
        "theta_name": space.theta_names[i],
        "internal_name": internal_names[i],
        "u_star": float(u_star_raw[i]),
        "anchor_grid_index": grid_meta["anchor_grid_index"],
        "u_grid": np.asarray(u_grid_i, dtype=float).tolist(),
        "grid_point_is_local": grid_meta["grid_point_is_local"],
        "local_grid_scale_u": grid_meta["local_grid_scale_u"],
        "local_grid_scale_source": grid_meta["local_grid_scale_source"],
        "profile_nll": [None if not v else float(x)
                        for x, v in zip(nll_i, valid_i)],
        "delta_nll": [None if not v else float(x)
                      for x, v in zip(dnll_i, valid_i)],
        "n_grid_points_failed": int(np.sum(~valid_i)),
        "n_grid_points_uncertified": int(np.sum(~certified_i)),
        "all_grid_points_certified": bool(np.all(certified_i)),
        "grid_point_certified": [bool(c) for c in certified_i],
        "grid_point_last_improvement_NLL": [
            None if not np.isfinite(x) else float(x) for x in last_imp_i],
        "theta_at_grid": [t if isinstance(t, list)
                          else np.asarray(t, dtype=float).tolist()
                          for t in theta_rows_i],
        **summary,
        "interval_internal_units": _interval_internal_units(
            space, i, u_star_raw, u_lo, u_hi),
        "interval_note": ("interval endpoints in the coordinate's own "
                          "internal units, obtained by mapping the box "
                          "fractions back through the transform"),
    }


def _interval_internal_units(space, i: int, u_star_raw: np.ndarray,
                             u_lo, u_hi) -> list:
    """Interval endpoints mapped to the coordinate's own internal units."""
    n_theta = space.n_theta
    internal_names = space.internal_names

    def _theta_at(u_val):
        if u_val is None:
            return None
        th = np.asarray(space.theta_from_u(
            jnp.asarray(np.where(np.arange(n_theta) == i,
                                 np.clip(u_val, 1e-12, 1 - 1e-12),
                                 u_star_raw))))
        return (float(th[i] / th[5]) if internal_names[i] == "r_res"
                else float(th[i]))

    return [_theta_at(u_lo), _theta_at(u_hi)]


def run_profile_likelihood(like, space, theta_star: np.ndarray,
                           n_coarse: int = 13,
                           crb_u=None,
                           local_crb_multiples: tuple[float, ...] =
                           (0.5, 1.0, 1.5, 2.0, 3.0),
                           anchor_iters: int = 2000,
                           step_iters: int = 400,
                           polish_ladder: tuple[int, ...] = (800, 2000, 4000),
                           lr: float = 0.03,
                           improvement_tol: float = SWEEP_IMPROVEMENT_TOL,
                           coordinates: list[str] | None = None) -> dict:
    """Profile every coordinate over its full box; see the module docstring.

    ``coordinates`` restricts the run to a subset of internal coordinate
    names: the grids are built, the march/ladder run and the readout reported
    on those coordinates only, and the artifact declares the subset. ``None``
    profiles every coordinate.
    """
    t0 = time.time()
    n_theta = space.n_theta
    internal_names = space.internal_names
    coarse = grid_fractions(n_coarse)

    sel = _resolve_subset(space, coordinates)
    sel_names = [internal_names[i] for i in sel]
    theta_star, nll_star, u_star_raw, z_star, warm_start = _optimum_context(
        like, space, theta_star)

    # ---- per-coordinate two-scale grid -----------------------------------
    # The coarse grid answers the box-scale question -- is the profile ever
    # below the cut away from the optimum, i.e. structural versus practical
    # non-identifiability. It cannot answer the interval: measured on form I,
    # phi's Cramer-Rao bound is 0.0024 box fractions against a 13-point grid
    # spacing of 0.071, thirty times coarser, so the 1.92 crossing falls
    # between two adjacent points and every reported width collapses onto the
    # grid step. The local grid is therefore spaced by the coordinate's OWN
    # Cramer-Rao bound, which is the scale the crossing actually lives on, and
    # the two are merged into one sorted curve per coordinate. Where no finite
    # bound exists (an exactly flat direction) the local grid falls back to a
    # declared fraction of the box, since there is no interval to resolve.
    crb = (np.asarray(crb_u, dtype=float) if crb_u is not None
           else np.full(n_theta, np.nan))
    crb_used = np.where(np.isfinite(crb) & (crb > 0.0), crb,
                        LOCAL_GRID_FALLBACK_U)
    mult = np.asarray(local_crb_multiples, dtype=float)
    n_pts = coarse.size + 2 * mult.size + 1
    rows_u = []
    for i in sel:
        offs = np.concatenate([-mult[::-1], [0.0], mult]) * crb_used[i]
        local = u_star_raw[i] + offs
        pts = np.clip(np.concatenate([coarse, local]),
                      Z_START_EPS, 1.0 - Z_START_EPS)
        rows_u.append(_dedupe_and_pad(pts, n_pts))
    u_grid = np.stack(rows_u)                       # (n_sel, n_grid)
    n_grid = u_grid.shape[1]
    n_rows = sel.size * n_grid
    z_fix = _logit(u_grid)
    is_local = np.zeros_like(u_grid, dtype=bool)
    for p in range(sel.size):
        is_local[p] = ~np.isin(u_grid[p], np.clip(coarse, Z_START_EPS,
                                                  1.0 - Z_START_EPS))

    # Each coordinate's anchor: the grid point nearest its value at theta*,
    # where the profile is known to touch the attained optimum. The local grid
    # always contains u* itself, so the anchor is exact.
    anchors = np.array([int(np.argmin(np.abs(u_grid[p] - u_star_raw[i])))
                        for p, i in enumerate(sel)])
    best_nll, best_z, masked_z_all, march_steps = _march(
        like, n_theta, n_grid, z_fix, anchors, z_star,
        anchor_iters, step_iters, lr, sel)
    (best_nll, best_z, masked_z_all, polish_record, certified,
     last_improvement) = _polish_ladder(
        like, n_theta, n_grid, z_fix, best_nll, best_z, masked_z_all,
        polish_ladder, lr, improvement_tol, sel)
    total_row_iterations = (int(sum(s["n_rows"] * s["n_iters"]
                                    for s in march_steps))
                            + int(sum(p["n_rows_run"] * p["n_iters"]
                                      for p in polish_record["passes"])))

    # ---- reference and per-coordinate readout ----------------------------
    finite = np.isfinite(best_nll)
    row_min = float(np.min(best_nll[finite])) if np.any(finite) else float("nan")
    beat_optimum = bool(np.any(finite) and row_min < nll_star)
    reference = min(nll_star, row_min) if np.any(finite) else nll_star
    theta_rows = np.asarray(jax.vmap(space.theta_from_z)(
        jnp.asarray(np.where(np.isfinite(masked_z_all), masked_z_all, 0.0))))

    per_param = {}
    for p, i in enumerate(sel):
        sl = slice(p * n_grid, (p + 1) * n_grid)
        nll_i = best_nll[sl]
        valid_i = np.isfinite(nll_i)
        dnll_i = np.where(valid_i, nll_i - reference, np.nan)
        per_param[internal_names[i]] = _readout_block(
            space, i, u_star_raw, u_grid[p], nll_i, valid_i, dnll_i,
            theta_rows[sl], certified[sl], last_improvement[sl],
            grid_meta={
                "anchor_grid_index": int(anchors[p]),
                "grid_point_is_local": is_local[p].tolist(),
                "local_grid_scale_u": float(crb_used[i]),
                "local_grid_scale_source": (
                    "Cramer-Rao bound in box fractions"
                    if np.isfinite(crb[i]) and crb[i] > 0.0
                    else "fallback (no finite bound)"),
            })

    return {
        "role": _ROLE_NOTE,
        "mode": "from-scratch",
        "coordinate_subset": sel_names,
        "n_coordinates_selected": int(sel.size),
        "delta_nll_cut": DELTA_NLL_95,
        "delta_nll_cut_basis": "chi-square(1) 0.95 quantile halved",
        "touch_rule": _touch_rule_record(),
        "per_row_certification_note": PER_ROW_CERTIFICATION_NOTE,
        "n_coarse": n_coarse,
        "n_grid_per_coordinate": n_grid,
        "local_crb_multiples": list(local_crb_multiples),
        "local_grid_fallback_u": LOCAL_GRID_FALLBACK_U,
        "grid_measure": ("two scales per coordinate, merged and sorted: a "
                         "coarse grid evenly spaced in the box fraction u for "
                         "the box-scale statement, plus a local grid spaced by "
                         "the coordinate's own Cramer-Rao bound for the "
                         "interval; endpoints excluded (u = 0 and u = 1 are "
                         "the box bounds)"),
        "anchor_iters": anchor_iters,
        "step_iters": step_iters,
        "polish_ladder": list(polish_ladder),
        "lr": lr,
        "improvement_tolerance_NLL": improvement_tol,
        "continuation": ("outward march from each coordinate's anchor grid "
                         "point, every row seeded from its already-converged "
                         "immediate neighbour, then one all-rows polish pass "
                         "re-seeded from the better neighbour"),
        "batching_note": ("one march step advances one row per coordinate, so "
                          "the batch is n_theta wide and the schedule compiles "
                          "once per step shape; that also bounds the "
                          "reverse-mode tape, which grows with batch size"),
        "march_steps": march_steps,
        "polish": polish_record,
        "converged": polish_record["converged"],
        "optimiser": _OPTIMISER_NOTE,
        "nll_at_theta_star": nll_star,
        "warm_start": warm_start,
        "u_star": u_star_raw.tolist(),
        "best_row_nll": row_min,
        "reference_nll": reference,
        "a_row_beat_theta_star": beat_optimum,
        "row_improvement_over_theta_star_NLL": (nll_star - row_min
                                                if beat_optimum else 0.0),
        "reference_note": ("Delta NLL is measured from the lower of the attained "
                           "optimum and the best profile row; a row beating the "
                           "optimum is reported, not absorbed"),
        "n_rows_failed": int(np.sum(~finite)),
        "failed_row_note": _FAILED_ROW_NOTE,
        "n_rows": n_rows,
        "total_row_iterations": total_row_iterations,
        "per_parameter": per_param,
        "classification_counts": _class_counts(per_param),
        "runtime_seconds": round(time.time() - t0, 1),
    }


def _class_counts(per_param: dict) -> dict:
    out = {}
    for v in per_param.values():
        out[v["classification"]] = out.get(v["classification"], 0) + 1
    return out


def _prior_certified_flags(prior: dict, internal_names, n_grid: int) -> dict:
    """Per-row certified flags of a prior artifact, keyed by internal name.

    A per-row-format artifact carries ``grid_point_certified`` in every
    per_param block; an older artifact carries the polish block's
    ``uncertified_rows`` list (coordinate INDEX + grid index), from which the
    flags are reconstructed exactly -- the list and the counts came from the
    same array, so nothing is inferred.
    """
    uncert = {(int(r["coordinate"]), int(r["grid_index"]))
              for r in prior.get("polish", {}).get("uncertified_rows", [])}
    flags = {}
    for idx, nm in enumerate(internal_names):
        blk = prior["per_parameter"][nm]
        if "grid_point_certified" in blk:
            flags[nm] = np.asarray(blk["grid_point_certified"], dtype=bool)
        else:
            flags[nm] = np.array([(idx, g) not in uncert
                                  for g in range(n_grid)], dtype=bool)
    return flags


def run_profile_continuation(like, space, theta_star: np.ndarray,
                             prior: dict, prior_path: str,
                             coordinates: list[str],
                             polish_ladder: tuple[int, ...] = (4000,),
                             lr: float = 0.03,
                             improvement_tol: float = SWEEP_IMPROVEMENT_TOL) -> dict:
    """Re-optimise a prior artifact's rows from their own stored points.

    For the selected coordinates the prior artifact supplies each row's grid
    point (bit-for-bit, never rebuilt), NLL and optimised theta vector; the
    march is skipped and every ladder pass seeds every row from its OWN
    current best -- never from a neighbour, because the question a
    continuation answers is whether the row's own claimed value is tight (a
    neighbour seed would answer whether the basin can be re-entered, which is
    the full re-profile's job). Rows with a missing or non-finite stored
    value fall back to the warm start at theta*.

    The result is a complete, self-standing replacement of the prior
    artifact's readout: the non-selected coordinates' stored rows are carried
    unchanged, every coordinate's delta_nll / interval / classification is
    re-read against this run's own reference (the lower of the optimum's NLL
    and every row NLL after re-optimisation, selected or carried), and a
    continuation record documents stored-vs-new NLL per re-run row.
    """
    t0 = time.time()
    n_theta = space.n_theta
    internal_names = space.internal_names
    if not coordinates:
        raise ValueError("a continuation requires an explicit coordinate "
                         "subset; the full list is a valid choice")
    sel = _resolve_subset(space, coordinates)
    sel_names = [internal_names[i] for i in sel]
    sel_set = set(sel.tolist())
    carried_idx = [i for i in range(n_theta) if i not in sel_set]
    carried_names = [internal_names[i] for i in carried_idx]

    theta_star, nll_star, u_star_raw, z_star, warm_start = _optimum_context(
        like, space, theta_star)

    n_grid = int(prior["n_grid_per_coordinate"])
    pp_prior = prior["per_parameter"]
    prior_flags = _prior_certified_flags(prior, internal_names, n_grid)

    # ---- restore the selected coordinates' rows --------------------------
    # u_grid comes over bit-for-bit; each row's seed is the z of its OWN
    # stored theta vector. A stored coordinate that saturated onto a box
    # bound recovers box fraction 0 or 1, where no finite z exists, so the
    # seed's fractions are pulled inside by the same margin the warm start
    # uses; the monotone merge with the stored NLL means no recorded row can
    # regress because of the pull.
    u_grid = np.stack([np.asarray(pp_prior[nm]["u_grid"], dtype=float)
                       for nm in sel_names])
    z_fix = _logit(u_grid)
    n_sel = int(sel.size)
    n_rerun = n_sel * n_grid
    pos_all = np.repeat(np.arange(n_sel), n_grid)
    coords_global = sel[pos_all]
    grids_all = np.tile(np.arange(n_grid), n_sel)

    stored_nll = np.full(n_rerun, np.inf)
    stored_ok = np.zeros(n_rerun, dtype=bool)
    seeds = np.zeros((n_rerun, n_theta))
    masked_z = np.zeros((n_rerun, n_theta))
    theta_stored = [None] * n_rerun
    fallback_rows = []
    seed_clipped_rows = []
    for p, i in enumerate(sel):
        blk = pp_prior[internal_names[i]]
        for g in range(n_grid):
            r = p * n_grid + g
            v = blk["profile_nll"][g]
            th_list = blk["theta_at_grid"][g]
            th = np.asarray(th_list, dtype=float)
            if v is not None and np.isfinite(v) and np.all(np.isfinite(th)):
                stored_nll[r] = float(v)
                stored_ok[r] = True
                theta_stored[r] = list(th_list)
                u_row = np.asarray(space.u_from_theta(th), dtype=float)
                u_cl = np.clip(u_row, Z_START_EPS, 1.0 - Z_START_EPS)
                n_cl = int(np.sum(u_cl != u_row))
                if n_cl:
                    seed_clipped_rows.append(
                        {"coordinate": internal_names[i], "grid_index": g,
                         "n_coordinates_pulled_inside": n_cl})
                seeds[r] = _logit(u_cl)
            else:
                fallback_rows.append(
                    {"coordinate": internal_names[i], "grid_index": g})
                seeds[r] = z_star
            masked_z[r] = seeds[r]
            masked_z[r, i] = z_fix[p, g]

    best_nll = stored_nll.copy()
    best_z = seeds.copy()
    ever_improved = np.zeros(n_rerun, dtype=bool)
    certified = np.zeros(n_rerun, dtype=bool)
    last_improvement = np.full(n_rerun, np.nan)
    records = []

    # ---- self-seeded ladder ----------------------------------------------
    for iters in polish_ladder:
        live = np.flatnonzero(~certified)
        if live.size == 0:
            break
        seed = np.array(best_z[live], dtype=float)
        nll, z, mk = _optimise_rows(
            like, n_theta, coords_global[live],
            z_fix[pos_all[live], grids_all[live]], seed, iters, lr)
        improved = np.isfinite(nll) & (nll < best_nll[live])
        gain = np.where(improved, best_nll[live] - nll, 0.0)
        rows_imp = live[improved]
        best_nll[rows_imp] = nll[improved]
        best_z[rows_imp] = z[improved]
        masked_z[rows_imp] = mk[improved]
        ever_improved[rows_imp] = True
        # certified: this pass, from the row's own best, bought no more than
        # the tolerance. A row recovered from no finite stored value has an
        # unbounded gain and stays uncertified this pass.
        certified[live[gain <= improvement_tol]] = True
        last_improvement[live] = gain
        gain_fin = gain[np.isfinite(gain)]
        imp_fin = improved & np.isfinite(gain)
        records.append({
            "stage": "continuation-polish", "n_iters": iters,
            "n_rows_run": int(live.size),
            "n_rows_improved": int(np.sum(improved)),
            "n_rows_recovered_from_no_stored_value": int(
                np.sum(improved & ~np.isfinite(gain))),
            "max_improvement_NLL": (float(np.max(gain_fin))
                                    if gain_fin.size else 0.0),
            "mean_improvement_NLL": (float(np.mean(gain[imp_fin]))
                                     if np.any(imp_fin) else 0.0),
            "n_rows_certified_after": int(np.sum(certified)),
            "n_rows_nonfinite": int(np.sum(~np.isfinite(nll))),
            "seeding": "self (each row from its own current best)",
        })

    polish_record = {
        "ladder": list(polish_ladder),
        "passes": records,
        "convergence_tolerance_NLL": improvement_tol,
        "n_rows_certified": int(np.sum(certified)),
        "n_rows_uncertified": int(np.sum(~certified)),
        "uncertified_rows": [
            {"coordinate": int(coords_global[r]),
             "grid_index": int(grids_all[r])}
            for r in np.flatnonzero(~certified)],
        "converged": bool(np.all(certified)),
        "seeding": ("self: every pass re-optimises each row from its OWN "
                    "current best, never from a neighbour"),
        "scope": ("the re-run (selected) rows only; carried rows ran no pass "
                  "in this process"),
        "reading": _CERTIFICATION_READING,
    }
    total_row_iterations = int(sum(p["n_rows_run"] * p["n_iters"]
                                   for p in records))

    # ---- theta vectors: optimiser output where a pass improved the row,
    # the stored vector VERBATIM where none did (so a carried value and its
    # theta stay the exact pair the prior artifact claimed).
    theta_new = np.asarray(jax.vmap(space.theta_from_z)(
        jnp.asarray(np.where(np.isfinite(masked_z), masked_z, 0.0))))
    theta_rows_out = [theta_stored[r]
                      if (not ever_improved[r] and theta_stored[r] is not None)
                      else theta_new[r]
                      for r in range(n_rerun)]

    # ---- reference over EVERY row, selected or carried --------------------
    carried_nll = {}
    for i in carried_idx:
        vals = pp_prior[internal_names[i]]["profile_nll"]
        arr = np.array([float(v) if v is not None else np.inf for v in vals])
        arr[~np.isfinite(arr)] = np.inf
        carried_nll[i] = arr
    all_nll = np.concatenate(
        [best_nll] + [carried_nll[i] for i in carried_idx])
    finite = np.isfinite(all_nll)
    row_min = (float(np.min(all_nll[finite])) if np.any(finite)
               else float("nan"))
    beat_optimum = bool(np.any(finite) and row_min < nll_star)
    reference = min(nll_star, row_min) if np.any(finite) else nll_star

    # ---- per-coordinate readout: fresh blocks for the selected, carried
    # rows re-read against this run's reference for the rest ----------------
    blocks = {}
    for p, i in enumerate(sel):
        sl = slice(p * n_grid, (p + 1) * n_grid)
        nll_i = best_nll[sl]
        valid_i = np.isfinite(nll_i)
        dnll_i = np.where(valid_i, nll_i - reference, np.nan)
        blk_prior = pp_prior[internal_names[i]]
        block = _readout_block(
            space, i, u_star_raw, u_grid[p], nll_i, valid_i, dnll_i,
            theta_rows_out[sl], certified[sl], last_improvement[sl],
            grid_meta={
                "anchor_grid_index": blk_prior.get("anchor_grid_index"),
                "grid_point_is_local": blk_prior.get("grid_point_is_local"),
                "local_grid_scale_u": blk_prior.get("local_grid_scale_u"),
                "local_grid_scale_source": blk_prior.get(
                    "local_grid_scale_source"),
            })
        block["carried_from_prior"] = False
        blocks[internal_names[i]] = block
    for i in carried_idx:
        nm = internal_names[i]
        blk = dict(pp_prior[nm])
        nll_i = carried_nll[i]
        valid_i = np.isfinite(nll_i)
        dnll_i = np.where(valid_i, nll_i - reference, np.nan)
        u_grid_i = np.asarray(blk["u_grid"], dtype=float)
        summary = _interval_and_class(u_grid_i, dnll_i, valid_i)
        u_lo, u_hi = summary["interval_u"]
        flags = prior_flags[nm]
        blk.update(summary)
        blk["u_star"] = float(u_star_raw[i])
        blk["delta_nll"] = [None if not v else float(x)
                            for x, v in zip(dnll_i, valid_i)]
        blk["n_grid_points_failed"] = int(np.sum(~valid_i))
        blk["n_grid_points_uncertified"] = int(np.sum(~flags))
        blk["all_grid_points_certified"] = bool(np.all(flags))
        blk["grid_point_certified"] = [bool(c) for c in flags]
        blk["grid_point_last_improvement_NLL"] = [None] * n_grid
        blk["interval_internal_units"] = _interval_internal_units(
            space, i, u_star_raw, u_lo, u_hi)
        blk["carried_from_prior"] = True
        blk["carried_note"] = (
            "rows (u_grid, profile_nll, theta_at_grid) and certified flags "
            "carried unchanged from the prior artifact; delta_nll, interval "
            "and classification re-read against this run's reference; no "
            "pass ran these rows in this process")
        blocks[nm] = blk
    per_param = {nm: blocks[nm] for nm in internal_names}

    # ---- the continuation record: stored vs new, per re-run row -----------
    rows_list = []
    for r in range(n_rerun):
        st = float(stored_nll[r]) if stored_ok[r] else None
        newv = float(best_nll[r]) if np.isfinite(best_nll[r]) else None
        rows_list.append({
            "coordinate": internal_names[coords_global[r]],
            "grid_index": int(grids_all[r]),
            "u": float(u_grid[pos_all[r], grids_all[r]]),
            "stored_nll": st,
            "new_nll": newv,
            "improvement_NLL": (st - newv
                                if st is not None and newv is not None
                                else None),
            "certified": bool(certified[r]),
        })
    improvements = np.array([r["improvement_NLL"] for r in rows_list
                             if r["improvement_NLL"] is not None])
    continuation_record = {
        "prior_artifact": str(prior_path),
        "prior_reference_nll": prior.get("reference_nll"),
        "selected_coordinates": sel_names,
        "carried_coordinates": carried_names,
        "n_rows_rerun": int(n_rerun),
        "n_rows_carried": int(len(carried_idx) * n_grid),
        "self_seed_rule": ("each row re-optimised from its OWN stored point, "
                           "later passes from its own current best, never "
                           "from a neighbour: the question a continuation "
                           "answers is whether the row's own claimed value "
                           "is tight"),
        "restore": {
            "n_rows_restored": int(np.sum(stored_ok)),
            "n_fallback_rows": len(fallback_rows),
            "fallback_rows": fallback_rows,
            "fallback_note": ("a row with a null or non-finite stored NLL or "
                              "a non-finite stored theta vector is re-seeded "
                              "from the warm start at theta*"),
            "n_seed_clipped_rows": len(seed_clipped_rows),
            "seed_clipped_rows": seed_clipped_rows,
            "seed_clip_note": ("a stored theta whose box fraction sits "
                               "outside [eps, 1-eps] has no movable z; the "
                               "SEED is pulled inside by the warm-start "
                               "margin while the stored NLL still enters the "
                               "monotone merge, so no recorded row regresses "
                               "because of the pull"),
        },
        "rows": rows_list,
        "totals": {
            "n_rows": int(n_rerun),
            "n_certified": int(np.sum(certified)),
            "n_uncertified": int(np.sum(~certified)),
            "n_improved": int(np.sum(ever_improved)),
            "total_improvement_NLL": (float(np.sum(improvements))
                                      if improvements.size else 0.0),
            "max_improvement_NLL": (float(np.max(improvements))
                                    if improvements.size else 0.0),
            "total_row_iterations": total_row_iterations,
        },
    }

    n_failed_total = int(np.sum(~np.isfinite(all_nll)))
    return {
        "role": _ROLE_NOTE,
        "mode": "continuation",
        "coordinate_subset": sel_names,
        "n_coordinates_selected": n_sel,
        "delta_nll_cut": DELTA_NLL_95,
        "delta_nll_cut_basis": "chi-square(1) 0.95 quantile halved",
        "touch_rule": _touch_rule_record(),
        "per_row_certification_note": PER_ROW_CERTIFICATION_NOTE,
        "n_coarse": prior.get("n_coarse"),
        "n_grid_per_coordinate": n_grid,
        "local_crb_multiples": prior.get("local_crb_multiples"),
        "local_grid_fallback_u": prior.get("local_grid_fallback_u"),
        "grid_measure": (str(prior.get("grid_measure"))
                         + " -- restored from the prior artifact, never "
                           "rebuilt"),
        "anchor_iters": None,
        "step_iters": None,
        "polish_ladder": list(polish_ladder),
        "lr": lr,
        "improvement_tolerance_NLL": improvement_tol,
        "continuation": ("continuation from a prior artifact: the march is "
                         "skipped; each selected row is restored (grid "
                         "point, NLL, theta vector) and re-optimised from "
                         "its OWN stored point by a self-seeded ladder; "
                         "non-selected coordinates' rows are carried "
                         "unchanged; every coordinate is re-read against "
                         "this run's own reference"),
        "batching_note": ("all not-yet-certified selected rows advance as "
                          "one batch per ladder pass"),
        "march_steps": [],
        "march_skipped": True,
        "polish": polish_record,
        "converged": polish_record["converged"],
        "optimiser": _OPTIMISER_NOTE,
        "nll_at_theta_star": nll_star,
        "warm_start": warm_start,
        "u_star": u_star_raw.tolist(),
        "best_row_nll": row_min,
        "reference_nll": reference,
        "a_row_beat_theta_star": beat_optimum,
        "row_improvement_over_theta_star_NLL": (nll_star - row_min
                                                if beat_optimum else 0.0),
        "reference_note": ("Delta NLL is measured from the lower of the "
                           "attained optimum and the best profile row, "
                           "selected or carried; a row beating the optimum "
                           "is reported, not absorbed"),
        "n_rows_failed": n_failed_total,
        "failed_row_note": _FAILED_ROW_NOTE,
        "n_rows": int(n_theta * n_grid),
        "n_rows_rerun": int(n_rerun),
        "n_rows_carried": int(len(carried_idx) * n_grid),
        "total_row_iterations": total_row_iterations,
        "per_parameter": per_param,
        "classification_counts": _class_counts(per_param),
        "continuation_record": continuation_record,
        "runtime_seconds": round(time.time() - t0, 1),
    }


# ---------------------------------------------------------------------------
# The band-ratio classification, recomputed for comparison
# ---------------------------------------------------------------------------


def band_ratio_classification(members: list[dict], space,
                              cuts: tuple[float, float] = BAND_RATIO_CUTS) -> dict:
    """Recompute the ensemble band-ratio R and its classes, for comparison.

    R is the width of the [2.5, 97.5] percentile band of the in-band members'
    box fractions u -- the classification being replaced. It is recomputed here
    from the ensemble artifact rather than read from it, so the comparison with
    the profile classification is made on one definition and one member set.
    """
    inb = [m for m in members if m.get("in_band") and not m.get("failed")]
    if not inb:
        raise RuntimeError("no in-band members; band ratio undefined")
    U = np.array([np.asarray(space.u_from_theta(np.asarray(m["theta_best"],
                                                           dtype=float)))
                  for m in inb])
    R = np.percentile(U, 97.5, axis=0) - np.percentile(U, 2.5, axis=0)
    lo_cut, hi_cut = cuts
    per = {}
    for i, nm in enumerate(space.internal_names):
        r = float(R[i])
        per[nm] = {
            "theta_name": space.theta_names[i],
            "R": r,
            "class": ("constrained" if r < lo_cut else
                      ("weak" if r < hi_cut else "unconstrained")),
        }
    return {
        "definition": ("width of the [2.5, 97.5] percentile band of the in-band "
                       "members' box fractions u"),
        "cuts": {"constrained": f"R < {lo_cut}", "weak": f"{lo_cut} <= R < {hi_cut}",
                 "unconstrained": f"R >= {hi_cut}"},
        "n_in_band_members": len(inb),
        "per_parameter": per,
    }


def compare_classifications(profile: dict, band: dict) -> dict:
    """Parameter-by-parameter comparison of the profile and band-ratio classes."""
    rows = []
    for nm, p in profile["per_parameter"].items():
        b = band["per_parameter"].get(nm, {})
        rows.append({
            "internal_name": nm,
            "theta_name": p["theta_name"],
            "profile_class": p["classification"],
            "profile_interval_width_u": p["interval_width_u"],
            "profile_max_delta_nll": p["max_delta_nll_over_box"],
            "lower_endpoint_at_box_edge": p["lower_endpoint_at_box_edge"],
            "upper_endpoint_at_box_edge": p["upper_endpoint_at_box_edge"],
            "band_ratio_R": b.get("R"),
            "band_ratio_class": b.get("class"),
        })
    agree_constrained = [r for r in rows
                         if r["band_ratio_class"] == "constrained"
                         and r["profile_class"] == CLASS_IDENTIFIED]
    disagree = [r for r in rows
                if (r["band_ratio_class"] == "constrained")
                != (r["profile_class"] == CLASS_IDENTIFIED)]
    return {
        "role": ("does the constrained/equifinal split reproduce under a "
                 "diagnostic that does not depend on the ensemble? measured, "
                 "reported either way"),
        "rows": rows,
        "n_band_constrained_and_profile_identified": len(agree_constrained),
        "n_disagreeing": len(disagree),
        "disagreeing": [
            {"internal_name": r["internal_name"],
             "band_ratio_class": r["band_ratio_class"],
             "profile_class": r["profile_class"]} for r in disagree],
    }
