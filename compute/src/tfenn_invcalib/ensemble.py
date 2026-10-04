"""Deep-ensemble multi-start gradient inversion through the frozen driver.

Role: bounded/transformed gradient optimization of the curve NLL in the
unconstrained z-space (theta_space transform enforces the optimization box
and c_res < c_peak by construction), from admissible multi-start samples,
with per-start convergence records and aggregation of optimiser outcomes.

Design decisions (documented):
- Optimizer: optax.adam in z-space (z coordinates are O(1) logits, so one rate
  serves every parameter -- 12 for form I, 14 for form II; the transform's
  Jacobian supplies per-parameter scaling), with the learning rate on a
  cosine schedule that decays from the
  peak `lr` to 0 over the first DECAY_FRACTION of the budget, then a frozen
  lr=0 tail. The decay is what lets the good basin actually converge: at a
  fixed lr adam micro-descends to the final iteration indefinitely (best-
  visited iterate pinned at n_iters even at 2000 iters), and decaying to zero
  only AT the budget end still leaves microscopic improvements that pin it
  there; decaying to zero BEFORE the end freezes the iterate at a fixed point,
  so the good basin plateaus with best_iteration ~ decay_steps < n_iters. No
  per-start early stop (the batch advances together); the reported per-start
  solution is the BEST-visited iterate (lowest NLL along the trajectory), a
  standard safeguarded choice for non-monotone first-order optimizers on
  piecewise-smooth objectives like the frozen form's.
- All starts advance simultaneously as one batched (S, n_theta) state through a
  vmapped value-and-grad and elementwise adam updates -- adam's update is
  elementwise, so batching preserves per-start independence while using the
  vectorized CPU path (the "vmap/parallel starts" design of the spec).
- A start whose NLL/gradient goes non-finite poisons only its own row; it is
  recorded as failed (never silently repaired).
- Convergence flag (recorded DIAGNOSTIC, never the band gate): plateau
  criterion -- the best-so-far NLL improved by less than REL_TOL (relative)
  over the trailing 20 percent of iterations, with a finite best NLL. It
  records whether a start stopped moving; it does NOT distinguish a good
  optimum from a bad one -- a stuck bad optimum plateaus while a still-
  descending good optimum does not -- so it must not gate band membership.
- Band membership (``select_band_members``): a finite non-failed member is
  admitted when its NLL increment from the best member is at most log(10),
  equivalently its relative likelihood is at least 0.1. This rule is invariant
  to an additive objective constant. The result is recorded per member
  as the boolean ``in_band`` so every downstream consumer (bands, LOO
  posterior-predictive, u_max propagation, coverage) selects from the SAME
  flag and can never disagree.
- Bands: per-component percentiles (2.5 / 97.5 by default) of theta across
  the ``in_band`` members. They describe optimiser outcomes, with no posterior
  or confidence-coverage interpretation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np

import jax
import jax.numpy as jnp
import optax

from .theta_space import space_for_form

REL_TOL_CONVERGED = 1e-5
BAND_PERCENTILES = (2.5, 97.5)

# Descriptive support cut for retained optimiser outcomes, not a coverage cut.
# Sensitivity is evaluated at relative likelihoods 0.5 and 0.01 in SR-16.
BAND_DELTA_NLL = float(np.log(10.0))

# Fraction of the iteration budget over which the learning rate decays to zero;
# the remaining tail runs at lr=0 (frozen). This is the mechanism that makes the
# good basin plateau with best_iteration < n_iters (see run_multistart): a
# per-start early stop was rejected because the batched vmap advances all starts
# together and would stall on the slowest row, and decaying to zero only AT the
# budget end still pins best_iteration at the last step.
DECAY_FRACTION = 0.8

# Production multi-start iteration budget (fixed length; lr decays to zero over
# the first DECAY_FRACTION of it = 1600 iters, then a frozen 1600-2000 tail).
# Set from the convergence probe on the 5 measured curves: the good basin
# reaches NLL 3626.2 / in-sample RMSE 4.14 MPa (0.04% above the fixed-lr
# asymptote 3624.6), vs the old 400-iter default where the worst good-basin
# start was still ~69% above its converged NLL (63/64 starts still descending
# at the budget). Every good-basin member's best-visited iterate settles at
# iter 1600 (= decay_steps, where lr hits zero); the probe measured
# best_iteration 1600/1600/1600 for all 9 good members, i.e. 0/9 pinned at the
# last iteration, and the frozen tail confirms the plateau adds nothing.
PRODUCTION_N_ITERS = 2000


@dataclass
class MemberRecord:
    start_index: int
    theta0: list = field(default_factory=list)
    initial_nll: float = float("nan")
    final_nll: float = float("nan")      # best-visited NLL
    best_iteration: int = -1
    iterations: int = 0
    converged: bool = False       # plateau DIAGNOSTIC only (never the band gate)
    failed: bool = False
    in_band: bool = False         # good-NLL-basin band membership (the gate)
    theta_best: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "start_index": self.start_index,
            "theta0": self.theta0,
            "initial_nll": self.initial_nll,
            "final_nll": self.final_nll,
            "best_iteration": self.best_iteration,
            "iterations": self.iterations,
            "converged": self.converged,
            "failed": self.failed,
            "in_band": self.in_band,
            "theta_best": self.theta_best,
        }


def make_batched_objective(nll_row):
    """Batch one z-row objective into the pair ``run_adam_z`` consumes.

    Returns ``(vg, nll_batch)``: a vmapped value-and-grad and a jitted vmapped
    value, both taking the (S, n) z-state as their single argument. A caller
    whose rows carry per-row constants (the profile lane freezes a different
    coordinate in every row) builds its own pair with those constants closed
    over, so the optimiser core stays a single implementation.
    """
    return jax.vmap(jax.value_and_grad(nll_row)), jax.jit(jax.vmap(nll_row))


def run_adam_z(vg, nll_batch, z0: np.ndarray, n_iters: int,
               lr: float = 0.03):
    """The batched adam-in-z core: every z-space optimisation runs through here.

    Extracted from ``run_multistart`` unchanged so that the multi-start
    inversion and the profile-likelihood re-optimisation are literally the
    same optimiser rather than two implementations that could drift apart.
    ``vg`` / ``nll_batch`` come from ``make_batched_objective`` (or from a
    caller that closed per-row constants over the same shapes).

    Returns a namespace with the best-visited state (``best_nll``, ``best_z``,
    ``best_it``), the full per-iteration NLL trajectory ``traj``
    ((n_iters + 1, S), the final iterate scored in the last row), and the
    schedule's ``decay_steps``.
    """
    z0 = np.asarray(z0, dtype=float)
    S = z0.shape[0]
    # Learning-rate schedule: cosine decay from `lr` to 0 over the first
    # DECAY_FRACTION of the budget, then a hard lr=0 tail. With a FIXED lr adam
    # inches downhill to the final iteration indefinitely on this smooth
    # objective (a convergence probe showed the good basin's best-visited
    # iterate pinned at n_iters even at 2000 iters); decaying to zero AT the
    # budget end still leaves microscopic improvements that pin best_iteration
    # at the last step. Decaying to zero BEFORE the end freezes the iterate at a
    # fixed point (lr=0 tail => no updates => no new bests), so the good basin
    # genuinely plateaus with best_iteration ~ decay_steps < n_iters, while the
    # best-visited iterate is still recorded (safeguard retained).
    decay_steps = max(1, int(round(DECAY_FRACTION * n_iters)))
    schedule = optax.cosine_decay_schedule(init_value=lr, decay_steps=decay_steps,
                                           alpha=0.0)
    opt = optax.adam(schedule)

    @jax.jit
    def step(z, opt_state, best_nll, best_z, best_it, it):
        nll, g = vg(z)
        improved = jnp.isfinite(nll) & (nll < best_nll)
        best_nll = jnp.where(improved, nll, best_nll)
        best_z = jnp.where(improved[:, None], z, best_z)
        best_it = jnp.where(improved, it, best_it)
        updates, opt_state = opt.update(g, opt_state, z)
        z = optax.apply_updates(z, updates)
        return z, opt_state, best_nll, best_z, best_it, nll

    z = jnp.asarray(z0)
    opt_state = opt.init(z)
    best_nll = jnp.full((S,), jnp.inf)
    best_z = z
    best_it = jnp.full((S,), -1, dtype=jnp.int32)

    traj = np.empty((n_iters + 1, S))
    for it in range(n_iters):
        z, opt_state, best_nll, best_z, best_it, nll_now = step(
            z, opt_state, best_nll, best_z, best_it, it)
        traj[it] = np.asarray(nll_now)
    # score the final iterate too
    nll_final = nll_batch(z)
    improved = jnp.isfinite(nll_final) & (nll_final < best_nll)
    best_nll = jnp.where(improved, nll_final, best_nll)
    best_z = jnp.where(improved[:, None], z, best_z)
    best_it = jnp.where(improved, n_iters, best_it)
    traj[n_iters] = np.asarray(nll_final)

    return SimpleNamespace(
        best_nll=np.asarray(best_nll),
        best_z=best_z,
        best_it=np.asarray(best_it),
        traj=traj,
        decay_steps=decay_steps,
        n_iters=n_iters,
        n_rows=S,
    )


def run_multistart(nll_z, theta_starts: np.ndarray, n_iters: int,
                   lr: float = 0.03, form: int = 1) -> list[MemberRecord]:
    """Batched multi-start adam on nll_z from admissible theta starts.

    ``form`` selects the z <-> theta transform pair (1 = frozen form I, the
    exact pre-existing code path; 2 = the 14-parameter form II); nll_z must
    have been built with the same form.
    """
    space = space_for_form(form)
    theta_starts = np.asarray(theta_starts, dtype=float)
    S = theta_starts.shape[0]
    z0 = np.stack([space.z_from_theta(theta_starts[i]) for i in range(S)])

    vg, nll_batch = make_batched_objective(nll_z)
    run = run_adam_z(vg, nll_batch, z0, n_iters, lr=lr)

    best_nll_np = run.best_nll
    best_it_np = run.best_it
    traj = run.traj
    theta_best = np.asarray(jax.vmap(space.theta_from_z)(run.best_z))

    # running best-so-far for the plateau convergence flag. Measured over the
    # PRODUCTIVE (decay) phase ending at decay_steps, not the full budget: the
    # lr=0 tail after decay_steps is frozen by construction and would make every
    # member trivially "plateaued", which would strip the diagnostic of meaning.
    minsofar = np.minimum.accumulate(np.where(np.isfinite(traj), traj, np.inf), axis=0)
    plateau_ref = min(run.decay_steps, n_iters)
    window = max(5, int(round(0.2 * plateau_ref)))
    window = min(window, plateau_ref)

    records: list[MemberRecord] = []
    for i in range(S):
        failed = not np.isfinite(best_nll_np[i])
        if failed:
            converged = False
        else:
            prev = minsofar[plateau_ref - window, i]
            last = minsofar[plateau_ref, i]
            if not np.isfinite(prev):
                converged = False
            else:
                rel_impr = (prev - last) / max(1.0, abs(last))
                converged = bool(rel_impr < REL_TOL_CONVERGED)
        records.append(MemberRecord(
            start_index=i,
            theta0=theta_starts[i].tolist(),
            initial_nll=float(traj[0, i]),
            final_nll=float(best_nll_np[i]),
            best_iteration=int(best_it_np[i]),
            iterations=n_iters,
            converged=converged,
            failed=failed,
            theta_best=theta_best[i].tolist(),
        ))
    # annotate good-NLL-basin band membership so every returned record already
    # carries the `in_band` gate (aggregate_bands re-derives it idempotently).
    select_band_members(records)
    return records


def band_threshold(best_nll: float, delta_nll: float = BAND_DELTA_NLL) -> float:
    """Additive-constant-invariant cutoff relative to the best fitted member."""
    if not np.isfinite(best_nll) or not np.isfinite(delta_nll) or delta_nll < 0:
        raise ValueError("Band reference must be finite and NLL increment nonnegative")
    return best_nll + delta_nll


def select_band_members(records: list[MemberRecord],
                        delta_nll: float = BAND_DELTA_NLL) -> list[MemberRecord]:
    """Mark ``in_band`` on every record by the good-NLL-basin rule and return
    the selected members (input order preserved).

    Gate (NOT the plateau ``converged`` flag): a member is in-band iff it is
    non-failed with a finite best-visited NLL no worse than
    ``band_threshold(best_final_nll, delta_nll)``. The best member always
    satisfies this, so the selection is never empty when a non-failed member
    exists. Failed members are set ``in_band = False``.
    """
    ok = [r for r in records if not r.failed and np.isfinite(r.final_nll)]
    if not ok:
        for r in records:
            r.in_band = False
        return []
    thr = band_threshold(min(r.final_nll for r in ok), delta_nll)
    selected: list[MemberRecord] = []
    for r in records:
        r.in_band = bool(not r.failed and np.isfinite(r.final_nll)
                         and r.final_nll <= thr)
        if r.in_band:
            selected.append(r)
    return selected


def aggregate_bands(records: list[MemberRecord],
                    percentiles: tuple[float, float] = BAND_PERCENTILES,
                    delta_nll: float = BAND_DELTA_NLL,
                    form: int = 1) -> dict:
    """Percentile spread of the admitted optimiser outcomes.

    Band membership is the good-NLL-basin gate (``select_band_members``), NOT
    the plateau ``converged`` flag: the latter is a stuck-vs-moving diagnostic
    that a bad local optimum can satisfy while a still-descending good optimum
    cannot, so it never selects members here. ``form`` only labels the output
    theta_names (1 = the 12 form-I names, 2 = the 14 form-II names).
    """
    ok = [r for r in records if not r.failed and np.isfinite(r.final_nll)]
    if not ok:
        raise RuntimeError("no non-failed ensemble members; bands undefined")
    members = select_band_members(records, delta_nll)
    thr = band_threshold(min(r.final_nll for r in ok), delta_nll)
    selection = (f"retained members (NLL - best NLL <= {delta_nll:g}; cutoff = "
                 f"{thr:.1f}): {len(members)} of {len(ok)} non-failed; plateau "
                 f"'converged' flag recorded as diagnostic only, not the gate")
    th = np.array([r.theta_best for r in members])
    lower = np.percentile(th, percentiles[0], axis=0)
    upper = np.percentile(th, percentiles[1], axis=0)
    median = np.percentile(th, 50.0, axis=0)
    best = min(members, key=lambda r: r.final_nll)
    return {
        "kind": "optimiser-basin spread",
        "percentiles": list(percentiles),
        "member_selection": selection,
        "band_delta_nll": delta_nll,
        "band_nll_threshold": float(thr),
        "n_members": len(members),
        "n_converged_diagnostic": int(sum(r.converged for r in records)),
        "n_failed": int(sum(r.failed for r in records)),
        "theta_names": space_for_form(form).theta_names,
        "lower": lower.tolist(),
        "upper": upper.tolist(),
        "median": median.tolist(),
        "best_theta": list(best.theta_best),
        "best_final_nll": best.final_nll,
    }


def multistart_summary(records: list[MemberRecord]) -> dict:
    finite_init = [r.initial_nll for r in records if np.isfinite(r.initial_nll)]
    ok = [r for r in records if not r.failed and np.isfinite(r.final_nll)]
    finite_final = [r.final_nll for r in ok]
    best = min(finite_final) if finite_final else None
    n_in_band = (sum(1 for r in ok if r.final_nll <= band_threshold(best))
                 if best is not None else 0)
    return {
        "n_starts": len(records),
        "n_failed": sum(r.failed for r in records),
        "n_converged": sum(r.converged for r in records),   # plateau diagnostic
        "n_in_band": n_in_band,                              # good-NLL-basin gate
        "best_initial_nll": (min(finite_init) if finite_init else None),
        "best_final_nll": best,
        "failed_start_rate": (sum(r.failed for r in records) / len(records)
                              if records else None),
    }
