"""Classical derivative-free staged-calibration baseline lane.

Role: the method-comparison lane of the claim contract -- a classical staged
calibration with scipy.optimize.least_squares on the SAME objective (the
likelihood module's NLL, including the fixed CU offset treatment) and the
SAME admissibility floor, with NO gradients through the driver: the driver is
only ever evaluated forward (values via the verified forward-equivalent
mirror, jitted, autodiff never invoked), and least_squares builds its
Jacobian by classical finite differencing in the transformed z-space
(explicit diff_step 1e-6, the G0a-validated FD scale).

Staging (documented classical protocol, block-coordinate on the full
objective):
  stage "elastic":  fit the 5 elastic parameters [E_p, E_o, G_op, nu_pp,
                    nu_op] with strength/softening parameters frozen;
  stage "strength": fit the 7 strength/softening parameters [c_peak, c_res,
                    phi_deg, c_j, phi_j_deg, gamma_c, k_sharpness] with the
                    elastic block frozen at the stage-1 result;
  stage "all":      optional final polish of all 12 parameters.
Every stage minimizes the same full-curve NLL residual vector (weighted
per-station residuals for all curves + the explicit CU offset penalty rows),
restricted to its parameter block; bounds are enforced exactly by the
z-space transform, so least_squares runs unbounded. Per-run records carry
initial/final NLL, nfev, status, and success per stage.

Form II (fabric-scaled matrix cohesion, 14 parameters) keeps the same three
stages: the fabric parameters scale the matrix cohesion, i.e. they are
strength parameters, so stage "strength" gains indices [12, 13] and stage
"all" becomes range(14); no new stage (STAGE_BLOCKS_F2).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

import jax
import jax.numpy as jnp

from . import mirror
from .curves import CurveSet
from .likelihood import LikelihoodConfig
from .theta_space import space_for_form

# Parameter blocks (indices into the 12-vector / z-vector).
STAGE_BLOCKS = {
    "elastic": [0, 1, 2, 3, 4],
    "strength": [5, 6, 7, 8, 9, 10, 11],
    "all": list(range(12)),
}
# Form-II blocks (indices into the 14-vector / z-vector): elastic unchanged,
# the fabric coordinates join the strength block (see module docstring).
STAGE_BLOCKS_F2 = {
    "elastic": [0, 1, 2, 3, 4],
    "strength": [5, 6, 7, 8, 9, 10, 11, 12, 13],
    "all": list(range(14)),
}
DIFF_STEP = 1e-6  # classical FD step in z-space (G0a-validated scale)


def _make_residual_fn(curveset: CurveSet, cfg: LikelihoodConfig,
                      form: int = 1):
    """Weighted residual vector f(z) with 0.5*sum(f^2) = NLL - const."""
    on_path = curveset.p_conf_seq is not None
    if form == 1:
        q_fn = (mirror.stations_path_fn(curveset.n_steps) if on_path
                else mirror.stations_fn(curveset.n_steps))
    elif form == 2:
        q_fn = (mirror.stations_path_f2_fn(curveset.n_steps) if on_path
                else mirror.stations_f2_fn(curveset.n_steps))
    else:
        raise ValueError(f"unknown form {form!r}; options (1, 2)")
    t_from_z = space_for_form(form).theta_from_z
    eps_sta = jnp.asarray(curveset.eps_stations)
    d_eps11 = curveset.d_eps11
    p_conf = curveset.p_conf_seq if on_path else curveset.p_conf

    def _forward_all(theta):
        return jax.vmap(
            lambda M, n, p, es: q_fn(theta, M, n, p, d_eps11, es)
        )(curveset.M_bond, curveset.n_hat, p_conf, eps_sta)

    forward_jit = jax.jit(lambda z: _forward_all(t_from_z(z)))

    s = cfg.sigma_q
    mu = cfg.cu_offset_mu
    t = cfg.cu_offset_sigma
    lo, hi = cfg.cu_offset_lo, cfg.cu_offset_hi
    q_meas = curveset.q_meas
    # On the measured effective-stress path there is no offset row and no
    # penalty row: every curve contributes its plain weighted residual, which
    # is what makes 0.5*sum(f^2) + const_total the same NLL the likelihood
    # reports in that mode.
    is_cu = np.zeros_like(curveset.is_cu) if on_path else curveset.is_cu

    def residuals(z: np.ndarray) -> np.ndarray:
        q_model = np.asarray(forward_jit(jnp.asarray(z)))
        rows: list[np.ndarray] = []
        for k in range(curveset.n_curves):
            m = q_meas[k] - q_model[k]
            if is_cu[k]:
                qmax = max(float(np.max(q_model[k])), 1e-6)
                w = q_model[k] / qmax
                denom = np.sum(w * w) / s ** 2 + 1.0 / t ** 2
                delta_hat = (mu / t ** 2 - np.sum(w * m) / s ** 2) / denom
                delta = float(np.clip(delta_hat, lo, hi))
                rows.append((m + delta * w) / s)
                rows.append(np.array([(delta - mu) / t]))
            else:
                rows.append(m / s)
        f = np.concatenate(rows)
        # scipy requires finite rows; a wild trial step can also produce
        # finite-but-enormous residuals whose squared sum overflows, so clip
        # to a large sentinel (the step is rejected either way).
        f = np.where(np.isfinite(f), f, 1e6)
        return np.clip(f, -1e6, 1e6)

    return residuals


def nll_from_residuals(f: np.ndarray, const_total: float) -> float:
    return 0.5 * float(np.sum(f ** 2)) + const_total


def run_staged_baseline(curveset: CurveSet, cfg: LikelihoodConfig,
                        theta0: np.ndarray, const_total: float,
                        stages: tuple[str, ...] = ("elastic", "strength"),
                        max_nfev_per_stage: int = 60,
                        form: int = 1) -> dict:
    """Run the staged least_squares baseline from one admissible start.

    ``form`` selects the constitutive form (1 = frozen form I, the exact
    pre-existing code path with STAGE_BLOCKS; 2 = the 14-parameter form II
    with STAGE_BLOCKS_F2); theta0 must be a vector of the form's length.
    """
    space = space_for_form(form)
    blocks = STAGE_BLOCKS if form == 1 else STAGE_BLOCKS_F2
    for st in stages:
        if st not in blocks:
            raise ValueError(f"unknown stage {st!r}")
    residuals = _make_residual_fn(curveset, cfg, form=form)
    z = space.z_from_theta(np.asarray(theta0, dtype=float))
    f0 = residuals(z)
    initial_obj = nll_from_residuals(f0, const_total)

    stage_records = []
    for stage in stages:
        block = blocks[stage]
        z_base = z.copy()

        def fun(x, _block=block, _z_base=z_base):
            z_full = _z_base.copy()
            z_full[_block] = x
            return residuals(z_full)

        nll_before = nll_from_residuals(residuals(z), const_total)
        sol = least_squares(fun, z[block], method="trf",
                            diff_step=DIFF_STEP, max_nfev=max_nfev_per_stage)
        z_cand = z.copy()
        z_cand[block] = sol.x
        nll_after = nll_from_residuals(residuals(z_cand), const_total)
        accepted = nll_after <= nll_before  # trf only accepts decreases; guard anyway
        if accepted:
            z = z_cand
        stage_records.append({
            "stage": stage,
            "block_params": [space.theta_names[i] for i in block],
            "initial_nll": nll_before,
            "final_nll": (nll_after if accepted else nll_before),
            "nfev": int(sol.nfev),
            "status": int(sol.status),
            "success": bool(sol.success),
            "accepted": bool(accepted),
        })

    theta_final = np.asarray(space.theta_from_z(jnp.asarray(z)))
    final_obj = nll_from_residuals(residuals(z), const_total)
    return {
        "lane": ("classical derivative-free staged calibration "
                 "(scipy least_squares, block-coordinate, FD Jacobian in "
                 "z-space; no AD gradients through the driver)"),
        "stages": stage_records,
        "diff_step": DIFF_STEP,
        "max_nfev_per_stage": max_nfev_per_stage,
        "theta0": np.asarray(theta0, dtype=float).tolist(),
        "theta_final": theta_final.tolist(),
        "initial_obj": initial_obj,
        "final_obj": final_obj,
    }
