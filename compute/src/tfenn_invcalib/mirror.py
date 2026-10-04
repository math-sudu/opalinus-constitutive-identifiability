"""Differentiable JAX mirror of the ESUBI-CWFS triaxial driver.

Provenance
----------
Adapted from the G0a gradient-sanity probe
``I:/GeoSkills/projects/tfenn-esubi/derive/results/d4_gradient_sanity_probe.py``
(source commit 29ab3d55ddbe4d381d72eefd8ae859e43b101acf), which mirrors the
frozen pure-NumPy CTC integrator ``run_triaxial_path`` in
``src/tfenn_esubi/data_gen/esubi_driver.py`` step for step. The G0a gate
verdict on that mirror: forward-equivalence max rel err ~7e-14 (criterion
1e-6), AD-vs-FD gradient median rel err 8.4e-9, 0/250 failed starts,
0 degenerate combos.

Constitutive and numerical scope
-------------------------------
Both forms use the same compliance, matrix return, branch thresholds and
accumulated principal-span softening law. The joint correction uses both
dyads without a half factor, so it closes the full excess shear traction.
The specimen controller prescribes axial strain, the two radial effective
stresses and zero specimen shear stresses, solving all five nonaxial strain
increments. Stress and strain use the same evaluated iterate; all five
controlled-stress residuals must meet an absolute and increment-scaled
tolerance, and an exhausted solve marks the path invalid. Invalid paths return
NaN q so station interpolation and the likelihood cannot silently accept them;
the finite stress/strain histories and per-step flags remain diagnostic output.
Constant and varying targets share this implementation. Station interpolation
and the measured loading paths are unchanged. Earlier three-evaluation results
belong to the source revision recorded in their provenance.

Sign/notation conventions follow the frozen driver: compression positive,
Mandel 6-vectors, bedding rotation about the global y axis.

Form II multiplies matrix cohesion by its existing fabric function evaluated
at each trial stress; d_fab=0 reduces to form I.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from functools import lru_cache, partial

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from tfenn_esubi.data_gen import esubi_driver as ref  # frozen NumPy reference

SOURCE_COMMIT = "29ab3d55ddbe4d381d72eefd8ae859e43b101acf"

SQRT2 = np.sqrt(2.0)
I3 = jnp.eye(3)

NEWTON_TOL = ref.LATERAL_TOL_MPA
NEWTON_INCREMENT_RTOL = ref.LATERAL_INCREMENT_RTOL
N_NEWTON = ref.LATERAL_MAX_EVALUATIONS

# G0a failed-start thresholds (recorded diagnostics, not physics):
FAILED_ABS_RESIDUAL_MPA = 1.0
FAILED_TRAIL_RESIDUAL_MPA = 0.1
TRAIL_FRACTION = 0.10


# ---------------------------------------------------------------------------
# Theta -> frozen-driver parameter mapping (G0a convention, verbatim)
# ---------------------------------------------------------------------------


def theta_to_ref_params(theta: np.ndarray) -> "ref.ESUBIParams":
    """Map the 12-vector theta onto the frozen driver's dataclasses.

    CWFS convention from the G0a probe: phi_ini = phi_peak = phi_deg (no
    friction strengthening), gamma_phi = 1.0.
    """
    (E_p, E_o, G_op, nu_pp, nu_op, c_peak, c_res,
     phi_deg, c_j, phi_j_deg, gamma_c, k_sharp) = [float(v) for v in theta]
    cwfs = ref.CWFSParams(
        c_peak=c_peak, c_res=c_res, gamma_c=gamma_c,
        phi_ini_deg=phi_deg, phi_peak_deg=phi_deg,
        gamma_phi=1.0, k_sharpness=k_sharp,
    )
    return ref.ESUBIParams(
        elastic=ref.TIElasticParams(E1=E_p, E3=E_o, nu12=nu_pp, nu13=nu_op, G13=G_op),
        matrix=ref.MatrixMCParams(c=c_peak, phi_deg=phi_deg),
        joint=ref.JointParams(c_j=c_j, phi_j_deg=phi_j_deg,
                              n_hat=np.array([0.0, 0.0, 1.0])),
        cwfs=cwfs,
    )


def run_reference_numpy(theta: np.ndarray, theta_deg: float, p_conf: float,
                        n_steps: int, eps_a_total: float,
                        n_newton: int = N_NEWTON) -> dict:
    """Run the frozen NumPy driver (the reference the mirror must reproduce)."""
    out = ref.run_triaxial_path(
        params=theta_to_ref_params(theta),
        theta_rad=np.radians(theta_deg),
        n_steps=n_steps,
        axial_strain_total=eps_a_total,
        confining_pressure=p_conf,
        n_newton=n_newton,
    )
    sig = out["sig"]
    return {"q": sig[:, 0] - sig[:, 2], "sig": sig, "gamma": out["gamma_ps"]}


def path_constants(theta_deg: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Bond matrix and unit bedding normal for a bedding angle (host-side)."""
    th_rad = np.radians(theta_deg)
    M = ref.bond_matrix_mandel(ref.rotation_matrix_Ry(th_rad))
    n_hat = ref.bedding_normal(th_rad)
    n_hat = n_hat / np.linalg.norm(n_hat)
    return jnp.asarray(M), jnp.asarray(n_hat)


# ---------------------------------------------------------------------------
# JAX mirror internals (adapted verbatim from the G0a probe)
# ---------------------------------------------------------------------------


def _mandel_to_tensor_j(v):
    return jnp.array([
        [v[0], v[5] / SQRT2, v[4] / SQRT2],
        [v[5] / SQRT2, v[1], v[3] / SQRT2],
        [v[4] / SQRT2, v[3] / SQRT2, v[2]],
    ])


def _tensor_to_mandel_j(T):
    return jnp.array([T[0, 0], T[1, 1], T[2, 2],
                      SQRT2 * T[1, 2], SQRT2 * T[0, 2], SQRT2 * T[0, 1]])


def _ti_stiffness_local_j(E1, E3, nu12, nu13, G13):
    """Mirror of ref.ti_stiffness_local, including the frozen construction
    order: Voigt compliance with nu31 = nu13*E3/E1 placed asymmetrically,
    Mandel scaling, inversion, then symmetrization 0.5*(C + C.T)."""
    nu31 = nu13 * E3 / E1
    G12 = E1 / (2.0 * (1.0 + nu12))
    z = jnp.zeros(())
    S_v = jnp.array([
        [1.0 / E1, -nu12 / E1, -nu31 / E3, z, z, z],
        [-nu12 / E1, 1.0 / E1, -nu31 / E3, z, z, z],
        [-nu13 / E3, -nu13 / E3, 1.0 / E3, z, z, z],
        [z, z, z, 1.0 / G13, z, z],
        [z, z, z, z, 1.0 / G13, z],
        [z, z, z, z, z, 1.0 / G12],
    ])
    d = jnp.array([1.0, 1.0, 1.0, 1.0 / SQRT2, 1.0 / SQRT2, 1.0 / SQRT2])
    S_m = S_v * jnp.outer(d, d)
    C = jnp.linalg.inv(S_m)
    return 0.5 * (C + C.T)


def _mc_return_j(sig_m, c_coh, phi_rad):
    """Mirror of ref.mohr_coulomb_return. Uses the algebraic identity
    V diag(p + alpha*d) V^T = p*I + alpha*(T - p*I) so only eigenvalues are
    needed. Branches (F<=0 elastic, |A|<1e-12 apex, alpha clip [0,1]) follow
    the frozen code; untaken where-branches are guard-protected divisions."""
    T = _mandel_to_tensor_j(sig_m)
    w = jnp.linalg.eigvalsh(T)          # ascending
    s1, s2, s3 = w[2], w[1], w[0]       # descending order as in reference
    sin_phi = jnp.sin(phi_rad)
    cos_phi = jnp.cos(phi_rad)
    F = (s1 - s3) - (s1 + s3) * sin_phi - 2.0 * c_coh * cos_phi
    p_mean = (s1 + s2 + s3) / 3.0
    d1 = s1 - p_mean
    d3 = s3 - p_mean
    A = (d1 - d3) - (d1 + d3) * sin_phi
    B = 2.0 * p_mean * sin_phi + 2.0 * c_coh * cos_phi
    apex = jnp.abs(A) < 1e-12
    A_safe = jnp.where(apex, 1.0, A)
    alpha_raw = B / A_safe
    alpha = jnp.clip(alpha_raw, 0.0, 1.0)
    sigma_apex = c_coh / jnp.maximum(jnp.tan(phi_rad), 1e-6)
    s_apex = jnp.minimum(p_mean, sigma_apex)
    T_scaled = p_mean * I3 + alpha * (T - p_mean * I3)
    T_apex = s_apex * I3
    T_new = jnp.where(apex, T_apex, T_scaled)
    out_plastic = _tensor_to_mandel_j(T_new)
    plastic = F > 0.0
    sig_out = jnp.where(plastic, out_plastic, sig_m)
    return sig_out


def _joint_return_j(sig_m, c_j, phi_j_rad, n_hat):
    """Mirror of ref.joint_return. Conditions evaluated on tau^2 to avoid a
    sqrt-at-zero in the primal (tau <= tau_allow <=> tau^2 <= tau_allow^2 for
    tau_allow >= 0; output is continuous across the boundary)."""
    T = _mandel_to_tensor_j(sig_m)
    n = n_hat  # unit-normalized constant, done host-side
    traction = T @ n
    sigma_n = traction @ n
    shear = traction - sigma_n * n
    tau2 = shear @ shear
    tau_allow = jnp.maximum(0.0, c_j + sigma_n * jnp.tan(phi_j_rad))
    elastic = (tau2 <= tau_allow ** 2) | (tau2 < 1e-24)
    tau_safe = jnp.sqrt(jnp.where(tau2 < 1e-30, 1.0, tau2))
    factor = tau_allow / tau_safe
    dT_vec = (factor - 1.0) * shear
    # Both dyads are required: their contraction with n gives dT_vec.
    outer = jnp.outer(dT_vec, n) + jnp.outer(n, dT_vec)
    out_plastic = _tensor_to_mandel_j(T + outer)
    sig_out = jnp.where(elastic, sig_m, out_plastic)
    return sig_out


def _epsp_span_j(deps_p):
    T = _mandel_to_tensor_j(deps_p)
    w = jnp.linalg.eigvalsh(T)
    return w[2] - w[0]


def mirror_forward(theta, M_bond, n_hat, p_conf, d_eps11, *,
                   n_steps: int, n_newton: int = N_NEWTON):
    """Form-I constant-target path using the shared verified integrator."""
    return _forward_path_generic(theta, M_bond, n_hat,
                                 jnp.full((n_steps,), p_conf), d_eps11,
                                 n_steps, n_newton, None)


def q_at_eps(theta, M_bond, n_hat, p_conf, d_eps11, eps_stations, *,
             n_steps: int, n_newton: int = N_NEWTON):
    """Differentiable q at arbitrary measured eps_a stations.

    The frozen integrator's axial-strain history is exactly i * d_eps11 (the
    CTC path prescribes uniform axial increments), so model q at a measured
    station is the piecewise-linear interpolation of the frozen q history at
    theta-independent abscissae: gradients flow through the q ordinates only,
    and the frozen form itself is untouched (its output graph is merely
    sampled between integration nodes).
    """
    out = mirror_forward(theta, M_bond, n_hat, p_conf, d_eps11,
                         n_steps=n_steps, n_newton=n_newton)
    eps_grid = jnp.arange(n_steps + 1) * d_eps11
    return jnp.interp(eps_stations, eps_grid, out["q"])


@lru_cache(maxsize=None)
def forward_jit(n_steps: int, n_newton: int = N_NEWTON):
    """Cached jitted mirror forward for a fixed step count."""
    return jax.jit(partial(mirror_forward, n_steps=n_steps, n_newton=n_newton))


@lru_cache(maxsize=None)
def stations_fn(n_steps: int, n_newton: int = N_NEWTON):
    """Un-jitted station evaluator (for composition into likelihoods)."""
    return partial(q_at_eps, n_steps=n_steps, n_newton=n_newton)


def path_health(res_end: np.ndarray, sig: np.ndarray) -> dict:
    """G0a-style failed-start diagnostics for one integrated path."""
    res_end = np.asarray(res_end)
    finite = bool(np.all(np.isfinite(np.asarray(sig))))
    n_trail = max(1, int(TRAIL_FRACTION * len(res_end)))
    res_max = float(np.max(res_end)) if finite else float("inf")
    res_trail = float(np.median(res_end[-n_trail:])) if finite else float("inf")
    failed = ((not finite) or res_max > FAILED_ABS_RESIDUAL_MPA
              or res_trail > FAILED_TRAIL_RESIDUAL_MPA)
    return {"finite": finite, "res_max_MPa": res_max,
            "res_trail_median_MPa": res_trail, "failed_start": failed}


# ---------------------------------------------------------------------------
# Forward-equivalence check (mirror vs frozen NumPy driver)
# ---------------------------------------------------------------------------


def forward_equivalence_check(theta: np.ndarray,
                              paths: list[tuple[float, float]],
                              n_steps: int, eps_a_total: float) -> dict:
    """Max relative q error, mirror vs frozen NumPy driver, per path.

    Normalization: per-curve max |q_numpy| over the full (n_steps+1) history.
    """
    d_eps11 = eps_a_total / n_steps
    fwd = forward_jit(n_steps)
    theta_j = jnp.asarray(theta)
    per_path = {}
    worst = 0.0
    for bed_deg, p_conf in paths:
        M, n_hat = path_constants(bed_deg)
        np_out = run_reference_numpy(theta, bed_deg, p_conf, n_steps, eps_a_total)
        jx_out = fwd(theta_j, M, n_hat, p_conf, d_eps11)
        q_np = np_out["q"]
        q_jx = np.asarray(jx_out["q"])
        if not np.isfinite(q_jx).all():
            raise ref.LateralConvergenceError("Invalid JAX path in forward-equivalence check")
        qmax = max(float(np.max(np.abs(q_np))), 1e-30)
        rel = float(np.max(np.abs(q_np - q_jx)) / qmax)
        per_path[f"{bed_deg:g}deg_{p_conf:g}MPa"] = rel
        worst = max(worst, rel)
    return {
        "max_rel_err": worst,
        "n_paths": len(paths),
        "normalization": "per-curve max|q_numpy| over full history",
        "n_steps": n_steps,
        "eps_a_total": eps_a_total,
        "per_path_rel_err": per_path,
    }


# ---------------------------------------------------------------------------
# Form II: fabric-scaled matrix cohesion (AG-03 two-form comparison)
# ---------------------------------------------------------------------------
# 14-vector theta_f2 = theta (12, form-I order) + [d_fab, q_fab]. The fabric
# scale multiplies the CWFS cohesion inside the Newton loop, mirroring the
# frozen driver's form-II branch step for step; every form-I function above
# is untouched (control arm). d_fab = 0 reproduces form I bit for bit.

N_THETA_F2 = 14
THETA_F2_EXTRA_NAMES = ["d_fab", "q_fab"]


def theta_f2_to_ref_params(theta_f2: np.ndarray) -> "ref.ESUBIParams":
    """Map the 14-vector onto the frozen driver's dataclasses (fabric set)."""
    params = theta_to_ref_params(np.asarray(theta_f2)[:12])
    params.fabric = ref.FabricParams(d_fab=float(theta_f2[12]),
                                     q_fab=float(theta_f2[13]))
    return params


def run_reference_numpy_f2(theta_f2: np.ndarray, theta_deg: float, p_conf: float,
                           n_steps: int, eps_a_total: float,
                           n_newton: int = N_NEWTON) -> dict:
    """Run the frozen NumPy driver on the form-II branch."""
    out = ref.run_triaxial_path(
        params=theta_f2_to_ref_params(theta_f2),
        theta_rad=np.radians(theta_deg),
        n_steps=n_steps,
        axial_strain_total=eps_a_total,
        confining_pressure=p_conf,
        n_newton=n_newton,
    )
    sig = out["sig"]
    return {"q": sig[:, 0] - sig[:, 2], "sig": sig, "gamma": out["gamma_ps"]}


def _fabric_scale_j(sig_m, n_hat, d_fab, q_fab):
    """Mirror of ref.fabric_cohesion_scale: g = 1 + zeta + q_fab zeta^2 with
    zeta = d_fab (1 - 3 |sigma n|^2 / |sigma|_F^2), floored at ref.G_FLOOR.
    |sigma|_F^2 is the Mandel self-product; hydrostatic states give g = 1."""
    T = _mandel_to_tensor_j(sig_m)
    t = T @ n_hat
    den = jnp.maximum(sig_m @ sig_m, 1e-30)
    l1sq = (t @ t) / den
    zeta = d_fab * (1.0 - 3.0 * l1sq)
    g = 1.0 + zeta + q_fab * zeta * zeta
    return jnp.maximum(g, ref.G_FLOOR)


def mirror_forward_f2(theta_f2, M_bond, n_hat, p_conf, d_eps11, *,
                      n_steps: int, n_newton: int = N_NEWTON):
    """Form-II constant-target path using the shared verified integrator."""
    return _forward_path_generic(theta_f2, M_bond, n_hat,
                                 jnp.full((n_steps,), p_conf), d_eps11,
                                 n_steps, n_newton,
                                 (theta_f2[12], theta_f2[13]))


def q_at_eps_f2(theta_f2, M_bond, n_hat, p_conf, d_eps11, eps_stations, *,
                n_steps: int, n_newton: int = N_NEWTON):
    """Differentiable form-II q at measured eps_a stations (cf. q_at_eps)."""
    out = mirror_forward_f2(theta_f2, M_bond, n_hat, p_conf, d_eps11,
                            n_steps=n_steps, n_newton=n_newton)
    eps_grid = jnp.arange(n_steps + 1) * d_eps11
    return jnp.interp(eps_stations, eps_grid, out["q"])


@lru_cache(maxsize=None)
def forward_f2_jit(n_steps: int, n_newton: int = N_NEWTON):
    """Cached jitted form-II forward for a fixed step count."""
    return jax.jit(partial(mirror_forward_f2, n_steps=n_steps, n_newton=n_newton))


@lru_cache(maxsize=None)
def stations_f2_fn(n_steps: int, n_newton: int = N_NEWTON):
    """Un-jitted form-II station evaluator (for composition into likelihoods)."""
    return partial(q_at_eps_f2, n_steps=n_steps, n_newton=n_newton)


def forward_equivalence_check_f2(theta_f2: np.ndarray,
                                 paths: list[tuple[float, float]],
                                 n_steps: int, eps_a_total: float) -> dict:
    """Form-II max relative q error, mirror vs frozen NumPy driver, per path."""
    d_eps11 = eps_a_total / n_steps
    fwd = forward_f2_jit(n_steps)
    theta_j = jnp.asarray(theta_f2)
    per_path = {}
    worst = 0.0
    for bed_deg, p_conf in paths:
        M, n_hat = path_constants(bed_deg)
        np_out = run_reference_numpy_f2(theta_f2, bed_deg, p_conf,
                                        n_steps, eps_a_total)
        jx_out = fwd(theta_j, M, n_hat, p_conf, d_eps11)
        q_np = np_out["q"]
        q_jx = np.asarray(jx_out["q"])
        if not np.isfinite(q_jx).all():
            raise ref.LateralConvergenceError("Invalid JAX path in form-II equivalence check")
        qmax = max(float(np.max(np.abs(q_np))), 1e-30)
        rel = float(np.max(np.abs(q_np - q_jx)) / qmax)
        per_path[f"{bed_deg:g}deg_{p_conf:g}MPa"] = rel
        worst = max(worst, rel)
    return {
        "max_rel_err": worst,
        "n_paths": len(paths),
        "normalization": "per-curve max|q_numpy| over full history",
        "n_steps": n_steps,
        "eps_a_total": eps_a_total,
        "per_path_rel_err": per_path,
    }


# ---------------------------------------------------------------------------
# Effective-stress-path surface (the constant-target surface above is untouched)
# ---------------------------------------------------------------------------
#
# The effective lateral target follows the measured pore-pressure path for CU
# specimens and is constant for CD specimens. The constant-target entry points
# above use the same integration body, including numerical failure handling.


def _forward_path_generic(theta, M_bond, n_hat, p_conf_seq, d_eps11,
                          n_steps, n_newton, fabric,
                          lateral_tol=NEWTON_TOL,
                          lateral_rtol=NEWTON_INCREMENT_RTOL,
                          instrument=False, early_exit=False):
    """Shared constitutive body with a bounded, verified traction solve.

    ``fabric`` is None for form I, or (d_fab, q_fab) for form II. The tolerance
    is capped in MPa and relative to the imposed elastic stress increment;
    it cannot become large relative to the increment as the grid is refined.
    The static loop bound supports both forward and reverse differentiation.
    """
    if n_steps < 1 or n_newton < 1 or lateral_tol <= 0 or lateral_rtol <= 0:
        raise ValueError("Positive step count, evaluation budget and tolerances required")
    (E_p, E_o, G_op, nu_pp, nu_op, c_peak, c_res,
     phi_deg, c_j, phi_j_deg, gamma_c, k_sharp) = [theta[i] for i in range(12)]
    C_local = _ti_stiffness_local_j(E_p, E_o, nu_pp, nu_op, G_op)
    C = M_bond @ C_local @ M_bond.T
    C_inv = jnp.linalg.inv(C)
    A_lat = C[1:, 1:]
    b_axial = C[1:, 0]
    phi_j_rad = jnp.deg2rad(phi_j_deg)

    lat0 = jnp.linalg.solve(A_lat, -b_axial * d_eps11)
    d_eps_init = jnp.concatenate((jnp.atleast_1d(d_eps11), lat0))

    p0 = p_conf_seq[0]
    sigma0 = jnp.array([p0, p0, p0, 0.0, 0.0, 0.0])

    def cwfs_static_mc(gamma_ps):
        x_c = gamma_ps / jnp.maximum(gamma_c, 1e-12)
        s_c = 0.5 * (1.0 + jnp.tanh(k_sharp * (x_c - 0.5)))
        c_coh = c_peak - (c_peak - c_res) * s_c
        return c_coh, jnp.deg2rad(phi_deg)   # phi_ini == phi_peak, as frozen

    def step(carry, p_target):
        sigma, gamma_ps = carry
        c_coh, phi_rad = cwfs_static_mc(gamma_ps)
        target = jnp.array([p_target, p_target, 0.0, 0.0, 0.0])
        stress_scale = jnp.maximum(jnp.max(jnp.abs(C @ d_eps_init)),
                                   jnp.max(jnp.abs(target - sigma[1:])))
        tolerance = jnp.minimum(lateral_tol, lateral_rtol * stress_scale)

        def evaluate(d_eps):
            sigma_tr = sigma + C @ d_eps
            if fabric is None:
                c_eff = c_coh
            else:
                c_eff = c_coh * _fabric_scale_j(sigma_tr, n_hat, *fabric)
            sig_mc = _mc_return_j(sigma_tr, c_eff, phi_rad)
            sig_j = _joint_return_j(sig_mc, c_j, phi_j_rad, n_hat)
            diagnostics = {}
            if instrument:
                traction = _mandel_to_tensor_j(sig_mc) @ n_hat
                normal = traction @ n_hat
                shear = traction - normal * n_hat
                distance = (jnp.maximum(0.0, c_j + normal * jnp.tan(phi_j_rad))
                            - jnp.linalg.norm(shear))
                accepted_traction = _mandel_to_tensor_j(sig_j) @ n_hat
                accepted_normal = accepted_traction @ n_hat
                accepted_shear = accepted_traction - accepted_normal * n_hat
                principal = jnp.linalg.eigvalsh(_mandel_to_tensor_j(sig_j))
                diagnostics = {
                    "joint_active_evaluations": jnp.asarray(
                        jnp.max(jnp.abs(sig_j - sig_mc)) > 1e-12, dtype=jnp.int32),
                    "joint_min_envelope_distance_MPa": distance,
                    "joint_accepted_excess_MPa": (
                        jnp.linalg.norm(accepted_shear)
                        - jnp.maximum(0.0, c_j + accepted_normal * jnp.tan(phi_j_rad))),
                    "matrix_accepted_excess_MPa": (
                        principal[-1] - principal[0]
                        - (principal[-1] + principal[0]) * jnp.sin(phi_rad)
                        - 2.0 * c_eff * jnp.cos(phi_rad)),
                }
            return sig_j, sig_j[1:] - target, diagnostics

        sigma_post, res, diagnostics = evaluate(d_eps_init)

        def iteration(_, state):
            d_eps, sigma_post, res, count, diagnostics, relaxation = state
            active = (jnp.all(jnp.isfinite(res))
                      & (jnp.max(jnp.abs(res)) > tolerance))

            def correct(state):
                d_eps, sigma_post, res, count, previous, relaxation = state
                trial_eps = d_eps.at[1:].add(-relaxation*jnp.linalg.solve(A_lat, res))
                # Every correction is evaluated before it can be recorded.
                trial_sigma, trial_res, trial_diagnostics = evaluate(trial_eps)
                # Aitken's residual secant accelerates slowly contracting
                # corrections and damps oscillations at a yield switch.
                # Keep the current factor when the secant predicts no
                # positive correction along the elastic search direction.
                difference = trial_res-res
                relaxation = jax.lax.cond(
                    jnp.dot(res, difference) < 0.,
                    lambda: relaxation*(-jnp.dot(res, difference)/jnp.dot(difference, difference)),
                    lambda: relaxation)
                d_eps, sigma_post, res, diagnostics = trial_eps, trial_sigma, trial_res, trial_diagnostics
                if instrument:
                    diagnostics["joint_active_evaluations"] = (
                        previous["joint_active_evaluations"] + trial_diagnostics["joint_active_evaluations"])
                    diagnostics["joint_min_envelope_distance_MPa"] = jnp.minimum(
                        previous["joint_min_envelope_distance_MPa"],
                        trial_diagnostics["joint_min_envelope_distance_MPa"])
                return d_eps, sigma_post, res, count + 1, diagnostics, relaxation

            return jax.lax.cond(active, correct, lambda x: x, state)

        initial = (d_eps_init, sigma_post, res, jnp.array(1), diagnostics, jnp.array(1.0))
        if early_exit:
            # Forward-mode fits do not need the static reverse-mode tape.
            # Stop the identical iteration as soon as its condition is false.
            def continuing(state):
                return ((state[3] < n_newton) & jnp.all(jnp.isfinite(state[2]))
                        & (jnp.max(jnp.abs(state[2])) > tolerance))
            final = jax.lax.while_loop(continuing, lambda state: iteration(0, state), initial)
        else:
            final = jax.lax.fori_loop(1, n_newton, iteration, initial)
        d_eps, sigma_new, res, evaluations, diagnostics, _ = final
        deps_p = d_eps - C_inv @ (sigma_new - sigma)
        gamma_new = gamma_ps + jnp.maximum(0.0, _epsp_span_j(deps_p))
        converged = (jnp.all(jnp.isfinite(sigma_new))
                     & jnp.all(jnp.isfinite(d_eps)) & jnp.isfinite(gamma_new)
                     & (jnp.max(jnp.abs(res)) <= tolerance))
        return (sigma_new, gamma_new), {"sigma": sigma_new, "gamma": gamma_new,
                                        "d_eps": d_eps,
                                        "res_end": jnp.max(jnp.abs(res)),
                                        "lateral_tolerance": tolerance,
                                        "lateral_evaluations": evaluations,
                                        "converged": converged,
                                        **diagnostics}

    (_, _), ys = jax.lax.scan(step, (sigma0, 0.0), p_conf_seq, length=n_steps)
    sig_hist = jnp.concatenate([sigma0[None, :], ys["sigma"]], axis=0)
    out = {
        "q": jnp.where(jnp.all(ys["converged"]),
                       sig_hist[:, 0] - sig_hist[:, 2], jnp.nan),
        "sig": sig_hist,
        "eps": jnp.concatenate([jnp.zeros((1, 6)), jnp.cumsum(ys["d_eps"], axis=0)], axis=0),
        "gamma": jnp.concatenate([jnp.zeros(1), ys["gamma"]]),
        "res_end": ys["res_end"],
        "lateral_tolerance": ys["lateral_tolerance"],
        "lateral_evaluations": ys["lateral_evaluations"],
        "converged": ys["converged"],
        "path_converged": jnp.all(ys["converged"]),
    }
    if instrument:
        out["mechanism"] = {
            "joint_return_evaluations": jnp.sum(ys["lateral_evaluations"]),
            "evaluations_active": jnp.sum(ys["joint_active_evaluations"]),
            "steps_with_joint_active": jnp.sum(ys["joint_active_evaluations"] > 0),
            "min_envelope_distance_MPa": jnp.min(ys["joint_min_envelope_distance_MPa"]),
            "max_accepted_joint_excess_MPa": jnp.max(ys["joint_accepted_excess_MPa"]),
            "max_accepted_matrix_excess_MPa": jnp.max(ys["matrix_accepted_excess_MPa"]),
        }
    return out


def mirror_forward_path(theta, M_bond, n_hat, p_conf_seq, d_eps11, *,
                        n_steps: int, n_newton: int = N_NEWTON, early_exit=False):
    """Form-I integrator with a per-step lateral-control target."""
    return _forward_path_generic(theta, M_bond, n_hat, p_conf_seq, d_eps11,
                                 n_steps, n_newton, None, early_exit=early_exit)


def mirror_forward_path_f2(theta_f2, M_bond, n_hat, p_conf_seq, d_eps11, *,
                           n_steps: int, n_newton: int = N_NEWTON, early_exit=False):
    """Form-II integrator with a per-step lateral-control target."""
    return _forward_path_generic(theta_f2, M_bond, n_hat, p_conf_seq, d_eps11,
                                 n_steps, n_newton,
                                 (theta_f2[12], theta_f2[13]), early_exit=early_exit)


def q_at_eps_path(theta, M_bond, n_hat, p_conf_seq, d_eps11, eps_stations, *,
                  n_steps: int, n_newton: int = N_NEWTON):
    """Differentiable form-I q at measured stations, on a prescribed path."""
    out = mirror_forward_path(theta, M_bond, n_hat, p_conf_seq, d_eps11,
                              n_steps=n_steps, n_newton=n_newton)
    return jnp.interp(eps_stations, jnp.arange(n_steps + 1) * d_eps11, out["q"])


def q_at_eps_path_f2(theta_f2, M_bond, n_hat, p_conf_seq, d_eps11,
                     eps_stations, *, n_steps: int, n_newton: int = N_NEWTON):
    """Differentiable form-II q at measured stations, on a prescribed path."""
    out = mirror_forward_path_f2(theta_f2, M_bond, n_hat, p_conf_seq, d_eps11,
                                 n_steps=n_steps, n_newton=n_newton)
    return jnp.interp(eps_stations, jnp.arange(n_steps + 1) * d_eps11, out["q"])


@lru_cache(maxsize=None)
def stations_path_fn(n_steps: int, n_newton: int = N_NEWTON):
    """Un-jitted form-I path station evaluator."""
    return partial(q_at_eps_path, n_steps=n_steps, n_newton=n_newton)


@lru_cache(maxsize=None)
def stations_path_f2_fn(n_steps: int, n_newton: int = N_NEWTON):
    """Un-jitted form-II path station evaluator."""
    return partial(q_at_eps_path_f2, n_steps=n_steps, n_newton=n_newton)


PATH_EQUIVALENCE_TOL_MPA = 1e-9


def path_equivalence_check(theta, theta_f2, paths, n_steps: int,
                           eps_a_total: float) -> dict:
    """Constant sequence through the path surface == the constant-target one.

    Residuals sit at float noise (~1e-14 MPa), the same order as the mirror's
    own forward equivalence to the frozen NumPy driver, because the target is
    carried as a traced sequence element rather than a Python scalar.
    """
    d_eps11 = eps_a_total / n_steps
    worst = {"form1": 0.0, "form2": 0.0}
    for bed_deg, p_conf in paths:
        M, n_hat = path_constants(bed_deg)
        seq = jnp.full((n_steps,), float(p_conf))
        a = forward_jit(n_steps)(jnp.asarray(theta), M, n_hat,
                                 float(p_conf), d_eps11)["q"]
        b = mirror_forward_path(jnp.asarray(theta), M, n_hat, seq, d_eps11,
                                n_steps=n_steps)["q"]
        if not bool(jnp.all(jnp.isfinite(a)) & jnp.all(jnp.isfinite(b))):
            raise ref.LateralConvergenceError("Invalid form-I path in target-equivalence check")
        worst["form1"] = max(worst["form1"], float(jnp.max(jnp.abs(a - b))))
        if theta_f2 is not None:
            a2 = forward_f2_jit(n_steps)(jnp.asarray(theta_f2), M, n_hat,
                                         float(p_conf), d_eps11)["q"]
            b2 = mirror_forward_path_f2(jnp.asarray(theta_f2), M, n_hat, seq,
                                        d_eps11, n_steps=n_steps)["q"]
            if not bool(jnp.all(jnp.isfinite(a2)) & jnp.all(jnp.isfinite(b2))):
                raise ref.LateralConvergenceError("Invalid form-II path in target-equivalence check")
            worst["form2"] = max(worst["form2"], float(jnp.max(jnp.abs(a2 - b2))))
    return {
        "max_abs_q_difference_MPa": worst,
        "criterion_MPa": PATH_EQUIVALENCE_TOL_MPA,
        "passes": all(v < PATH_EQUIVALENCE_TOL_MPA for v in worst.values()),
    }
