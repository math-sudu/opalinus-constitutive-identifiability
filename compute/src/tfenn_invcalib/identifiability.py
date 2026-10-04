"""Ensemble-free identifiability diagnostics at an attained optimum.

Role: the sensitivity matrices, Fisher information, Cramer-Rao bounds and null
space of the fitted model AT a given parameter vector. Every quantity here is a
property of the likelihood and the data at that point. None of them depends on
how many multi-start members were run, how wide the prior box was drawn, or
which admission rule selected an ensemble band, which is what separates a
structural flat direction from a practical one that a finite optimiser budget
merely failed to explore.

Which sensitivity matrix (three, because they answer three questions)
---------------------------------------------------------------------
  J_raw    d q_driver / d theta            the literal model sensitivity, no
                                           offset term at all
  J_fixed  d q_pred / d theta at delta=mu_d matches the scoring rule the gap
                                           decomposition uses
  A_joint  d q_pred / d(theta, delta)      the profiled likelihood's own
                                           sensitivity, nuisance included;
                                           the source of the profile Fisher
                                           information

In which coordinates
--------------------
Each is recorded twice: in physical theta units, and in BOX-NORMALISED
coordinates u -- the transform-native measure, log for the log-sampled
coordinates and linear otherwise, each scaled by its own box width. The
normalised form is the one to read and plot. A raw-unit singular vector is
dominated by whichever parameter carries the largest unit (E_p in MPa against a
dimensionless Poisson ratio), so raw-unit loadings would describe the choice of
units rather than the directions along which distinct parameter sets reproduce
the same measured curves. The internal coordinate is r_res = c_res / c_peak,
because that is the coordinate the box, the transform and the prior actually
use.

Fisher information and the delta nuisance
-----------------------------------------
With Sigma = sigma_q^2 I,

    F_joint = A_joint^T Sigma^-1 A_joint + diag(0_theta, sigma_d^-2 I)
    F_prof  = F_theta,theta - F_theta,delta F_delta,delta^-1 F_delta,theta

F_prof is the profile information for theta and is the primary reported object,
because the profiled delta is the likelihood that was optimised. F_fixed =
J_fixed^T Sigma^-1 J_fixed is reported beside it as the information under the
decomposition's scoring rule.

Clip handling is explicit, not silent. A CU curve whose profiled offset sits on
a budget bound at the optimum has a nuisance with no local freedom: it cannot
trade against theta, so its Schur row and column are DROPPED and the curve is
listed in a recorded active bound set. Which curves those are is measured at the
optimum, never assumed. The Schur complement is invariant to any diagonal
rescaling of the nuisance, so the choice of delta units does not enter F_prof.

These are Gauss-Newton informations: the outer-product term only. On this fit
the residuals are several sigma_q, so the dropped term sum_i r_i d^2 q_i/dp^2 is
not small, and the Gauss-Newton information is NOT the curvature of the attained
NLL. ``verify_delta_nuisance`` measures that difference instead of assuming it
away, and separates it from the nuisance algebra, which is checked exactly.

Declared tolerances
-------------------
Null space, on the normalised spectrum: sigma_i / sigma_1 < 1e-8 is numerically
null, < 1e-3 is practically flat. For a Fisher matrix the same cuts are applied
to sqrt(eigenvalue) ratios, which is the singular-value scale.

Pseudo-inverse: eigenvalues below FISHER_PINV_RTOL times the largest are treated
as null. D4's numerical singular-value cut of 1e-8 squares to 1e-16 on a Gram
matrix, which is below the double-precision noise floor of that matrix, so it
cannot be used as a pseudo-inverse cut; FISHER_PINV_RTOL = 1e-14 is the
attainable floor and is stated rather than inherited. The bound at the practical
cut (1e-6 on eigenvalues, the square of the 1e-3 singular-value cut) is reported
beside it, and so is the plain inverse where the matrix is invertible, so a
reader can see which bound is which.
"""

from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp

# Null-space cuts on a normalised singular-value spectrum (declared, D4).
NULL_SPACE_RTOL_NUMERICAL = 1e-8
NULL_SPACE_RTOL_PRACTICAL = 1e-3
# Spectral cuts for the Fisher pseudo-inverse, on eigenvalue ratios.
FISHER_PINV_RTOL = 1e-14
FISHER_PRACTICAL_RTOL = NULL_SPACE_RTOL_PRACTICAL ** 2
# How many smallest directions are reported with their loadings.
N_SMALLEST_DIRECTIONS = 5


# ---------------------------------------------------------------------------
# Sensitivity matrices
# ---------------------------------------------------------------------------


def build_jacobians(theta_star: np.ndarray, like, space) -> dict:
    """The three sensitivity matrices at ``theta_star``, in both coordinates.

    Returns the raw arrays (not JSON) plus the linearisation point: the
    profiled offsets used, and the box fractions u* of theta_star.
    """
    n_theta = space.n_theta
    K = like.n_curves
    cu = np.asarray(like.cu_indices, dtype=int)
    mu = like.cfg.cu_offset_mu
    th = jnp.asarray(np.asarray(theta_star, dtype=float))
    u_star = np.asarray(space.u_from_theta(theta_star), dtype=float)
    uj = jnp.asarray(u_star)

    if like.on_effective_stress_path:
        return _build_jacobians_no_nuisance(th, uj, theta_star, u_star,
                                            like, space, n_theta, K)

    _, delta_star = like.predict_jit(th)
    delta_star = np.asarray(delta_star)
    fixed_vec = jnp.full((K,), mu)
    cu_j = jnp.asarray(cu)

    def _raw(t):
        return like.forward(t).reshape(-1)

    def _fixed(t):
        return like.predict_at_delta(t, fixed_vec).reshape(-1)

    def _joint(t, d_cu):
        d_full = jnp.zeros((K,)).at[cu_j].set(d_cu)
        return like.predict_at_delta(t, d_full).reshape(-1)

    d_cu_star = jnp.asarray(delta_star[cu])

    J_raw = np.asarray(jax.jacfwd(_raw)(th))
    J_fixed = np.asarray(jax.jacfwd(_fixed)(th))
    A_t = np.asarray(jax.jacfwd(_joint, argnums=0)(th, d_cu_star))
    A_d = np.asarray(jax.jacfwd(_joint, argnums=1)(th, d_cu_star))

    J_raw_n = np.asarray(jax.jacfwd(lambda u: _raw(space.theta_from_u(u)))(uj))
    J_fixed_n = np.asarray(jax.jacfwd(lambda u: _fixed(space.theta_from_u(u)))(uj))
    A_t_n = np.asarray(jax.jacfwd(
        lambda u: _joint(space.theta_from_u(u), d_cu_star))(uj))

    return {
        "n_theta": n_theta,
        "n_cu": int(cu.size),
        "n_rows": int(J_raw.shape[0]),
        "theta_star": np.asarray(theta_star, dtype=float),
        "u_star": u_star,
        "delta_star": delta_star,
        "cu_indices": cu,
        "physical": {"J_raw": J_raw, "J_fixed": J_fixed,
                     "A_theta": A_t, "A_delta": A_d},
        "normalised": {"J_raw": J_raw_n, "J_fixed": J_fixed_n,
                       "A_theta": A_t_n, "A_delta": A_d},
        "coordinate_note": (
            "physical: d/d theta in each parameter's own unit; normalised: "
            "d/d u with u the position in the parameter's own box on the box's "
            "native scale (log for the log coordinates), internal coordinate 6 "
            "being r_res = c_res / c_peak. The nuisance columns A_delta are in "
            "MPa in both, because delta is not a box coordinate; the Schur "
            "complement is invariant to rescaling it."),
    }


def _build_jacobians_no_nuisance(th, uj, theta_star, u_star, like, space,
                                 n_theta: int, K: int) -> dict:
    """``build_jacobians`` on the measured effective-stress path.

    With the offset gone the three sensitivity matrices are one matrix: the
    predicted mean IS the driver output, so the raw model sensitivity, the
    fixed-offset sensitivity and the joint sensitivity's theta block all equal
    d q_driver / d theta, and the nuisance block has no columns. Everything
    downstream then does the right thing without a special case: the Schur
    complement over an empty nuisance is the identity operation, so the
    profile information equals the joint information equals the fixed-rule
    information, which is the correct statement and not a fallback.
    """
    def _raw(t):
        return like.forward(t).reshape(-1)

    J = np.asarray(jax.jacfwd(_raw)(th))
    J_n = np.asarray(jax.jacfwd(lambda u: _raw(space.theta_from_u(u)))(uj))
    A_d = np.zeros((J.shape[0], 0))
    return {
        "n_theta": n_theta,
        "n_cu": 0,
        "n_rows": int(J.shape[0]),
        "theta_star": np.asarray(theta_star, dtype=float),
        "u_star": u_star,
        "delta_star": np.zeros(K),
        "cu_indices": np.asarray([], dtype=int),
        "physical": {"J_raw": J, "J_fixed": J, "A_theta": J, "A_delta": A_d},
        "normalised": {"J_raw": J_n, "J_fixed": J_n, "A_theta": J_n,
                       "A_delta": A_d},
        "coordinate_note": (
            "physical: d/d theta in each parameter's own unit; normalised: "
            "d/d u with u the position in the parameter's own box on the "
            "box's native scale (log for the log coordinates), internal "
            "coordinate 6 being r_res = c_res / c_peak. The undrained curves "
            "are driven along the measured effective-stress path, so there is "
            "no offset nuisance: J_raw, J_fixed and A_theta are one matrix and "
            "A_delta has no columns."),
    }

# ---------------------------------------------------------------------------
# Fisher information, Schur complement, Cramer-Rao
# ---------------------------------------------------------------------------


def fisher_blocks(A_theta: np.ndarray, A_delta: np.ndarray,
                  sigma_q: float, sigma_d: float,
                  bound_active: np.ndarray) -> dict:
    """Joint Gauss-Newton information and its profile Schur complement.

    ``bound_active`` is the measured per-CU-curve boolean: True where the
    profiled offset sits on a budget bound at this point, so that nuisance has
    no local freedom and is dropped from the Schur correction rather than
    being credited with a trade it cannot make.
    """
    n = A_theta.shape[1]
    m = A_delta.shape[1]
    A = np.concatenate([A_theta, A_delta], axis=1)
    F = A.T @ A / sigma_q ** 2
    F[n:, n:] += np.eye(m) / sigma_d ** 2
    F_tt = F[:n, :n]
    F_td = F[:n, n:]
    F_dd = F[n:, n:]
    free = np.flatnonzero(~np.asarray(bound_active, dtype=bool))
    if free.size:
        Ftd = F_td[:, free]
        F_prof = F_tt - Ftd @ np.linalg.solve(F_dd[np.ix_(free, free)], Ftd.T)
    else:
        F_prof = F_tt.copy()
    F_prof = 0.5 * (F_prof + F_prof.T)
    return {"F_joint": F, "F_tt": F_tt, "F_td": F_td, "F_dd": F_dd,
            "F_prof": F_prof, "free_nuisance_indices": free}


def fisher_fixed(J_fixed: np.ndarray, sigma_q: float) -> np.ndarray:
    """Information under the decomposition's scoring rule (delta held at mu_d)."""
    F = J_fixed.T @ J_fixed / sigma_q ** 2
    return 0.5 * (F + F.T)


# A coordinate losing more than this fraction of its unit vector to the
# discarded (null) subspace has an UNBOUNDED Cramer-Rao bound, not a small one.
CRB_NULL_PROJECTION_TOL = 1e-8


def cramer_rao(F: np.ndarray, names: list[str]) -> dict:
    """Cramer-Rao bounds from a Fisher matrix at the declared spectral cuts.

    Reports the pseudo-inverse bound at the numerical cut, the bound at the
    practical cut, the plain-inverse bound where the matrix is invertible, the
    condition number and the numerical rank, so the reader can see which bound
    is which rather than being handed one number.

    Null directions are reported as UNBOUNDED, not as zero. ``sqrt(diag(F^+))``
    returns ~0 for a coordinate lying in the discarded subspace, because the
    pseudo-inverse sets that direction's variance to zero -- which reads as
    "perfectly determined" and means the exact opposite. The fix is to measure
    how much of each coordinate's unit vector survives the retained subspace:
    a coordinate that loses any of itself to a discarded direction has no
    finite bound at all, and its entry is infinity with the lost fraction
    recorded beside it. On a form whose joint mechanism never activates, that
    is the difference between reporting its two parameters as the
    best-determined in the table and reporting them as undetermined.
    """
    w, V = np.linalg.eigh(F)
    w = np.maximum(w, 0.0) if np.all(w > -1e-8 * max(w.max(), 1.0)) else w
    wmax = float(np.max(w))
    n = F.shape[0]

    def _pinv_sd(rtol: float):
        keep = w > rtol * wmax
        if not np.any(keep):
            return (np.full(n, np.inf), 0, np.ones(n))
        Pinv = (V[:, keep] / w[keep]) @ V[:, keep].T
        sd = np.sqrt(np.maximum(np.diag(Pinv), 0.0))
        # fraction of each coordinate's unit vector lost to discarded directions
        lost = 1.0 - np.sum(V[:, keep] ** 2, axis=1)
        lost = np.clip(lost, 0.0, 1.0)
        sd = np.where(lost > CRB_NULL_PROJECTION_TOL, np.inf, sd)
        return sd, int(np.sum(keep)), lost

    sd_num, rank_num, lost_num = _pinv_sd(FISHER_PINV_RTOL)
    sd_prac, rank_prac, lost_prac = _pinv_sd(FISHER_PRACTICAL_RTOL)
    wmin = float(np.min(w))
    cond = float(wmax / wmin) if wmin > 0.0 else float("inf")
    invertible = bool(wmin > 0.0 and cond < 1.0 / np.finfo(float).eps)
    sd_inv = None
    if invertible:
        sd_inv = np.sqrt(np.maximum(np.diag(np.linalg.inv(F)), 0.0)).tolist()
    unbounded = [names[i] for i in range(n) if not np.isfinite(sd_num[i])]
    return {
        "eigenvalues": w.tolist(),
        "eigenvalue_max": wmax,
        "eigenvalue_min": wmin,
        "condition_number": cond,
        "numerical_rank_at_pinv_rtol": rank_num,
        "numerical_rank_at_practical_rtol": rank_prac,
        "pinv_rtol": FISHER_PINV_RTOL,
        "practical_rtol": FISHER_PRACTICAL_RTOL,
        "crb_pinv": dict(zip(names, sd_num.tolist())),
        "crb_practical_cut": dict(zip(names, sd_prac.tolist())),
        "crb_plain_inverse": (dict(zip(names, sd_inv)) if sd_inv else None),
        "unbounded_coordinates_at_pinv_rtol": unbounded,
        "null_projection_at_pinv_rtol": dict(zip(names, lost_num.tolist())),
        "null_projection_at_practical_rtol": dict(zip(names, lost_prac.tolist())),
        "null_projection_tolerance": CRB_NULL_PROJECTION_TOL,
        "invertible": invertible,
        "note": ("crb_pinv uses the pseudo-inverse at the declared numerical "
                 "cut, with any coordinate projecting onto a discarded "
                 "direction reported as UNBOUNDED rather than as the zero the "
                 "pseudo-inverse would return; crb_practical_cut additionally "
                 "discards every direction flatter than the practical cut, so "
                 "it is the more optimistic bound on the coordinates that "
                 "survive and unbounded on more of them; crb_plain_inverse is "
                 "present only when the matrix is invertible in double "
                 "precision"),
    }


# ---------------------------------------------------------------------------
# Spectra and null space
# ---------------------------------------------------------------------------


def _spectrum(singular_values: np.ndarray, right_vectors: np.ndarray,
              names: list[str], source: str) -> dict:
    """Full spectrum + the smallest directions with their component loadings.

    ``right_vectors`` has one direction per COLUMN, ordered with
    ``singular_values`` descending.
    """
    s = np.asarray(singular_values, dtype=float)
    ratios = s / s[0] if s[0] > 0.0 else np.full_like(s, np.inf)
    n_small = min(N_SMALLEST_DIRECTIONS, s.size)
    directions = []
    for k in range(s.size - n_small, s.size):
        v = right_vectors[:, k]
        order = np.argsort(-np.abs(v))
        directions.append({
            "index": int(k),
            "singular_value": float(s[k]),
            "ratio_to_largest": float(ratios[k]),
            "numerically_null": bool(ratios[k] < NULL_SPACE_RTOL_NUMERICAL),
            "practically_flat": bool(ratios[k] < NULL_SPACE_RTOL_PRACTICAL),
            "loadings": {names[i]: float(v[i]) for i in range(len(names))},
            "dominant_components": [
                {"name": names[int(i)], "loading": float(v[int(i)])}
                for i in order[:3]],
        })
    return {
        "source": source,
        "singular_values": s.tolist(),
        "ratio_to_largest": ratios.tolist(),
        "cuts": {"numerically_null": NULL_SPACE_RTOL_NUMERICAL,
                 "practically_flat": NULL_SPACE_RTOL_PRACTICAL},
        "n_numerically_null": int(np.sum(ratios < NULL_SPACE_RTOL_NUMERICAL)),
        "n_practically_flat": int(np.sum(ratios < NULL_SPACE_RTOL_PRACTICAL)),
        "smallest_directions": directions,
        "direction_note": ("right singular vectors in normalised box-fraction "
                           "coordinates; a loading is a step in units of the "
                           "parameter's own prior-box width"),
    }


def jacobian_spectrum(J: np.ndarray, names: list[str], source: str) -> dict:
    _, s, Vt = np.linalg.svd(J, full_matrices=False)
    return _spectrum(s, Vt.T, names, source)


def fisher_spectrum(F: np.ndarray, names: list[str], sigma_q: float,
                    source: str) -> dict:
    """Spectrum of a Fisher matrix on the SINGULAR-VALUE scale.

    F = J^T Sigma^-1 J has eigenvalues s_i^2 / sigma_q^2, so sqrt(w) * sigma_q
    puts a Fisher spectrum on the same scale as a Jacobian's singular values
    and the same declared ratio cuts apply to both.
    """
    w, V = np.linalg.eigh(F)
    order = np.argsort(-w)
    w = np.maximum(w[order], 0.0)
    V = V[:, order]
    return _spectrum(np.sqrt(w) * sigma_q, V, names, source)


# ---------------------------------------------------------------------------
# Verification: the AD Jacobian against central differences
# ---------------------------------------------------------------------------


# Central-difference error below this is round-off, not truncation, so a rise
# across it says nothing about the model.
FD_ROUNDOFF_FLOOR = 1e-8
FD_STEP_SWEEP = (1e-3, 1e-4, 1e-5, 1e-6, 1e-7)


def _smoothness_from_sweep(rel: list[float], steps: tuple[float, ...]) -> dict:
    """Separate truncation error from a yield-surface switch, by convergence.

    An absolute threshold cannot do this. At a perfectly smooth point the
    central-difference error is O(h^2), so it is LARGE at a large step purely
    from truncation; reading that as "the model stops being smooth here" would
    call every strongly-curved coordinate non-smooth. What distinguishes the
    two is the direction of travel as the step shrinks:

      truncation  error falls ~100x per decade of h, until round-off floors it;
      a switch    error GROWS as h shrinks, because the difference quotient
                  keeps straddling two integration branches until the whole
                  interval finally fits on one side, and then collapses.

    So a switch is reported only where the error rises across consecutive steps
    while still above the round-off floor, and the smooth radius is the largest
    step from which the sequence descends without such a rise.
    """
    rel = list(rel)
    rises = []
    for i in range(len(rel) - 1):
        if (np.isfinite(rel[i]) and np.isfinite(rel[i + 1])
                and rel[i] > FD_ROUNDOFF_FLOOR and rel[i + 1] > rel[i]):
            rises.append({"from_h": steps[i], "to_h": steps[i + 1],
                          "rel_from": rel[i], "rel_to": rel[i + 1]})
    # largest step from which the sequence descends with no above-floor rise
    radius = None
    for i in range(len(steps)):
        tail_ok = all(not (rel[j] > FD_ROUNDOFF_FLOOR and rel[j + 1] > rel[j])
                      for j in range(i, len(rel) - 1))
        if tail_ok:
            radius = steps[i]
            break
    orders = [float(np.log10(rel[i] / rel[i + 1]))
              for i in range(len(rel) - 1)
              if rel[i + 1] > 0 and rel[i] > FD_ROUNDOFF_FLOOR]
    return {
        "switch_detected": bool(rises),
        "above_floor_rises": rises,
        "smooth_radius": radius,
        "convergence_orders_per_decade": orders,
        "roundoff_floor": FD_ROUNDOFF_FLOOR,
        "reading": ("smooth radius = the largest step from which the "
                    "central-difference error descends without an above-floor "
                    "rise; a detected switch means the model is only "
                    "differentiable inside a neighbourhood smaller than the "
                    "smallest step swept, so a linearised bound reported at a "
                    "wider scale is formal rather than informative"),
    }


def verify_jacobian_fd(fn, x0: np.ndarray, J_ad: np.ndarray,
                       names: list[str],
                       steps: tuple[float, ...] = FD_STEP_SWEEP) -> dict:
    """Central-difference check of an AD Jacobian, column by column, over a
    STEP SWEEP.

    ``fn`` maps an (n,) point to the (m,) prediction vector; ``J_ad`` is the
    autodiff Jacobian at ``x0``. The per-coordinate discrepancy is the column
    norm of the difference relative to the column norm of the AD column, which
    is the quantity that matters for a direction in parameter space.

    Why a sweep and not one step. A single step cannot tell a wrong gradient
    from a right one measured across a branch boundary, and this driver's
    forward map has yield-surface switches. At a smooth point the central
    difference converges as h^2 until round-off; near a switch the discrepancy
    instead grows as h shrinks toward the boundary distance and only collapses
    once the whole interval fits on one side. Both were observed here: the
    form-II ensemble optimum converges cleanly from h = 1e-4, while the
    lower-NLL point the profile rows found reproduces the AD column only at
    h = 1e-7, i.e. it sits within a millionth of a box width of a seam.

    So each column reports its best discrepancy over the sweep -- the honest
    verification -- AND the smoothness reading of ``_smoothness_from_sweep``,
    which separates truncation error from a switch by the DIRECTION the error
    travels as the step shrinks. An absolute threshold cannot make that
    separation: at a smooth point with strong curvature the error at a large
    step is large purely from truncation, and reading that as the end of the
    smooth region would call every sharply-curved coordinate non-smooth. A
    detected switch is what says a Cramer-Rao bound reported at a wider scale
    is formal rather than informative.

    A column that is IDENTICALLY ZERO is reported as a structural zero, not as
    a relative error of 0/0. That case is not a numerical accident here: a
    parameter whose mechanism never activates along the loading path leaves the
    prediction exactly unchanged, so both the AD and the finite-difference
    column vanish, and the agreement is exact. Such a column is a structural
    flat direction and is flagged as one; it is excluded from the median and
    maximum, which describe the columns that carry sensitivity.
    """
    x0 = np.asarray(x0, dtype=float)
    n = x0.size
    cols = []
    for i in range(n):
        col_ad = J_ad[:, i]
        den = float(np.linalg.norm(col_ad))
        per_step = []
        for h in steps:
            e = np.zeros(n)
            e[i] = h
            f_plus = np.asarray(fn(x0 + e), dtype=float)
            f_minus = np.asarray(fn(x0 - e), dtype=float)
            col_fd = (f_plus - f_minus) / (2.0 * h)
            num = float(np.linalg.norm(col_fd - col_ad))
            fd_norm = float(np.linalg.norm(col_fd))
            structural_zero = bool(den == 0.0 and fd_norm == 0.0)
            per_step.append({
                "h": h, "fd_column_norm": fd_norm, "abs_discrepancy": num,
                "rel_discrepancy": (0.0 if structural_zero else
                                    (num / den if den > 0.0 else float("nan"))),
            })
        rels = np.array([p["rel_discrepancy"] for p in per_step], dtype=float)
        structural_zero = bool(den == 0.0
                               and all(p["fd_column_norm"] == 0.0
                                       for p in per_step))
        smooth = _smoothness_from_sweep(list(rels), tuple(steps))
        best_j = int(np.nanargmin(rels)) if np.any(np.isfinite(rels)) else 0
        cols.append({
            "name": names[i],
            "ad_column_norm": den,
            "fd_column_norm": per_step[best_j]["fd_column_norm"],
            "abs_discrepancy": per_step[best_j]["abs_discrepancy"],
            "rel_discrepancy": per_step[best_j]["rel_discrepancy"],
            "best_step": per_step[best_j]["h"],
            "per_step": per_step,
            **smooth,
            "structurally_zero_column": structural_zero,
        })
    live = [c for c in cols if not c["structurally_zero_column"]]
    rel = np.array([c["rel_discrepancy"] for c in live], dtype=float)
    finite = rel[np.isfinite(rel)]
    zeros = [c["name"] for c in cols if c["structurally_zero_column"]]
    radii = [c["smooth_radius"] for c in live if c["smooth_radius"] is not None]
    unverified = [c["name"] for c in live if c["smooth_radius"] is None]
    switched = [c["name"] for c in live if c["switch_detected"]]
    return {
        "steps": list(steps),
        "scheme": ("central difference in the same coordinates as the AD "
                   "Jacobian, swept over the step sizes above"),
        "per_coordinate": cols,
        "structurally_zero_columns": zeros,
        "n_structurally_zero_columns": len(zeros),
        "median_rel_discrepancy": (float(np.median(finite)) if finite.size else None),
        "max_rel_discrepancy": (float(np.max(finite)) if finite.size else None),
        "max_rel_coordinate": (live[int(np.nanargmax(rel))]["name"]
                               if finite.size else None),
        "roundoff_floor": FD_ROUNDOFF_FLOOR,
        "coordinates_with_a_switch_inside_the_sweep": switched,
        "n_coordinates_with_a_switch": len(switched),
        "smooth_radius_min": (min(radii) if radii else None),
        "smooth_radius_max": (max(radii) if radii else None),
        "coordinates_without_a_smooth_radius": unverified,
        "smooth_radius_note": (
            "the largest step from which the central-difference error descends "
            "with no above-round-off rise, per coordinate and in the "
            "Jacobian's own coordinates. It bounds the neighbourhood over which a "
            "linearisation of this model -- a Fisher information, a Cramer-Rao "
            "bound, a quadratic interval -- describes anything. A bound wider "
            "than its own smooth radius is formal, not informative, and the "
            "profile likelihood is then the instrument that survives, since it "
            "never linearises."),
        "statistics_note": ("discrepancies are each column's best over the step "
                            "sweep; median and maximum are over the columns "
                            "carrying sensitivity; identically-zero columns "
                            "agree exactly and are listed separately"),
    }


# ---------------------------------------------------------------------------
# Verification: the delta-nuisance algebra
# ---------------------------------------------------------------------------


def _fd_hessian(f, x0: np.ndarray, h: float) -> np.ndarray:
    """Central-difference Hessian of a scalar function."""
    x0 = np.asarray(x0, dtype=float)
    n = x0.size
    H = np.zeros((n, n))
    f0 = float(f(x0))
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = h
        H[i, i] = (float(f(x0 + ei)) - 2.0 * f0 + float(f(x0 - ei))) / h ** 2
        for j in range(i + 1, n):
            ej = np.zeros(n)
            ej[j] = h
            val = (float(f(x0 + ei + ej)) - float(f(x0 + ei - ej))
                   - float(f(x0 - ei + ej)) + float(f(x0 - ei - ej))) / (4.0 * h * h)
            H[i, j] = H[j, i] = val
    return H


def _rel_frobenius(A: np.ndarray, B: np.ndarray) -> float:
    den = np.linalg.norm(B)
    return float(np.linalg.norm(A - B) / den) if den > 0.0 else float("nan")


def verify_delta_nuisance(theta_star: np.ndarray, like, space, jac: dict,
                          fisher: dict,
                          steps: tuple[float, ...] = (1e-5, 3e-5, 1e-4)) -> dict:
    """Check the profile Fisher information against independent numerical routes.

    Three separate questions, kept separate because they have different right
    answers:

    (1) Is the Schur complement computed correctly? Checked against Woodbury:
        F_prof = A_t^T (Sigma + A_d S_d^-1 A_d^T)^-1 A_t, a different algebraic
        route to the same object. This is exact linear algebra and is accepted
        at 1e-10 relative Frobenius, ~4 orders above double-precision rounding
        on matrices of this condition and far below any difference that could
        change a reported bound.

    (2) Is profiling the nuisance out of the ACTUAL likelihood the same
        operation? Checked by finite-differencing two Hessians of the true NLL
        at the optimum -- the joint one in (u, free deltas) and the profiled one
        in u -- and testing the Schur relation between them. This holds exactly
        at a nuisance optimum whatever the residual size, so the only error is
        the finite-difference one. The check is run over a SWEEP of steps rather
        than at one chosen step, because a single step cannot distinguish an
        algebra error from a badly-chosen step: the sweep shows the usual
        round-off/truncation minimum, and the accepted number is the best of it.
        The tolerance is 1e-4 relative Frobenius, which the measured floor at
        the best step clears by well over an order of magnitude while sitting
        far below any difference that could change a reported bound.

    (3) How far is the Gauss-Newton information from the actual curvature? This
        is NOT a tolerance: the residuals here are several sigma_q, so the
        dropped term sum_i r_i d^2 q_i / dp^2 is genuinely present. It is
        measured and reported, because a Cramer-Rao bound read off a
        Gauss-Newton information at a poorly-fitting optimum is a bound on a
        linearised model, and the size of that gap is what says how much.

    Bound-active nuisances are held at their clipped value throughout, matching
    the Schur restriction; whether the active set survives the finite-difference
    neighbourhood is measured rather than assumed, and a run in which it does
    not is flagged, because the profiled NLL is only piecewise smooth across an
    active-set change.
    """
    sigma_q = like.cfg.sigma_q
    sigma_d = like.cfg.cu_offset_sigma
    lo, hi = like.cfg.cu_offset_lo, like.cfg.cu_offset_hi
    u_star = jac["u_star"]
    delta_star = jac["delta_star"]
    cu = jac["cu_indices"]
    K = like.n_curves
    free = fisher["free_nuisance_indices"]

    # ---- (1) Woodbury cross-check of the Schur complement -----------------
    A_t = jac["normalised"]["A_theta"]
    A_d = jac["normalised"]["A_delta"][:, free] if free.size else None
    if A_d is not None and A_d.size:
        C = sigma_q ** 2 * np.eye(A_t.shape[0]) + sigma_d ** 2 * (A_d @ A_d.T)
        F_wood = A_t.T @ np.linalg.solve(C, A_t)
    else:
        F_wood = A_t.T @ A_t / sigma_q ** 2
    F_wood = 0.5 * (F_wood + F_wood.T)
    rel_wood = _rel_frobenius(F_wood, fisher["F_prof"])

    # ---- active-set stability over the finite-difference neighbourhood ----
    probe_steps = [s * sign for s in steps for sign in (1.0, -1.0)]
    margins = []
    for step in probe_steps:
        for i in range(u_star.size):
            uu = u_star.copy()
            uu[i] += step
            hat = np.asarray(like.delta_hat_unclipped_jit(
                jnp.asarray(np.asarray(space.theta_from_u(jnp.asarray(uu))))))
            margins.append(hat[cu])
    margins = np.asarray(margins)
    bound_active = np.zeros(cu.size, dtype=bool)
    bound_active[np.setdiff1d(np.arange(cu.size), free)] = True
    stable = True
    for j in range(cu.size):
        col = margins[:, j]
        if bound_active[j]:
            stable &= bool(np.all((col <= lo) | (col >= hi)))
        else:
            stable &= bool(np.all((col > lo) & (col < hi)))

    # ---- (2) finite-difference Schur relation on the true NLL -------------
    def _theta_of(u):
        return space.theta_from_u(jnp.asarray(np.asarray(u, dtype=float)))

    def _profiled(u):
        return float(like.nll_theta_jit(_theta_of(u)))

    delta_base = jnp.asarray(delta_star)

    def _joint(x):
        u = x[:u_star.size]
        d_free_scaled = x[u_star.size:]
        d = delta_base
        if free.size:
            d = d.at[jnp.asarray(cu[free])].set(
                jnp.asarray(d_free_scaled) * sigma_d)
        return float(like.nll_at_delta_jit(_theta_of(u), d))

    x0 = np.concatenate([u_star, delta_star[cu[free]] / sigma_d]) \
        if free.size else u_star.copy()
    n_u = u_star.size

    sweep = []
    for step in steps:
        H_prof_fd = _fd_hessian(_profiled, u_star, step)
        H_joint_fd = _fd_hessian(_joint, x0, step)
        H_uu = H_joint_fd[:n_u, :n_u]
        if free.size:
            H_ud = H_joint_fd[:n_u, n_u:]
            H_dd = H_joint_fd[n_u:, n_u:]
            H_schur = H_uu - H_ud @ np.linalg.solve(H_dd, H_ud.T)
        else:
            H_schur = H_uu
        sweep.append({
            "step": step,
            "rel_frobenius_schur_vs_profiled_hessian":
                _rel_frobenius(H_schur, H_prof_fd),
            "rel_frobenius_gauss_newton_vs_profiled_hessian":
                _rel_frobenius(fisher["F_prof"], H_prof_fd),
            "profiled_hessian_frobenius": float(np.linalg.norm(H_prof_fd)),
            "gauss_newton_frobenius": float(np.linalg.norm(fisher["F_prof"])),
        })
    best = min(sweep, key=lambda r: r["rel_frobenius_schur_vs_profiled_hessian"])
    gn = [r["rel_frobenius_gauss_newton_vs_profiled_hessian"] for r in sweep]

    tol_wood = 1e-10
    tol_schur = 1e-4
    return {
        "role": "verification of the profile Fisher information, three routes",
        "woodbury_cross_check": {
            "rel_frobenius": rel_wood,
            "tolerance": tol_wood,
            "passes": bool(rel_wood < tol_wood),
            "route": ("F_prof = A_t^T (Sigma + A_d sigma_d^2 A_d^T)^-1 A_t, "
                      "the marginalised-nuisance form of the same object"),
            "exactness": "exact linear algebra; a discrepancy here is a bug",
        },
        "finite_difference_schur_check": {
            "step_sweep": sweep,
            "best_step": best["step"],
            "best_rel_frobenius":
                best["rel_frobenius_schur_vs_profiled_hessian"],
            "tolerance": tol_schur,
            "passes": bool(best["rel_frobenius_schur_vs_profiled_hessian"]
                           < tol_schur),
            "route": ("Schur relation between the finite-difference Hessians of "
                      "the TRUE joint and profiled NLL at the optimum"),
            "exactness": ("exact at a nuisance optimum irrespective of residual "
                          "size; limited only by finite-difference accuracy, "
                          "which the step sweep exhibits"),
        },
        "gauss_newton_gap": {
            "rel_frobenius":
                best["rel_frobenius_gauss_newton_vs_profiled_hessian"],
            "range_over_step_sweep": [float(min(gn)), float(max(gn))],
            "is_a_tolerance": False,
            "reading": ("the residual-curvature term the Gauss-Newton "
                        "information omits, measured at this optimum; a large "
                        "value means the Cramer-Rao bounds below are bounds for "
                        "the linearised model, not the attained curvature. Its "
                        "near-constancy across the step sweep is what says it is "
                        "a property of the fit and not a finite-difference "
                        "artifact"),
        },
        "active_set": {
            "free_nuisance_indices": free.tolist(),
            "n_bound_active": int(cu.size - free.size),
            "stable_over_fd_neighbourhood": stable,
            "note": ("the profiled NLL is only piecewise smooth across an "
                     "active-set change; an unstable set invalidates the "
                     "finite-difference Hessian, so it is checked"),
        },
        "nuisance_scaling": ("nuisance finite-differenced in units of sigma_d; "
                             "the Schur complement is invariant to that choice"),
    }
