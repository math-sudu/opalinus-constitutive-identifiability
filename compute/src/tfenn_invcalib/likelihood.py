"""Negative-log-likelihood data misfit over the measured triaxial curves.

Noise model (documented; floors reference the project noise budget)
-------------------------------------------------------------------
Per-station observation noise on q is iid Gaussian with a single fixed
standard deviation

    sigma_q = max(sigma_q_base, noise_floor),

where sigma_q_base = 1.0 MPa (package default, config-recorded) and
noise_floor is READ AT RUNTIME from ``results/data/noise_budget.json`` entry
``frozen_driver_end_of_step_transient_noise_floor`` (0.6 MPa): the G0a-measured
frozen-driver end-of-step lateral-control transient perturbs q at the
0.1-0.6 MPa scale, so no likelihood component may claim tighter observation
precision than that floor. Context band: the budget's
``surrogate_scale_RMSE_q`` (1.86-3.03 MPa) is a surrogate-scale error, not a
driver-scale observation noise, and is not used as a floor here. The full NLL
constant (N log(sigma_q sqrt(2 pi)) per curve) is included so recorded values
are proper negative log likelihoods.

CU fixed-sigma_r CTC systematic-offset treatment (THE fixed choice)
-------------------------------------------------------------------
CU tests are undrained: effective sigma_r evolves with pore pressure, while
the frozen driver integrates a CTC path at constant sigma_r = sigma_r0 (the
digitisation artifact's cu_handling_note). This protocol mismatch is a known
systematic that the published budget bounds to 5-10 MPa at peak
(noise_budget.json entry ``CU_fixed_sigma_r_CTC_systematic_offset``). It
enters the likelihood as an EXPLICIT term, fixed as follows (one documented
choice; there is NO runtime switch):

- Predicted measurement mean for a CU curve:
      q_pred(eps) = q_driver(eps; theta) - delta_k * w(eps),
  with mobilization weight w(eps) = q_driver(eps) / max_eps q_driver, so the
  offset vanishes at zero deviatoric load (no pore-pressure buildup, CTC
  approximation exact) and reaches the full delta_k exactly at the model peak
  ("systematic offset at peak", budget definition). The frozen driver output
  itself is untouched.
- delta_k is one scalar per CU curve with Gaussian penalty
  N(mu_d, sigma_d^2), mu_d = midpoint of the published budget (7.5 MPa),
  sigma_d = quarter-width (1.25 MPa) so that mu_d +/- 2 sigma_d spans exactly
  the published 5-10 MPa budget; the penalty term
  (delta_k - mu_d)^2 / (2 sigma_d^2) + log(sigma_d sqrt(2 pi)) is explicit in
  the NLL. Budget numbers are READ from noise_budget.json, not hardcoded.
- At each theta evaluation delta_k is profiled analytically (the joint
  objective is quadratic in delta_k) and then CLIPPED to the published
  [5, 10] MPa budget interval, so the nuisance can never absorb more
  systematic offset than the published budget allows. Profiling equals
  marginalization up to a theta-independent constant for this linear-Gaussian
  nuisance. delta_k* is a smooth function of theta except at the clip
  boundary (piecewise-constant there; subgradient zero), keeping the NLL
  differentiable for the gradient inversion.
- CD curves carry NO offset term: q_pred = q_driver.

Effective-stress-path mode (SR-03(h))
-------------------------------------
The offset above is a surrogate for one thing: the undrained tests are driven
at a constant radial stress equal to the INITIAL effective confining pressure,
while their true effective radial stress follows the pore pressure. The source
publishes that channel for these three specimens, so the surrogate can be
replaced by the measurement. When the curve set carries ``p_conf_seq``
(``curves.attach_effective_stress_path``), the likelihood integrates the
frozen form along sigma_r_eff(eps_a) = sigma_r0_eff - p_w(eps_a) and the CU
offset term is GONE, not merely fixed: no delta, no Gaussian penalty, no
budget clip, and no CU penalty constant in the NLL. CU and CD curves then
share one observation model, which also makes the held-out rule and the
in-sample rule the same rule -- ``predict`` and ``predict_fixed_delta``
coincide, and every share of a decomposition is scored the way it was fitted
without a convention having to be declared.

The delta-explicit surface is refused in this mode rather than returned
degenerate: there is no nuisance to profile out, so a Schur complement over
delta is not the right object and a caller asking for one is asking under the
wrong model.

Held-out posterior-predictive convention (documented): when evaluating a
held-out CU curve (LOO), delta is FIXED at mu_d = 7.5 MPa instead of profiled,
so no held-out data leaks into the prediction; the nominal-null RMSE uses the
same rule. In-sample (calibration/oracle) RMSE uses the profiled clipped
delta, i.e. exactly the likelihood's own predicted mean.

Delta-explicit surface (added for the identifiability and gap-decomposition
instruments; the default objective is untouched)
------------------------------------------------------------------------
The profiled offset is the likelihood's own choice, but three consumers need
the offset as an EXPLICIT argument rather than an internal profile:

- the joint sensitivity matrix A_joint = d q_pred / d(theta, delta_1..delta_K)
  whose Schur complement is the profile Fisher information: delta must be a
  differentiable input, not a value profiled inside the graph;
- the raw model sensitivity J_raw = d q_driver / d theta, which carries no
  offset term at all;
- the fixed-delta objective, which FITS under the same rule the held-out
  lanes SCORE under (delta held at mu_d), so a fit and its score share one
  offset convention.

``_curve_terms_at_delta`` states the observation model at a supplied offset
and backs all three. It is deliberately a SECOND writing of the model rather
than a helper the profiled path also calls: routing ``_curve_terms`` through
it changes the traced graph enough to move the gradient by ~1e-15 relative,
which is enough to move the form-I ensemble trajectory off the bit-for-bit
reproduction the control arm of the two-form comparison requires. The
duplication is held closed by ``check_delta_explicit_equivalence``, an
executable check that the two writings agree to machine precision at a given
theta; every artifact that uses the delta-explicit surface records its result.
``nll_theta``, ``predict`` and ``predict_fixed_delta`` are byte-identical to
what they were before this surface existed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import jax
import jax.numpy as jnp

from . import mirror
from .curves import CurveSet
from .runio import NOISE_BUDGET_REL
from .theta_space import theta_from_z, theta_from_z_f2

SIGMA_Q_BASE_DEFAULT = 1.0  # MPa, package default observation std (pre-floor)


@dataclass(frozen=True)
class LikelihoodConfig:
    sigma_q: float            # effective per-station observation std (MPa)
    sigma_q_base: float       # configured base before flooring (MPa)
    noise_floor: float        # frozen-driver transient floor from budget (MPa)
    cu_offset_lo: float       # published budget lower bound (MPa)
    cu_offset_hi: float       # published budget upper bound (MPa)
    cu_offset_mu: float       # Gaussian penalty center = budget midpoint (MPa)
    cu_offset_sigma: float    # Gaussian penalty std = budget quarter-width (MPa)
    noise_budget_source: str  # provenance of the budget numbers

    def as_dict(self) -> dict:
        return {
            "sigma_q_MPa": self.sigma_q,
            "sigma_q_base_MPa": self.sigma_q_base,
            "noise_floor_MPa": self.noise_floor,
            "noise_floor_entry": "frozen_driver_end_of_step_transient_noise_floor",
            "cu_offset_budget_MPa": [self.cu_offset_lo, self.cu_offset_hi],
            "cu_offset_mu_MPa": self.cu_offset_mu,
            "cu_offset_sigma_MPa": self.cu_offset_sigma,
            "cu_offset_entry": "CU_fixed_sigma_r_CTC_systematic_offset",
            "cu_treatment": ("mobilization-weighted per-CU-curve scalar offset, "
                             "analytically profiled then clipped to the published "
                             "budget, Gaussian penalty explicit in the NLL; "
                             "CD curves carry no offset term"),
            "noise_budget_source": self.noise_budget_source,
        }


def load_likelihood_config(root: str | Path,
                           sigma_q_base: float = SIGMA_Q_BASE_DEFAULT) -> LikelihoodConfig:
    """Build the likelihood config from the frozen project noise budget."""
    path = Path(root) / NOISE_BUDGET_REL
    budget = json.loads(path.read_text(encoding="utf-8"))
    entries = {e["name"]: e for e in budget["entries"]}
    floor = float(entries["frozen_driver_end_of_step_transient_noise_floor"]["value_MPa"])
    lo, hi = (float(v) for v in
              entries["CU_fixed_sigma_r_CTC_systematic_offset"]["value_MPa_range"])
    return LikelihoodConfig(
        sigma_q=max(float(sigma_q_base), floor),
        sigma_q_base=float(sigma_q_base),
        noise_floor=floor,
        cu_offset_lo=lo,
        cu_offset_hi=hi,
        cu_offset_mu=0.5 * (lo + hi),
        cu_offset_sigma=0.25 * (hi - lo),
        noise_budget_source=str(path),
    )


# ---------------------------------------------------------------------------
# NLL construction
# ---------------------------------------------------------------------------

_LOG_2PI_HALF = 0.5 * np.log(2.0 * np.pi)


def make_likelihood(curveset: CurveSet, cfg: LikelihoodConfig,
                    form: int = 1) -> SimpleNamespace:
    """Build the NLL and predictive callables for one curve set.

    ``form`` selects the constitutive form the NLL differentiates through:
    1 (default) binds the frozen form-I mirror and transform exactly as
    before; 2 binds the fabric-scaled form-II mirror and the 14-parameter
    transform. The noise model, CU offset treatment, and NLL constants are
    form-independent frozen protocol (there is no explicit theta prior: the
    optimization box + admissibility floor ARE the prior for both forms).

    Returns a namespace with:
      nll_theta(theta)              pure jax scalar NLL (physical theta space)
      nll_z(z)                      NLL composed with the z -> theta transform
      predict(theta)                (q_pred (K,N), delta (K,)) profiled CU delta
      predict_fixed_delta(theta)    same but CU delta fixed at mu_d (held-out rule)
      jitted variants: nll_theta_jit, nll_z_jit, grad_nll_theta_jit,
                       predict_jit, predict_fixed_delta_jit
      const_total                   theta-independent NLL constant included
    """
    on_path = curveset.p_conf_seq is not None
    if form == 1:
        q_fn = (mirror.stations_path_fn(curveset.n_steps) if on_path
                else mirror.stations_fn(curveset.n_steps))
        t_from_z = theta_from_z
    elif form == 2:
        q_fn = (mirror.stations_path_f2_fn(curveset.n_steps) if on_path
                else mirror.stations_f2_fn(curveset.n_steps))
        t_from_z = theta_from_z_f2
    else:
        raise ValueError(f"unknown form {form!r}; options (1, 2)")
    eps_sta = jnp.asarray(curveset.eps_stations)
    q_meas = jnp.asarray(curveset.q_meas)
    # On the measured effective-stress path the CU offset term does not exist,
    # so every curve takes the offset-free branch.
    is_cu = jnp.zeros_like(jnp.asarray(curveset.is_cu)) if on_path \
        else jnp.asarray(curveset.is_cu)
    M_b, n_h = curveset.M_bond, curveset.n_hat
    p_c = curveset.p_conf_seq if on_path else curveset.p_conf
    d_eps11 = curveset.d_eps11
    s = cfg.sigma_q
    mu = cfg.cu_offset_mu
    t = cfg.cu_offset_sigma
    lo, hi = cfg.cu_offset_lo, cfg.cu_offset_hi
    n_pts = curveset.q_meas.shape[1]
    n_cu = 0 if on_path else int(np.sum(curveset.is_cu))
    # theta-independent constants: N log(s sqrt(2pi)) per curve + CU penalty const
    const_total = (curveset.n_curves * n_pts * (np.log(s) + _LOG_2PI_HALF)
                   + n_cu * (np.log(t) + _LOG_2PI_HALF))

    def _forward_all(theta):
        return jax.vmap(
            lambda M, n, p, es: q_fn(theta, M, n, p, d_eps11, es)
        )(M_b, n_h, p_c, eps_sta)

    def _curve_terms(q_model, q_meas_k, is_cu_k, profile_delta: bool):
        m = q_meas_k - q_model
        # ---- CD branch
        nll_cd = 0.5 * jnp.sum((m / s) ** 2)
        # ---- CU branch: mobilization-weighted profiled offset
        qmax = jnp.maximum(jnp.max(q_model), 1e-6)
        w = q_model / qmax
        if profile_delta:
            denom = jnp.sum(w * w) / s ** 2 + 1.0 / t ** 2
            delta_hat = (mu / t ** 2 - jnp.sum(w * m) / s ** 2) / denom
            delta = jnp.clip(delta_hat, lo, hi)
        else:
            delta = jnp.asarray(mu)
        e = m + delta * w
        nll_cu = 0.5 * jnp.sum((e / s) ** 2) + 0.5 * ((delta - mu) / t) ** 2
        nll_k = jnp.where(is_cu_k, nll_cu, nll_cd)
        q_pred = jnp.where(is_cu_k, q_model - delta * w, q_model)
        delta_out = jnp.where(is_cu_k, delta, 0.0)
        return nll_k, q_pred, delta_out

    def _curve_terms_at_delta(q_model, q_meas_k, is_cu_k, delta):
        """Observation model of one curve at a GIVEN offset.

        States the same CU/CD observation model as ``_curve_terms``, with the
        offset supplied instead of profiled. It is a second writing of the
        model rather than a shared helper because expressing ``_curve_terms``
        through it changes the XLA graph and moves the gradient by ~1e-15
        relative, which moves the form-I ensemble trajectory off bit-for-bit
        reproduction of the frozen control arm. The two writings are held
        together by an executable check rather than by inspection:
        ``check_delta_explicit_equivalence`` asserts, at any theta, that
        ``nll_at_delta(theta, profiled_delta) == nll_theta(theta)`` and
        ``predict_at_delta(theta, profiled_delta) == predict(theta)`` to
        machine precision whenever the profiled offset is interior to the
        budget, so a silent divergence cannot survive a run.
        """
        m = q_meas_k - q_model
        nll_cd = 0.5 * jnp.sum((m / s) ** 2)
        qmax = jnp.maximum(jnp.max(q_model), 1e-6)
        w = q_model / qmax
        e = m + delta * w
        nll_cu = 0.5 * jnp.sum((e / s) ** 2) + 0.5 * ((delta - mu) / t) ** 2
        nll_k = jnp.where(is_cu_k, nll_cu, nll_cd)
        q_pred = jnp.where(is_cu_k, q_model - delta * w, q_model)
        return nll_k, q_pred

    def _delta_hat_unclipped(q_model, q_meas_k):
        """The profiled offset BEFORE the budget clip (bound-margin diagnostic)."""
        m = q_meas_k - q_model
        qmax = jnp.maximum(jnp.max(q_model), 1e-6)
        w = q_model / qmax
        denom = jnp.sum(w * w) / s ** 2 + 1.0 / t ** 2
        return (mu / t ** 2 - jnp.sum(w * m) / s ** 2) / denom

    def nll_theta(theta):
        q_model = _forward_all(theta)
        nll_k, _, _ = jax.vmap(_curve_terms, in_axes=(0, 0, 0, None))(
            q_model, q_meas, is_cu, True)
        return jnp.sum(nll_k) + const_total

    def nll_z(z):
        return nll_theta(t_from_z(z))

    def _predict(theta, profile_delta: bool):
        q_model = _forward_all(theta)
        _, q_pred, delta = jax.vmap(_curve_terms, in_axes=(0, 0, 0, None))(
            q_model, q_meas, is_cu, profile_delta)
        return q_pred, delta

    predict = lambda theta: _predict(theta, True)              # noqa: E731
    predict_fixed_delta = lambda theta: _predict(theta, False)  # noqa: E731

    # ---- delta-explicit surface (see the module docstring) ----------------

    def forward(theta):
        """Raw driver q at the measured stations (K, N); NO offset term."""
        return _forward_all(theta)

    def predict_at_delta(theta, delta_vec):
        """Predicted mean (K, N) at an EXPLICIT per-curve offset vector."""
        q_model = _forward_all(theta)
        _, q_pred = jax.vmap(_curve_terms_at_delta)(
            q_model, q_meas, is_cu, delta_vec)
        return q_pred

    def nll_at_delta(theta, delta_vec):
        """Joint NLL L(theta, delta) at an EXPLICIT per-curve offset vector.

        Same constant as ``nll_theta``, so the two are directly comparable;
        L(theta, delta*(theta)) equals nll_theta(theta) whenever the profiled
        offset is interior to the budget.
        """
        q_model = _forward_all(theta)
        nll_k, _ = jax.vmap(_curve_terms_at_delta)(
            q_model, q_meas, is_cu, delta_vec)
        return jnp.sum(nll_k) + const_total

    def nll_theta_fixed_delta(theta):
        """NLL under the held-out offset rule (delta held at mu_d).

        This is the objective a lane FITS under when the fit and the score
        are required to share one offset rule; the default objective still
        profiles the offset.
        """
        q_model = _forward_all(theta)
        nll_k, _, _ = jax.vmap(_curve_terms, in_axes=(0, 0, 0, None))(
            q_model, q_meas, is_cu, False)
        return jnp.sum(nll_k) + const_total

    def nll_z_fixed_delta(z):
        return nll_theta_fixed_delta(t_from_z(z))

    def delta_hat_unclipped(theta):
        """Profiled offsets BEFORE the budget clip (K,), CD entries meaningless.

        The clip status at an optimum is a measured fact, not an assumption:
        the distance between this vector and the budget bounds is what says
        whether a nuisance is free or bound-pinned there.
        """
        q_model = _forward_all(theta)
        return jax.vmap(_delta_hat_unclipped)(q_model, q_meas)

    def _no_nuisance(*_args, **_kwargs):
        raise NotImplementedError(
            "the CU offset does not exist on the measured effective-stress "
            "path: there is no nuisance to supply, profile or marginalise")

    if on_path:
        predict_at_delta = nll_at_delta = _no_nuisance
        delta_hat_unclipped = _no_nuisance
        nll_theta_fixed_delta, nll_z_fixed_delta = nll_theta, nll_z

    return SimpleNamespace(
        curveset=curveset,
        cfg=cfg,
        form=form,
        on_effective_stress_path=on_path,
        const_total=float(const_total),
        n_curves=int(curveset.n_curves),
        n_stations=int(n_pts),
        cu_indices=([] if on_path
                    else np.flatnonzero(np.asarray(curveset.is_cu)).tolist()),
        nll_theta=nll_theta,
        nll_z=nll_z,
        predict=predict,
        predict_fixed_delta=predict_fixed_delta,
        forward=forward,
        predict_at_delta=predict_at_delta,
        nll_at_delta=nll_at_delta,
        nll_theta_fixed_delta=nll_theta_fixed_delta,
        nll_z_fixed_delta=nll_z_fixed_delta,
        delta_hat_unclipped=delta_hat_unclipped,
        nll_theta_jit=jax.jit(nll_theta),
        nll_z_jit=jax.jit(nll_z),
        grad_nll_theta_jit=jax.jit(jax.grad(nll_theta)),
        predict_jit=jax.jit(predict),
        predict_fixed_delta_jit=jax.jit(predict_fixed_delta),
        forward_jit=jax.jit(forward),
        predict_at_delta_jit=(predict_at_delta if on_path
                              else jax.jit(predict_at_delta)),
        nll_at_delta_jit=(nll_at_delta if on_path
                          else jax.jit(nll_at_delta)),
        nll_theta_fixed_delta_jit=jax.jit(nll_theta_fixed_delta),
        nll_z_fixed_delta_jit=jax.jit(nll_z_fixed_delta),
        delta_hat_unclipped_jit=(delta_hat_unclipped if on_path
                                 else jax.jit(delta_hat_unclipped)),
    )


def heldout_predictive_rmse(thetas, like_out: SimpleNamespace) -> dict:
    """Ensemble posterior-predictive RMSE on ONE held-out curve (MPa).

    The predictive is the per-station MEDIAN over the supplied members'
    predicted means, scored under the held-out rule (CU offset fixed at the
    budget midpoint, so no held-out datum enters the prediction). Returned
    alongside the per-member spread, which says whether the median is
    representative of the members or an artifact of averaging disagreeing
    ones.

    Single source for this quantity: the leave-one-curve-out lane and the gap
    decomposition both call it, so the decomposition can never be built on a
    predictive computed differently from the one the lane reported.
    """
    q_meas = like_out.curveset.q_meas[0]
    q_preds = []
    member_rmse = []
    for th in thetas:
        q_pred, _ = like_out.predict_fixed_delta_jit(jnp.asarray(np.asarray(th)))
        q_pred = np.asarray(q_pred)[0]
        q_preds.append(q_pred)
        member_rmse.append(float(np.sqrt(np.mean((q_meas - q_pred) ** 2))))
    q_med = np.median(np.stack(q_preds), axis=0)
    rmse_post = float(np.sqrt(np.mean((q_meas - q_med) ** 2)))
    return {
        "rmse_posterior_predictive_MPa": rmse_post,
        "n_members_used": len(member_rmse),
        "member_rmse_MPa": {
            "min": float(np.min(member_rmse)),
            "median": float(np.median(member_rmse)),
            "max": float(np.max(member_rmse)),
        },
        "q_median": q_med,
    }


def cu_offset_status(theta: np.ndarray, like: SimpleNamespace) -> dict:
    """Measured CU-offset state at a parameter vector (never assumed).

    Reports, per CU curve, the profiled offset the likelihood actually uses,
    the same offset BEFORE the budget clip, and whether the clip is active
    there. A bound-pinned offset is not a free nuisance: it has no local
    freedom to trade against theta, which is why the profile Fisher
    information drops its Schur row and column, and why the profiled NLL is
    only piecewise smooth in theta at that point. ``bound_margin_MPa`` is the
    unclipped offset's distance from the violated bound -- a small margin
    means the active set is about to change and any local expansion around
    this point is fragile, so the margin is reported rather than only the
    boolean.
    """
    th = jnp.asarray(np.asarray(theta, dtype=float))
    _, delta = like.predict_jit(th)
    delta = np.asarray(delta)
    hat = np.asarray(like.delta_hat_unclipped_jit(th))
    cfg = like.cfg
    keys = like.curveset.keys
    per_curve = {}
    active = []
    for k in like.cu_indices:
        at_lo = bool(hat[k] <= cfg.cu_offset_lo)
        at_hi = bool(hat[k] >= cfg.cu_offset_hi)
        if at_lo:
            margin = float(cfg.cu_offset_lo - hat[k])
        elif at_hi:
            margin = float(hat[k] - cfg.cu_offset_hi)
        else:
            margin = float(min(hat[k] - cfg.cu_offset_lo, cfg.cu_offset_hi - hat[k]))
        per_curve[keys[k]] = {
            "delta_used_MPa": float(delta[k]),
            "delta_unclipped_MPa": float(hat[k]),
            "clip_active": bool(at_lo or at_hi),
            "bound": ("lower" if at_lo else ("upper" if at_hi else None)),
            "bound_margin_MPa": margin,
        }
        if at_lo or at_hi:
            active.append(keys[k])
    return {
        "budget_MPa": [cfg.cu_offset_lo, cfg.cu_offset_hi],
        "per_cu_curve": per_curve,
        "active_bound_set": active,
        "n_bound_active": len(active),
        "n_cu_curves": len(like.cu_indices),
    }


def check_delta_explicit_equivalence(theta: np.ndarray,
                                     like: SimpleNamespace) -> dict:
    """Assert the delta-explicit surface reproduces the profiled likelihood.

    The delta-explicit observation model (``_curve_terms_at_delta``) is a
    second writing of the profiled path's model, kept separate so the default
    objective stays bit-for-bit (see the module docstring). This check closes
    the duplication: fed the likelihood's own profiled offsets, the explicit
    surface must reproduce ``nll_theta`` and ``predict`` to machine precision.

    The identity is exact only where the profiled offset is INTERIOR to the
    budget; at a clipped bound the two still agree, because the clipped value
    is what ``predict`` used. Returns the measured discrepancies and a
    ``passes`` flag at a 1e-9 relative tolerance (well above double-precision
    rounding on a ~1e3-1e4 NLL, well below any physical scale here).
    """
    th = jnp.asarray(np.asarray(theta, dtype=float))
    q_ref, delta_ref = like.predict_jit(th)
    nll_ref = float(like.nll_theta_jit(th))
    q_exp = np.asarray(like.predict_at_delta_jit(th, delta_ref))
    nll_exp = float(like.nll_at_delta_jit(th, delta_ref))
    dq = float(np.max(np.abs(np.asarray(q_ref) - q_exp)))
    dn = abs(nll_exp - nll_ref)
    rel_n = dn / max(1.0, abs(nll_ref))
    rel_q = dq / max(1.0, float(np.max(np.abs(np.asarray(q_ref)))))
    tol = 1e-9
    return {
        "role": ("equivalence of the delta-explicit observation model with the "
                 "profiled likelihood at the same offsets"),
        "nll_profiled": nll_ref,
        "nll_at_explicit_delta": nll_exp,
        "nll_abs_diff": dn,
        "nll_rel_diff": rel_n,
        "q_pred_max_abs_diff_MPa": dq,
        "q_pred_rel_diff": rel_q,
        "tolerance_rel": tol,
        "passes": bool(rel_n < tol and rel_q < tol),
    }


def rmse_per_curve(theta: np.ndarray, like: SimpleNamespace,
                   profile_delta: bool = True) -> np.ndarray:
    """RMSE_q per curve (MPa) of the observation-model predicted mean.

    profile_delta=True  -> in-sample rule (likelihood's own profiled clipped
                           CU delta; CD unaffected).
    profile_delta=False -> held-out rule (CU delta fixed at the budget
                           midpoint mu_d; no held-out data used).
    """
    fn = like.predict_jit if profile_delta else like.predict_fixed_delta_jit
    q_pred, _ = fn(jnp.asarray(np.asarray(theta, dtype=float)))
    resid = like.curveset.q_meas - np.asarray(q_pred)
    return np.sqrt(np.mean(resid ** 2, axis=1))
