"""12-parameter theta space for the ESUBI-CWFS inverse calibration.

Role: single source of truth for (a) the project-wide 12-parameter order and
units (identical to the G0a probe THETA_NAMES convention), (b) loading the
nominal Method-A parameter vector from the frozen data artifact, (c) the
bounded optimization box and the smooth bijective transform between physical
theta and unconstrained z-space, (d) the G0a admissibility floor, and (e)
admissible multi-start sampling schemes.

Theta order (units): [E_p, E_o, G_op, nu_pp, nu_op, c_peak, c_res, phi_deg,
c_j, phi_j_deg, gamma_c, k_sharpness] in [MPa, MPa, MPa, -, -, MPa, MPa, deg,
MPa, deg, -, -].

Transform design (documented module-level decision):
- Internal coordinate 6 is the ratio r_res = c_res / c_peak in (0.01, 0.95),
  so the admissibility constraint c_res < c_peak holds by construction for
  every z; all other coordinates are the physical parameters.
- "log" coordinates map z through a logistic onto [log lo, log hi] and
  exponentiate (bounded log-space box); "linear" coordinates map z through a
  logistic onto [lo, hi]. Both are smooth and bijective, so unconstrained
  gradient optimization in z respects all box bounds exactly.
- The box is an optimization-design choice (wide enough to contain the G0a
  sampling neighborhood and published Opalinus corridors); it is NOT part of
  the frozen constitutive form.

Admissibility floor (from the G0a probe, kept verbatim): all moduli,
cohesions, gamma_c, k_sharpness > 0; c_res < c_peak; phi_deg and phi_j_deg in
(5, 60) deg; symmetrized local TI stiffness positive definite (eigvalsh > 0,
computed with the frozen driver's own ti_stiffness_local).

Form II (fabric-scaled matrix cohesion, AG-03): the 14-parameter space
theta_f2 = theta (12, form-I order) + [d_fab, q_fab] gets its own parallel
section at the bottom of this module (box, transform, admissibility floor,
start sampling) plus the ``space_for_form`` dispatch namespace. Every form-I
definition above stays byte-for-byte untouched (control arm).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import jax.numpy as jnp
from jax.nn import sigmoid

from tfenn_esubi.data_gen import esubi_driver as ref  # frozen NumPy reference

from .mirror import N_THETA_F2, THETA_F2_EXTRA_NAMES  # form-II single source

THETA_NAMES = [
    "E_p", "E_o", "G_op", "nu_pp", "nu_op", "c_peak", "c_res",
    "phi_deg", "c_j", "phi_j_deg", "gamma_c", "k_sharpness",
]
THETA_UNITS = [
    "MPa", "MPa", "MPa", "-", "-", "MPa", "MPa",
    "deg", "MPa", "deg", "-", "-",
]
N_THETA = 12

# Aliases used by the frozen parameter-set artifact (opalinus_method_a.json).
PARAM_SET_ALIASES = {"phi": "phi_deg", "phi_j": "phi_j_deg", "k": "k_sharpness"}

# Internal transform coordinates: index 6 is r_res = c_res / c_peak.
INTERNAL_NAMES = [
    "E_p", "E_o", "G_op", "nu_pp", "nu_op", "c_peak", "r_res",
    "phi_deg", "c_j", "phi_j_deg", "gamma_c", "k_sharpness",
]
# name -> (lo, hi, kind); kind "log" = logistic in log-space, "linear" = logistic.
BOX = {
    "E_p": (1.0e3, 5.0e4, "log"),
    "E_o": (5.0e2, 2.0e4, "log"),
    "G_op": (2.0e2, 1.0e4, "log"),
    "nu_pp": (0.01, 0.45, "linear"),
    "nu_op": (0.01, 0.45, "linear"),
    "c_peak": (0.5, 30.0, "log"),
    "r_res": (0.01, 0.95, "linear"),
    "phi_deg": (5.0, 60.0, "linear"),
    "c_j": (0.1, 20.0, "log"),
    "phi_j_deg": (5.0, 60.0, "linear"),
    "gamma_c": (5.0e-4, 0.2, "log"),
    "k_sharpness": (0.5, 30.0, "log"),
}

_LO = np.array([BOX[n][0] for n in INTERNAL_NAMES])
_HI = np.array([BOX[n][1] for n in INTERNAL_NAMES])
_IS_LOG = np.array([BOX[n][2] == "log" for n in INTERNAL_NAMES])
_LOG_LO = np.where(_IS_LOG, np.log(_LO), _LO)
_LOG_HI = np.where(_IS_LOG, np.log(_HI), _HI)

_LO_J = jnp.asarray(_LO)
_HI_J = jnp.asarray(_HI)
_IS_LOG_J = jnp.asarray(_IS_LOG)
_LOG_LO_J = jnp.asarray(_LOG_LO)
_LOG_HI_J = jnp.asarray(_LOG_HI)


def load_nominal_theta(root: str | Path) -> np.ndarray:
    """Load the nominal Method-A theta from the frozen parameter-set artifact.

    Maps artifact names through PARAM_SET_ALIASES and returns the values in
    THETA_NAMES order; raises if any of the 12 components is missing.
    """
    path = Path(root) / "results/data/parameter_sets/opalinus_method_a.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    values: dict[str, float] = {}
    for entry in payload["params"]:
        name = PARAM_SET_ALIASES.get(entry["name"], entry["name"])
        values[name] = float(entry["value"])
    missing = [n for n in THETA_NAMES if n not in values]
    if missing:
        raise KeyError(f"parameter-set artifact missing components: {missing}")
    return np.array([values[n] for n in THETA_NAMES], dtype=float)


# ---------------------------------------------------------------------------
# z <-> theta transform
# ---------------------------------------------------------------------------


def theta_from_z(z):
    """Map unconstrained z (12,) to physical theta (12,) (jax-differentiable)."""
    u = _LOG_LO_J + (_LOG_HI_J - _LOG_LO_J) * sigmoid(z)
    internal = jnp.where(_IS_LOG_J, jnp.exp(u), u)
    theta = internal.at[6].set(internal[6] * internal[5])  # c_res = r_res*c_peak
    return theta


def internal_from_theta(theta: np.ndarray) -> np.ndarray:
    internal = np.array(theta, dtype=float)
    internal[6] = theta[6] / theta[5]  # r_res
    return internal


def z_from_theta(theta: np.ndarray) -> np.ndarray:
    """Inverse transform (host-side numpy). Raises if theta is outside the box."""
    internal = internal_from_theta(np.asarray(theta, dtype=float))
    if np.any(internal <= _LO) or np.any(internal >= _HI):
        bad = [INTERNAL_NAMES[i] for i in range(N_THETA)
               if not (_LO[i] < internal[i] < _HI[i])]
        raise ValueError(f"theta outside optimization box for: {bad}")
    v = np.where(_IS_LOG, np.log(internal), internal)
    frac = (v - _LOG_LO) / (_LOG_HI - _LOG_LO)
    frac = np.clip(frac, 1e-12, 1.0 - 1e-12)
    return np.log(frac / (1.0 - frac))


def in_box(theta: np.ndarray) -> bool:
    internal = internal_from_theta(np.asarray(theta, dtype=float))
    return bool(np.all(internal > _LO) and np.all(internal < _HI))


# ---------------------------------------------------------------------------
# Box-normalised coordinates u (the transform-native measure)
# ---------------------------------------------------------------------------
# u_i in (0, 1) is the position of internal coordinate i inside its own box on
# the box's NATIVE scale -- log for the "log" coordinates, linear otherwise --
# i.e. exactly sigmoid(z). Two things need this coordinate and neither can use
# raw physical units:
#   - sensitivity matrices whose singular vectors are read as directions in
#     parameter space: a raw-unit loading is dominated by whichever parameter
#     carries the largest unit (E_p in MPa against a dimensionless Poisson
#     ratio), so raw-unit directions are a statement about units, not about
#     the data;
#   - any width reported "as a fraction of the prior box", which is the only
#     form directly comparable with the ensemble band ratio R (defined on the
#     same u).
# u is the affine-in-native-measure sibling of z (u = sigmoid(z)); it is used
# where a finite box fraction is wanted and z where an unconstrained optimiser
# variable is wanted. The optimisation box itself is unchanged.


def theta_from_u(u):
    """Map box fractions u (12,) in (0, 1) to physical theta (jax-differentiable)."""
    v = _LOG_LO_J + (_LOG_HI_J - _LOG_LO_J) * u
    internal = jnp.where(_IS_LOG_J, jnp.exp(v), v)
    return internal.at[6].set(internal[6] * internal[5])  # c_res = r_res*c_peak


def u_from_theta(theta: np.ndarray) -> np.ndarray:
    """Inverse (host-side numpy): physical theta -> box fractions u (12,).

    Computed directly in the native measure rather than as sigmoid(z_from_theta)
    so no logistic round-trip is taken; the two agree to rounding for any theta
    strictly inside the box.
    """
    internal = internal_from_theta(np.asarray(theta, dtype=float))
    v = np.where(_IS_LOG, np.log(internal), internal)
    return (v - _LOG_LO) / (_LOG_HI - _LOG_LO)


# ---------------------------------------------------------------------------
# Admissibility floor (G0a)
# ---------------------------------------------------------------------------


def is_admissible(theta: np.ndarray) -> tuple[bool, list[str]]:
    """G0a admissibility floor; returns (ok, list of violated rules)."""
    th = np.asarray(theta, dtype=float)
    reasons: list[str] = []
    if not np.all(np.isfinite(th)):
        reasons.append("nonfinite component")
        return False, reasons
    if not np.all(th[[0, 1, 2, 5, 6, 8, 10, 11]] > 0.0):
        reasons.append("moduli/cohesions/gamma_c/k_sharpness must be > 0")
    if not th[6] < th[5]:
        reasons.append("c_res < c_peak violated")
    if not 5.0 < th[7] < 60.0:
        reasons.append("phi_deg outside (5, 60) deg")
    if not 5.0 < th[9] < 60.0:
        reasons.append("phi_j_deg outside (5, 60) deg")
    if not reasons:
        C_loc = ref.ti_stiffness_local(ref.TIElasticParams(
            E1=th[0], E3=th[1], nu12=th[3], nu13=th[4], G13=th[2]))
        if not bool(np.all(np.linalg.eigvalsh(C_loc) > 0.0)):
            reasons.append("symmetrized local TI stiffness not positive definite")
    return (not reasons), reasons


# ---------------------------------------------------------------------------
# Admissible start sampling
# ---------------------------------------------------------------------------

G0A_JITTER_MULT_IDX = [0, 1, 2, 3, 4, 5, 6, 10, 11]
G0A_JITTER_ADD_IDX = [7, 9]

START_SCHEMES = ("g0a_jitter", "box_loguniform")


def sample_starts(rng: np.random.Generator, n: int, scheme: str,
                  nominal: np.ndarray, max_reject: int = 100_000) -> np.ndarray:
    """Draw n admissible starts (theta space).

    Schemes:
      - "g0a_jitter": the G0a-verified local scheme -- multiplicative
        U(0.85, 1.15) on [E_p, E_o, G_op, nu_pp, nu_op, c_peak, c_res,
        gamma_c, k_sharpness] and additive U(-3, +3) deg on
        [phi_deg, phi_j_deg] around the nominal vector (0/250 failed starts
        in the G0a gate).
      - "box_loguniform": wide production scheme -- independent draws over
        the full optimization box (uniform in the transform-native measure:
        log-uniform for "log" coordinates, uniform for "linear" ones;
        r_res drawn uniformly and multiplied onto the drawn c_peak).

    Every draw passes the G0a admissibility floor (including TI positive
    definiteness) and the box-membership check; rejected draws are re-drawn.
    """
    if scheme not in START_SCHEMES:
        raise ValueError(f"unknown start scheme {scheme!r}; options {START_SCHEMES}")
    out: list[np.ndarray] = []
    n_reject = 0
    while len(out) < n:
        if n_reject > max_reject:
            raise RuntimeError(f"start sampling rejected {n_reject} draws; scheme={scheme}")
        if scheme == "g0a_jitter":
            th = np.array(nominal, dtype=float)
            th[G0A_JITTER_MULT_IDX] *= rng.uniform(0.85, 1.15, size=len(G0A_JITTER_MULT_IDX))
            th[G0A_JITTER_ADD_IDX] += rng.uniform(-3.0, 3.0, size=len(G0A_JITTER_ADD_IDX))
        else:
            internal = np.empty(N_THETA)
            for i in range(N_THETA):
                if _IS_LOG[i]:
                    internal[i] = np.exp(rng.uniform(_LOG_LO[i], _LOG_HI[i]))
                else:
                    internal[i] = rng.uniform(_LO[i], _HI[i])
            th = internal.copy()
            th[6] = internal[6] * internal[5]
        ok, _ = is_admissible(th)
        if ok and in_box(th):
            out.append(th)
        else:
            n_reject += 1
    return np.asarray(out)


# ---------------------------------------------------------------------------
# Form II: fabric-scaled matrix cohesion (14-parameter space)
# ---------------------------------------------------------------------------
# theta_f2 = theta (12, form-I order) + [d_fab, q_fab] (the frozen driver's
# FabricParams; mirror.py form-II section). Box design decision (documented):
# the 12 shared coordinates keep BOX verbatim (frozen protocol); the fabric
# coordinates inherit the AG-03 gate's search box
# (selfrev/evidence/AG-03_gate_anisotropy_reach.json ``setup.bounds``), which
# contained the gate optimum strictly interior with wide margin. Both fabric
# coordinates are "linear" kind, exactly like nu_pp/nu_op.

THETA_F2_NAMES = THETA_NAMES + THETA_F2_EXTRA_NAMES
THETA_F2_UNITS = THETA_UNITS + ["-", "-"]
INTERNAL_F2_NAMES = INTERNAL_NAMES + THETA_F2_EXTRA_NAMES
BOX_F2 = {
    **BOX,
    "d_fab": (-1.5, 1.5, "linear"),
    "q_fab": (-1.0, 8.0, "linear"),
}

_LO_F2 = np.array([BOX_F2[n][0] for n in INTERNAL_F2_NAMES])
_HI_F2 = np.array([BOX_F2[n][1] for n in INTERNAL_F2_NAMES])
_IS_LOG_F2 = np.array([BOX_F2[n][2] == "log" for n in INTERNAL_F2_NAMES])
# Masked (not np.where) log: the fabric bounds are negative and np.where
# would evaluate np.log on them; the first 12 entries equal the form-I arrays.
_LOG_LO_F2 = _LO_F2.copy()
_LOG_LO_F2[_IS_LOG_F2] = np.log(_LO_F2[_IS_LOG_F2])
_LOG_HI_F2 = _HI_F2.copy()
_LOG_HI_F2[_IS_LOG_F2] = np.log(_HI_F2[_IS_LOG_F2])

_IS_LOG_F2_J = jnp.asarray(_IS_LOG_F2)
_LOG_LO_F2_J = jnp.asarray(_LOG_LO_F2)
_LOG_HI_F2_J = jnp.asarray(_LOG_HI_F2)


def theta_from_z_f2(z):
    """Map unconstrained z (14,) to physical theta_f2 (14,) (jax-differentiable).

    Same construction as the form-I transform: internal coordinate 6 is
    r_res = c_res / c_peak; the fabric coordinates are plain "linear" logits.
    """
    u = _LOG_LO_F2_J + (_LOG_HI_F2_J - _LOG_LO_F2_J) * sigmoid(z)
    internal = jnp.where(_IS_LOG_F2_J, jnp.exp(u), u)
    theta = internal.at[6].set(internal[6] * internal[5])  # c_res = r_res*c_peak
    return theta


def internal_from_theta_f2(theta_f2: np.ndarray) -> np.ndarray:
    internal = np.array(theta_f2, dtype=float)
    internal[6] = theta_f2[6] / theta_f2[5]  # r_res
    return internal


def z_from_theta_f2(theta_f2: np.ndarray) -> np.ndarray:
    """Inverse form-II transform (host-side numpy); raises outside the box."""
    internal = internal_from_theta_f2(np.asarray(theta_f2, dtype=float))
    if np.any(internal <= _LO_F2) or np.any(internal >= _HI_F2):
        bad = [INTERNAL_F2_NAMES[i] for i in range(N_THETA_F2)
               if not (_LO_F2[i] < internal[i] < _HI_F2[i])]
        raise ValueError(f"theta_f2 outside optimization box for: {bad}")
    v = internal.copy()
    v[_IS_LOG_F2] = np.log(internal[_IS_LOG_F2])
    frac = (v - _LOG_LO_F2) / (_LOG_HI_F2 - _LOG_LO_F2)
    frac = np.clip(frac, 1e-12, 1.0 - 1e-12)
    return np.log(frac / (1.0 - frac))


def in_box_f2(theta_f2: np.ndarray) -> bool:
    internal = internal_from_theta_f2(np.asarray(theta_f2, dtype=float))
    return bool(np.all(internal > _LO_F2) and np.all(internal < _HI_F2))


def theta_from_u_f2(u):
    """Map box fractions u (14,) to physical theta_f2 (jax-differentiable).

    Same construction as the form-I box-fraction map; the fabric coordinates
    are "linear" kind, so their box fraction is the plain position in
    [-1.5, 1.5] / [-1, 8].
    """
    v = _LOG_LO_F2_J + (_LOG_HI_F2_J - _LOG_LO_F2_J) * u
    internal = jnp.where(_IS_LOG_F2_J, jnp.exp(v), v)
    return internal.at[6].set(internal[6] * internal[5])  # c_res = r_res*c_peak


def u_from_theta_f2(theta_f2: np.ndarray) -> np.ndarray:
    """Inverse form-II box-fraction map (host-side numpy)."""
    internal = internal_from_theta_f2(np.asarray(theta_f2, dtype=float))
    v = internal.copy()
    v[_IS_LOG_F2] = np.log(internal[_IS_LOG_F2])
    return (v - _LOG_LO_F2) / (_LOG_HI_F2 - _LOG_LO_F2)


# ---------------------------------------------------------------------------
# Form-II admissibility floor: the 12 form-I rules + fabric positivity
# ---------------------------------------------------------------------------

# Analytic fabric-positivity floor: g_min(d_fab, q_fab) >= 0.05, the same
# criterion the AG-03 gate declared (AG-03_gate_anisotropy_reach.json
# ``setup.criteria.g_min``). 0.05 keeps admissible samples four-plus orders
# of magnitude above the frozen driver's numerical clamp ref.G_FLOOR = 1e-6
# (whose docstring requires search boxes to keep g well above it), so no
# start begins in the floored zero-gradient regime, while leaving the
# strong-anisotropy region of interest uncut.
FABRIC_G_MIN_FLOOR = 0.05


def fabric_g_min(d_fab: float, q_fab: float) -> float:
    """Analytic minimum of the fabric cohesion scale over the reachable domain.

    g(zeta) = 1 + zeta + q_fab*zeta^2 with zeta = d_fab*(1 - 3*lambda) and
    lambda = |sigma n|^2 / |sigma|_F^2 in [0, 1] (Cauchy-Schwarz), so the
    reachable interval is zeta in [min(d_fab, -2*d_fab), max(d_fab, -2*d_fab)].
    The parabola's minimum over that closed interval is the smaller endpoint
    value or, when q_fab > 0 and the vertex zeta* = -1/(2*q_fab) lies inside,
    the vertex value 1 - 1/(4*q_fab). Evaluated BEFORE the driver's G_FLOOR
    clamp (this is the a-priori whole-domain form of the gate's
    realized-along-path check, i.e. at least as strict).
    """
    d, q = float(d_fab), float(q_fab)
    z_lo, z_hi = min(d, -2.0 * d), max(d, -2.0 * d)
    gmin = min(1.0 + z_lo + q * z_lo * z_lo, 1.0 + z_hi + q * z_hi * z_hi)
    if q > 0.0:
        z_star = -1.0 / (2.0 * q)
        if z_lo <= z_star <= z_hi:
            gmin = min(gmin, 1.0 - 1.0 / (4.0 * q))
    return gmin


def is_admissible_f2(theta_f2: np.ndarray) -> tuple[bool, list[str]]:
    """Form-II admissibility floor; returns (ok, list of violated rules).

    The 12 form-I rules apply unchanged to the matrix block, plus the
    analytic fabric-positivity rule fabric_g_min >= FABRIC_G_MIN_FLOOR.
    """
    th = np.asarray(theta_f2, dtype=float)
    if th.shape != (N_THETA_F2,):
        raise ValueError(f"expected a {N_THETA_F2}-vector, got shape {th.shape}")
    if not np.all(np.isfinite(th)):
        return False, ["nonfinite component"]
    reasons = list(is_admissible(th[:N_THETA])[1])
    gmin = fabric_g_min(th[12], th[13])
    if not gmin >= FABRIC_G_MIN_FLOOR:
        reasons.append(f"fabric g_min {gmin:.6g} below floor {FABRIC_G_MIN_FLOOR}")
    return (not reasons), reasons


# ---------------------------------------------------------------------------
# Form-II admissible start sampling
# ---------------------------------------------------------------------------

# g0a_jitter fabric half-widths (SMOKE-ONLY; see sample_starts_f2): 5 percent
# of each fabric box width, additive because the canonical zero-fabric
# nominal makes multiplicative jitter inert.
G0A_JITTER_D_FAB_HALFWIDTH = 0.15
G0A_JITTER_Q_FAB_HALFWIDTH = 0.45


def nominal_f2_from(nominal: np.ndarray) -> np.ndarray:
    """Canonical form-II embedding of a 12-vector nominal: zero fabric.

    d_fab = 0 reproduces form I bit for bit in the frozen driver (AG-03 gate
    check A), so the Method-A vector's form-II embedding appends [0.0, 0.0].
    """
    return np.concatenate([np.asarray(nominal, dtype=float), [0.0, 0.0]])


def sample_starts_f2(rng: np.random.Generator, n: int, scheme: str,
                     nominal_f2: np.ndarray,
                     max_reject: int = 100_000) -> np.ndarray:
    """Draw n admissible form-II starts (14-vector theta space).

    Schemes (same names as form I):
      - "g0a_jitter" (SMOKE-ONLY for form II): the form-I jitter on the 12
        matrix coordinates plus additive U(-0.15, +0.15) on d_fab and
        U(-0.45, +0.45) on q_fab around the nominal's fabric components --
        additive because the canonical form-II nominal has zero fabric and
        multiplicative jitter is inert at 0. Production lanes never use this
        scheme (script defaults keep box_loguniform).
      - "box_loguniform": wide production scheme -- independent draws over
        the full 14-coordinate box (log-uniform for "log" coordinates,
        uniform for "linear" ones; the fabric coordinates are "linear" and
        draw uniformly over their box, exactly like nu_pp/nu_op; r_res drawn
        uniformly and multiplied onto the drawn c_peak).

    Every draw passes the form-II admissibility floor (the 12 form-I rules
    plus the analytic fabric-positivity rule) and box membership; rejected
    draws are re-drawn.
    """
    if scheme not in START_SCHEMES:
        raise ValueError(f"unknown start scheme {scheme!r}; options {START_SCHEMES}")
    nominal_f2 = np.asarray(nominal_f2, dtype=float)
    if nominal_f2.shape != (N_THETA_F2,):
        raise ValueError(f"nominal_f2 must be a {N_THETA_F2}-vector, "
                         f"got shape {nominal_f2.shape}")
    out: list[np.ndarray] = []
    n_reject = 0
    while len(out) < n:
        if n_reject > max_reject:
            raise RuntimeError(f"start sampling rejected {n_reject} draws; scheme={scheme}")
        if scheme == "g0a_jitter":
            th = nominal_f2.copy()
            th[G0A_JITTER_MULT_IDX] *= rng.uniform(0.85, 1.15, size=len(G0A_JITTER_MULT_IDX))
            th[G0A_JITTER_ADD_IDX] += rng.uniform(-3.0, 3.0, size=len(G0A_JITTER_ADD_IDX))
            th[12] += rng.uniform(-G0A_JITTER_D_FAB_HALFWIDTH,
                                  G0A_JITTER_D_FAB_HALFWIDTH)
            th[13] += rng.uniform(-G0A_JITTER_Q_FAB_HALFWIDTH,
                                  G0A_JITTER_Q_FAB_HALFWIDTH)
        else:
            internal = np.empty(N_THETA_F2)
            for i in range(N_THETA_F2):
                if _IS_LOG_F2[i]:
                    internal[i] = np.exp(rng.uniform(_LOG_LO_F2[i], _LOG_HI_F2[i]))
                else:
                    internal[i] = rng.uniform(_LO_F2[i], _HI_F2[i])
            th = internal.copy()
            th[6] = internal[6] * internal[5]
        ok, _ = is_admissible_f2(th)
        if ok and in_box_f2(th):
            out.append(th)
        else:
            n_reject += 1
    return np.asarray(out)


# ---------------------------------------------------------------------------
# Per-form dispatch
# ---------------------------------------------------------------------------


def space_for_form(form: int) -> SimpleNamespace:
    """Single dispatch point for the per-form parameter-space interface.

    form 1 returns the frozen 12-parameter form-I space (the module-level
    functions verbatim, so form-I callers keep the identical code path);
    form 2 returns the 14-parameter fabric-scaled form-II space.
    ``extend_nominal`` maps the 12-vector Method-A nominal into the form's
    theta space (identity for form I; zero-fabric embedding for form II).

    Box-fraction members (``theta_from_u`` / ``u_from_theta`` and the box
    arrays) expose the transform-native measure the sensitivity matrices and
    the profile grid are expressed in: ``internal_names`` names the internal
    coordinates (index 6 is r_res, not c_res), ``box_lo`` / ``box_hi`` are the
    physical bounds and ``box_native_lo`` / ``box_native_hi`` the same bounds
    on the box's native scale (log for the "log" coordinates), so a width in
    u is directly a fraction of the prior box.
    """
    if form == 1:
        return SimpleNamespace(
            form=1, n_theta=N_THETA, theta_names=THETA_NAMES,
            theta_from_z=theta_from_z, z_from_theta=z_from_theta,
            in_box=in_box, is_admissible=is_admissible,
            sample_starts=sample_starts,
            extend_nominal=lambda nominal: np.asarray(nominal, dtype=float),
            theta_from_u=theta_from_u, u_from_theta=u_from_theta,
            internal_names=INTERNAL_NAMES,
            box_lo=_LO, box_hi=_HI, box_is_log=_IS_LOG,
            box_native_lo=_LOG_LO, box_native_hi=_LOG_HI,
        )
    if form == 2:
        return SimpleNamespace(
            form=2, n_theta=N_THETA_F2, theta_names=THETA_F2_NAMES,
            theta_from_z=theta_from_z_f2, z_from_theta=z_from_theta_f2,
            in_box=in_box_f2, is_admissible=is_admissible_f2,
            sample_starts=sample_starts_f2,
            extend_nominal=nominal_f2_from,
            theta_from_u=theta_from_u_f2, u_from_theta=u_from_theta_f2,
            internal_names=INTERNAL_F2_NAMES,
            box_lo=_LO_F2, box_hi=_HI_F2, box_is_log=_IS_LOG_F2,
            box_native_lo=_LOG_LO_F2, box_native_hi=_LOG_HI_F2,
        )
    raise ValueError(f"unknown form {form!r}; options (1, 2)")
