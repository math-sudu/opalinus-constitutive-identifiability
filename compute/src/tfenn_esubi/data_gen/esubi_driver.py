"""Phase-1 ESUBI constitutive driver (no CWFS softening).

Implements a stress-path integrator for a transversely-isotropic (TI) elastic
matrix with two plastic yield mechanisms:

  1. Matrix Mohr-Coulomb (isotropic cohesion/friction on principal stresses).
  2. Ubiquitous-joint weakness plane with orientation n_hat.

Sign convention
---------------
Compression positive (geotechnics). Principal ordering s1 >= s2 >= s3.

Tensor notation
---------------
Mandel 6-vectors for stress/strain with index order
  [sigma_11, sigma_22, sigma_33, sqrt(2) sigma_23, sqrt(2) sigma_13, sqrt(2) sigma_12].

This matches TFENN's Mandel convention (see scripts/verify_hexagonal.py).

Phase 1 scope
-------------
- TI elastic stiffness assembled from (E1, E3, nu12, nu13, G13).
- Matrix yield: F_m = (s1 - s3) - (s1 + s3) sin(phi) - 2 c cos(phi).
  Return mapping: closest-point projection that preserves hydrostatic pressure
  and scales the principal deviator to sit on the Mohr-Coulomb surface.
- Joint yield: tau_j - c_j - sigma_n tan(phi_j) on plane with normal n_hat.
  Return mapping: reduce shear-traction magnitude while preserving normal stress.
- No strain softening.  Cohesion and friction are constant.

Form II extension (optional, additive)
--------------------------------------
``ESUBIParams.fabric`` switches on a microstructure-tensor scaling of the
matrix cohesion (Pietruszczak-Mroz line, one deviatoric parameter plus the
n=1 higher-order term); ``fabric=None`` leaves every form-I path untouched
and ``d_fab=0`` reproduces form I bit for bit.  See ``FabricParams``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SQRT2 = np.sqrt(2.0)

# Numerical controls only; neither yield return nor the material domain changes.
LATERAL_TOL_MPA = 1e-9
LATERAL_INCREMENT_RTOL = 1e-6
# The repaired inclined boundary needs 262 evaluations at a measured fold
# starting vector. Keep the same tolerance and accepted-state iteration while
# allowing this slowly contracting joint return to reach its stress targets.
LATERAL_MAX_EVALUATIONS = 512


class LateralConvergenceError(RuntimeError):
    """The returned stress did not satisfy the radial and shear targets."""

# ---------------------------------------------------------------------------
# Mandel / tensor helpers
# ---------------------------------------------------------------------------


def mandel_to_tensor(v: np.ndarray) -> np.ndarray:
    """Convert a 6-vector in Mandel notation to a 3x3 symmetric tensor."""
    return np.array([
        [v[0], v[5] / SQRT2, v[4] / SQRT2],
        [v[5] / SQRT2, v[1], v[3] / SQRT2],
        [v[4] / SQRT2, v[3] / SQRT2, v[2]],
    ])


def tensor_to_mandel(T: np.ndarray) -> np.ndarray:
    return np.array([T[0, 0], T[1, 1], T[2, 2],
                     SQRT2 * T[1, 2], SQRT2 * T[0, 2], SQRT2 * T[0, 1]])


def rotation_matrix_Ry(theta_rad: float) -> np.ndarray:
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([[c, 0.0, s],
                     [0.0, 1.0, 0.0],
                     [-s, 0.0, c]])


def bond_matrix_mandel(R: np.ndarray) -> np.ndarray:
    """Return the 6x6 Mandel Bond matrix M such that
    sigma_mandel_rot = M @ sigma_mandel for rotation R.

    Built by applying R to each Mandel-basis unit stress and reading the
    rotated tensor back into Mandel form.  sigma_rot = R sigma R^T.
    """
    idx = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    M = np.zeros((6, 6))
    for J in range(6):
        e = np.zeros(6)
        e[J] = 1.0
        T = mandel_to_tensor(e)
        T_rot = R @ T @ R.T
        M[:, J] = tensor_to_mandel(T_rot)
    return M


# ---------------------------------------------------------------------------
# TI elastic stiffness
# ---------------------------------------------------------------------------


@dataclass
class TIElasticParams:
    E1: float   # in-plane Young's modulus (horizontal, parallel to bedding)
    E3: float   # out-of-plane Young's modulus (along bedding normal)
    nu12: float  # Poisson ratio in bedding plane
    nu13: float  # Poisson ratio between bedding plane and normal
    G13: float  # out-of-plane shear modulus


def ti_stiffness_local(p: TIElasticParams) -> np.ndarray:
    """Return the 6x6 Mandel stiffness with bedding-plane isotropy about axis z
    (local frame: bedding plane = xy, normal = z).

    Builds the compliance first and inverts it to avoid sign-convention
    surprises.  Mandel scaling: C_mandel[3:, 3:] = 2 * G (since 2 epsilon_ij
    becomes sqrt(2) eps when using Mandel, then squared → factor 2 on stiffness).
    """
    E1, E3, nu12, nu13, G13 = p.E1, p.E3, p.nu12, p.nu13, p.G13
    nu31 = nu13 * E3 / E1
    G12 = E1 / (2.0 * (1.0 + nu12))
    # Voigt compliance (engineering strain convention)
    S_v = np.array([
        [1.0 / E1, -nu12 / E1, -nu31 / E3, 0, 0, 0],
        [-nu12 / E1, 1.0 / E1, -nu31 / E3, 0, 0, 0],
        [-nu13 / E3, -nu13 / E3, 1.0 / E3, 0, 0, 0],
        [0, 0, 0, 1.0 / G13, 0, 0],
        [0, 0, 0, 0, 1.0 / G13, 0],
        [0, 0, 0, 0, 0, 1.0 / G12],
    ])
    # Convert Voigt->Mandel: scale shear rows/cols so that C is the same linear
    # operator in Mandel. For compliance, S_mandel[i,j] with i or j >=3 is divided by 2.
    S_m = S_v.copy()
    for i in range(3, 6):
        S_m[i, :] /= SQRT2
        S_m[:, i] /= SQRT2
    C = np.linalg.inv(S_m)
    # Symmetrise to kill numerical noise
    C = 0.5 * (C + C.T)
    return C


def ti_stiffness_global(p: TIElasticParams, theta_rad: float) -> np.ndarray:
    """Return stiffness in the global frame for a bedding plane whose normal
    makes angle theta with the global y axis in the xz-plane (rotation about
    the global y axis).  theta=0: bedding plane is horizontal (xy-plane).
    """
    C_local = ti_stiffness_local(p)
    R = rotation_matrix_Ry(theta_rad)
    M = bond_matrix_mandel(R)
    return M @ C_local @ M.T


# ---------------------------------------------------------------------------
# Plasticity
# ---------------------------------------------------------------------------


@dataclass
class MatrixMCParams:
    c: float        # cohesion
    phi_deg: float  # friction angle


@dataclass
class CWFSParams:
    """Smooth CWFS (Cohesion Weakening Friction Strengthening) parameters.

    Plastic shear strain `gamma_ps` drives a single scalar progress variable
    s(gamma) = 0.5*(1 + tanh(k*(gamma/gamma_c - 0.5))), with s(0) ~ 0 and
    s(>> gamma_c) ~ 1.  Cohesion decays from peak to residual; friction grows
    from initial to peak over the same scale.
    """
    c_peak: float
    c_res: float
    gamma_c: float
    phi_ini_deg: float
    phi_peak_deg: float
    gamma_phi: float
    k_sharpness: float = 6.0  # steepness of the tanh transition

    def cohesion(self, gamma_ps: float) -> float:
        x = gamma_ps / max(self.gamma_c, 1e-12)
        s = 0.5 * (1.0 + np.tanh(self.k_sharpness * (x - 0.5)))
        return self.c_peak - (self.c_peak - self.c_res) * s

    def friction_deg(self, gamma_ps: float) -> float:
        x = gamma_ps / max(self.gamma_phi, 1e-12)
        s = 0.5 * (1.0 + np.tanh(self.k_sharpness * (x - 0.5)))
        return self.phi_ini_deg + (self.phi_peak_deg - self.phi_ini_deg) * s

    def as_static_MC(self, gamma_ps: float) -> "MatrixMCParams":
        return MatrixMCParams(c=self.cohesion(gamma_ps),
                              phi_deg=self.friction_deg(gamma_ps))


@dataclass
class JointParams:
    c_j: float       # joint cohesion
    phi_j_deg: float  # joint friction angle
    n_hat: np.ndarray  # unit normal to joint plane, shape (3,)


@dataclass
class FabricParams:
    """Microstructure-tensor scaling of the matrix cohesion (form II).

    Pietruszczak-Mroz line with one deviatoric fabric parameter and the n=1
    higher-order term: the deviatoric microstructure tensor is
    diag(-2d, d, d) in the material frame (bedding normal first), its
    projection on the generalized loading vector reduces to
    zeta = d * (1 - 3 |sigma n|^2 / |sigma|_F^2), and the cohesion scale is
    g = 1 + zeta + q_fab * zeta^2 (floored at G_FLOOR). d_fab = 0 gives
    g = 1 exactly, i.e. the frozen form I bit for bit.
    """
    d_fab: float   # deviatoric fabric parameter (dimensionless)
    q_fab: float   # second-order expansion coefficient (dimensionless)


G_FLOOR = 1e-6  # cohesion-scale floor; search boxes must keep g well above it


def fabric_cohesion_scale(sigma_m: np.ndarray, n_hat: np.ndarray,
                          fp: FabricParams) -> float:
    """Cohesion scale g(sigma, n_hat) of the form-II fabric expansion.

    |sigma|_F^2 is the Mandel self-product; at a hydrostatic state the
    projection gives zeta = 0 and g = 1 for every d_fab.
    """
    T = mandel_to_tensor(sigma_m)
    t = T @ n_hat
    den = float(sigma_m @ sigma_m)
    l1sq = float(t @ t) / max(den, 1e-30)
    zeta = fp.d_fab * (1.0 - 3.0 * l1sq)
    g = 1.0 + zeta + fp.q_fab * zeta * zeta
    return max(g, G_FLOOR)


def mohr_coulomb_return(sigma_tr_m: np.ndarray, mp: MatrixMCParams) -> np.ndarray:
    """Project trial stress onto matrix MC surface in principal space.

    Preserves hydrostatic component; scales principal deviator by alpha such
    that (s1 - s3) - (s1+s3) sin phi - 2 c cos phi = 0.
    Falls back to the apex (isotropic) point when the surface cannot be
    reached by scaling alone (tensile/low-confinement regime).
    """
    T_tr = mandel_to_tensor(sigma_tr_m)
    w, V = np.linalg.eigh(T_tr)
    order = np.argsort(-w)  # descending
    w = w[order]
    V = V[:, order]
    s1, s2, s3 = w
    phi = np.radians(mp.phi_deg)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    F = (s1 - s3) - (s1 + s3) * sin_phi - 2.0 * mp.c * cos_phi
    if F <= 0.0:
        return sigma_tr_m

    p_mean = (s1 + s2 + s3) / 3.0
    d = np.array([s1 - p_mean, s2 - p_mean, s3 - p_mean])
    # Target principal stresses satisfy the yield exactly while sharing p_mean.
    # Write s1 = p_mean + alpha d1, s3 = p_mean + alpha d3 and solve F=0.
    A = (d[0] - d[2]) - (d[0] + d[2]) * sin_phi
    B = 2.0 * p_mean * sin_phi + 2.0 * mp.c * cos_phi
    if abs(A) < 1e-12:
        # Apex: drop to isotropic pressure that just satisfies tensile cut-off
        sigma_apex = mp.c / max(np.tan(phi), 1e-6)
        s_new = np.full(3, min(p_mean, sigma_apex))
    else:
        alpha = B / A
        alpha = max(0.0, min(1.0, alpha))
        s_new = p_mean + alpha * d

    T_new = V @ np.diag(s_new) @ V.T
    return tensor_to_mandel(T_new)


def joint_return(sigma_tr_m: np.ndarray, jp: JointParams) -> np.ndarray:
    """Return mapping for ubiquitous-joint weakness plane.

    Decomposes traction T = sigma n_hat into normal sigma_n and shear tau.
    If F_j = tau - c_j - sigma_n tan(phi_j) > 0, shear magnitude is reduced to
    put the state on the joint surface while keeping sigma_n and the in-plane
    stresses unchanged.
    """
    T = mandel_to_tensor(sigma_tr_m)
    n = jp.n_hat / np.linalg.norm(jp.n_hat)
    traction = T @ n
    sigma_n = float(traction @ n)
    shear_vec = traction - sigma_n * n
    tau = float(np.linalg.norm(shear_vec))
    phi_j = np.radians(jp.phi_j_deg)
    tau_allow = jp.c_j + sigma_n * np.tan(phi_j)
    tau_allow = max(0.0, tau_allow)
    if tau <= tau_allow:
        return sigma_tr_m
    # scale shear to the allowable value
    if tau < 1e-12:
        return sigma_tr_m
    factor = tau_allow / tau
    shear_new = factor * shear_vec
    traction_new = sigma_n * n + shear_new
    # dT is tangential. The sum of the two dyads changes traction by dT
    # while preserving normal traction and the in-plane stress block.
    dT = traction_new - traction
    outer = np.outer(dT, n) + np.outer(n, dT)
    T_new = T + outer
    return tensor_to_mandel(T_new)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


@dataclass
class ESUBIParams:
    elastic: TIElasticParams
    matrix: MatrixMCParams
    joint: JointParams
    cwfs: CWFSParams | None = None  # if provided, matrix softens with gamma_ps
    fabric: FabricParams | None = None  # if provided, matrix cohesion is fabric-scaled (form II)


def bedding_normal(theta_rad: float) -> np.ndarray:
    """Bedding-plane normal when the TI local frame (z local = bedding normal)
    is rotated about the global y axis by theta_rad.  theta=0 gives n=(0,0,1).
    """
    R = rotation_matrix_Ry(theta_rad)
    return R @ np.array([0.0, 0.0, 1.0])


def equivalent_plastic_shear_increment(deps_p_mandel: np.ndarray) -> float:
    """Max minus min principal plastic strain rate, standard CWFS driver.

    Input: plastic strain increment in Mandel notation (6-vector).
    """
    T = mandel_to_tensor(deps_p_mandel)
    w = np.linalg.eigvalsh(T)
    return float(w.max() - w.min())


def run_path(
    params: ESUBIParams,
    theta_rad: float,
    strain_increments: np.ndarray,
    sigma_init: np.ndarray | None = None,
    return_history: bool = False,
) -> np.ndarray:
    """Integrate a strain-controlled path.

    Parameters
    ----------
    params : ESUBIParams
        Elastic + plastic parameters.  The joint normal in `params.joint.n_hat`
        is overridden by the bedding normal derived from `theta_rad`.
        If ``params.cwfs`` is provided, the matrix cohesion / friction evolve
        with the accumulated equivalent plastic shear strain ``gamma_ps``.
    theta_rad : float
        Bedding inclination (rotation about global y axis).  theta=0 -> horizontal.
    strain_increments : np.ndarray
        Array of shape (N, 6) in Mandel notation (compression positive).
    sigma_init : np.ndarray | None
        Initial stress in Mandel notation (e.g. confining pressure).
    return_history : bool
        If True, also returns ``gamma_ps`` history.

    Returns
    -------
    np.ndarray of shape (N+1, 6) with the stress path in Mandel notation.
    If ``return_history`` is True, returns ``(stress_path, gamma_history)``.
    """
    C = ti_stiffness_global(params.elastic, theta_rad)
    C_inv = np.linalg.inv(C)
    n_hat = bedding_normal(theta_rad)
    jp = JointParams(
        c_j=params.joint.c_j,
        phi_j_deg=params.joint.phi_j_deg,
        n_hat=n_hat,
    )

    sigma = np.zeros(6) if sigma_init is None else np.array(sigma_init, dtype=float)
    gamma_ps = 0.0
    path = [sigma.copy()]
    gamma_history = [gamma_ps]
    for deps in strain_increments:
        mp = (params.cwfs.as_static_MC(gamma_ps)
              if params.cwfs is not None else params.matrix)
        sigma_tr = sigma + C @ deps
        if params.fabric is not None:
            g = fabric_cohesion_scale(sigma_tr, n_hat, params.fabric)
            mp = MatrixMCParams(c=mp.c * g, phi_deg=mp.phi_deg)
        sigma_post_matrix = mohr_coulomb_return(sigma_tr, mp)
        sigma_post_joint = joint_return(sigma_post_matrix, jp)
        sigma_new = sigma_post_joint
        # plastic strain increment: total - elastic
        d_sigma = sigma_new - sigma
        deps_p = deps - C_inv @ d_sigma
        gamma_ps += max(0.0, equivalent_plastic_shear_increment(deps_p))
        sigma = sigma_new
        path.append(sigma.copy())
        gamma_history.append(gamma_ps)
    arr = np.asarray(path)
    if return_history:
        return arr, np.asarray(gamma_history)
    return arr


# ---------------------------------------------------------------------------
# Convenience: canonical triaxial compression path
# ---------------------------------------------------------------------------


def triaxial_compression_increments(
    n_steps: int,
    axial_strain_total: float,
    confining_pressure: float,
    C_global: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct elastic CTC increments with radial pressure and zero shear stress.

    During elastic loading, sigma_22 = sigma_33 = confining_pressure and all
    three specimen shear stresses remain zero while eps_11 increases linearly.
    Solve for all five remaining Mandel strain increments. For paths with
    plastic returns, ``run_triaxial_path`` enforces these stress targets after
    the constitutive update at every step.

    Returns
    -------
    sigma_init : np.ndarray (6,)
        Hydrostatic initial stress at `confining_pressure`.
    strain_increments : np.ndarray (n_steps, 6)
        Mandel strain increments for the path.
    """
    sigma_init = np.array([confining_pressure, confining_pressure, confining_pressure,
                           0.0, 0.0, 0.0])
    d_eps11 = axial_strain_total / n_steps
    # Enforce zero increments in the two radial and three shear stresses.
    A = C_global[1:, 1:]
    b = -C_global[1:, 0] * d_eps11
    deps = np.zeros(6)
    deps[0] = d_eps11
    deps[1:] = np.linalg.solve(A, b)
    increments = np.tile(deps, (n_steps, 1))
    return sigma_init, increments


def run_triaxial_path(
    params: ESUBIParams,
    theta_rad: float,
    n_steps: int,
    axial_strain_total: float,
    confining_pressure: float,
    n_newton: int = LATERAL_MAX_EVALUATIONS,
    lateral_tol: float = LATERAL_TOL_MPA,
    lateral_rtol: float = LATERAL_INCREMENT_RTOL,
) -> dict:
    """Integrate a CTC path with radial pressure and zero specimen shear stress.

    At each step, prescribe ``d_eps_11`` and solve the other five Mandel strain
    increments so that the returned radial stresses equal the confining
    pressure and all three specimen shear stresses are zero. The corresponding
    5x5 elastic block remains the fixed Jacobian. A positive residual secant
    sets the Aitken relaxation factor, accelerating slow contractions and
    damping oscillation at yield switches.
    Accept only a stress/strain pair evaluated at the same iterate; raise
    LateralConvergenceError if the evaluation budget does not meet tolerance.

    Returns a dict with ``eps``, ``sig``, ``gamma_ps`` history (each shape
    ``(n_steps+1, ...)``). ``eps`` includes the solved shear strains. The
    existing ``lateral_*`` diagnostics and tolerance arguments apply to all
    five controlled stress components in Mandel notation.
    """
    if n_steps < 1 or n_newton < 1 or lateral_tol <= 0 or lateral_rtol <= 0:
        raise ValueError("Positive step count, evaluation budget and tolerances required")
    C = ti_stiffness_global(params.elastic, theta_rad)
    C_inv = np.linalg.inv(C)
    A_free = C[1:, 1:]  # radial/shear sub-block, used as the fixed Jacobian
    b_axial = C[1:, 0]
    stress_target = np.array([confining_pressure, confining_pressure, 0, 0, 0])
    n_hat = bedding_normal(theta_rad)
    jp = JointParams(
        c_j=params.joint.c_j,
        phi_j_deg=params.joint.phi_j_deg,
        n_hat=n_hat,
    )

    sigma = np.array(
        [confining_pressure, confining_pressure, confining_pressure, 0, 0, 0],
        dtype=float,
    )
    eps = np.zeros(6)
    gamma_ps = 0.0

    d_eps11 = axial_strain_total / n_steps

    eps_hist = [eps.copy()]
    sig_hist = [sigma.copy()]
    gamma_hist = [gamma_ps]
    res_hist, tol_hist, eval_hist = [], [], []

    for _ in range(n_steps):
        # The elastic guess keeps both radial stresses and all shears fixed.
        d_eps = np.zeros(6)
        d_eps[0] = d_eps11
        d_eps[1:] = np.linalg.solve(A_free, -b_axial * d_eps11)
        tolerance = min(lateral_tol, lateral_rtol * np.max(np.abs(C @ d_eps)))

        def evaluate(increment):
            mp = (params.cwfs.as_static_MC(gamma_ps)
                  if params.cwfs is not None else params.matrix)
            sigma_tr = sigma + C @ increment
            if params.fabric is not None:
                g = fabric_cohesion_scale(sigma_tr, n_hat, params.fabric)
                mp = MatrixMCParams(c=mp.c * g, phi_deg=mp.phi_deg)
            sigma_post = mohr_coulomb_return(sigma_tr, mp)
            sigma_post = joint_return(sigma_post, jp)
            return sigma_post, sigma_post[1:] - stress_target

        sigma_post, res = evaluate(d_eps)
        relaxation = 1.0
        for _it in range(n_newton):
            residual = float(np.max(np.abs(res)))
            if np.isfinite(residual) and residual <= tolerance:
                break
            if _it == n_newton - 1 or not np.isfinite(residual):
                raise LateralConvergenceError(
                    f"Step {len(res_hist) + 1}: residual {residual:.6g} MPa "
                    f"exceeds {tolerance:.6g} after {_it + 1} evaluations")
            # Correct all five unknown increments using the elastic Jacobian.
            trial_eps = d_eps.copy()
            trial_eps[1:] -= relaxation*np.linalg.solve(A_free, res)
            trial_sigma, trial_res = evaluate(trial_eps)
            difference = trial_res-res
            if res @ difference < 0.:
                relaxation *= -(res @ difference)/(difference @ difference)
            d_eps, sigma_post, res = trial_eps, trial_sigma, trial_res

        # Finalise step
        sigma_new = sigma_post
        d_sigma = sigma_new - sigma
        deps_p = d_eps - C_inv @ d_sigma
        gamma_ps += max(0.0, equivalent_plastic_shear_increment(deps_p))
        sigma = sigma_new
        eps = eps + d_eps
        eps_hist.append(eps.copy())
        sig_hist.append(sigma.copy())
        gamma_hist.append(gamma_ps)
        res_hist.append(residual)
        tol_hist.append(tolerance)
        eval_hist.append(_it + 1)

    return {
        "eps": np.asarray(eps_hist),
        "sig": np.asarray(sig_hist),
        "gamma_ps": np.asarray(gamma_hist),
        "res_end": np.asarray(res_hist),
        "lateral_tolerance": np.asarray(tol_hist),
        "lateral_evaluations": np.asarray(eval_hist),
    }


def default_params() -> ESUBIParams:
    """Canonical layered-mudstone set used for Phase-1 data generation."""
    return ESUBIParams(
        elastic=TIElasticParams(E1=30_000.0, E3=15_000.0, nu12=0.20, nu13=0.30, G13=8_000.0),
        matrix=MatrixMCParams(c=15.0, phi_deg=35.0),
        joint=JointParams(c_j=5.0, phi_j_deg=25.0, n_hat=np.array([0.0, 0.0, 1.0])),
    )


def default_params_cwfs() -> ESUBIParams:
    """Phase-2 parameter set: same elastic/joint as Phase-1 plus CWFS softening.

    CWFS tuned so peak behaves like Phase-1, residual cohesion ~1/3 of peak,
    friction rises from 30 to 40 deg over ~1% plastic shear strain.
    """
    cwfs = CWFSParams(
        c_peak=15.0, c_res=5.0, gamma_c=0.005,
        phi_ini_deg=30.0, phi_peak_deg=40.0, gamma_phi=0.008,
    )
    return ESUBIParams(
        elastic=TIElasticParams(E1=30_000.0, E3=15_000.0, nu12=0.20, nu13=0.30, G13=8_000.0),
        matrix=MatrixMCParams(c=cwfs.c_peak, phi_deg=cwfs.phi_ini_deg),  # unused seed
        joint=JointParams(c_j=5.0, phi_j_deg=25.0, n_hat=np.array([0.0, 0.0, 1.0])),
        cwfs=cwfs,
    )


def opalinus_params() -> ESUBIParams:
    """Candidate-sites Opalinus Clay, anisotropic MC with CWFS softening.

    Source: Nordas, Brauchart, Anthi, Anagnostou (2024). Rock Mech. Rock Eng.
      DOI 10.1007/s00603-023-03509-7 - 73 CD/CU triaxial tests, Method A.

    Notes
    -----
    * Uses ``Eo`` and ``Gop`` at confining pressure 0 (linear in sigma_r0 is
      not modelled here because our ESUBI driver uses constant stiffness).
    * For confining pressure dependency we pick the 5 MPa reference (i.e.
      Eo ~ 3000 MPa, Gop ~ 1365 MPa); user can override.
    * Friction strengthening is absent in Nordas et al.; we keep peak = peak
      friction for matrix, only cohesion softens.  Residual cohesion 1 MPa
      (matrix) and 1 MPa (bedding).
    """
    sigma_r0_ref = 5.0
    Eo = 2100.0 + 180.0 * sigma_r0_ref       # = 3000 MPa
    Ep = 3.0 * Eo                             # n = 3, so Ep = 9000 MPa
    Gop = 955.0 + 82.0 * sigma_r0_ref         # = 1365 MPa
    nu_pp = 0.15
    nu_op = 0.10
    cwfs = CWFSParams(
        c_peak=5.0, c_res=1.0, gamma_c=0.008,
        phi_ini_deg=33.0, phi_peak_deg=33.0,  # no friction strengthening in Nordas
        gamma_phi=1.0,
    )
    return ESUBIParams(
        elastic=TIElasticParams(
            E1=Ep,     # in-plane (parallel to bedding)
            E3=Eo,     # out-of-plane (perpendicular to bedding)
            nu12=nu_pp, nu13=nu_op, G13=Gop,
        ),
        matrix=MatrixMCParams(c=cwfs.c_peak, phi_deg=cwfs.phi_ini_deg),
        joint=JointParams(c_j=3.0, phi_j_deg=29.0, n_hat=np.array([0.0, 0.0, 1.0])),
        cwfs=cwfs,
    )
