"""TI porous extension of the weak-plane storage energy.

Compression is positive; fluid content is positive for added fluid. Mandel
order is (11,22,33,sqrt(2)*23,sqrt(2)*13,sqrt(2)*12). The exact mixed-control
update advances normal contact and volumetric compaction at fixed matrix
sliding strain, tangential plane modes and crack density.
"""

from dataclasses import dataclass
from itertools import product

import numpy as np

from .energetic_coupling import VOLUME, mandel, plane_basis


def mandel_rotation(rotation):
    tensors = [np.diag(row) for row in np.eye(3)]
    for i, j in ((1, 2), (0, 2), (0, 1)):
        tensor = np.zeros((3, 3))
        tensor[i, j] = tensor[j, i] = 1 / np.sqrt(2)
        tensors.append(tensor)
    return np.column_stack([mandel(rotation @ t @ rotation.T) for t in tensors])


@dataclass(frozen=True)
class TIElasticity:
    """Reciprocal engineering constants; nu_pn = -eps_normal/eps_parallel
    under uniaxial stress parallel to bedding. Thus nu_np/E_normal equals
    nu_pn/E_parallel. Moduli are MPa.
    """

    E_parallel: float
    E_normal: float
    nu_parallel: float
    nu_pn: float
    G_normal: float

    def at_confinement(self, initial_pressure, reference_pressure, pressure_fraction):
        """Fix the reference stiffness for a prepared specimen before shearing.

        The moduli vary linearly with initial effective confinement; their
        anisotropy and Poisson ratios stay fixed. pressure_fraction is the
        pressure-dependent fraction of stiffness at reference_pressure.
        This is a family of reference energies, not a current-stress tangent.
        """
        if initial_pressure < 0 or reference_pressure <= 0 or not 0 <= pressure_fraction <= 1:
            raise ValueError("Preparation pressures and stiffness fraction are inadmissible.")
        scale = 1+pressure_fraction*(initial_pressure/reference_pressure-1)
        if scale <= 0:
            raise ValueError("Prepared stiffness must be positive.")
        return TIElasticity(self.E_parallel*scale, self.E_normal*scale,
                            self.nu_parallel, self.nu_pn, self.G_normal*scale)

    def local_compliance(self):
        ep, en, nu, cross, g = (self.E_parallel, self.E_normal,
                               self.nu_parallel, self.nu_pn, self.G_normal)
        if min(ep, en, g) <= 0:
            raise ValueError("TI moduli must be positive.")
        S = np.diag([1/ep, 1/ep, 1/en, 1/(2*g), 1/(2*g), (1+nu)/ep])
        S[0, 1] = S[1, 0] = -nu/ep
        S[0, 2] = S[2, 0] = S[1, 2] = S[2, 1] = -cross/ep
        try:
            np.linalg.cholesky(S)
        except np.linalg.LinAlgError as exc:
            raise ValueError("TI engineering compliance must be positive definite.") from exc
        return S

    def compliance(self, normal):
        n = np.asarray(normal, dtype=float)
        if np.linalg.norm(n) == 0:
            raise ValueError("The bedding normal must be nonzero.")
        n = n / np.linalg.norm(n)
        t1 = np.cross(n, np.eye(3)[np.argmin(np.abs(n))])
        t1 /= np.linalg.norm(t1)
        rotation = np.column_stack((t1, np.cross(n, t1), n))
        Q = mandel_rotation(rotation)
        return Q @ self.local_compliance() @ Q.T


class PorousEnergyClosure:
    """One energy for TI elasticity, stored slip, contact and compaction.

    Plane compliance is a projected-host closure L B.T (S0+d*Sc) B L,
    with Sc=matrix_slope*S0. It is not an exact anisotropic penny-crack
    homogenization. The two positive plane factors define L squared.
    Grain elasticity and constant reference porosity determine b and M.
    plane_reference_damage freezes Hp in the entire energy for the matched
    fixed-storage counterpart; it does not remove one force selectively.
    """

    def __init__(self, elastic, normal, *, grain_bulk, fluid_bulk, porosity,
                 matrix_slope, plane_normal_factor, plane_shear_factor,
                 compaction_onset, compaction_hardening, plane_reference_damage=None):
        if not (min(grain_bulk, fluid_bulk, matrix_slope, plane_normal_factor,
                    plane_shear_factor, compaction_onset, compaction_hardening) > 0
                and 0 < porosity < 1):
            raise ValueError("Positive moduli, factors and resistances, and 0<porosity<1 are required.")
        self.S0 = elastic.compliance(normal)
        self.C0 = np.linalg.inv(self.S0)
        self.Sc = matrix_slope * self.S0
        self.Hc = np.linalg.inv(self.Sc)
        self.B = plane_basis(normal)
        L = np.diag(np.sqrt([plane_normal_factor, plane_shear_factor, plane_shear_factor]))
        self.plane_S0 = L @ self.B.T @ self.S0 @ self.B @ L
        self.plane_Sprime = L @ self.B.T @ self.Sc @ self.B @ L
        if plane_reference_damage is not None and plane_reference_damage <= 0:
            raise ValueError("A fixed plane-storage reference must have positive damage density.")
        self.plane_reference_damage = plane_reference_damage
        self.biot = VOLUME - self.C0 @ VOLUME / (3*grain_bulk)
        storage = (self.biot @ VOLUME / 3 - porosity) / grain_bulk + porosity / fluid_bulk
        if storage <= 0:
            raise ValueError("The chosen skeleton and constituents give nonpositive fluid storage.")
        self.M = 1 / storage
        self.compaction_mode = VOLUME / 3
        self.compaction_onset = compaction_onset
        self.compaction_hardening = compaction_hardening

    def plane_storage(self, d):
        reference = self.plane_reference_damage
        H = np.linalg.inv(self.plane_S0 + (d if reference is None else reference)*self.plane_Sprime)
        return H, -H @ self.plane_Sprime @ H if reference is None else np.zeros((3, 3))

    def inelastic_strain(self, matrix_strain, plane_modes, compaction):
        return np.asarray(matrix_strain) + self.B @ plane_modes + compaction*self.compaction_mode

    def fluid_content(self, strain, matrix_strain, plane_modes, compaction, pressure):
        ei = self.inelastic_strain(matrix_strain, plane_modes, compaction)
        return pressure/self.M - self.biot @ (np.asarray(strain)-ei) - VOLUME @ ei

    def response(self, strain, matrix_strain, plane_modes, compaction, d, D, fluid_content):
        if d <= 0 or D <= 0:
            raise ValueError("Positive crack and plane densities are required.")
        cm, z = np.asarray(matrix_strain), np.asarray(plane_modes)
        ei = self.inelastic_strain(cm, z, compaction)
        ee = np.asarray(strain) - ei
        pressure = self.M * (fluid_content + self.biot @ ee + VOLUME @ ei)
        elastic_stress = self.C0 @ ee
        total_stress = elastic_stress + self.biot*pressure
        plastic_stress = total_stress - VOLUME*pressure
        H, Hp = self.plane_storage(d)
        yc = .5 * cm @ self.Hc @ cm / d**2
        ycross = -.5 * z @ Hp @ z / D
        energy = (.5 * ee @ self.C0 @ ee + .5 * cm @ self.Hc @ cm / d
                  + .5 * z @ H @ z / D
                  + .5 * self.compaction_hardening * compaction**2
                  + .5 * pressure**2 / self.M)
        return {
            "energy": energy, "pressure": pressure, "total_stress": total_stress,
            "elastic_stress": elastic_stress, "plastic_stress": plastic_stress,
            "matrix_force": plastic_stress - self.Hc @ cm / d,
            "plane_force": self.B.T @ plastic_stress - H @ z / D,
            "compaction_force": (self.compaction_mode @ plastic_stress
                                 - self.compaction_hardening*compaction),
            "matrix_self_release": yc, "plane_to_matrix_release": ycross,
            "matrix_release": yc+ycross, "plane_release": .5 * z @ H @ z / D**2,
            "elastic_volume": VOLUME @ ee,
            "inelastic_volume": VOLUME @ ei,
            "total_volume": VOLUME @ np.asarray(strain),
        }

    def normal_compaction_step(self, axial_strain, transverse_stress, *,
                               matrix_strain, tangential_modes, d, D,
                               previous_compaction=0., fluid_content=None,
                               pressure=None):
        """Exact contact/cap update under axial strain and five total tractions.

        Specify either fluid_content (undrained) or pressure (drained).
        Normal plane strain z0<=0 means opening; closure has z0=0.
        Compaction cannot decrease. Tangential slip, matrix sliding and
        damage are supplied states, not advanced by this subproblem.
        """
        if (fluid_content is None) == (pressure is None):
            raise ValueError("Specify exactly one fluid boundary condition.")
        if d <= 0 or D <= 0 or previous_compaction < 0:
            raise ValueError("Positive densities and nonnegative prior compaction are required.")
        cm, jt = np.asarray(matrix_strain), np.asarray(tangential_modes)
        offset = cm + self.B[:, 1:] @ jt
        # Unknown vector x=(six total strains, z0, compaction).
        A = np.column_stack((np.eye(6), -self.B[:, 0], -self.compaction_mode))
        t = np.r_[self.biot, (VOLUME-self.biot) @ self.B[:, 0],
                  (VOLUME-self.biot) @ self.compaction_mode]
        H, _ = self.plane_storage(d)
        quadratic = A.T @ self.C0 @ A
        linear = -A.T @ self.C0 @ offset
        quadratic[6, 6] += H[0, 0]/D
        linear[6] += H[0, 1:] @ jt / D
        quadratic[7, 7] += self.compaction_hardening
        linear[7] += self.compaction_onset
        if pressure is None:
            quadratic += self.M*np.outer(t, t)
            linear += self.M*t*(fluid_content+(VOLUME-self.biot) @ offset)
        else:
            # Legendre potential Psi-u*zeta, at prescribed u.
            linear += pressure*t
        linear[1:6] -= np.asarray(transverse_stress)

        candidates = []
        for opened, compacting in product((False, True), repeat=2):
            fixed = [0] + ([] if opened else [6]) + ([] if compacting else [7])
            free = [i for i in range(8) if i not in fixed]
            x = np.r_[axial_strain, np.zeros(6), previous_compaction]
            x[free] = np.linalg.solve(quadratic[np.ix_(free, free)],
                                     -linear[free]-quadratic[np.ix_(free, fixed)] @ x[fixed])
            if x[6] <= 0 and x[7] >= previous_compaction:
                candidates.append((.5*x @ quadratic @ x + linear @ x, x))
        x = min(candidates, key=lambda item: item[0])[1]
        z = np.r_[x[6], jt]
        zeta = (fluid_content if pressure is None else
                self.fluid_content(x[:6], cm, z, x[7], pressure))
        response = self.response(x[:6], cm, z, x[7], d, D, zeta)
        return {"strain": x[:6], "plane_modes": z, "compaction": x[7],
                "fluid_content": zeta, "response": response}
