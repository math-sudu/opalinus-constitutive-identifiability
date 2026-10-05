"""Energy closure for interacting matrix cracks and finite weak planes.

This is the constitutive research prototype in paper/DERIVATION_PACKAGE.md.
It is independent of the calibrated Nordas mirror. Stress/energy units are MPa;
strains and crack densities are dimensionless. Tensor vectors use Mandel order
(11, 22, 33, sqrt(2)*23, sqrt(2)*13, sqrt(2)*12).
"""

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.optimize import brentq


I6 = np.eye(6)
VOLUME = np.array([1., 1., 1., 0., 0., 0.])
J = np.outer(VOLUME, VOLUME) / 3
KDEV = I6 - J


def mandel(tensor):
    return np.array([tensor[0, 0], tensor[1, 1], tensor[2, 2],
                     np.sqrt(2) * tensor[1, 2],
                     np.sqrt(2) * tensor[0, 2],
                     np.sqrt(2) * tensor[0, 1]])


def plane_basis(normal):
    """Orthonormal normal/shear strain modes, not a six-dimensional inverse."""
    n = np.asarray(normal, dtype=float)
    length = np.linalg.norm(n)
    if length == 0:
        raise ValueError("The weak-plane normal must be nonzero.")
    n = n / length
    axis = np.eye(3)[np.argmin(np.abs(n))]
    t1 = np.cross(n, axis)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    return np.column_stack([
        mandel(np.outer(n, n)),
        mandel((np.outer(n, t1) + np.outer(t1, n)) / np.sqrt(2)),
        mandel((np.outer(n, t2) + np.outer(t2, n)) / np.sqrt(2)),
    ])


@dataclass(frozen=True)
class EnergyClosure:
    E0: float
    nu0: float
    bulk_slope: float
    shear_slope: float

    def __post_init__(self):
        if not (self.E0 > 0 and -1 < self.nu0 < .5
                and self.bulk_slope > 0 and self.shear_slope > 0):
            raise ValueError("Positive moduli/compliance slopes and -1 < nu0 < .5 are required.")

    @property
    def bulk(self):
        return self.E0 / (3 * (1 - 2 * self.nu0))

    @property
    def shear(self):
        return self.E0 / (2 * (1 + self.nu0))

    @property
    def C0(self):
        return 3 * self.bulk * J + 2 * self.shear * KDEV

    @property
    def Sc(self):
        return (self.bulk_slope / (3 * self.bulk) * J
                + self.shear_slope / (2 * self.shear) * KDEV)

    @property
    def Hc(self):
        return np.linalg.inv(self.Sc)

    def plane_compliance(self, d):
        """Return (s_n,s_t,s_t) and their derivatives with respect to d.

        s_n = 16(1-nu^2)/(3E), s_t = s_n/(2-nu), evaluated in
        the CURRENT cracked matrix. The x=1/K, y=1/G expressions
        expose positivity of both derivatives without differencing.
        """
        x = (1 + self.bulk_slope * d) / self.bulk
        y = (1 + self.shear_slope * d) / self.shear
        xp = self.bulk_slope / self.bulk
        yp = self.shear_slope / self.shear
        sn = 4 * y / 3 + 4 * x * y / (3 * y + x)
        st = 8 * y / 9 + 16 * x * y / (9 * (3 * y + 2 * x))
        snp = 4 * yp / 3 + 4 * (3 * y*y * xp + x*x * yp) / (3 * y + x)**2
        stp = 8 * yp / 9 + 16 * (3 * y*y * xp + 2 * x*x * yp) / (9 * (3 * y + 2 * x)**2)
        return np.array([sn, st, st]), np.array([snp, stp, stp])

    def energy(self, strain, matrix_strain, plane_modes, d, D, normal):
        B = plane_basis(normal)
        e = np.asarray(strain) - matrix_strain - B @ plane_modes
        s, _ = self.plane_compliance(d)
        return (.5 * e @ self.C0 @ e
                + .5 / d * matrix_strain @ self.Hc @ matrix_strain
                + .5 / D * np.sum(plane_modes**2 / s))

    def forces(self, strain, matrix_strain, plane_modes, d, D, normal):
        B = plane_basis(normal)
        stress = self.C0 @ (np.asarray(strain) - matrix_strain - B @ plane_modes)
        s, sp = self.plane_compliance(d)
        yc = .5 / d**2 * matrix_strain @ self.Hc @ matrix_strain
        ycross = .5 / D * np.sum(plane_modes**2 * sp / s**2)
        return {
            "stress": stress,
            "matrix_force": stress - self.Hc @ matrix_strain / d,
            "plane_force": B.T @ stress - plane_modes / (D * s),
            "matrix_self_release": yc,
            "plane_to_matrix_release": ycross,
            "matrix_release": yc + ycross,
            "plane_release": .5 / D**2 * np.sum(plane_modes**2 / s),
        }

    def condensed_compliance(self, d, D, normal):
        """Frictionless relaxed compliance; not a locked unloading tangent."""
        B = plane_basis(normal)
        s, _ = self.plane_compliance(d)
        return np.linalg.inv(self.C0) + d * self.Sc + D * (B * s) @ B.T


@dataclass(frozen=True)
class MonotonicShear:
    """Constant-Poisson, two-mode slice with fixed friction resistances.

    m and j increase monotonically; d is irreversible and D is fixed.
    This slice has no pore fluid, compaction or weak-plane growth law.
    Setting evolving_localization=False selects a DIFFERENT energy with
    h_t fixed at d0. It does not delete a force from the evolving energy.
    """

    elastic: EnergyClosure
    d0: float
    D: float
    overlap: float
    matrix_friction: float
    plane_friction: float
    resistance: float
    evolving_localization: bool = True

    def __post_init__(self):
        if self.elastic.bulk_slope != self.elastic.shear_slope:
            raise ValueError("The convex shear slice requires constant Poisson ratio.")
        if not (self.d0 > 0 and self.D > 0 and 0 <= self.overlap < 1
                and self.matrix_friction > 0 and self.plane_friction > 0
                and self.resistance > 0):
            raise ValueError("Positive densities/resistances and 0 <= overlap < 1 are required.")

    @property
    def shear_stiffness(self):
        return 2 * self.elastic.shear

    @property
    def matrix_storage(self):
        return self.shear_stiffness / self.elastic.shear_slope

    def plane_storage(self, d):
        s, sp = self.elastic.plane_compliance(d if self.evolving_localization else self.d0)
        h = 1 / s[1]
        hp = -sp[1] / s[1]**2 if self.evolving_localization else 0.
        return h, hp

    def energy(self, strain, state):
        m, j, d = state
        h, _ = self.plane_storage(d)
        elastic = self.shear_stiffness / 2 * ((strain - m)**2 + j*j
                                             - 2 * self.overlap * (strain - m) * j)
        return elastic + self.matrix_storage * m*m / (2*d) + h*j*j / (2*self.D)

    def response(self, strain, state):
        m, j, d = state
        k, rho = self.shear_stiffness, self.overlap
        h, hp = self.plane_storage(d)
        stress = k * (strain - m - rho*j)
        yc = self.matrix_storage * m*m / (2*d*d)
        ycross = -hp * j*j / (2*self.D)
        return {"stress": stress, "matrix_force": stress - self.matrix_storage*m/d,
                "plane_force": k*(rho*(strain-m)-j) - h*j/self.D,
                "matrix_self_release": yc, "plane_to_matrix_release": ycross,
                "matrix_release": yc + ycross, "energy": self.energy(strain, state)}

    def _slips(self, strain, d, previous):
        """Exact two-dimensional bound-constrained quadratic minimizer."""
        k, rho = self.shear_stiffness, self.overlap
        h, _ = self.plane_storage(d)
        A = np.array([[k + self.matrix_storage/d, k*rho], [k*rho, k + h/self.D]])
        b = np.array([k*strain-self.matrix_friction, k*rho*strain-self.plane_friction])
        lower = np.asarray(previous[:2])
        candidates = []
        for flags in product((False, True), repeat=2):
            free = np.flatnonzero(flags)
            fixed = np.flatnonzero(np.logical_not(flags))
            x = lower.copy()
            if len(free):
                x[free] = np.linalg.solve(A[np.ix_(free, free)], b[free] - A[np.ix_(free, fixed)] @ x[fixed])
            if np.all(x >= lower):
                candidates.append((.5 * x @ A @ x - b @ x, x))
        return min(candidates, key=lambda pair: pair[0])[1]

    def step(self, strain, previous):
        """Minimize energy + friction work + R*(d-d_previous).

        With equal compliance slopes the potential is jointly convex:
        m^2/d and j^2/(1+a*d) are quadratic perspectives. After the
        exact slip solve, the envelope derivative is R-Y. Its unique
        root, when damage activates, supplies the damage update.
        """
        lower = previous[2]

        def derivative(d):
            m, j = self._slips(strain, d, previous)
            return self.resistance - self.response(strain, (m, j, d))["matrix_release"]

        if derivative(lower) >= 0:
            d = lower
        else:
            upper = max(2*lower, 1.)
            while derivative(upper) < 0:
                upper *= 2
            d = brentq(derivative, lower, upper, xtol=1e-12)
        m, j = self._slips(strain, d, previous)
        return np.array([m, j, d])

    def plane_driven_onset(self):
        """Analytical onset on the m=0 branch, provided matrix friction is inactive."""
        h, hp = self.plane_storage(self.d0)
        if hp == 0 or self.overlap == 0:
            raise ValueError("No plane-driven onset exists for this slice.")
        j = np.sqrt(2*self.D*self.resistance / (-hp))
        strain = (self.plane_friction + (self.shear_stiffness+h/self.D)*j) / (self.shear_stiffness*self.overlap)
        matrix_force = self.shear_stiffness*(strain-self.overlap*j)
        if matrix_force >= self.matrix_friction:
            raise ValueError("Matrix friction activates before the plane-only damage onset.")
        return {"strain": strain, "plane_mode": j, "matrix_force": matrix_force}
