"""Implicit matrix/plane friction and damage in the TI porous energy.

Compression is positive. Matrix flow is a TI quadratic deviatoric potential
with independent dilation; flat unbonded planes obey Coulomb contact. All
forces, pore pressure and damage release are evaluated at the same end state.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, least_squares

from .energetic_coupling import KDEV, VOLUME


@dataclass(frozen=True)
class FrictionDamageLaw:
    matrix_cohesion: float
    matrix_friction: float
    matrix_dilatancy: float
    parallel_weight: float
    shear_weight: float
    plane_friction: float
    damage_resistance: float
    damage_peak_density: float | None = None
    damage_residual_fraction: float = 0.
    damage_shape: float = 1.
    dilation_residual_fraction: float = 1.
    parallel_residual_weight: float | None = None
    dilation_density_ratio: float = 1.
    dilation_initial_density: float | None = None
    matrix_contraction: float = 0.
    friction_residual_fraction: float = 1.

    def __post_init__(self):
        if not (self.matrix_cohesion > 0 and self.damage_resistance > 0
                and min(self.parallel_weight, self.shear_weight) > 0
                and 0 <= self.matrix_dilatancy <= self.matrix_friction
                and self.plane_friction >= 0):
            raise ValueError("Positive strengths/weights and 0 <= dilation <= friction are required.")
        if (self.damage_peak_density is not None and self.damage_peak_density <= 0
                or not 0 <= self.damage_residual_fraction < 1 or self.damage_shape <= 0
                or not 0 <= self.dilation_residual_fraction <= 1
                or self.dilation_density_ratio <= 0
                or self.dilation_initial_density is not None and self.dilation_initial_density < 0
                or self.matrix_contraction < 0
                or self.matrix_contraction > 0 and (self.dilation_initial_density is None
                                                    or self.damage_peak_density is None)
                or not 0 <= self.friction_residual_fraction <= 1
                or self.friction_residual_fraction != 1 and self.damage_peak_density is None
                or self.parallel_residual_weight is not None and self.parallel_residual_weight <= 0):
            raise ValueError("Positive damage peak density/shape, 0 <= resistance fraction < 1, "
                             "positive flow scales/weights and 0 <= dilation fraction <= 1 are required. "
                             "Shear contraction needs a nonnegative amplitude and a density origin/scale; "
                             "friction evolution needs a peak density and a residual fraction in [0, 1].")

    def friction_scale(self, d):
        """Peak-to-residual friction using the existing resistance-peak density."""
        if self.damage_peak_density is None or d <= self.damage_peak_density:
            return 1., 0.
        transient = (1-self.friction_residual_fraction)*self.damage_peak_density/d
        return self.friction_residual_fraction+transient, -transient/d

    def friction(self, d):
        scale, derivative = self.friction_scale(d)
        return self.matrix_friction*scale, self.matrix_friction*derivative

    def parallel_metric_weight(self, d):
        """Relax the directional weight on the descending damage branch.

        The peak and residual metrics remain positive. Damage resistance and
        strength directionality have distinct roles; neither residual isotropy
        nor permanent peak anisotropy is imposed on the material fit.
        """
        if (self.parallel_residual_weight is None or self.damage_peak_density is None
                or d <= self.damage_peak_density):
            return self.parallel_weight, 0.
        difference = self.parallel_weight-self.parallel_residual_weight
        ratio = self.damage_peak_density/d
        return self.parallel_residual_weight+difference*ratio, -difference*ratio/d

    def resistance(self, d):
        """Positive resistance and derivative in the matrix-density coordinate.

        A missing peak density selects the constant-resistance limit. Otherwise
        the rational rise/fall generalizes Ren et al. (2025), Eq. (16): an
        unshifted density ratio, adjustable exponent and optional residual floor.
        """
        if self.damage_peak_density is None:
            return self.damage_resistance, 0.
        x = (d/self.damage_peak_density)**self.damage_shape
        r = self.damage_residual_fraction
        value = self.damage_resistance*(r+(1-r)*2*x/(1+x*x))
        derivative = (self.damage_resistance*(1-r)*2*self.damage_shape*x*(1-x*x)
                      / (d*(1+x*x)**2))
        return value, derivative

    def dilatancy(self, d):
        if self.damage_peak_density is None:
            return self.matrix_dilatancy, 0.
        scale = self.damage_peak_density*self.dilation_density_ratio
        r = self.dilation_residual_fraction
        friction_scale, friction_prime = self.friction_scale(d)
        amplitude = self.matrix_dilatancy*friction_scale
        amplitude_prime = self.matrix_dilatancy*friction_prime
        if self.dilation_initial_density is not None:
            # Distributed shear can compact before dilation mobilizes. Its
            # contraction term decays as the existing damage density grows.
            # beta <= friction(d) still gives nonnegative sliding work;
            # that condition does not require beta to be nonnegative.
            x = max(0., d-self.dilation_initial_density)/scale
            value = 4*x/(1+x)**2
            derivative = (4*(1-x)/(scale*(1+x)**3)
                          if d >= self.dilation_initial_density else 0.)
            contraction = self.matrix_contraction/(1+x)**2
            contraction_prime = (-2*self.matrix_contraction/(scale*(1+x)**3)
                                 if d >= self.dilation_initial_density else 0.)
            shape = r+(1-r)*value
            return (amplitude*shape-contraction,
                    amplitude_prime*shape+amplitude*(1-r)*derivative-contraction_prime)
        x = d/scale
        shape = r+(1-r)/(1+x)
        return (amplitude*shape,
                amplitude_prime*shape-amplitude*(1-r)/(scale*(1+x)**2))


def complementarity(a, b):
    """Fischer--Burmeister residual: a>=0, b>=0, a*b=0."""
    return np.hypot(a, b)-a-b


def complementarity_partials(a, b):
    radius = np.hypot(a, b)
    return (a/radius-1, b/radius-1) if radius > 0 else (-1., -1.)


class CoupledFrictionDamage:
    def __init__(self, energy, law, plane_density):
        if plane_density <= 0:
            raise ValueError("Plane density must be positive.")
        self.energy, self.law, self.D = energy, law, plane_density
        normal_mode = np.sqrt(1.5)*(energy.B[:, 0]-VOLUME/3)
        normal_projector = np.outer(normal_mode, normal_mode)
        shear_projector = energy.B[:, 1:] @ energy.B[:, 1:].T
        parallel_projector = KDEV-normal_projector-shear_projector
        self.parallel_metric = 1.5*parallel_projector
        # Each projector is orthogonal. Equal weights recover von Mises q.
        self.metric = 1.5*(normal_projector + law.parallel_weight*parallel_projector
                          + law.shear_weight*shear_projector)

    def matrix_metric(self, d):
        if d is None:
            return self.metric, np.zeros((6, 6))
        weight, derivative = self.law.parallel_metric_weight(d)
        return (self.metric+(weight-self.law.parallel_weight)*self.parallel_metric,
                derivative*self.parallel_metric)

    def matrix_surface(self, force, d=None):
        deviator = KDEV @ force
        metric, _ = self.matrix_metric(d)
        mapped = metric @ deviator
        q = np.sqrt(max(0., deviator @ mapped))
        mean = force @ VOLUME/3
        direction = mapped/q if q > 0 else np.zeros(6)
        beta = self.law.matrix_dilatancy if d is None else self.law.dilatancy(d)[0]
        if mean > 0:
            direction = direction-beta*VOLUME/3
        friction = self.law.matrix_friction if d is None else self.law.friction(d)[0]
        strength = self.law.matrix_cohesion+friction*max(0., mean)
        return q-strength, direction, q, mean

    def consolidated_state(self, effective_confinement, d, pressure=0.):
        """Drained isotropic consolidation, including any attained cap strain."""
        if effective_confinement < 0 or d <= 0:
            raise ValueError("Nonnegative confinement and positive matrix density are required.")
        m = self.energy
        total = (effective_confinement+pressure)*VOLUME
        kappa = max(0., (effective_confinement-m.compaction_onset)/m.compaction_hardening)
        strain = m.S0 @ (total-m.biot*pressure)+kappa*m.compaction_mode
        cm, z = np.zeros(6), np.zeros(3)
        zeta = m.fluid_content(strain, cm, z, kappa, pressure)
        return dict(strain=strain, matrix_strain=cm, plane_modes=z, compaction=kappa,
                    d=d, fluid_content=zeta,
                    response=m.response(strain, cm, z, kappa, d, self.D, zeta))

    def _mixed_kinematics(self, axial_strain, transverse_stress, pressure, fluid_content):
        """Eliminate the five free strains exactly, for any inelastic strain."""
        m = self.energy
        if pressure is None:
            stiffness = m.C0+m.M*np.outer(m.biot, m.biot)
            coupling = -m.C0+m.M*np.outer(m.biot, VOLUME-m.biot)
            offset = m.M*m.biot*fluid_content
        else:
            stiffness, coupling, offset = m.C0, -m.C0, m.biot*pressure
        base = np.r_[axial_strain, np.linalg.solve(
            stiffness[1:, 1:], np.asarray(transverse_stress)
            -stiffness[1:, 0]*axial_strain-offset[1:])]
        transform = np.vstack((np.zeros(6), np.linalg.solve(
            stiffness[1:, 1:], -coupling[1:, :])))
        return base, transform

    def step(self, axial_strain, transverse_stress, previous, *,
             pressure=None, fluid_content=None):
        """Backward-Euler mixed-control solve of all five active processes.

        Unknowns are matrix strain, both tangential slips, normal opening,
        compaction, log(d/d_previous), and the matrix friction multiplier. Normal
        contact, cap, damage and friction consistency use complementarity;
        total tractions and the fluid constraint are eliminated algebraically.
        Plane flow uses the equivalent proximal Coulomb equation, which remains
        defined at open contact where tangential force vanishes during slip.
        A nonzero constitutive residual raises rather than returning a state.
        """
        if (pressure is None) == (fluid_content is None):
            raise ValueError("Specify exactly one fluid boundary condition.")
        m, law = self.energy, self.law
        base, transform = self._mixed_kinematics(
            axial_strain, transverse_stress, pressure, fluid_content)
        cm0, z0, k0, d0 = (previous["matrix_strain"], previous["plane_modes"],
                           previous["compaction"], previous["d"])
        strain_scale = 1e-3
        stress_scale = np.linalg.norm(m.C0, ord=2)*strain_scale
        identity = np.eye(12)
        dcm = strain_scale*identity[:6]
        dz = strain_scale*identity[[8, 6, 7]]
        dk = strain_scale*identity[9]
        dei = dcm+m.B @ dz+np.outer(m.compaction_mode, dk)
        dee = (transform-np.eye(6)) @ dei
        du = (m.M*(m.biot @ dee+VOLUME @ dei) if pressure is None else np.zeros(12))
        dtau = m.C0 @ dee+np.outer(m.biot-VOLUME, du)

        def evaluate(x):
            cm = x[:6]*strain_scale
            z = np.r_[x[8], x[6:8]]*strain_scale
            kappa, d = x[9]*strain_scale, d0*np.exp(x[10])
            lm = x[11]*strain_scale
            ei = m.inelastic_strain(cm, z, kappa)
            strain = base+transform @ ei
            zeta = (fluid_content if pressure is None else
                    m.fluid_content(strain, cm, z, kappa, pressure))
            f = m.response(strain, cm, z, kappa, d, self.D, zeta)
            fm, flow, q, mean = self.matrix_surface(f["matrix_force"], d)
            tangent = f["plane_force"][1:]
            norm = np.linalg.norm(tangent)
            reaction = f["plane_force"][0]
            # Mandel traction has magnitude sqrt(2)*physical shear traction.
            friction = np.sqrt(2)*law.plane_friction*max(0., reaction)
            fp = norm-friction
            slip = (z[1:]-z0[1:])/strain_scale
            proximal = slip+tangent/stress_scale
            radius = np.linalg.norm(proximal)
            threshold = friction/stress_scale
            shrink = (1-threshold/radius)*proximal if radius > threshold else np.zeros(2)
            resistance, resistance_prime = law.resistance(d)
            residual = np.r_[
                (cm-cm0-lm*flow)/strain_scale,
                slip-shrink,
                complementarity(-x[8], reaction/stress_scale),
                complementarity((kappa-k0)/strain_scale,
                                (m.compaction_onset-f["compaction_force"])/stress_scale),
                complementarity(x[10], 1-f["matrix_release"]/resistance),
                complementarity(x[11], -fm/stress_scale),
            ]
            state = dict(strain=strain, matrix_strain=cm, plane_modes=z, compaction=kappa,
                         d=d, fluid_content=zeta, response=f,
                         matrix_multiplier=lm, plane_multiplier=np.linalg.norm(z[1:]-z0[1:]),
                         matrix_yield=fm, plane_yield=fp,
                         matrix_equivalent_force=q, matrix_mean_force=mean,
                         damage_resistance=resistance)

            dd = d*identity[10]
            H, Hp = m.plane_storage(d)
            Hpp = (2*H @ m.plane_Sprime @ H @ m.plane_Sprime @ H
                   if m.plane_reference_damage is None else np.zeros((3, 3)))
            dmatrix = dtau-m.Hc @ dcm/d+np.outer(m.Hc @ cm/d**2, dd)
            dplane = m.B.T @ dtau-H @ dz/self.D-np.outer(Hp @ z/self.D, dd)
            metric, metric_prime = self.matrix_metric(d)
            force = f["matrix_force"]
            mapped = metric @ force
            dmapped = metric @ dmatrix+np.outer(metric_prime @ force, dd)
            dq = ((mapped @ dmatrix+.5*(force @ metric_prime @ force)*dd)/q
                  if q > 0 else np.zeros(12))
            dflow = (dmapped/q-np.outer(mapped, dq)/q**2
                     if q > 0 else np.zeros((6, 12)))
            if mean > 0:
                dflow -= np.outer(VOLUME/3, law.dilatancy(d)[1]*dd)
            mu, mu_prime = law.friction(d)
            dfm = dq-mu*(VOLUME @ dmatrix)/3*(mean > 0)-mu_prime*max(0., mean)*dd
            dy = (cm @ m.Hc @ dcm/d**2-2*f["matrix_self_release"]*dd/d
                  -z @ Hp @ dz/self.D-.5*(z @ Hpp @ z)*dd/self.D)
            jacobian = np.zeros((12, 12))
            jacobian[:6] = (dcm-lm*dflow)/strain_scale-np.outer(flow, identity[11])
            dslip = dz[1:]/strain_scale
            jacobian[6:8] = dslip
            if radius > threshold:
                dproximal = dslip+dplane[1:]/stress_scale
                dthreshold = np.sqrt(2)*law.plane_friction*dplane[0]*(reaction > 0)/stress_scale
                shrink_jacobian = ((1-threshold/radius)*np.eye(2)
                                  +threshold*np.outer(proximal, proximal)/radius**3)
                jacobian[6:8] -= (shrink_jacobian @ dproximal
                                   -np.outer(proximal/radius, dthreshold))
            pairs = [
                (-x[8], reaction/stress_scale, -identity[8], dplane[0]/stress_scale),
                ((kappa-k0)/strain_scale, (m.compaction_onset-f["compaction_force"])/stress_scale,
                 identity[9], -(m.compaction_mode @ dtau-m.compaction_hardening*dk)/stress_scale),
                (x[10], 1-f["matrix_release"]/resistance, identity[10],
                 -dy/resistance+f["matrix_release"]*resistance_prime*dd/resistance**2),
                (x[11], -fm/stress_scale, identity[11], -dfm/stress_scale),
            ]
            for row, (a, b, da, db) in enumerate(pairs, start=8):
                pa, pb = complementarity_partials(a, b)
                jacobian[row] = pa*da+pb*db
            return residual, state, jacobian

        cached_x, cached_value = None, None

        def cached(x):
            nonlocal cached_x, cached_value
            if cached_x is None or not np.array_equal(x, cached_x):
                cached_x, cached_value = x.copy(), evaluate(x)
            return cached_value

        initial = np.r_[cm0/strain_scale, z0[1:]/strain_scale, z0[0]/strain_scale,
                        k0/strain_scale, 0., 0.]
        solution = least_squares(lambda x: cached(x)[0], initial, jac=lambda x: cached(x)[2],
                                 x_scale="jac", ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=200)
        residual, state, _ = cached(solution.x)
        if np.max(np.abs(residual)) > 1e-7 and law.damage_peak_density is not None:
            # The softening branch can fold under axial control. Continuing
            # only from the old state then stalls at the vanished local root.
            # At each fixed density solve the other forces, then bracket the
            # damage consistency equation along increasing density. This
            # retains the implicit flow, contact and fluid equations during
            # the branch search rather than guessing their new strains.
            fixed_states = {}

            def fixed_density(log_ratio):
                if log_ratio in fixed_states:
                    candidate = fixed_states[log_ratio]
                else:
                    seed = (fixed_states[min(fixed_states, key=lambda v: abs(v-log_ratio))].x.copy()
                            if fixed_states else initial.copy())
                    seed[10] = log_ratio

                    def fixed_residual(x):
                        result = cached(x)[0].copy()
                        result[10] = x[10]-log_ratio
                        return result

                    def fixed_jacobian(x):
                        result = cached(x)[2].copy()
                        result[10] = identity[10]
                        return result

                    candidate = least_squares(fixed_residual, seed, jac=fixed_jacobian,
                        x_scale="jac", ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=200)
                    if not np.all(np.isfinite(candidate.fun)) or np.max(np.abs(candidate.fun)) > 1e-7:
                        raise RuntimeError(f"Fixed-density constitutive solve failed at "
                                           f"axial strain {axial_strain:g}, log(d/d_previous)={log_ratio:g}.")
                    fixed_states[log_ratio] = candidate
                result = cached(candidate.x)[1]
                return result["response"]["matrix_release"]/result["damage_resistance"]-1

            lower = 0.
            if fixed_density(lower) <= 0:
                log_ratio = lower
            else:
                for _ in range(40):
                    upper = lower+np.log(2.)
                    if fixed_density(upper) <= 0:
                        log_ratio = brentq(fixed_density, lower, upper, xtol=1e-11)
                        fixed_density(log_ratio)
                        break
                    lower = upper
                else:
                    raise RuntimeError(f"Damage consistency root was not bracketed at axial strain {axial_strain:g}.")
            solution = fixed_states[log_ratio]
            residual, state, _ = cached(solution.x)
        # The optimizer flag describes its own stopping rule. A reached
        # iteration limit can still supply a root of the constitutive equations;
        # conversely, a small optimization step need not supply one.
        if not np.all(np.isfinite(residual)) or np.max(np.abs(residual)) > 1e-7:
            raise RuntimeError(f"Coupled constitutive solve failed at axial strain {axial_strain:g}: "
                               f"{solution.message}; scaled residual={max(abs(residual)):.3g}")
        f = state["response"]
        state["dissipation_increments"] = {
            "matrix_sliding": f["matrix_force"] @ (state["matrix_strain"]-cm0),
            "plane_sliding": f["plane_force"][1:] @ (state["plane_modes"][1:]-z0[1:]),
            "normal_contact": f["plane_force"][0]*(state["plane_modes"][0]-z0[0]),
            "compaction": f["compaction_force"]*(state["compaction"]-k0),
            "damage": f["matrix_release"]*(state["d"]-d0),
        }
        return state
