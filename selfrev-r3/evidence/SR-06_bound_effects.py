"""SR-06: active-bound sensitivity of the current measured-curve likelihood.

Every stored ordinate is evaluated with the current stress-path mirror.  The
primary declared domain is unchanged; the domains here are separate experiments.

Producer actions: --reference [--scenario ...], --profile QUANTITY, --audit
QUANTITY, and --boundary, each with --form and the selected/comparison --steps.
They resume their one current role file in WORK. --finalize writes result.json
only after decisive real-model replay. Parent entry: --verify.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.dont_write_bytecode = True
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "results/iterative/sr06-bound-effects"
PROMPT = ROOT / "results/driver_codex_exec/revision-r3-sr06-bound-effects/prompt.md"
sys.path.insert(0, str(ROOT / "compute/src"))
import psutil
proc = psutil.Process()
cpus = proc.cpu_affinity()
offset = int(os.environ.get("SR06_CPU_OFFSET", "0"))
proc.cpu_affinity(cpus[offset:offset+8] or cpus[:8])
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares, minimize
from tfenn_invcalib import current_inputs, curves, likelihood, mirror, theta_space, numerics
from tfenn_invcalib import operative_elastic as elastic

LEAST_SQUARES_FTOL = inspect.signature(least_squares).parameters["ftol"].default


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT/path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


mapping = load_module("sr06_primary", "selfrev-r2/evidence/SR-09a_map.py")
CUT = mapping.CUT
STEPS = mapping.STEPS
FINE = mapping.COMPARISON_STEPS
SCENARIOS = ("primary", "elastic", "friction", "fabric", "coupled")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, data):
    mapping.save(path, data)


def primary(form, steps):
    data = read(mapping.output(form))
    current_inputs.require_current(data["recipe"])
    if steps == STEPS[form]:
        return data["reference"]
    if steps == FINE[form]:
        current_inputs.require_current(data["grid_comparison"]["recipe"]["inputs"])
        return data["grid_comparison"]["reference"]
    raise ValueError("Use the selected or comparison integration grid")


def identity(form, steps):
    return current_inputs.identity(form, steps, (
        "compute/src/tfenn_invcalib/operative_elastic.py",
        "selfrev-r2/evidence/SR-09a_map.py"))


def feasible_fd_jacobian(fun,x,step,lower,upper,indices=None):
    """Use two actual feasible probes, including one-sided physical boundaries.

    A failed real path is not assigned a derivative or a successful residual.
    Change the local stencil when the requested probe is outside the actual
    constitutive domain; fail explicitly if no representable stencil remains.
    """
    x=np.asarray(x,dtype=float)
    f0=np.asarray(fun(x))
    def column(i):
        h=step
        failures=[]
        while True:
            probes=[]
            seen=set()
            for multiplier in (-1,1,-2,2):
                point=x.copy()
                point[i]=np.clip(x[i]+multiplier*h,lower[i],upper[i])
                offset=point[i]-x[i]
                if offset==0 or offset in seen:
                    continue
                seen.add(offset)
                try:
                    value=np.asarray(fun(point))
                except ValueError as exc:
                    failures.append({"offset":float(offset),"reason":str(exc)})
                    continue
                probes.append((offset,value))
                if len(probes)==2:
                    break
            if len(probes)==2:
                break
            h/=2
            if x[i]+h==x[i] and x[i]-h==x[i]:
                raise ValueError(f"No representable feasible derivative stencil for coordinate {i}: {failures}")
        (a,fa),(b,fb)=probes
        result=(b*(fa-f0)/a-a*(fb-f0)/b)/(b-a)
        change=None
        if failures or h!=step:
            change={"coordinate":i,"requested_step":step,"actual_offsets":[float(a),float(b)],
                    "rejected_probes":failures}
        return result,change
    coordinates=range(len(x)) if indices is None else indices
    workers=int(os.environ.get("SR07_FD_WORKERS","1"))
    if workers>1:
        # Each task calls the identical single-vector compiled primal.
        # There is no vmap/batched constitutive arithmetic, shared residual
        # cache or fit/checkpoint mutation in these derivative probes.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results=list(pool.map(column,coordinates))
    else:
        results=[column(i) for i in coordinates]
    columns=[r[0] for r in results]
    changes=[r[1] for r in results if r[1] is not None]
    jac=np.column_stack(columns)
    if not np.isfinite(jac).all():
        raise ValueError("Nonfinite derivative from the actual feasible residuals")
    return jac,changes


class Model:
    def __init__(self, form, steps, scenario="coupled"):
        self.form, self.steps, self.scenario = form, steps, scenario
        base = theta_space.space_for_form(form)
        self.names = base.theta_names
        self.internal_names = base.internal_names
        self.lo, self.hi = base.box_native_lo.copy(), base.box_native_hi.copy()
        self.log = base.box_is_log
        self.release_v = scenario in ("elastic", "coupled")
        self.release_phi = scenario in ("friction", "coupled")
        self.release_q = form == 2 and scenario in ("fabric", "coupled")
        if self.release_v:
            self.hi[3] = 1.0
        if self.release_phi:
            self.lo[9] = 0.0
        if self.release_q:
            self.lo[13] = -100.0  # affine origin/scale only; lower bound is infinite
        self.span = self.hi-self.lo
        self.lower, self.upper = np.zeros(len(self.names)), np.ones(len(self.names))
        if form == 2:
            # A compensated tail can have d << 1 while q*d**2 remains finite.
            # The old -1.5+3*u map lost d to cancellation around u=0.5,
            # corrupting the actual quadratic fabric term.  Keep exactly the
            # same [-1.5,1.5] physical domain with a zero-origin coordinate.
            self.lo[12], self.hi[12], self.span[12] = 0., 3., 3.
            self.lower[12], self.upper[12] = -.5, .5
        if self.release_q:
            self.lower[13]=-np.inf
        self.upper[7] = np.nextafter(1., 0.)
        self.upper[9] = np.nextafter(1., 0.)
        if not self.release_phi:
            self.lower[9] = (np.nextafter(5., 60.)-5.)/55.
        self.lower[7] = (np.nextafter(5., 60.)-5.)/55.
        if self.release_v:
            self.upper[3] = np.nextafter(1., 0.)
        self.cs = curves.load_measured_curveset(ROOT/"compute", n_steps=steps,
                                              effective_stress_path=True)
        cfg = likelihood.load_likelihood_config(ROOT/"compute")
        self.sigma = cfg.sigma_q
        self.constant = self.cs.q_meas.size*math.log(self.sigma*math.sqrt(2*math.pi))
        forward = mirror.mirror_forward_path if form == 1 else mirror.mirror_forward_path_f2
        cs = self.cs
        def residual(u):
            theta = self.theta(u)
            def one(arg):
                M, n, pressure, stations = arg
                out = forward(theta, M, n, pressure, cs.d_eps11,
                              n_steps=steps, early_exit=True)
                return jnp.interp(stations, jnp.arange(steps+1)*cs.d_eps11, out["q"])
            return ((jax.lax.map(one, (cs.M_bond, cs.n_hat, cs.p_conf_seq,
                     jnp.asarray(cs.eps_stations)))-jnp.asarray(cs.q_meas))/self.sigma).ravel()
        self.r = jax.jit(residual)
        self.rb = jax.jit(jax.vmap(residual))
        self.j = jax.jit(jax.jacfwd(residual))
        self.e = jax.jit(lambda u: elastic.engineering(self.theta(u)))
        self.ej = jax.jit(jax.jacfwd(self.e))
        self.m = jax.jit(self.margins)
        self.mj = jax.jit(jax.jacfwd(self.margins))

    def theta(self, u):
        v = jnp.asarray(self.lo)+jnp.asarray(self.span)*u
        internal = jnp.where(jnp.asarray(self.log), jnp.exp(v), v)
        return internal.at[6].set(internal[6]*internal[5])

    def u(self, theta):
        v = np.asarray(theta, dtype=float).copy()
        v[6] /= v[5]
        v[self.log] = np.log(v[self.log])
        return (v-self.lo)/self.span

    def margins(self, u):
        th = self.theta(u)
        spd = elastic.positive_definiteness_margin(th)
        if self.form == 1:
            return jnp.array([spd])
        d, q = th[12:14]
        a, b = jnp.minimum(d, -2*d), jnp.maximum(d, -2*d)
        z = jnp.clip(-1/(2*jnp.where(q > 0, q, 1.)), a, b)
        ends = jnp.minimum(1+a+q*a*a, 1+b+q*b*b)
        g = jnp.where(q > 0, jnp.minimum(ends, 1+z+q*z*z), ends)
        # The original .05 floor is retained only in the primary and isolated
        # elastic/friction scenarios.  Released fabric uses strict positivity.
        return jnp.array([spd, g-(0. if self.release_q else .05)])

    def valid(self, u):
        return (np.isfinite(u).all() and np.all(u >= self.lower-1e-14)
                and np.all(u <= self.upper+1e-14)
                and np.all(np.asarray(self.m(jnp.asarray(u))) > 0))

    def record(self, u):
        u = np.asarray(u)
        if not self.valid(u):
            raise ValueError(f"Inadmissible {self.scenario} vector: {u}")
        theta = np.asarray(self.theta(jnp.asarray(u)))
        r = np.asarray(self.r(jnp.asarray(u)))
        if not np.isfinite(r).all():
            raise ValueError("Nonfinite or unconverged current real-model path")
        C = np.asarray(mirror._ti_stiffness_local_j(theta[0], theta[1], theta[3], theta[4], theta[2]))
        eig = np.linalg.eigvalsh(C)
        if eig.min() <= 0:
            raise ValueError("Integrated stiffness is not SPD")
        eng = np.asarray(self.e(jnp.asarray(u)))
        S = np.linalg.inv(C)
        direct = np.array([1/S[0,0], 1/S[2,2], 1/(2*S[3,3]), -S[0,1]/S[0,0],
                           -S[0,2]/S[0,0], -S[0,2]/S[2,2], 1/(2*S[5,5]), S[2,2]/S[0,0]])
        condition=float(np.linalg.cond(C))
        np.testing.assert_allclose(eng,direct,
            rtol=max(2e-10,32*np.finfo(float).eps*condition),atol=2e-10)
        return {"u": u.tolist(), "theta": theta.tolist(),
            "coordinate_system": "zero-origin-fabric" if self.form == 2 else "native-affine",
            "nll": float(r@r/2+self.constant),
            "engineering": dict(zip(elastic.NAMES, eng.tolist())),
            "elastic_rho":float(eng[5]*np.sqrt(2*eng[0]/((1-eng[3])*eng[1]))),
            "normal_compliance_spd_margin":float(1-eng[3]-2*eng[5]**2*eng[0]/eng[1]),
            "spd_margin": float(elastic.positive_definiteness_margin(theta)),
            "stiffness_min_eigenvalue_MPa": float(eig.min()),
            "stiffness_condition_number":condition,
            "engineering_mapping_relative_error":float(np.max(abs(eng-direct)/np.maximum(abs(eng),1.))),
            "fabric_g_min": theta_space.fabric_g_min(*theta[12:14]) if self.form == 2 else None,
            "active_constraints": [{"name": self.internal_names[i], "side": side,
                "value": float(theta[i])} for i,x in enumerate(u)
                for side,edge in (("lower", self.lower[i]), ("upper", self.upper[i])) if abs(x-edge)<1e-7]}

    def value(self, row, name):
        if name.startswith("operative_"):
            return row["engineering"][name.removeprefix("operative_")]
        return row["theta"][self.names.index(name)]

    def chart(self, fixed, base):
        if fixed is None:
            return np.arange(len(base)), lambda x: x.copy(), lambda u: np.eye(len(base))
        name, value = fixed
        if name == "operative_nu_pp":
            pivot = 4
            def original(x):
                u = base.copy()
                u[free] = x
                p,o,_,v,_ = np.asarray(self.theta(jnp.asarray(u)))[:5]
                alpha = (p+o)**2/(2*p*o)
                den = alpha*(1-value)*(1+v)-2*(1+value)*(1-v)
                square = 2*(1-v)*(v-value)/den if den else -1.
                w = np.sqrt(square) if square > 0 else np.nan
                u[4] = (w-self.lo[4])/self.span[4]
                return u
            def differential(u):
                d = np.asarray(self.ej(jnp.asarray(u)))[3]
                T = np.eye(len(base))[:,free]
                T[4] = -d[free]/d[4]
                return T
        else:
            pivot = self.names.index(name)
            def original(x):
                u = base.copy()
                u[free] = x
                native = np.log(value) if self.log[pivot] else value
                if pivot == 6:
                    native = value/float(self.theta(jnp.asarray(u))[5])
                u[pivot] = (native-self.lo[pivot])/self.span[pivot]
                return u
            def differential(u):
                T = np.eye(len(base))[:,free]
                if pivot == 6:
                    T[6, list(free).index(5)] = -(self.lo[6]+u[6]*self.span[6])*self.span[5]/self.span[6]
                return T
        free = np.array([i for i in range(len(base)) if i != pivot])
        return free, original, differential

    def fit(self, seed, fixed=None, checkpoint=None, method="auto"):
        if method=="auto":
            method="engineering" if self.scenario=="coupled" else "trf"
        if method=="engineering":
            return self.fit_engineering(seed,fixed,checkpoint)
        started = time.monotonic()
        base = np.clip(self.u(seed["theta"]), self.lower, self.upper)
        serialized_initial=bool(fixed is None
            and seed.get("coordinate_system")==("zero-origin-fabric" if self.form==2 else "native-affine")
            and np.array_equal(np.asarray(self.theta(jnp.asarray(seed["u"]))),np.asarray(seed["theta"]))
            and self.valid(np.asarray(seed["u"])))
        if serialized_initial:
            # In the free native chart u itself is the optimizer variable.
            # Preserve it as x0, rather than retaining it only as a best-point
            # witness beside a round-tripped, different actual solver start.
            base=np.asarray(seed["u"],dtype=float).copy()
        free, original, differential = self.chart(fixed, base)
        x = base[free]
        if fixed and fixed[0]=="operative_nu_pp" and not self.valid(original(x)):
            eng=dict(zip(elastic.NAMES,np.asarray(self.e(jnp.asarray(base)))))
            eng["nu_pp"]=fixed[1]
            # Keep the seed's axial moduli for initialization.  If necessary,
            # move its cross Poisson ratio into the target compliance's SPD
            # domain before taking the exact assembly inverse.
            limit=np.sqrt((1-fixed[1])*eng["E_o"]/(2*eng["E_p"]))
            eng["nu_op"]=min(eng["nu_op"],.9*limit)
            candidates=elastic.assembly_candidates(**{k:eng[k] for k in
                ("E_p","E_o","G_op","nu_pp","nu_op")})
            for candidate in candidates:
                th=np.asarray(seed["theta"]).copy()
                th[:5]=candidate
                candidate_u=self.u(th)
                if self.valid(candidate_u):
                    x=candidate_u[free]
                    break
        # Exact operative constraint may need a feasible elastic starting point.
        if not self.valid(original(x)):
            if fixed is None:
                raise ValueError("Inadmissible initial reference")
            name,value = fixed
            def quantity(u):
                if name == "operative_nu_pp":
                    return self.e(u)[3]
                return self.theta(u)[self.names.index(name)]
            q = jax.jit(quantity)
            qj = jax.jit(jax.grad(quantity))
            scale = max(abs(value),1.)
            cons = [{"type":"ineq", "fun":lambda z: np.asarray(self.m(jnp.asarray(z))),
                     "jac":lambda z:np.asarray(self.mj(jnp.asarray(z)))},
                    {"type":"eq", "fun":lambda z:(float(q(jnp.asarray(z)))-value)/scale,
                     "jac":lambda z:np.asarray(qj(jnp.asarray(z)))/scale}]
            sol = minimize(lambda z: (float((z-base)@(z-base)/2),z-base),base,jac=True,
                           method="SLSQP",bounds=list(zip(self.lower,self.upper)),
                           constraints=cons,options={"ftol":1e-12,"maxiter":200})
            x = sol.x[free]
        best = self.record(original(x))
        initial = best["nll"]
        calls = rejected = 0
        def residual(z):
            nonlocal best,calls,rejected
            calls += 1
            u = original(z)
            r = np.asarray(self.r(jnp.asarray(u))) if self.valid(u) else None
            if r is None or not np.isfinite(r).all():
                rejected += 1
                return np.full(self.cs.q_meas.size,1e100)  # rejected trial only
            if float(r@r/2+self.constant) < best["nll"]:
                best = self.record(u)
                if checkpoint:
                    checkpoint(best)
            return r
        stencil_changes=[]
        def jacobian(z):
            nonlocal best
            u = original(z)
            if getattr(self,"real_fd_step",None):
                probes=[]
                def actual(point):
                    trial=original(point)
                    if not self.valid(trial):
                        raise ValueError("Invalid physical vector in native finite-difference stencil")
                    r=np.asarray(self.r(jnp.asarray(trial)))
                    if not np.isfinite(r).all():
                        raise ValueError("Invalid real path in native finite-difference stencil")
                    probes.append((trial.copy(),float(r@r/2+self.constant)))
                    return r
                jac,changes=feasible_fd_jacobian(actual,z,self.real_fd_step,self.lower[free],self.upper[free])
                trial,nll=min(probes,key=lambda item:(item[1],tuple(item[0])))
                if nll<best["nll"]:
                    best={**self.record(trial),"best_point_origin":"actual finite-difference nuisance probe"}
                    if checkpoint:
                        checkpoint(best)
                if changes:
                    stencil_changes.append({"optimizer_x":z.tolist(),"changes":changes})
                return jac
            J = np.asarray(self.j(jnp.asarray(u)))@differential(u)
            if not np.isfinite(J).all():
                raise ValueError("Nonfinite constrained real-model Jacobian")
            return J
        passes = []
        completed_starts=[]
        while True:
            requested_start=x.copy()
            if method == "slsqp":
                def objective(z):
                    u=original(z)
                    if not self.valid(u):
                        margins=np.asarray(self.m(jnp.asarray(u)))
                        negative=np.minimum(margins,0.)
                        return (1e12+1e12*float(negative@negative),
                            2e12*negative@np.asarray(self.mj(jnp.asarray(u)))@differential(u))
                    r=residual(z)
                    return float(r@r/2)/self.cs.q_meas.size, jacobian(z).T@r/self.cs.q_meas.size
                constraints=[{"type":"ineq","fun":lambda z:np.asarray(self.m(jnp.asarray(original(z)))),
                              "jac":lambda z:np.asarray(self.mj(jnp.asarray(original(z))))@differential(original(z))}]
                sol=minimize(objective,x,jac=True,method="SLSQP",
                    bounds=list(zip(self.lower[free],self.upper[free])),constraints=constraints,
                    options={"ftol":1e-9,"maxiter":100})
                passes.append({"status":int(sol.status),"message":str(sol.message),"nfev":int(sol.nfev)})
                unfinished=sol.status==9
            else:
                sol = least_squares(residual,x,jac=jacobian,x_scale="jac",tr_solver="exact",
                                    bounds=(self.lower[free],self.upper[free]),max_nfev=100)
                passes.append({"status":int(sol.status),"message":str(sol.message),
                               "nfev":int(sol.nfev),"optimality":float(sol.optimality)})
                unfinished=sol.status==0
            print("fit",self.form,self.steps,self.scenario,fixed,"nll",best["nll"],
                  "status",sol.status,"calls",calls,"rejected",rejected,flush=True)
            if sol.success:
                completed_starts.append(requested_start)
            if (not unfinished and method!="slsqp" and sol.success
                    and best.get("best_point_origin")=="actual finite-difference nuisance probe"
                    and best["nll"]<float(sol.fun@sol.fun/2+self.constant)):
                candidate=np.asarray(best["u"])[free]
                if np.array_equal(candidate,sol.x):
                    raise ValueError("The same native optimizer point has inconsistent replayed likelihoods")
                if any(np.array_equal(candidate,point) for point in completed_starts):
                    passes[-1]["lower_probe_at_already_terminated_start"]=best["nll"]
                    break  # TRF shifts exact bound initials into the interior; identical restarts add no search.
                passes[-1]["continued_from_lower_actual_probe_nll"]=best["nll"]
                x=candidate
                continue
            if not unfinished:
                break
            x = sol.x
        if not sol.success:
            raise ValueError(f"Unresolved nuisance fit: {sol.message}")
        return {**best,"fixed":fixed,"initial_nll":initial,"solver":method+" exact constrained chart",
                "derivative_method":"three-point actual-residual differences" if getattr(self,"real_fd_step",None) else "automatic differentiation",
                "derivative_absolute_step":getattr(self,"real_fd_step",None),
                "single_primal_derivative_workers":int(os.environ.get("SR07_FD_WORKERS","1")),
                "serialized_native_initial":serialized_initial,
                "actual_derivative_probes_retained":bool(getattr(self,"real_fd_step",None)),
                "feasible_stencil_changes":stencil_changes,
                "termination":passes,"residual_calls":calls,"rejected_trials":rejected,
                "seconds":time.monotonic()-started}

    def fit_engineering(self,seed,fixed=None,checkpoint=None):
        """Optimize the same law in a regular engineering-compliance chart.

        rho = nu_op sqrt(2 E_p / ((1-nu_pp) E_o)); |rho|<1 is the
        normal-compliance SPD condition.  The assembly inverse is exact and
        the residual/Jacobian still come from this model's actual mirror.
        """
        started=time.monotonic()
        fixed_items=dict(fixed.items() if isinstance(fixed,dict) else [fixed] if fixed else [])
        fabric_chart=self.form==2
        strength_chart=fabric_chart and getattr(self,"strength_chart",False)
        log_gap_chart=getattr(self,"log_gap_chart",False)
        if strength_chart and not (getattr(self,"direct_search",False) or getattr(self,"real_fd_step",None)):
            raise ValueError("The effective-strength chart uses actual-residual differences or direct search")
        fabric_sign=(float(np.sign(fixed_items.get("d_fab",seed["theta"][12])))
                     if fabric_chart else 1.)
        chart_identity={"kind":"engineering-compliance-v2","fabric_sign":fabric_sign,
                        "strength_chart":strength_chart,"log_gap_chart":log_gap_chart}
        if fabric_chart and fabric_sign==0:
            raise ValueError("The signed log-fabric chart needs a nonzero amplitude")
        free_q=fabric_chart and self.release_q and "q_fab" not in fixed_items
        def fabric_floor(d):
            boundary=jnp.maximum(-1-d,(2*d-1)/4)
            return jnp.where(d>1,d*d/4,boundary)
        def coordinates(u):
            e=self.e(u)
            x=u.at[0].set((jnp.log(e[0])-self.lo[0])/self.span[0])
            x=x.at[1].set((jnp.log(e[1])-self.lo[1])/self.span[1])
            x=x.at[3].set((e[3]+1)/2)
            x=x.at[4].set(e[5]*jnp.sqrt(2*e[0]/((1-e[3])*e[1])))
            if log_gap_chart:
                x=x.at[4].set(jnp.log1p(-x[4]))
            if fabric_chart:
                th=self.theta(u)
                x=x.at[12].set(jnp.log(jnp.abs(th[12])))
                if free_q:
                    floor=fabric_floor(th[12]);ceiling=8*th[12]**2
                    x=x.at[13].set((th[13]*th[12]**2-floor)/(ceiling-floor))
                if strength_chart:
                    b=th[13]*th[12]**2
                    x=x.at[5].set((jnp.log(th[5]*(1+th[12]+b))-self.lo[5])/self.span[5])
                    x=x.at[13].set((jnp.log(th[5]*(1-2*th[12]+4*b))-self.lo[5])/self.span[5])
            return x
        coordinate_j=jax.jit(jax.jacfwd(coordinates))
        seed_theta=np.asarray(seed["theta"]).copy()
        if fixed and not isinstance(fixed,dict) and self.form==2:
            name,value=fixed
            # Continue the actual quadratic fabric coefficient b=q*d**2.
            # Keeping d unchanged at a large negative-q step would create a
            # spurious high-loss branch before nuisance optimization starts.
            if name=="c_peak" and strength_chart:
                # Changing peak cohesion alone can invalidate the explicit
                # path before nuisance fitting begins. Continue its actual
                # fabric compensation: retain the two effective cohesions
                # and the residual ratio while fixing the requested c_peak.
                c,d,q=seed_theta[5],seed_theta[12],seed_theta[13]
                if getattr(self,"c_peak_start_projection","effective_strength")=="native_cohesion":
                    # A previously observed invalid compensated projection
                    # can still have a valid start with d and q retained.
                    # Only the requested cohesion and residual ratio change;
                    # the actual constitutive residual decides feasibility.
                    seed_theta[5],seed_theta[6]=value,seed_theta[6]*value/c
                else:
                    A,B=c*(1+d+q*d*d),c*(1-2*d+4*q*d*d)
                    new_d=((4*A-B)/(3*value)-1)/2
                    if new_d*d>0:
                        b=A/value-1-new_d
                        seed_theta[5],seed_theta[6]=value,seed_theta[6]*value/c
                        seed_theta[12],seed_theta[13]=new_d,b/(new_d*new_d)
            elif name=="q_fab" and seed_theta[13]*value>0:
                seed_theta[12]*=np.sqrt(seed_theta[13]/value)
                seed_theta[13]=value
            elif name=="d_fab" and value!=0:
                if strength_chart:
                    c,d,q=seed_theta[5],seed_theta[12],seed_theta[13]
                    A,B=c*(1+d+q*d*d),c*(1-2*d+4*q*d*d)
                    new_c=(4*A-B)/(3*(1+2*value))
                    if new_c<=0:
                        raise ValueError("Fixed fabric amplitude outside this positive-strength chart component")
                    b=A/new_c-1-value
                    retained_c=np.clip(new_c,np.exp(self.lo[5]),np.exp(self.hi[5]))
                    seed_theta[5],seed_theta[6]=retained_c,seed_theta[6]*retained_c/c
                    seed_theta[12],seed_theta[13]=value,b/value**2
                else:
                    seed_theta[13]*=(seed_theta[12]/value)**2
                    seed_theta[12]=value
        base=np.asarray(coordinates(jnp.asarray(self.u(seed_theta)))).copy()
        saved_chart=seed.get("engineering_chart")
        chart_reuse=False
        chart_reuse_kind="physical-vector reconstruction"
        previous_fixed=saved_chart.get("fixed",{}) if saved_chart else {}
        changing_constraints=set(previous_fixed)|set(fixed_items)
        compatible_release=(not changing_constraints.intersection(("c_peak","q_fab"))
                            and (strength_chart or "d_fab" not in changing_constraints))
        if (saved_chart and saved_chart["identity"]==chart_identity
                and (fixed_items==previous_fixed or compatible_release)):
            # The inverse assembly map is accurate to roundoff, but roundoff
            # can excite the diagnosed explicit-update instability. Resume
            # the actual optimizer point instead of recomputing its inverse.
            # Releasing or changing a simple fixed coordinate only changes
            # its free/fixed membership below. Its full chart vector is still
            # valid. Fixed c_peak/q alter the strength reconstruction and
            # require the physical-vector conversion above instead.
            base=np.asarray(saved_chart["coordinates"],dtype=float).copy()
            chart_reuse=True
            chart_reuse_kind="compatible full chart"
        elif saved_chart and saved_chart["identity"]==chart_identity:
            # c_peak/q changes can require new fabric-strength coordinates.
            # Preserve all unaffected coordinates, especially the elastic
            # inverse near SPD, instead of perturbing them by a round trip.
            saved_coordinates=np.asarray(saved_chart["coordinates"],dtype=float)
            unaffected=[i for i in range(len(base)) if i not in (5,6,12,13)]
            base[unaffected]=saved_coordinates[unaffected]
            chart_reuse_kind="unchanged coordinates retained; fabric-strength coordinates reconstructed"
        lower,upper=self.lower.copy(),self.upper.copy()
        lower[:2]=-2.
        lower[3:5]=[np.nextafter(0.,1.),0.]
        upper[3:5]=np.nextafter(1.,0.)
        if log_gap_chart:
            lower[4],upper[4]=-np.inf,0.
        if fabric_chart:
            lower[12],upper[12]=-np.inf,np.log(1.5)
            if free_q:
                lower[13],upper[13]=np.nextafter(0.,1.),1.
            if strength_chart:
                # Positive effective cohesions at lambda=0 and 1. The original
                # c_peak, residual-ratio, q and global-positivity constraints
                # are checked on every recovered original parameter vector.
                lower[[5,13]],upper[[5,13]]=-np.inf,np.inf
        pivots=[]
        fixed_encoding_reused=[]
        if fixed:
            items=fixed.items() if isinstance(fixed,dict) else [fixed]
            for name,value in items:
                pivot=3 if name=="operative_nu_pp" else (4 if name=="elastic_rho" else self.names.index(name))
                pivots.append(pivot)
                if pivot==6:
                    raise ValueError("Physical residual cohesion needs its own engineering chart")
                if (chart_reuse and name=="operative_nu_pp"
                        and seed.get("engineering",{}).get("nu_pp")==value):
                    # This exact requested physical quantity is already held
                    # by the saved vector. Inverting C and re-encoding that
                    # value can perturb the chart by roundoff and excite the
                    # diagnosed path instability. Keep its actual encoding.
                    fixed_encoding_reused.append(name)
                    continue
                base[pivot]=((value+1)/2 if name=="operative_nu_pp" else value if name=="elastic_rho" else
                    np.log(abs(value)) if name=="d_fab" else (
                    ((np.log(value) if self.log[pivot] else value)-self.lo[pivot])/self.span[pivot]))
                if log_gap_chart and name=="elastic_rho":
                    base[pivot]=np.log1p(-value)
        free=np.array([i for i in range(len(base)) if i not in pivots])
        def unpack(z):
            x=base.copy();x[free]=z
            ep,eo=np.exp(self.lo[:2]+self.span[:2]*x[:2])
            nu=2*x[3]-1
            rho=-np.expm1(x[4]) if log_gap_chart else x[4]
            cross=rho*np.sqrt((1-nu)*eo/(2*ep))
            th=np.asarray(self.theta(jnp.asarray(x))).copy()
            if fabric_chart:
                th[12]=fabric_sign*np.exp(x[12])
                if free_q:
                    floor=float(fabric_floor(th[12]));ceiling=8*th[12]**2
                    th[13]=(floor+x[13]*(ceiling-floor))/th[12]**2
                if strength_chart:
                    A,B=np.exp(self.lo[5]+self.span[5]*x[[5,13]])
                    d=th[12]
                    if "c_peak" in fixed_items:
                        c=fixed_items["c_peak"]
                        A=(3*c*(1+2*d)+B)/4
                        b=A/c-1-d
                    elif "q_fab" in fixed_items:
                        b=fixed_items["q_fab"]*d*d
                        c=A/(1+d+b)
                    else:
                        if 1+2*d==0:
                            raise ValueError("Singular effective-strength chart at d=-0.5")
                        c=(4*A-B)/(3*(1+2*d))
                        b=A/c-1-d
                    if not np.isfinite(c) or c<=0:
                        raise ValueError("Nonpositive recovered matrix cohesion")
                    th[5],th[6],th[13]=c,(self.lo[6]+self.span[6]*x[6])*c,b/(d*d)
            candidates=elastic.assembly_candidates(E_p=ep,E_o=eo,G_op=th[2],nu_pp=nu,nu_op=cross)
            valid=[]
            for candidate in candidates:
                th[:5]=candidate
                u=self.u(th)
                if self.valid(u):
                    valid.append(u)
            if not valid:
                raise ValueError("Engineering target outside the retained assembly domain")
            return min(valid,key=lambda u:np.linalg.norm(u[:5]-self.u(seed["theta"])[:5]))
        best=None;count=rejected=0
        if (seed.get("coordinate_system")==("zero-origin-fabric" if self.form==2 else "native-affine")
                and np.array_equal(np.asarray(self.theta(jnp.asarray(seed["u"]))),np.asarray(seed["theta"]))):
            seed_u=np.asarray(seed["u"])
            candidate=self.record(seed_u)
            def holds(name,value):
                actual=candidate["elastic_rho"] if name=="elastic_rho" else self.value(candidate,name)
                tolerance=16*np.finfo(float).eps*max(1.,abs(value))
                if name in ("operative_nu_pp","elastic_rho"):
                    # Reuse the conditioning allowance already used by
                    # record() to compare the operative mapping with C^-1.
                    # This only recognizes roundoff in the exact constraint;
                    # it does not relax the optimization equality or domain.
                    tolerance=max(tolerance,32*np.finfo(float).eps*candidate["stiffness_condition_number"]*max(1.,abs(value)))
                return abs(actual-value)<=tolerance
            if all(holds(name,value) for name,value in fixed_items.items()):
                best=candidate
                best["best_point_origin"]="exact serialized feasible initial point"
                if "engineering_chart" in seed:
                    best["engineering_chart"]=seed["engineering_chart"]
        derivative_repairs=[]
        rejected_examples={}
        cache={}
        def residual(z):
            nonlocal best,count,rejected
            count+=1
            try:
                u=unpack(z)
            except ValueError as exc:
                rejected+=1
                rejected_examples.setdefault(str(exc),{"reason":str(exc),"optimizer_x":z.tolist()})
                if best is None:
                    raise
                return np.full(self.cs.q_meas.size,1e100)
            r=np.asarray(self.r(jnp.asarray(u)))
            if not np.isfinite(r).all():
                rejected+=1
                rejected_examples.setdefault("nonfinite_path",{"reason":"Nonfinite or unconverged actual path",
                    "optimizer_x":z.tolist(),"theta":np.asarray(self.theta(jnp.asarray(u))).tolist()})
                if best is None:
                    raise ValueError("Invalid initial engineering-chart integration path")
                return np.full(self.cs.q_meas.size,1e100)
            if best is None or r@r/2+self.constant<best["nll"]:
                best=self.record(u)
                best["best_point_origin"]="actual optimizer residual evaluation"
                full=base.copy();full[free]=z
                best["engineering_chart"]={"identity":chart_identity,"coordinates":full.tolist(),
                                           "fixed":fixed_items}
                if checkpoint:
                    checkpoint(best)
            cache.update(z=z.copy(),u=u)
            return r
        def jacobian(z):
            nonlocal best
            u=cache["u"] if np.array_equal(cache.get("z"),z) else unpack(z)
            fd_step=getattr(self,"real_fd_step",None)
            if fd_step is not None:
                if getattr(self,"real_fd_batch",False):
                    points=[np.asarray(z).copy()]
                    stencils=[]
                    for i in range(len(z)):
                        left,right=z[i]-lower[free][i],upper[free][i]-z[i]
                        offsets=(-fd_step,fd_step) if min(left,right)>=fd_step else (
                            (1 if right>=left else -1)*min(fd_step,max(left,right)/2),
                            (1 if right>=left else -1)*2*min(fd_step,max(left,right)/2))
                        actual_offsets=[]
                        for offset in offsets:
                            point=np.asarray(z).copy()
                            point[i]=np.clip(z[i]+offset,lower[free][i],upper[free][i])
                            actual_offsets.append(point[i]-z[i]);points.append(point)
                        stencils.append(actual_offsets)
                    vectors=[];valid_indices=[];probe_failures=[]
                    for index,point in enumerate(points):
                        try:
                            vectors.append(unpack(point));valid_indices.append(index)
                        except ValueError as exc:
                            vectors.append(None)
                            probe_failures.append({"probe_index":index,"reason":str(exc)})
                    if vectors[0] is None:
                        raise ValueError("Current optimizer point is outside the physical domain")
                    # Keep the compiled batch shape fixed. Rejected probes
                    # are padded with the real central vector solely for
                    # evaluation; their values are discarded and never used
                    # as derivatives or as successful trial evidence.
                    padded=[vectors[0] if vector is None else vector for vector in vectors]
                    values=np.asarray(self.rb(jnp.asarray(np.stack(padded))))
                    residuals={index:values[index] for index in valid_indices if np.isfinite(values[index]).all()}
                    for index in valid_indices:
                        r=values[index]
                        if not np.isfinite(r).all():
                            probe_failures.append({"probe_index":index,"reason":"Invalid actual batched constitutive path"})
                    columns=[]
                    for i,(a,b) in enumerate(stencils):
                        if a==0 or b==0 or a==b:
                            raise ValueError("Finite-difference stencil collapsed")
                        indices=(0,2*i+1,2*i+2)
                        if all(index in residuals for index in indices):
                            f0,fa,fb=(residuals[index] for index in indices)
                            columns.append((b*(fa-f0)/a-a*(fb-f0)/b)/(b-a))
                        else:
                            def actual(point):
                                r=np.asarray(self.r(jnp.asarray(unpack(point))))
                                if not np.isfinite(r).all():
                                    raise ValueError("Invalid actual constitutive path in replacement stencil")
                                return r
                            column,changes=feasible_fd_jacobian(actual,z,fd_step,lower[free],upper[free],indices=[i])
                            columns.append(column[:,0])
                            derivative_repairs.append({"reason":"A batched derivative stencil contains an inadmissible vector or failed real path",
                                "optimizer_x":z.tolist(),"coordinate":i,"rejected_probes":[r for r in probe_failures if r["probe_index"] in indices],
                                "replacement":"three-point actual feasible residuals","stencil_changes":changes})
                    return np.column_stack(columns)
                probes=[]
                def actual(y):
                    trial=unpack(y)
                    r=np.asarray(self.r(jnp.asarray(trial)))
                    if not np.isfinite(r).all():
                        raise ValueError("Invalid real path in finite-difference nuisance derivative")
                    probes.append((np.asarray(y).copy(),trial.copy(),float(r@r/2+self.constant)))
                    return r
                jac,changes=feasible_fd_jacobian(actual,z,fd_step,lower[free],upper[free])
                point,trial,nll=min(probes,key=lambda item:(item[2],tuple(item[0])))
                if best is None or nll<best["nll"]:
                    best=self.record(trial)
                    best["best_point_origin"]="actual finite-difference nuisance probe"
                    full=base.copy();full[free]=point
                    best["engineering_chart"]={"identity":chart_identity,"coordinates":full.tolist(),"fixed":fixed_items}
                    if checkpoint:
                        checkpoint(best)
                if changes:
                    derivative_repairs.append({"reason":"Adapt the stencil to the actual feasible constitutive domain",
                        "optimizer_x":z.tolist(),"stencil_changes":changes})
                return jac
            B=np.asarray(coordinate_j(jnp.asarray(u)))
            T=np.linalg.solve(B,np.eye(len(u))[:,free])
            J=np.asarray(self.j(jnp.asarray(u)))@T
            try:
                if not np.isfinite(J).all():
                    raise FloatingPointError("Nonfinite analytic derivative")
                with np.errstate(over="raise",invalid="raise"):
                    np.sum(J**2,axis=0)
            except FloatingPointError as exc:
                # The differentiated Aitken/controller branch has produced an
                # overflowing sensitivity at a real fine-grid k endpoint.
                # Recompute the derivative from actual residual evaluations in
                # this regular chart; no penalty/default residual is admitted.
                def actual(y):
                    r=np.asarray(self.r(jnp.asarray(unpack(y))))
                    if not np.isfinite(r).all():
                        raise ValueError("Invalid real path in derivative repair")
                    return r
                J=numerics.absolute_fd_jacobian(actual,z,1e-6,lower[free],upper[free])
                derivative_repairs.append({"reason":str(exc),"theta":np.asarray(self.theta(jnp.asarray(u))).tolist(),
                    "method":"three-point absolute finite difference of actual real-model residuals in engineering/log-d/fabric-coefficient chart",
                    "absolute_step":1e-6,"maximum_absolute_derivative":float(np.max(abs(J)))})
                print("real-residual derivative repair",self.form,self.steps,fixed,flush=True)
            return J
        x=base[free]
        initial_residual=residual(x)
        initial=float(initial_residual@initial_residual/2+self.constant)
        passes=[]
        if getattr(self,"direct_search",False):
            direct_method=self.direct_search if isinstance(self.direct_search,str) else "Nelder-Mead"
            simplex=None
            directions=np.eye(len(x))*.01
            while True:
                def objective(z):
                    if np.any(z<lower[free]) or np.any(z>upper[free]):
                        return 1e200
                    r=residual(z)
                    return float(r@r/2+self.constant)
                if direct_method=="COBYQA":
                    sol=minimize(objective,x,method="COBYQA",bounds=list(zip(lower[free],upper[free])),
                        options={"initial_tr_radius":.02})
                    optimality={"gradient":None,"gradient_note":"COBYQA does not report gradient optimality.",
                        "maximum_constraint_violation":float(sol.maxcv),
                        "initial_trust_region_radius":.02,"final_trust_region_radius":1e-6,
                        "termination_message":str(sol.message)}
                elif direct_method=="Powell":
                    sol=minimize(objective,x,method="Powell",options={"maxiter":20,"direc":directions,"ftol":1e-8})
                    optimality={"gradient":None,"gradient_note":"Powell does not compute a gradient optimality measure.",
                        "terminal_objective":float(sol.fun),"directions":sol.direc.tolist()}
                else:
                    options={"adaptive":True,"maxiter":100}
                    if simplex is None and getattr(self,"feasible_simplex",False):
                        vertices=[x.copy()]
                        for j,i in enumerate(free):
                            delta=.05 if i==12 or (i==4 and log_gap_chart) else .01
                            while True:
                                candidates=[]
                                for sign in (-1,1):
                                    point=x.copy()
                                    point[j]=np.clip(x[j]+sign*delta,lower[i],upper[i])
                                    if point[j]!=x[j]:
                                        value=objective(point)
                                        if np.isfinite(value) and value<1e99:
                                            candidates.append((value,point))
                                if candidates:
                                    vertices.append(min(candidates,key=lambda v:v[0])[1])
                                    break
                                delta/=2
                                if x[j]+delta==x[j]:
                                    raise ValueError("No feasible simplex displacement in a nuisance coordinate")
                        simplex=np.asarray(vertices)
                    if simplex is not None:
                        options["initial_simplex"]=simplex
                    sol=minimize(objective,x,method="Nelder-Mead",bounds=list(zip(lower[free],upper[free])),options=options)
                    vertices,values=sol.final_simplex
                    optimality={"simplex_objective_spread":float(np.ptp(values)),
                        "simplex_coordinate_diameter":float(np.max(np.ptp(vertices,axis=0))),
                        "gradient":None,"gradient_note":"This derivative-free solver does not compute a gradient optimality measure."}
                passes.append({"solver_kind":direct_method,"status":int(sol.status),"success":bool(sol.success),
                    "message":str(sol.message),"nfev":int(sol.nfev),"nit":int(sol.nit),
                    "optimality":optimality})
                print("direct fit",self.form,self.steps,fixed,best["nll"],sol.status,"calls",count,flush=True)
                if sol.status not in ((5,6) if direct_method=="COBYQA" else (1,2)):
                    break
                x=sol.x
                if direct_method=="Powell":
                    directions=sol.direc
                elif direct_method=="Nelder-Mead":
                    simplex=vertices
            if not sol.success:
                raise ValueError(f"Unresolved direct nuisance fit: {sol.message}")
            return {**best,"fixed":fixed,"initial_nll":initial,"solver":"engineering compliance chart "+direct_method,
                "termination":passes,"residual_calls":count,"rejected_trials":rejected,
                "rejected_examples":list(rejected_examples.values()),
                "derivative_method":f"{direct_method} direct search of the actual residual likelihood",
                "strength_chart":strength_chart,
                "log_gap_chart":log_gap_chart,
                "seconds":time.monotonic()-started}
        completed_starts=[]
        while True:
            requested_start=x.copy()
            sol=least_squares(residual,x,jac=jacobian,x_scale="jac",tr_solver="exact",
                bounds=(lower[free],upper[free]),max_nfev=100,ftol=LEAST_SQUARES_FTOL)
            passes.append({"status":int(sol.status),"message":str(sol.message),
                "nfev":int(sol.nfev),"optimality":float(sol.optimality)})
            print("engineering fit",self.form,self.steps,fixed,best["nll"],sol.status,
                "calls",count,"rejected",rejected,flush=True)
            if sol.success:
                completed_starts.append(requested_start)
            if (sol.status!=0 and sol.success
                    and best.get("best_point_origin")=="actual finite-difference nuisance probe"
                    and best["nll"]<float(sol.fun@sol.fun/2+self.constant)):
                candidate=np.asarray(best["engineering_chart"]["coordinates"])[free]
                if np.array_equal(candidate,sol.x):
                    raise ValueError("The same engineering optimizer point has inconsistent replayed likelihoods")
                terminal_cost=float(sol.fun@sol.fun/2)
                probe_improvement=terminal_cost+self.constant-best["nll"]
                if probe_improvement<=LEAST_SQUARES_FTOL*terminal_cost:
                    # Keep the actual lower probe without restarting a successful
                    # solve for changes within that solver's objective tolerance.
                    passes[-1]["lower_probe_within_solver_ftol"]={
                        "nll":best["nll"],"terminal_nll":terminal_cost+self.constant,
                        "objective_improvement":probe_improvement}
                    break
                if any(np.array_equal(candidate,point) for point in completed_starts):
                    passes[-1]["lower_probe_at_already_terminated_start"]=best["nll"]
                    break
                passes[-1]["continued_from_lower_actual_probe_nll"]=best["nll"]
                x=candidate
                continue
            if sol.status!=0:
                break
            x=sol.x
        if not sol.success:
            raise ValueError(f"Unresolved engineering-chart fit: {sol.message}")
        return {**best,"fixed":fixed,"initial_nll":initial,"solver":"engineering compliance chart dense TRF",
            "derivative_method":("three-point actual-residual finite differences" if getattr(self,"real_fd_step",None) else "automatic differentiation"),
            "derivative_absolute_step":getattr(self,"real_fd_step",None),
            "derivative_batched":getattr(self,"real_fd_batch",False),
            "single_primal_derivative_workers":int(os.environ.get("SR07_FD_WORKERS","1")),
            "initial_chart_reused":chart_reuse,
            "initial_chart_reuse_kind":chart_reuse_kind,
            "initial_chart_previous_fixed":previous_fixed if chart_reuse else None,
            "fixed_encoding_reused":fixed_encoding_reused,
            "c_peak_start_projection":getattr(self,"c_peak_start_projection","effective_strength"),
            "actual_derivative_probes_retained":bool(getattr(self,"real_fd_step",None)) and not getattr(self,"real_fd_batch",False),
            "probe_restart_ftol":LEAST_SQUARES_FTOL,
            "termination":passes,"residual_calls":count,"rejected_trials":rejected,
            "rejected_examples":list(rejected_examples.values()),
            "derivative_repairs":derivative_repairs,
            "seconds":time.monotonic()-started}


def refine_reference(form,steps):
    path=ref_path(form,steps,"coupled")
    data=read(path)
    model=Model(form,steps,"coupled")
    seeds=[("retained",data["fit"])]
    for s in ("elastic","friction","fabric"):
        p=ref_path(form,steps,s)
        if p.exists():
            seeds.append((s,read(p)["fit"]))
    for p in WORK.glob(f"f{form}_{steps}_coupled_profile_*.json"):
        d=read(p)
        seeds.append((p.stem,min(d["rows"],key=lambda r:r["nll"])))
    best=data["fit"]
    records=[]
    for label,seed in seeds:
        def checkpoint(row):
            data["in_progress"]={**row,"start":label}
            save(path,data)
        fit=model.fit(seed,checkpoint=checkpoint,method="slsqp")
        fit=model.fit(fit,checkpoint=checkpoint)
        records.append({"start":label,"fit":fit})
        if fit["nll"]<best["nll"]:
            best=fit
    data.update(fit=best,multistart=records,status="complete")
    data.pop("in_progress",None)
    save(path,data)
    print("refined reference",form,best["nll"],flush=True)
    return data


def ref_path(form,steps,scenario):
    return WORK/f"f{form}_{steps}_{scenario}_reference.json"


def run_reference(form, steps, scenario):
    path = ref_path(form,steps,scenario)
    data = read(path) if path.exists() else {}
    ident = identity(form,steps)
    if data:
        if data["inputs"] != ident:
            raise ValueError("Changed input in reference checkpoint")
        if data.get("status") == "complete":
            return data
    model = Model(form,steps,scenario)
    seed = data.get("in_progress", primary(form,steps))
    selected=ref_path(form,STEPS[form],scenario)
    if steps != STEPS[form] and not data and selected.exists():
        seed=read(selected)["fit"]
    if scenario == "coupled" and not data:
        candidates = [seed]
        for s in ("elastic","friction","fabric"):
            p = ref_path(form,steps,s)
            if p.exists() and read(p).get("status") == "complete":
                candidates.append(read(p)["fit"])
        seed = min(candidates,key=lambda r:model.record(model.u(r["theta"]))["nll"])
    data.update(form=form,n_steps=steps,scenario=scenario,inputs=ident,status="running")
    original = model.record(model.u(primary(form,steps)["theta"]))
    data["primary_replay"] = original
    def checkpoint(row):
        data["in_progress"] = row
        save(path,data)
    fit = model.fit(seed,checkpoint=checkpoint)
    continuation = []
    while True:
        nxt = model.fit(fit,checkpoint=checkpoint)
        change = fit["nll"]-nxt["nll"]
        continuation.append({"start_nll":fit["nll"],"nll":nxt["nll"],"termination":nxt["termination"]})
        if change > 0:
            fit = nxt
        if change <= 1e-8*(fit["nll"]-model.constant):
            break
    data.update(status="complete",fit=fit,self_continuation=continuation)
    data.pop("in_progress",None)
    save(path,data)
    return data


def profile_path(form,steps,name,scenario="coupled"):
    return WORK/f"f{form}_{steps}_{scenario}_profile_{name}.json"


def audit_profile(form,steps,name):
    path=profile_path(form,steps,name)
    data=run_profile(form,steps,name)
    if data.get("audit_method")=="current engineering-chart adjacent-interior nuisance continuation":
        return data
    model=Model(form,steps)
    checks=[]
    for side in ("-1","1"):
        end=data["endpoints"][side]
        seed=end.get("inside",data["reference"])
        target=end["outside"]["value"] if end["kind"]=="likelihood_crossing" else end["value"]
        def checkpoint(row):
            data["audit_in_progress"]={"side":side,"value":target,"row":row}
            save(path,data)
        candidate=model.fit(seed,(name,target),checkpoint)
        if name in ("d_fab","q_fab"):
            other="q_fab" if name=="d_fab" else "d_fab"
            other_path=profile_path(form,steps,other)
            if other_path.exists():
                witnesses=read(other_path)["rows"]
                second=min(witnesses,key=lambda r:abs(np.log(abs(model.value(r,name)/target))))
                alternate=model.fit(second,(name,target),checkpoint)
                checks.append({"side":side,"seed_role":"cross-parameter compensated witness",
                               "seed":second,"fit":alternate})
                if alternate["nll"]<candidate["nll"]:
                    candidate=alternate
        candidate["value"]=target
        old=next(r for r in data["rows"] if r["value"]==target)
        if candidate["nll"]<old["nll"]:
            data["rows"][data["rows"].index(old)]=candidate
        checks.append({"side":side,"seed":seed,"fit":candidate})
        data.pop("audit_in_progress",None)
        if end["kind"]=="likelihood_crossing" and candidate["nll"]-data["reference"]["nll"]<CUT:
            data.update(status="running")
            data.setdefault("audit_checks",[]).extend(checks)
            save(path,data)
            run_profile(form,steps,name)
            return audit_profile(form,steps,name)
    data.setdefault("audit_checks",[]).extend(checks)
    for side in ("-1","1"):
        end=data["endpoints"][side]
        if end["kind"]=="likelihood_crossing":
            inside,outside=bracket(data["rows"],model.value(data["reference"],name),
                                   data["reference"]["nll"],int(side))
            fraction=(data["reference"]["nll"]+CUT-inside["nll"])/(outside["nll"]-inside["nll"])
            end.update(inside=inside,outside=outside,
                value=inside["value"]+fraction*(outside["value"]-inside["value"]),
                inside_delta_nll=inside["nll"]-data["reference"]["nll"],
                outside_delta_nll=outside["nll"]-data["reference"]["nll"])
    data["interval"]=[data["endpoints"][s]["value"] for s in ("-1","1")]
    data["audit_method"]="current engineering-chart adjacent-interior nuisance continuation"
    save(path,data)
    return data


def boundary_support(form,steps):
    """Nuisance refits at successively less restrictive compliance-SPD guards."""
    path=WORK/f"f{form}_{steps}_spd_boundary.json"
    data=read(path) if path.exists() else {}
    model=Model(form,steps)
    ref=read(ref_path(form,steps,"coupled"))["fit"]
    fixed={}
    if form==1:
        profile=read(profile_path(form,steps,"operative_nu_pp"))
        seed=profile["endpoints"]["1"]["outside"]
        fixed["operative_nu_pp"]=seed["value"]
    else:
        seed=ref
    rows=(data.get("rows",[]) if data.get("reference",{}).get("theta")==ref["theta"] else [])
    for gap in (1e-3,1e-4,1e-5,1e-6):
        existing=next((r for r in rows if r["rho_gap"]==gap),None)
        if existing:
            seed=existing
            continue
        fixed["elastic_rho"]=1-gap
        def checkpoint(row):
            save(path,{"inputs":identity(form,steps),"form":form,"n_steps":steps,
                       "reference":ref,"rows":rows,"in_progress":row,"status":"running"})
        fit=model.fit(seed,dict(fixed),checkpoint)
        fit["rho_gap"]=gap
        rows.append(fit)
        seed=fit
        save(path,{"inputs":identity(form,steps),"form":form,"n_steps":steps,
                   "reference":ref,"rows":rows,"status":"running"})
    data={"inputs":identity(form,steps),"form":form,"n_steps":steps,"rows":rows,
          "reference":ref,"status":"complete","fixed":fixed,
          "physical_boundary":"rho=1 is excluded by strict SPD; every stored witness is strictly admissible"}
    save(path,data)
    return data


def bracket(rows,center,ref_nll,direction):
    ordered=sorted((r for r in rows if direction*(r["value"]-center)>=0),
                   key=lambda r:direction*(r["value"]-center))
    inside=ordered[0]
    for r in ordered[1:]:
        if r["nll"]-ref_nll >= CUT:
            return inside,r
        inside=r
    return inside,None


def run_profile(form,steps,name,scenario="coupled"):
    path=profile_path(form,steps,name,scenario)
    data=read(path) if path.exists() else {}
    ident=identity(form,steps)
    refdata=read(ref_path(form,steps,scenario))
    if refdata["status"] != "complete":
        raise ValueError("Reference fit is incomplete")
    ref=refdata["fit"]
    if data:
        if data["inputs"] != ident:
            raise ValueError("Profile checkpoint has different inputs")
        if data["reference"]["theta"] != ref["theta"]:
            data["reference"]=ref
            data["status"]="running"
            data.pop("in_progress",None)
            data.pop("endpoints",None)
            data.pop("audit_method",None)
            data.pop("audit_checks",None)
        if data.get("status")=="complete":
            return data
    model=Model(form,steps,scenario)
    center=model.value(ref,name)
    rows=data.get("rows",[{**ref,"value":center}])
    if not any(r["value"]==center for r in rows):
        rows.append({**ref,"value":center})
    data.update(form=form,n_steps=steps,scenario=scenario,name=name,inputs=ident,
                reference=ref,rows=rows,status="running")
    def fit_row(value,seed,force=False):
        prior=next((r for r in rows if r["value"]==value),None)
        if prior and not force:
            return prior
        pending=data.get("in_progress")
        if pending and pending["value"]==value:
            seed=pending["row"]
        def checkpoint(row):
            data["in_progress"]={"value":value,"row":row}
            save(path,data)
        fit=model.fit(seed,(name,value),checkpoint)
        passes=[]
        while True:
            nxt=model.fit(fit,(name,value),checkpoint)
            change=fit["nll"]-nxt["nll"]
            passes.append({"start_nll":fit["nll"],"nll":nxt["nll"],"termination":nxt["termination"]})
            if change>0:
                fit=nxt
            if change<=1e-8*(fit["nll"]-model.constant):
                break
        fit.update(value=value,seed_theta=seed["theta"],self_continuation=passes)
        if prior is None:
            rows.append(fit)
        elif fit["nll"]<prior["nll"]:
            rows[rows.index(prior)]=fit
        else:
            fit=prior
        data.pop("in_progress",None)
        save(path,data)
        print("profile",form,steps,name,value,"delta",fit["nll"]-ref["nll"],flush=True)
        return fit
    if data.get("in_progress"):
        p=data["in_progress"]
        fit_row(p["value"],p["row"],True)
    u=model.u(ref["theta"])
    J=np.asarray(model.j(jnp.asarray(u)))
    free=np.array([i for i in range(len(u)) if min(u[i]-model.lower[i],model.upper[i]-u[i])>1e-7])
    if name=="operative_nu_pp":
        derivative=np.asarray(model.ej(jnp.asarray(u)))[3]
        lo,hi=-1.,1.
    else:
        i=model.names.index(name)
        derivative=np.asarray(jax.jacfwd(model.theta)(jnp.asarray(u)))[i]
        if i==6:
            lo,hi=0.,30.
        else:
            native_lo=model.lo[i]+model.span[i]*model.lower[i]
            native_hi=model.lo[i]+model.span[i]*model.upper[i]
            lo,hi=(np.exp(native_lo),np.exp(native_hi)) if model.log[i] else (native_lo,native_hi)
            if name=="q_fab" and model.release_q:
                lo=-np.inf
    cov=np.linalg.pinv(J[:,free].T@J[:,free])
    spacing=float(np.sqrt(max(0,2*CUT*derivative[free]@cov@derivative[free])))
    if name=="operative_nu_pp":
        spacing=min(spacing,.05)  # finite continuation step, not an interval cutoff
    if name=="q_fab":
        spacing=max(spacing,abs(center)*.3)
    if not np.isfinite(spacing) or spacing<=np.finfo(float).eps:
        # The active friction coordinate has zero conditional variance.  The
        # original measured profile width is a real-data marching scale.
        orig=read(mapping.output(form))["per_parameter"].get(name)
        if orig:
            a,b=orig["interval_internal_units"]
            spacing=abs(b-a)
        if not np.isfinite(spacing) or spacing<=0:
            raise ValueError(f"No measured-data profile spacing for {name}")
    data["initial_spacing"]=spacing
    endpoints={}
    for direction in (-1,1):
        inside,outside=bracket(rows,center,ref["nll"],direction)
        offset=max(spacing,abs(inside["value"]-center)*1.7)
        while outside is None:
            value=float(np.clip(center+direction*offset,lo,hi))
            if name=="d_fab" and direction<0 and center>0:
                value=max(value,inside["value"]*.5)
            if abs(value-center)<1e-8*max(1,abs(center)):
                value=lo if direction<0 else hi
            if value==inside["value"]:
                endpoints[str(direction)]={"kind":"domain_boundary","value":value,
                    "witness":inside,"delta_nll":inside["nll"]-ref["nll"]}
                break
            row=fit_row(value,inside)
            inside,outside=bracket(rows,center,ref["nll"],direction)
            if outside is None and value in (lo,hi):
                endpoints[str(direction)]={"kind":"domain_boundary","value":value,
                    "witness":row,"delta_nll":row["nll"]-ref["nll"]}
                break
            offset*=1.7
        if outside is None:
            continue
        # Retain an explicit crossing bracket.  Resolution is tied to the
        # interval's width: reporting sub-percent endpoint shifts suffices for
        # this domain-effect comparison; root digits are not the objective.
        while True:
            width=abs(outside["value"]-inside["value"])
            scale=max(abs(inside["value"]-center),abs(outside["value"]-center))
            # The extreme fabric tail is reported as a physical-coordinate
            # bracket. Its point estimate is grid-sensitive; extra root digits
            # do not answer whether releasing q changes identifiability.
            fraction_resolution=.5 if name=="q_fab" and direction<0 else .01
            if width<=fraction_resolution*scale:
                before=outside["nll"]
                outside=fit_row(outside["value"],inside,True)
                if outside["nll"]-ref["nll"]>=CUT:
                    data.setdefault("adjacent_continuation",[]).append({
                        "direction":direction,"inside_value":inside["value"],
                        "outside_value":outside["value"],"before_nll":before,
                        "continued_nll":outside["nll"]})
                    break
                inside,outside=bracket(rows,center,ref["nll"],direction)
                if outside is None:
                    # Continue from the new interior branch until a current
                    # nuisance-optimized exterior is obtained.
                    while outside is None:
                        value=float(np.clip(center+direction*offset,lo,hi))
                        if name=="d_fab" and direction<0 and center>0:
                            value=max(value,inside["value"]*.5)
                        fit_row(value,inside,True)
                        inside,outside=bracket(rows,center,ref["nll"],direction)
                        offset*=1.7
                        if outside is None and value in (lo,hi):
                            raise ValueError("Continuation reached a new domain edge; inspect before classifying")
                continue
            fraction=(ref["nll"]+CUT-inside["nll"])/(outside["nll"]-inside["nll"])
            fraction=float(np.clip(fraction,.2,.8))
            value=inside["value"]+fraction*(outside["value"]-inside["value"])
            fit_row(value,inside)
            inside,outside=bracket(rows,center,ref["nll"],direction)
        fraction=(ref["nll"]+CUT-inside["nll"])/(outside["nll"]-inside["nll"])
        endpoints[str(direction)]={"kind":"likelihood_crossing",
            "value":inside["value"]+fraction*(outside["value"]-inside["value"]),
            "inside":inside,"outside":outside,
            "inside_delta_nll":inside["nll"]-ref["nll"],
            "outside_delta_nll":outside["nll"]-ref["nll"]}
        data["endpoints"]=endpoints
        save(path,data)
    data.update(status="complete",endpoints=endpoints,
                interval=[endpoints[str(s)]["value"] for s in (-1,1)],
                classification=("two-sided likelihood crossings" if all(e["kind"]=="likelihood_crossing"
                    for e in endpoints.values()) else "one-sided likelihood; domain boundary on other side"))
    save(path,data)
    return data


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--form",type=int,choices=(1,2),default=1)
    ap.add_argument("--steps",type=int)
    ap.add_argument("--scenario",choices=SCENARIOS,default="coupled")
    ap.add_argument("--reference",action="store_true")
    ap.add_argument("--profile")
    ap.add_argument("--refine-reference",action="store_true")
    ap.add_argument("--audit")
    ap.add_argument("--boundary",action="store_true")
    ap.add_argument("--finalize",action="store_true")
    ap.add_argument("--verify",action="store_true")
    a=ap.parse_args()
    if a.finalize:
        finalize()
    elif a.boundary:
        boundary_support(a.form,a.steps or STEPS[a.form])
    elif a.audit:
        audit_profile(a.form,a.steps or STEPS[a.form],a.audit)
    elif a.refine_reference:
        refine_reference(a.form,a.steps or STEPS[a.form])
    elif a.reference:
        run_reference(a.form,a.steps or STEPS[a.form],a.scenario)
    elif a.profile:
        run_profile(a.form,a.steps or STEPS[a.form],a.profile,a.scenario)
    elif a.verify:
        result=read(WORK/"result.json")
        if result["prompt_sha256"]!=current_inputs.digest(PROMPT):
            raise ValueError("Immutable prompt snapshot changed")
        if result["producer_sha256"]!=current_inputs.digest(Path(__file__)):
            raise ValueError("Task producer changed after finalization")
        for relative,expected in result["baseline_artifact_sha256"].items():
            if current_inputs.digest(ROOT/relative)!=expected:
                raise ValueError(f"Changed primary evidence: {relative}")
        if result.get("reconciliation_source"):
            recovery=load_module("sr06_profile_recovery",result["reconciliation_source"]["path"])
            verification=recovery.verify()
            for name in recovery.REFINED_QUANTITIES:
                selected_path=profile_path(2,51200,name)
                current={"expanded_selected":recovery.profile_summary(read(selected_path),selected_path),
                         "expanded_comparison":recovery.paired_profile_summary(name)}
                for role,summary in current.items():
                    if result["forms"]["2"]["profiles"][name][role]!=summary:
                        raise ValueError("Reconciled SR-06 result differs from current actual profile evidence")
            print(json.dumps(verification,indent=2))
            return
        verification=verify_data()
        for f,n,q in required_profiles():
            stored=result["forms"][str(f)]["profiles"][q]["expanded_selected" if n==STEPS[f] else "expanded_comparison"]
            actual=read(profile_path(f,n,q))
            if stored["interval"]!=actual["interval"] or stored["classification"]!=actual["classification"]:
                raise ValueError("Structured result interpretation differs from replayed evidence")
        print(json.dumps(verification,indent=2))
    else:
        ap.error("Select an action")


CORE=("operative_nu_pp","c_j","phi_j_deg","G_op")
FOLLOW=("c_peak","gamma_c","k_sharpness","d_fab","q_fab")


def required_profiles():
    return [(f,n,q) for f in (1,2) for n in (STEPS[f],FINE[f])
            for q in CORE+(FOLLOW if f==2 else ())]


def verify_data():
    """Replay decisive REAL residuals and admissibility, not status flags."""
    models={}
    replays={}
    largest_error=0.
    def model(f,n,s="coupled"):
        key=(f,n,s)
        if key not in models:
            models[key]=Model(*key)
        return models[key]
    def replay(m,row):
        nonlocal largest_error
        key=(m.form,m.steps,m.scenario,tuple(row["theta"]))
        if key in replays:
            return replays[key]
        # Historical witnesses retain their physical theta.  Reconstruct the
        # current zero-origin fabric coordinate instead of interpreting an
        # old affine-u vector as the new chart.
        u=m.u(row["theta"])
        np.testing.assert_allclose(np.asarray(m.theta(jnp.asarray(u))),row["theta"],rtol=2e-12,atol=2e-10)
        now=m.record(u)
        error=abs(now["nll"]-row["nll"])
        largest_error=max(largest_error,error)
        # This replay tolerance is far below the likelihood classification
        # threshold; the crossing signs themselves are tested separately.
        if error>CUT*1e-3:
            raise ValueError(f"Changed real-model likelihood: {error}")
        replays[key]=now
        return now
    references={}
    for path in sorted(WORK.glob("*_reference.json")):
        d=read(path)
        current_inputs.require_current(d["inputs"])
        if d["status"]!="complete":
            raise ValueError(f"Incomplete reference {path}")
        m=model(d["form"],d["n_steps"],d["scenario"])
        references[(d["form"],d["n_steps"],d["scenario"])]=replay(m,d["fit"])
        replay(m,d["primary_replay"])
    crossing_sides=physical_edges=0
    for f,n,q in required_profiles():
        d=read(profile_path(f,n,q))
        current_inputs.require_current(d["inputs"])
        if d.get("status")!="complete" or not d.get("audit_method"):
            raise ValueError(f"Missing decisive nuisance evidence: {f}, {n}, {q}")
        refdata=read(ref_path(f,n,"coupled"))["fit"]
        if d["reference"]["theta"]!=refdata["theta"]:
            raise ValueError(f"Stale profile reference: {f}, {n}, {q}")
        m=model(f,n)
        ref=replay(m,d["reference"])
        crossings=0
        for side in ("-1","1"):
            end=d["endpoints"][side]
            if end["kind"]=="likelihood_crossing":
                inside,outside=replay(m,end["inside"]),replay(m,end["outside"])
                if not inside["nll"]-ref["nll"]<CUT<=outside["nll"]-ref["nll"]:
                    raise ValueError(f"Crossing sign fails current replay: {f} {n} {q} {side}")
                a,b=sorted((m.value(inside,q),m.value(outside,q)))
                if not a<=end["value"]<=b:
                    raise ValueError("Reported interval is outside its real-model bracket")
                crossings+=1;crossing_sides+=1
                for r in (end["inside"],end["outside"]):
                    np.testing.assert_allclose(m.value(replay(m,r),q),r["value"],rtol=2e-7,atol=2e-8)
            elif end["kind"]=="domain_boundary" and q=="phi_j_deg" and end["value"]==0:
                boundary=replay(m,end["witness"])
                if boundary["nll"]-ref["nll"]>=CUT or abs(m.value(boundary,q))>1e-8:
                    raise ValueError("Frictionless boundary interpretation fails")
                physical_edges+=1
            else:
                raise ValueError(f"Unresolved or computational boundary: {end['kind']}")
        expected="two-sided likelihood crossings" if crossings==2 else "one-sided likelihood; domain boundary on other side"
        if d["classification"]!=expected:
            raise ValueError("Stored classification does not follow from witnesses")
        for check in d["audit_checks"]:
            fit=check["fit"]
            replay(m,fit)
            if not fit["termination"] or any(p["status"]==0 for p in fit["termination"][-1:]):
                raise ValueError("Unfinished decisive nuisance fit")
    for f in (1,2):
        for n in (STEPS[f],FINE[f]):
            d=read(WORK/f"f{f}_{n}_spd_boundary.json")
            current_inputs.require_current(d["inputs"])
            if d["status"]!="complete":
                raise ValueError("Incomplete physical-boundary continuation")
            for r in d["rows"]:
                now=replay(model(f,n),r)
                np.testing.assert_allclose(now["elastic_rho"],1-r["rho_gap"],rtol=0,atol=2e-7)
    return {"status":"passed","method":"current measured-curve residual and integrated-stiffness replay; explicit likelihood signs, engineering constraints and physical boundary checks",
        "real_model_witnesses_replayed":len(replays),"crossing_sides_checked":crossing_sides,
        "frictionless_boundary_sides_checked":physical_edges,"maximum_NLL_replay_difference":largest_error}


def finalize():
    verification=verify_data()
    result={"kind":"iterative_exec_result","schema_version":1,"status":"completed",
        "stage_id":"revision-r3-sr06-bound-effects","task":"sr06-bound-effects",
        "prompt_sha256":current_inputs.digest(PROMPT),"producer_sha256":current_inputs.digest(Path(__file__)),
        "verify_cmd":"python selfrev-r3/evidence/SR-06_bound_effects.py --verify",
        "verification":verification,"forms":{}}
    result["domains"]={
        "role":"Separate sensitivity experiments. The original declared-domain map is not replaced.",
        "elastic_release":"assembly nu_pp upper ceiling 0.45 is removed: 0.01 <= v < 1, with p,o,G>0 and 1-v-w^2*(p+o)^2/(2*p*o)>0. The lower 0.01 bound and other inactive assembly boxes remain declared calibration choices.",
        "engineering_currency":"Engineering constants are read from the inverse of the symmetrized stiffness actually integrated. Operative nu_pp is not assembly v. Normal compliance SPD is rho^2=2*nu_op^2*E_p/((1-nu_pp)*E_o)<1. G_op is invariant under the mapping.",
        "joint_release":"0 <= phi_j_deg < 60: zero friction is physically admissible; the retained 60 degree ceiling is a calibration choice, not a physical limit. Positive joint cohesion remains within the inactive original [0.1,20] MPa box.",
        "fabric_release":"q_fab has no finite lower bound; q<=8 and -1.5<=d<=1.5 remain inactive original calibration bounds. Require g_min(d,q)>0 over all lambda in [0,1]. Neither the original q=-1 box floor nor the original g_min>=0.05 computational guard is a physical admissibility boundary.",
        "fabric_continuation":"Use the actual quadratic coefficient b=q*d^2 and log(d) on the attained positive-d connected branch, with exact conversion to finite original theta. All reported ordinates use the original real forward model; no limiting or synthetic substitute is scored.",
        "guards":"Strict SPD and finite successful stress-path integration are required. rho=1 is excluded; a sequence of fixed-rho nuisance refits distinguishes approach to this open physical boundary from the old assembly ceiling. The initial q=-100 computational guard was removed when reached.",
        "other_coordinates":"Unchanged original calibration boxes and c_res/c_peak transform in theta_space.py. They are not presented as universal physical restrictions.",
        "likelihood":"Five measured curves, 70 measured stress stations per curve, measured effective-stress paths, sigma_q=1 MPa, Delta NLL=1.92; corrected CU-Z angle and the current converged shear-free mirror."}
    result["numerical_method"]={
        "objective":"Unmodified current real-model residuals and Gaussian NLL; every accepted witness has a finite converged forward path.",
        "coordinates":"Original assembly theta is retained. Exact engineering-compliance inversion regularizes elastic nuisance fitting near SPD; log(d) and b=q*d^2 regularize the extended fabric compensation direction.",
        "profiles":"Fix the named physical quantity and optimize every remaining nuisance coordinate. March connected witnesses, bracket Delta NLL=1.92, then continue the adjacent interior nuisance solution to the exterior. The d/q checks additionally start from the other compensated profile's witness.",
        "termination":"Actual solver termination, optimality and continuation records are stored with each fit. Evaluation chunks resume rather than terminate the calculation. Iteration budgets are not scientific acceptance criteria.",
        "tail_reporting":"The very remote negative-q crossing is retained as an actual sign-changing physical-coordinate bracket. Its scalar interpolation is only a location summary; paired-grid variation prevents a precise material-bound interpretation."}
    for f in (1,2):
        primary_data=read(mapping.output(f))
        block={"declared_primary_reference":primary_data["reference"],"scenario_refits":{},"profiles":{}}
        block["theta_names"]=theta_space.space_for_form(f).theta_names
        block["theta_units"]=theta_space.THETA_UNITS if f==1 else theta_space.THETA_F2_UNITS
        block["declared_primary_reference"]["engineering"]=read(ref_path(f,STEPS[f],"coupled"))["primary_replay"]["engineering"]
        for scenario in (("elastic","friction","coupled") if f==1 else ("elastic","friction","fabric","coupled")):
            d=read(ref_path(f,STEPS[f],scenario))
            block["scenario_refits"][scenario]={"fit":d["fit"],"NLL_change_from_primary":d["fit"]["nll"]-d["primary_replay"]["nll"],
                "path":ref_path(f,STEPS[f],scenario).relative_to(ROOT).as_posix()}
        block["comparison_reference"]=read(ref_path(f,FINE[f],"coupled"))["fit"]
        block["admissibility_continuation"]={}
        for n in (STEPS[f],FINE[f]):
            p=WORK/f"f{f}_{n}_spd_boundary.json"
            d=read(p)
            block["admissibility_continuation"][str(n)]={"path":p.relative_to(ROOT).as_posix(),
                "rows":[{"rho_gap":r["rho_gap"],"nll":r["nll"],"delta_nll":r["nll"]-d["reference"]["nll"],
                         "operative_nu_pp":r["engineering"]["nu_pp"],"spd_margin":r["spd_margin"],
                         "fabric_g_min":r["fabric_g_min"]} for r in d["rows"]]}
        for q in CORE+(FOLLOW if f==2 else ()):
            selected=read(profile_path(f,STEPS[f],q));fine=read(profile_path(f,FINE[f],q))
            if q=="operative_nu_pp":
                oldpath=ROOT/f"results/iterative/sr03-operative-elastic/f{f}_{STEPS[f]}_profile_nu_pp.json"
                old=read(oldpath)
                baseline={"interval":old["interval"],"classification":old["classification"],
                          "reference":old["reference"]["engineering"]["nu_pp"],"source":oldpath.relative_to(ROOT).as_posix()}
            else:
                old=primary_data["per_parameter"][q]
                baseline={"interval":old["interval_internal_units"],"classification":old["classification"],
                          "reference":primary_data["reference"]["theta"][theta_space.space_for_form(f).theta_names.index(q)],
                          "source":mapping.output(f).relative_to(ROOT).as_posix()}
            def summary(d):
                endings={}
                for side,e in d["endpoints"].items():
                    if e["kind"]=="likelihood_crossing":
                        endings[side]={"kind":e["kind"],"bracket":sorted((e["inside"]["value"],e["outside"]["value"])),
                            "inside_delta_nll":e["inside"]["nll"]-d["reference"]["nll"],
                            "outside_delta_nll":e["outside"]["nll"]-d["reference"]["nll"]}
                    else:
                        endings[side]={"kind":e["kind"],"value":e["value"],"delta_nll":e["witness"]["nll"]-d["reference"]["nll"]}
                return {"interval":d["interval"],"classification":d["classification"],"endpoints":endings,
                        "witness_path":profile_path(f,d["n_steps"],q).relative_to(ROOT).as_posix()}
            block["profiles"][q]={"primary":baseline,"expanded_selected":summary(selected),"expanded_comparison":summary(fine)}
            block["profiles"][q]["integration_support"]={
                "classification_agrees":selected["classification"]==fine["classification"],
                "endpoint_change_comparison_minus_selected":[b-a for a,b in zip(selected["interval"],fine["interval"])],
                "interpretation":"Compare the physical brackets; extreme fabric lower endpoints are poorly resolved across grids and are not precise material bounds." if q in ("q_fab","d_fab") else
                                 "Selected and comparison grids directly test the retained classification and the material domain-effect conclusion."}
        result["forms"][str(f)]=block
    result["findings"]=[
        "Active calibration bounds truncate consequential nuisance compensation on both forms. Expanded-domain fits and intervals cannot be described as reproducing the original material-parameter estimates.",
        "Operative in-plane Poisson intervals shift upward and broaden substantially. Assembly nu_pp=0.45 was not a physical SPD limit. The expanded fits/profile endpoints approach the normal-compliance SPD boundary; the fixed-rho continuation retains strictly admissible real-model witnesses.",
        "The joint-friction floor moves from the arbitrary 5 degree box to physically admissible zero friction. Joint cohesion remains two-sided but its interval shifts upward on both forms. Shear modulus retains two crossings on both selected and comparison grids, with shifted estimates and intervals.",
        "Form-II fabric compensation is strongly coupled: very negative q is admissible as d decreases while q*d^2 remains finite. The former narrow d interval and one-sided q bound are conditional on q>=-1. Extreme expanded lower endpoints must be read with their actual brackets and paired-grid variation, not as well-resolved material constants.",
        "The consequential form-II matrix peak cohesion and softening coordinates were also profiled after their references moved outside the old narrow intervals. Their expanded intervals and paired-grid evidence are reported in physical units.",
        "This is a separate domain-sensitivity experiment. No robustness claim for the full original 10/12 and 11/14 identified counts follows from this targeted set, and no expanded-domain total is asserted."]
    result["parent_dependencies"]=[
        {"paths":["paper/sections/03_methods.tex:subsec:inversion","paper/sections/04_results.tex:subsec:idmap,tab:params,tab:app:elastic",
                  "paper/sections/05_discussion.tex:subsec:disc-idmap,subsec:disc-practice","selfrev-answers.md:assembly/operative Poisson entry"],
         "work":"Add this separate sensitivity result in matching engineering currency, explain physical SPD/friction admissibility versus calibration boxes, and condition material-constant and identified-count interpretations on the declared domain. Preserve primary declared-domain numbers unless the parent deliberately chooses a new primary experiment."}]
    result["shared_producer_defect"]=None
    decisive=[]
    for f,n,q in required_profiles():
        for e in read(profile_path(f,n,q))["endpoints"].values():
            decisive.extend([e["inside"],e["outside"]] if e["kind"]=="likelihood_crossing" else [e["witness"]])
    result["guard_observations"]={
        "minimum_global_fabric_scale_at_decisive_witnesses":min(r["fabric_g_min"] for r in decisive if r["fabric_g_min"] is not None),
        "forward_fabric_clamp":float(mirror.ref.G_FLOOR),
        "other_active_calibration_bounds_at_decisive_witnesses":sorted({a["name"] for r in decisive for a in r["active_constraints"]
            if a["name"] not in ("phi_j_deg","nu_pp","q_fab")}),
        "meaning":"The realized witnesses distinguish global fabric positivity from both the 0.05 admission guard and the forward clamp. Nuisance approach to elastic SPD is separately retained, not relabeled as a calibration-box edge."}
    result["runtime"]={"python":sys.version.split()[0],"numpy":np.__version__,"jax":jax.__version__,
                       "scipy":__import__("scipy").__version__,"backend":"cpu","floating_point":"float64"}
    result["unresolved_dependencies"]=[]
    result["primary_replacement_required"]=False
    paths=[Path(__file__),*sorted(WORK.glob("*.json"))]
    paths=[p for p in paths if p.name!="result.json"]+[WORK/"result.json"]
    result["result_paths"]=[p.relative_to(ROOT).as_posix() for p in paths]
    result["changed_files"]=result["result_paths"]
    result["input_identity"]={str(f):identity(f,STEPS[f]) for f in (1,2)}
    result["baseline_artifact_sha256"]={p.relative_to(ROOT).as_posix():current_inputs.digest(p)
        for p in [mapping.output(1),mapping.output(2),*[ROOT/f"results/iterative/sr03-operative-elastic/f{f}_{STEPS[f]}_profile_nu_pp.json" for f in (1,2)]]}
    save(WORK/"result.json",result)
    print(json.dumps({"status":"completed","verification":verification},indent=2))


if __name__ == "__main__":
    main()
