"""Resumable bounded least squares on the actual selected model residuals."""
from __future__ import annotations

import time
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares


def fit(model, theta, train=None, fixed=None, checkpoint=None, chunk=100):
    base = np.clip(np.asarray(model.space.u_from_theta(np.asarray(theta))),
                   model.lower, model.upper)
    if fixed is not None:
        base[fixed[0]] = fixed[1]
    unobserved = getattr(model, "unobserved_coordinates", ())
    free = np.array([i for i in range(len(base))
                     if i not in unobserved and (fixed is None or i != fixed[0])])
    if train is None:
        train = np.arange(model.cs.q_meas.size)
    const = len(train) * model.constant / model.cs.q_meas.size
    best = None
    calls = rejected = 0
    started = time.monotonic()

    def unpack(x):
        u = base.copy()
        u[free] = x
        return u

    def residual(x):
        nonlocal best, calls, rejected
        calls += 1
        u = unpack(x)
        th = np.asarray(model.space.theta_from_u(jnp.asarray(u)))
        valid, _ = model.space.is_admissible(th)
        r = np.asarray(model.r(jnp.asarray(u)))[train] if valid else None
        if r is None or not np.isfinite(r).all():
            rejected += 1
            if best is None:
                raise ValueError("Invalid initial calibration path")
            # This is a rejected trust-region trial, never a model likelihood.
            return np.full(len(train), 1e100)
        nll = float(r @ r / 2 + const)
        if best is None or nll < best["nll"]:
            best = {"u": u.tolist(), "theta": th.tolist(), "nll": nll}
            if checkpoint:
                checkpoint(best)
        return r

    def jacobian(x):
        jac = np.asarray(model.j(jnp.asarray(unpack(x))))[train][:, free]
        if not np.isfinite(jac).all():
            raise ValueError("Nonfinite calibration sensitivity")
        with np.errstate(over="raise", invalid="raise"):
            np.sum(jac**2, axis=0)
        return jac

    initial = None
    total_nfev = 0
    try:
        residual(base[free])
        initial = best["nll"]
        x = base[free]
        while True:
            # Inactive yield branches leave exact null directions; LSMR's
            # regularized step avoids the dense solve's slow drift along them.
            # Its two-dimensional subspace is unavailable with one free variable.
            sol = least_squares(residual, x, jac=jacobian, x_scale="jac",
                                tr_solver="lsmr" if len(free) > 1 else "exact",
                                bounds=(model.lower[free], model.upper[free]),
                                max_nfev=chunk)
            total_nfev += sol.nfev
            if sol.status != 0:
                break
            # An evaluation chunk is a checkpoint interval, not termination.
            x = sol.x
        termination = {"solver_status": int(sol.status), "nfev": int(total_nfev),
                       "solver_message": str(sol.message),
                       "optimality": float(sol.optimality)}
    except (ValueError, FloatingPointError) as exc:
        termination = {"solver_status": "failed", "error": str(exc)}
    return {"best": best, "initial_nll": initial, **termination,
            "residual_calls": calls, "rejected_trials": rejected,
            "seconds": time.monotonic() - started}
