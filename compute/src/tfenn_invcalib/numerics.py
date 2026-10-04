"""Absolute finite differences in the existing bounded internal coordinates."""
from __future__ import annotations

import numpy as np


def solver_bounds_u(space):
    """Existing box with adjacent representable, strictly admissible angles."""
    lower, upper = np.zeros(space.n_theta), np.ones(space.n_theta)
    for name in ("phi_deg", "phi_j_deg"):
        i = space.internal_names.index(name)
        lo, hi = space.box_lo[i], space.box_hi[i]
        lower[i] = (np.nextafter(lo, hi) - lo) / (hi - lo)
        upper[i] = (np.nextafter(hi, lo) - lo) / (hi - lo)
    return lower, upper


def absolute_fd_jacobian(fun, x, step, lower, upper):
    """Three-point derivative with absolute steps and inward boundary stencils.

    The step is in unit-box coordinates, never multiplied by abs(x). Use a
    centred stencil where both probes fit, otherwise two inward probes.
    Actual representable offsets determine the weights. Bounds are supplied
    by the caller, including any strictly open physical angle endpoints.
    """
    x = np.asarray(x, dtype=float)
    lower, upper = np.broadcast_arrays(lower, upper, x)[:2]
    steps = np.broadcast_to(np.asarray(step, dtype=float), x.shape)
    if (not np.isfinite(x).all() or not np.isfinite(steps).all()
            or np.any(steps <= 0) or np.any(lower >= upper)
            or np.any(x < lower) or np.any(x > upper)):
        raise ValueError("Finite in-bound coordinates and positive absolute steps required")
    f0 = np.asarray(fun(x), dtype=float).ravel()
    columns = []
    for i, requested in enumerate(steps):
        left, right = x[i] - lower[i], upper[i] - x[i]
        if min(left, right) >= requested:
            offsets = (-requested, requested)
        else:
            direction = 1 if right >= left else -1
            h = min(requested, max(left, right) / 2)
            offsets = (direction * h, direction * 2 * h)
        probes = []
        for offset in offsets:
            probe = x.copy()
            probe[i] = np.clip(x[i] + offset, lower[i], upper[i])
            probes.append((probe[i] - x[i], np.asarray(fun(probe), dtype=float).ravel()))
        (a, fa), (b, fb) = probes
        if a == 0 or b == 0 or a == b:
            raise ValueError("Finite-difference stencil collapses at machine precision")
        columns.append((b * (fa - f0) / a - a * (fb - f0) / b) / (b - a))
    jac = np.column_stack(columns)
    if not np.isfinite(f0).all() or not np.isfinite(jac).all():
        raise ValueError("Nonfinite residual or finite-difference derivative")
    return jac
