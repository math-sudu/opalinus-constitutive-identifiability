"""Engineering constants of the stiffness actually integrated by the mirror.

Directions use Nordas' convention: ``nu_op`` is in-plane transverse strain
under normal loading, and ``nu_po`` is normal strain under in-plane loading.
The input theta still contains the original assembly coordinates.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

NAMES = ("E_p", "E_o", "G_op", "nu_pp", "nu_po", "nu_op", "G_pp", "E_p_over_E_o")
SCALES = np.array([1e4, 5e3, 2e3, 1., 1., 1., 5e3, 1.])


def engineering(theta):
    """Differentiable scalar inverse of the symmetrised TI stiffness.

    H=C11+C12, K=C11-C12, F=C13, C=C33. The normal-block
    determinant is H*C-2*F**2. No engineering-Voigt shear factor enters
    these normal compliances; G_op is unchanged by the assembly.
    """
    p, o, g, v, w = theta[:5]
    delta = 1-v-2*w*w
    H, K = p/delta, p/(1+v)
    F, C = w*(p+o)/(2*delta), (1-v)*o/delta
    determinant = H*C-2*F*F
    s11 = (C/determinant+1/K)/2
    s12 = (C/determinant-1/K)/2
    s13, s33 = -F/determinant, H/determinant
    ep, eo = 1/s11, 1/s33
    return jnp.stack((ep, eo, g, -s12/s11, -s13/s11, -s13/s33,
                      K/2, ep/eo))


def positive_definiteness_margin(theta):
    """Normal-block SPD margin; positive moduli and -1<v<1 are separate."""
    p, o, _, v, w = theta[:5]
    return 1-v-w*w*(p+o)**2/(2*p*o)


def assembly_candidates(*, E_p, E_o, G_op, nu_pp, nu_op):
    """Invert five independent engineering constants without changing the law.

    Enumerate the real positive roots for r=o/p of
    ((H+K)*t-2*K*r)*(1+r)**2 - 2*H*h**2*r = 0,
    where t=C/H, h=2*F/H. Then v=1-t/r, p=K*(1+v),
    o=r*p and w=h/(1+r). Candidates are returned in assembly order
    (p,o,G_op,v,w). The caller must enforce its declared assembly domain;
    physical target constants do not guarantee membership of that box.
    """
    if min(E_p, E_o, G_op) <= 0 or not -1 < nu_pp < 1:
        raise ValueError("Non-positive modulus or inadmissible in-plane ratio")
    S = np.array([[1/E_p, -nu_pp/E_p, -nu_op/E_o],
                  [-nu_pp/E_p, 1/E_p, -nu_op/E_o],
                  [-nu_op/E_o, -nu_op/E_o, 1/E_o]])
    if np.linalg.eigvalsh(S).min() <= 0:
        raise ValueError("Supplied engineering compliance is not positive definite")
    stiffness = np.linalg.inv(S)
    H = stiffness[0, 0]+stiffness[0, 1]
    K = stiffness[0, 0]-stiffness[0, 1]
    C, F = stiffness[2, 2], stiffness[0, 2]
    t, h = C/H, 2*F/H
    A = (H+K)*t
    roots = np.roots([-2*K, A-4*K, 2*A-2*K-2*H*h*h, A])
    result = []
    for root in roots:
        if abs(root.imag) > 64*np.finfo(float).eps*max(1, abs(root.real)) or root.real <= 0:
            continue
        r = root.real
        v = 1-t/r
        p = K*(1+v)
        o, w = r*p, h/(1+r)
        candidate = np.array([p, o, G_op, v, w])
        if p > 0 and o > 0 and -1 < v < 1 and 1-v-2*w*w > 0:
            result.append(candidate)
    if not result:
        raise ValueError("No positive-modulus assembly inverse")
    return result
