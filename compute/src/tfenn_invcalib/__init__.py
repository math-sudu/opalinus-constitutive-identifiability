"""tfenn_invcalib: inverse-calibration core stack for the tfenn-invcalib paper.

Deep-ensemble multi-start gradient inversion of the 12 ESUBI-CWFS
constitutive parameters from 5 digitised measured candidate-sites Opalinus
triaxial curves (3 CU + 2 CD), through a differentiable JAX mirror of the
frozen NumPy driver (src/tfenn_esubi/data_gen/esubi_driver.py, absorbed base,
read-only). The frozen constitutive form is reproduced exactly -- no
smoothing, no regularization, no added damping.

Modules:
  mirror         differentiable JAX mirror of the frozen driver (provenance:
                 G0a probe, source commit 29ab3d5) + forward-equivalence check
  theta_space    12-parameter convention, nominal loading, optimization box,
                 z <-> theta transform, G0a admissibility floor, start sampling
  curves         measured-curve loading, exact station alignment, LOO folds,
                 synthetic-truth curves
  likelihood     NLL data misfit with the fixed CU systematic-offset term and
                 the budget-floored noise model
  ensemble       batched multi-start adam inversion + ensemble-based
                 approximate credible bands
  oracle         best-attainable-fit oracles: all-curve, per-fold four-curve,
                 and the fixed-delta-objective sensitivity lane
  optimum        the attained optimum theta*, lowest NLL over both optimisers
                 with a derivative-free polish; read by every instrument below
  identifiability sensitivity matrices, Fisher information, Cramer-Rao bounds
                 and null space at the attained optimum, with their
                 verifications
  profile_likelihood  per-coordinate profile likelihood over the full prior
                 box, its intervals, prior-edge flags and classification
  decomposition  four-bucket gap decomposition on one offset footing, signed
                 and unclipped (pure function + deterministic unit check)
  coverage       synthetic-truth coverage harness for the ensemble bands
  baseline       classical derivative-free staged-calibration lane
  runio          seeds, sha256 fingerprints, config echo, effort accounting,
                 JSON output helpers

Environment pins: CPU JAX with x64 (set before any jax import below).
"""

import os as _os

_os.environ.setdefault("JAX_PLATFORMS", "cpu")

from . import runio          # noqa: E402,F401
from . import mirror         # noqa: E402,F401  (enables jax x64 on import)
from . import theta_space    # noqa: E402,F401
from . import curves         # noqa: E402,F401
from . import likelihood     # noqa: E402,F401
from . import ensemble       # noqa: E402,F401
from . import oracle         # noqa: E402,F401
from . import decomposition  # noqa: E402,F401
from . import coverage       # noqa: E402,F401
from . import baseline       # noqa: E402,F401
from . import optimum        # noqa: E402,F401
from . import identifiability      # noqa: E402,F401
from . import profile_likelihood   # noqa: E402,F401

__version__ = "0.1.0"
