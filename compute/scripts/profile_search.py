"""Compare nuisance starts and propagate an improved likelihood reference."""
from __future__ import annotations


class ReferenceImproved(RuntimeError):
    def __init__(self, reference):
        self.reference = reference
        super().__init__("A profile found a lower reference; rebuild the connected profiles")


def compare_starts(fit_one, seeds):
    attempts, valid, seen = [], [], set()
    for label, seed in seeds:
        key = tuple(seed["u"])
        if key in seen:
            continue
        seen.add(key)
        try:
            fit = fit_one(seed)
        except ValueError as exc:
            attempts.append({"start": label, "status": "failed", "error": str(exc)})
            continue
        attempts.append({"start": label, "status": "terminated", "nll": fit["nll"]})
        valid.append(fit)
    if not valid:
        raise ValueError(f"All nuisance starts failed: {attempts}")
    return {"fit": min(valid, key=lambda row: row["nll"]), "attempts": attempts}


def update_reference(reference, candidate, unconstrained_fit):
    if candidate["nll"] >= reference["nll"]:
        return reference
    fit = unconstrained_fit(candidate)
    if fit["solver_status"] == "failed":
        raise ValueError(f"Unconstrained reference continuation failed: {fit}")
    return min((candidate, fit["best"]), key=lambda row: row["nll"])
