"""Four-bucket gap decomposition (pure function + unit check).

Splits each leave-one-curve-out held-out residual, in RMSE_q units (MPa),
into four shares that sum to the nominal-parameter null exactly:

    RMSE_null = b1 + b2 + b_gen + b3

    b1    = RMSE_null    - RMSE_post      parameter-reducible
    b2    = RMSE_post    - RMSE_oracle4   optimiser-attributable
    b_gen = RMSE_oracle4 - RMSE_oracle5   generalisation
    b3    = RMSE_oracle5                  form-irreducible, conditional on the
                                          form at the attained optimum

where, all on the SAME held-out curve,
  RMSE_null     nominal-parameter forward null,
  RMSE_post     the fold ensemble's posterior-predictive,
  RMSE_oracle4  the fold's four-curve oracle -- the best fit attainable on
                exactly the data and objective the fold's ensemble had,
  RMSE_oracle5  the five-curve attained optimum, the best over both
                optimisers.

Why four and not three. With only a five-curve oracle, the middle share
differences a fit that saw four curves against a fit that saw five, so it adds
the cost of predicting an unseen curve to the optimiser gap it is named for.
The four-curve oracle removes that: b2 compares two fits with identical
information, so it is an optimiser gap, and b_gen carries the generalisation
cost on its own.

Two rules, stated rather than implied
-------------------------------------
*Scoring.* All four quantities are scored with the CU offset FIXED at the
budget midpoint mu_d -- the held-out rule the null and the posterior-predictive
already use. Previously the oracle alone was scored with its offset profiled,
so it was handed a free parameter the other two lanes were denied, which
understated the irreducible share and overstated the optimiser share by
construction. Each fit still uses the paper's own likelihood (profiled offset)
on its own calibration set: what is made common is the scoring rule, not the
model. Driven along the measured effective-stress path there is no offset
to fix or to profile, so every lane is already on one footing and the scoring
rule the artifact reports says that instead.

*No clipping.* No share is clipped at zero. Every share is reported signed, so
the sum identity holds exactly and unconditionally -- which is the honest form
of the condition the identity needs: it holds because nothing is clipped, and
the earlier three-bucket identity failed exactly when a clip fired. A negative
share is reported as measured, with its reading:
  b1 < 0     the calibrated posterior predicts the held-out curve WORSE than
             the nominal null;
  b2 < 0     the fold's posterior-predictive beats the four-curve oracle on the
             held-out curve -- an ensemble-median effect, not a better fit, and
             a sign the oracle's own attainment should be checked;
  b_gen < 0  the four-curve oracle predicts the held-out curve better than the
             five-curve optimum does, i.e. fitting the held-out curve too
             traded held-out accuracy away.

"Dominant bucket" is defined once and here: the share of LARGEST MAGNITUDE
among the four for that fold, under this scoring rule.

Resolution labelling: a share is ``noise_dominated`` when its MAGNITUDE lies
below the 0.6 MPa frozen-driver end-of-step transient noise floor (noise-budget
entry); magnitude, not signed value, because the shares are signed here. The
headline claim margin is applied at claim level, not inside this function.
"""

from __future__ import annotations

import json
from pathlib import Path

from .runio import CLAIM_CONTRACT_REL, NOISE_BUDGET_REL

FORM_CONDITION_LABEL = ("conditional on the frozen ESUBI-CWFS form family "
                        "at the attained optimum")
FORM2_CONDITION_LABEL = ("conditional on the fabric-scaled form-II "
                         "(anisotropic-matrix) family at the attained optimum")

SCORING_RULE_LABEL = ("every share scored on the held-out curve with the CU "
                      "offset fixed at the budget midpoint mu_d (the held-out "
                      "rule), for all four quantities alike")

NO_OFFSET_SCORING_RULE_LABEL = ("every share scored on the held-out curve with "
                                "the undrained curves driven along the measured "
                                "effective-stress path, so there is no offset "
                                "to fix or to profile and all four quantities "
                                "are already on one footing")

BUCKET_KEYS = ("parameter_reducible", "optimizer_attributable",
               "generalization", "form_irreducible")
BUCKET_SYMBOLS = {"parameter_reducible": "b1", "optimizer_attributable": "b2",
                  "generalization": "b_gen", "form_irreducible": "b3"}


def four_bucket_decomposition(rmse_null_MPa: float,
                              rmse_posterior_MPa: float,
                              rmse_oracle4_MPa: float,
                              rmse_oracle5_MPa: float,
                              noise_floor_MPa: float,
                              condition_label: str = FORM_CONDITION_LABEL,
                              scoring_rule_label: str = SCORING_RULE_LABEL) -> dict:
    """Pure signed four-bucket split of one fold's measured-holdout gap.

    All four inputs must already be on the fixed-delta scoring rule; the
    function does not rescore and cannot detect a mixed-rule input, so the
    producer is responsible for feeding it one footing (see SCORING_RULE_LABEL).
    ``condition_label`` names the form family the form-irreducible share is
    conditional on; ``scoring_rule_label`` names the footing the caller put the
    four inputs on, which differs between the two drives.
    """
    b1 = rmse_null_MPa - rmse_posterior_MPa
    b2 = rmse_posterior_MPa - rmse_oracle4_MPa
    b_gen = rmse_oracle4_MPa - rmse_oracle5_MPa
    b3 = rmse_oracle5_MPa

    shares = {"parameter_reducible": b1, "optimizer_attributable": b2,
              "generalization": b_gen, "form_irreducible": b3}
    total = b1 + b2 + b_gen + b3
    residual = total - rmse_null_MPa

    def _flags(name: str, share: float) -> dict:
        return {
            "symbol": BUCKET_SYMBOLS[name],
            "share_MPa": share,
            "sign": ("negative" if share < 0.0 else "non-negative"),
            "noise_dominated": abs(share) < noise_floor_MPa,
        }

    out_shares = {name: _flags(name, shares[name]) for name in BUCKET_KEYS}
    out_shares["optimizer_attributable"]["bounded_by"] = (
        "the fold's four-curve oracle: the same calibration curves and the "
        "same objective the fold ensemble had")
    out_shares["generalization"]["measures"] = (
        "the cost of never having seen the held-out curve: four-curve oracle "
        "minus five-curve attained optimum, both on the held-out curve")
    out_shares["form_irreducible"]["conditional_on_attained_optimum"] = True
    out_shares["form_irreducible"]["condition_label"] = condition_label

    dominant = max(BUCKET_KEYS, key=lambda k: abs(shares[k]))
    return {
        "inputs": {
            "rmse_null_MPa": rmse_null_MPa,
            "rmse_posterior_MPa": rmse_posterior_MPa,
            "rmse_oracle4_MPa": rmse_oracle4_MPa,
            "rmse_oracle5_MPa": rmse_oracle5_MPa,
            "noise_floor_MPa": noise_floor_MPa,
        },
        "scoring_rule": scoring_rule_label,
        "clipping": "none; every share is reported signed",
        **out_shares,
        "shares_sum_MPa": total,
        "identity_residual_MPa": residual,
        "sum_matches_null": abs(residual) <= 1e-12 * max(1.0, abs(rmse_null_MPa)),
        "dominant_bucket": dominant,
        "dominant_bucket_definition": ("the share of largest magnitude among "
                                       "the four, for this fold, under this "
                                       "scoring rule"),
        "dominant_bucket_share_MPa": shares[dominant],
    }


def load_preregistered_thresholds(root: str | Path) -> dict:
    """Noise floor + claim margin read from the frozen project artifacts."""
    root = Path(root)
    budget = json.loads((root / NOISE_BUDGET_REL).read_text(encoding="utf-8"))
    entries = {e["name"]: e for e in budget["entries"]}
    floor = float(entries["frozen_driver_end_of_step_transient_noise_floor"]["value_MPa"])
    contract = json.loads((root / CLAIM_CONTRACT_REL).read_text(encoding="utf-8"))
    claims = {c["claim_id"]: c for c in contract["claims"]}
    margin = claims["identifiability-map-posterior"]["target_relevance"]["discriminating_margin"]
    c5_margin = claims["c5-gap-decomposition"]["target_relevance"]["discriminating_margin"]
    return {
        "noise_floor_MPa": floor,
        "identifiability_margin_threshold_MPa": float(margin["threshold_value"]),
        "identifiability_margin_comparator": margin["comparator"],
        "c5_share_threshold_MPa": float(c5_margin["threshold_value"]),
        "sources": {
            "noise_floor": str(root / NOISE_BUDGET_REL),
            "margins": str(root / CLAIM_CONTRACT_REL),
        },
    }


def unit_check(scoring_rule_label: str = SCORING_RULE_LABEL) -> dict:
    """Deterministic unit check of the pure share computation.

    Hand-constructed residual inputs; expected values follow the definition
    above exactly. The cases deliberately include the two negative shares the
    decomposition must report rather than clip -- a negative ``b1`` (the
    calibrated posterior predicts the held-out curve worse than the nominal
    null, already observed on a real fold) and a negative ``b_gen`` (the
    four-curve oracle predicts the held-out curve better than the five-curve
    optimum) -- and an all-negative-middle case, because the sum identity is
    only unconditional if it survives them. Returns {"passes": bool, ...}.
    """
    tol = 1e-12
    cases = []
    ok_all = True

    def _record(name: str, got: dict, expect: dict, checks: list[tuple[str, bool]]):
        nonlocal ok_all
        passed = all(flag for _, flag in checks)
        ok_all = ok_all and passed
        cases.append({
            "name": name,
            "inputs": got["inputs"],
            "expected": expect,
            "got_shares": {k: got[k]["share_MPa"] for k in BUCKET_KEYS},
            "dominant_bucket": got["dominant_bucket"],
            "identity_residual_MPa": got["identity_residual_MPa"],
            "checks": [{"check": c, "ok": bool(f)} for c, f in checks],
            "passed": passed,
        })

    def _sum_checks(d: dict, null: float) -> list[tuple[str, bool]]:
        return [
            ("shares sum to null exactly", d["sum_matches_null"]),
            ("identity residual is zero", abs(d["identity_residual_MPa"]) < tol),
        ]

    # Case A: all four shares positive; the parameter share dominates.
    d = four_bucket_decomposition(10.0, 4.0, 3.5, 3.0, 0.6)
    _record("A_all_positive", d, {"b1": 6.0, "b2": 0.5, "b_gen": 0.5, "b3": 3.0},
            [("b1 == null - post", abs(d["parameter_reducible"]["share_MPa"] - 6.0) < tol),
             ("b2 == post - oracle4", abs(d["optimizer_attributable"]["share_MPa"] - 0.5) < tol),
             ("b_gen == oracle4 - oracle5", abs(d["generalization"]["share_MPa"] - 0.5) < tol),
             ("b3 == oracle5", abs(d["form_irreducible"]["share_MPa"] - 3.0) < tol),
             ("b2 noise_dominated (0.5 < 0.6)",
              d["optimizer_attributable"]["noise_dominated"] is True),
             ("dominant bucket is b1", d["dominant_bucket"] == "parameter_reducible"),
             ("b3 conditional on the attained optimum",
              d["form_irreducible"]["conditional_on_attained_optimum"] is True),
             *_sum_checks(d, 10.0)])

    # Case B: NEGATIVE b1 -- the calibrated posterior predicts the held-out
    # curve worse than the nominal null (observed on the form-II CD-P fold).
    d = four_bucket_decomposition(4.0, 4.863, 4.5, 4.2, 0.6)
    _record("B_negative_b1", d,
            {"b1": -0.863, "b2": 0.363, "b_gen": 0.3, "b3": 4.2},
            [("b1 == -0.863", abs(d["parameter_reducible"]["share_MPa"] + 0.863) < 1e-9),
             ("b1 reported signed, not clipped",
              d["parameter_reducible"]["share_MPa"] < 0.0),
             ("b1 sign flagged negative",
              d["parameter_reducible"]["sign"] == "negative"),
             ("dominant bucket is b3", d["dominant_bucket"] == "form_irreducible"),
             *_sum_checks(d, 4.0)])

    # Case C: NEGATIVE b_gen -- the four-curve oracle predicts the held-out
    # curve better than the five-curve optimum does.
    d = four_bucket_decomposition(9.0, 5.0, 3.0, 3.8, 0.6)
    _record("C_negative_b_gen", d,
            {"b1": 4.0, "b2": 2.0, "b_gen": -0.8, "b3": 3.8},
            [("b_gen == -0.8", abs(d["generalization"]["share_MPa"] + 0.8) < 1e-9),
             ("b_gen reported signed, not clipped",
              d["generalization"]["share_MPa"] < 0.0),
             ("b_gen sign flagged negative",
              d["generalization"]["sign"] == "negative"),
             ("b_gen not noise_dominated (|-0.8| > 0.6)",
              d["generalization"]["noise_dominated"] is False),
             ("dominant bucket is b1", d["dominant_bucket"] == "parameter_reducible"),
             *_sum_checks(d, 9.0)])

    # Case D: both middle shares negative at once -- the identity must still
    # close, which is the whole point of removing the clip.
    d = four_bucket_decomposition(6.0, 6.5, 7.0, 6.4, 0.6)
    _record("D_negative_b1_and_b2_and_b_gen", d,
            {"b1": -0.5, "b2": -0.5, "b_gen": 0.6, "b3": 6.4},
            [("b1 == -0.5", abs(d["parameter_reducible"]["share_MPa"] + 0.5) < 1e-9),
             ("b2 == -0.5", abs(d["optimizer_attributable"]["share_MPa"] + 0.5) < 1e-9),
             ("b_gen == 0.6", abs(d["generalization"]["share_MPa"] - 0.6) < 1e-9),
             ("dominant bucket is b3", d["dominant_bucket"] == "form_irreducible"),
             *_sum_checks(d, 6.0)])

    # Case E: a negative share whose MAGNITUDE is below the noise floor is
    # noise-dominated (the flag reads magnitude, not signed value).
    d = four_bucket_decomposition(5.0, 5.2, 4.9, 4.5, 0.6)
    _record("E_negative_below_noise_floor", d,
            {"b1": -0.2, "b2": 0.3, "b_gen": 0.4, "b3": 4.5},
            [("b1 == -0.2", abs(d["parameter_reducible"]["share_MPa"] + 0.2) < 1e-9),
             ("b1 noise_dominated by magnitude",
              d["parameter_reducible"]["noise_dominated"] is True),
             *_sum_checks(d, 5.0)])

    return {"passes": ok_all, "n_cases": len(cases),
            "scoring_rule": scoring_rule_label,
            "clipping": "none", "cases": cases}
