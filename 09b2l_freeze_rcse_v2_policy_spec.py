#!/usr/bin/env python
"""
09b2l_freeze_rcse_v2_policy_spec.py

Writes the frozen RCSE v2 policy specification used before Step 09b.2m+.
The content is intentionally fixed; do not edit parameters after observing
RCSE v2 simulation results.
"""
from pathlib import Path

SPEC_JSON = r"""{
  "step": "09b.2l",
  "name": "RCSE_V2_FROZEN_POLICY_SPEC",
  "status": "FROZEN_BEFORE_RCSE_V2_SIMULATION",
  "analysis_status": "Secondary revised analysis motivated by diagnostics after the original RCSE v1 simulation. Must be reported as such; it does not replace the original frozen v1 results.",
  "architecture": {
    "evidence_gate": {
      "name": "EVIDENCE_GATE_V2",
      "states": [
        "VALID",
        "INCOMPLETE",
        "CONTRADICTORY"
      ],
      "precedence": "CONTRADICTORY overrides INCOMPLETE; otherwise VALID",
      "value_consistency_delta": 0.05,
      "delta_rationale": "0.05 is frozen because it equals the minimum contradiction strength already present in the controlled benchmark design; it is not selected as the best-performing test threshold.",
      "incomplete_rules": [
        "po_value_initial_eur is missing",
        "invoice_value_t0_eur is missing",
        "goods_receipt_required=True AND gr_available_at_t0=False",
        "vendor_invoice_seen_at_t0=False",
        "po_created_at_t0=False"
      ],
      "process_consistency_rules": [
        "gr_available_at_t0=False AND n_goods_receipts_before_t0>0",
        "gr_available_at_t0=True AND n_goods_receipts_before_t0==0",
        "had_gr_cancellation_before_t0=True AND n_cancel_gr_before_t0==0",
        "had_invoice_cancellation_before_t0=True AND n_cancel_invoice_before_t0==0",
        "had_price_change_before_t0=True AND n_price_change_before_t0==0",
        "had_quantity_change_before_t0=True AND n_quantity_change_before_t0==0",
        "had_payment_block_before_t0=True AND (n_set_payment_block_before_t0+n_remove_payment_block_before_t0)==0"
      ],
      "value_consistency_rule": "inv_po_abs_rel_diff >= 0.05 OR (gr_available_at_t0=True AND inv_gr_abs_rel_diff >= 0.05)",
      "guardrail": "inv_po_abs_rel_diff and inv_gr_abs_rel_diff are used only as deterministic evidence-consistency checks, not as learned historical REVIEW labels or learned safety-model features."
    },
    "t0_execution_safety_belief": {
      "model": "Dedicated D1_CORE_NATURAL binary safety estimator",
      "primary_score": "p_safe_t0_cal",
      "raw_score": "p_safe_t0_raw",
      "positive_class": "SAFE_TO_EXECUTE",
      "training_scope": "Natural D1-resolved frozen source cases only",
      "synthetic_labels_used_for_training": false,
      "future_outcomes_used_as_features": false,
      "exposure_or_risk_used_as_features": false,
      "rejected_mismatch_fields_used_as_learned_features": false,
      "interpretation": "Calibrated safety probability is used as the primary execution risk belief because the RCSE expected-loss rule requires a probability-like belief. OOD calibration degradation is evaluated as an empirical robustness limitation; no R4-specific override is allowed."
    },
    "post_gather_safety_belief": {
      "score": "p_safe_t1_for_primary_rcse",
      "role": "Post-GATHER selective execution belief",
      "universal_calibration_claim": false
    },
    "planning_baseline": {
      "frozen_multiclass_model_modified": false,
      "role": "The original frozen multiclass action model remains an action/planning baseline and comparator. Its p(EXECUTE) is not used as the RCSE v2 execution-safety probability."
    }
  },
  "consequence": {
    "primary": "LOG_TRAIN_P90",
    "formula": "C(v)=log1p(|v|)/log1p(P90_train_exposure)",
    "normalization": "source-case-level TRAIN-only exposure percentile per regime",
    "sensitivity": [
      "UNIFORM",
      "LOG_TRAIN_P50",
      "LOG_TRAIN_P99"
    ]
  },
  "costs": {
    "reference": {
      "c_G": 0.02,
      "c_H": 0.1,
      "c_A": 0.5,
      "rho_H": 0.0
    },
    "inheritance": "Reference costs are unchanged from the frozen RCSE v1 specification to preserve comparability."
  },
  "risk_and_selectivity": {
    "reference": {
      "tau_safe_t0": 0.95,
      "beta0": 0.05,
      "tau_safe_t1": 0.95,
      "beta1": 0.05
    },
    "tau_safe_t0_grid": [
      0.9,
      0.95,
      0.97,
      0.99
    ],
    "beta0_grid": [
      0.01,
      0.02,
      0.05,
      0.1,
      0.2
    ],
    "tau_safe_t1_grid": [
      0.9,
      0.95,
      0.97,
      0.99
    ],
    "beta1_grid": [
      0.01,
      0.02,
      0.05,
      0.1
    ],
    "reference_point_interpretation": "Illustrative frozen reference configuration only. Main claims must be based on the full predeclared frontier/sensitivity analysis, not on whichever point performs best on the test set."
  },
  "action_feasibility": {
    "EXECUTE": {
      "VALID": "Feasible iff p_safe_t0_cal >= tau_safe_t0 AND (1-p_safe_t0_cal)*C(v) <= beta0",
      "INCOMPLETE": "INFEASIBLE",
      "CONTRADICTORY": "INFEASIBLE"
    },
    "GATHER": {
      "VALID": "Feasible iff gather_policy_eligible=True; evaluated by expected GATHER cost plus post-GATHER continuation.",
      "INCOMPLETE": "Feasible iff gather_policy_eligible=True; this is the preferred information-acquisition path when missing evidence can be obtained.",
      "CONTRADICTORY": "INFEASIBLE in the primary v2 policy. Contradictory evidence is not assumed resolvable by the available one-step GATHER action."
    },
    "ESCALATE": {
      "VALID": "FEASIBLE",
      "INCOMPLETE": "FEASIBLE",
      "CONTRADICTORY": "FEASIBLE"
    },
    "ABSTAIN": {
      "VALID": "FEASIBLE",
      "INCOMPLETE": "FEASIBLE",
      "CONTRADICTORY": "FEASIBLE"
    }
  },
  "expected_loss": {
    "EXECUTE": "L_E=(1-p_safe_t0_cal)*C(v)",
    "ESCALATE": "L_H=c_H+rho_H*C(v)",
    "ABSTAIN": "L_A=c_A",
    "GATHER": "L_G=c_G + min feasible post-GATHER continuation loss, using p_safe_t1_for_primary_rcse, tau_safe_t1, and beta1.",
    "post_gather_execute": "Feasible iff p_safe_t1_for_primary_rcse >= tau_safe_t1 AND (1-p_safe_t1_for_primary_rcse)*C(v) <= beta1"
  },
  "gate_routing": {
    "VALID": "Choose minimum expected-loss action among all feasible actions.",
    "INCOMPLETE": "EXECUTE is blocked. Compare feasible GATHER, ESCALATE, ABSTAIN.",
    "CONTRADICTORY": "EXECUTE and GATHER are blocked. Choose minimum expected-loss action between ESCALATE and ABSTAIN."
  },
  "policies_to_compare": [
    {
      "name": "ARGMAX_V1",
      "description": "Original frozen multiclass argmax comparator."
    },
    {
      "name": "CONFIDENCE_THRESHOLD_V1",
      "description": "Original multiclass p(EXECUTE) confidence-threshold comparator."
    },
    {
      "name": "VOI_BLIND_V1",
      "description": "Original consequence-blind information-acquisition comparator."
    },
    {
      "name": "RCSE_V1_FULL",
      "description": "Original frozen RCSE v1 policy result, preserved unchanged."
    },
    {
      "name": "V2_SAFETY_NO_GATE",
      "description": "Uses dedicated t0 safety belief and consequence-aware execution risk but does not apply evidence-gate blocking. Diagnostic ablation only."
    },
    {
      "name": "V2_GATE_MULTICLASS_EXECUTE",
      "description": "Applies Evidence Gate v2 but uses the original multiclass p(EXECUTE) for direct-execution risk. Diagnostic ablation isolating the gate."
    },
    {
      "name": "RCSE_V2_NO_GATHER",
      "description": "Evidence Gate v2 + dedicated t0 safety belief + consequence-aware execution risk, with GATHER disabled."
    },
    {
      "name": "RCSE_V2_FULL",
      "description": "Evidence Gate v2 + dedicated t0 safety belief + consequence-aware expected loss + one-step GATHER/VoI."
    }
  ],
  "primary_evaluation_metrics": [
    "autonomous_execution_coverage",
    "unsafe_execution_rate",
    "raw_exposure_weighted_unsafe_rate",
    "consequence_weighted_unsafe_rate",
    "mean_realized_normalized_loss",
    "human_escalation_rate",
    "gather_rate",
    "abstention_rate"
  ],
  "required_v2_ablations": [
    "RCSE_V2_FULL vs RCSE_V2_NO_GATHER: information-acquisition contribution",
    "RCSE_V2_FULL vs V2_SAFETY_NO_GATE: evidence-gate contribution",
    "RCSE_V2_FULL vs V2_GATE_MULTICLASS_EXECUTE: dedicated safety-belief contribution",
    "RCSE_V2_FULL vs VOI_BLIND_V1: consequence-aware revised policy comparison",
    "RCSE_V2_FULL vs RCSE_V1_FULL: revised architecture comparison"
  ],
  "governance": [
    "Do not modify the evidence-gate delta after RCSE v2 results are observed.",
    "Do not choose tau_safe_t0, beta0, tau_safe_t1, or beta1 from test-set winners.",
    "Report the frozen reference point together with the full predeclared grids/frontiers.",
    "Do not introduce an R4-specific policy override in the primary analysis.",
    "Do not train on synthetic intervention labels.",
    "Do not use intervention_type or synthetic_intervention as policy inputs.",
    "Do not use future outcomes as t0 policy features.",
    "Do not use transaction exposure/risk as learned safety-model features; exposure enters only through C(v).",
    "Preserve original RCSE v1 results and identify v2 as a diagnostically motivated secondary/revised analysis.",
    "Do not claim universal probability calibration if R4 calibration remains poor.",
    "Do not claim RCSE v2 dominance unless the predeclared frontier/ablation results support it."
  ],
  "next_step": "Step 09b.2m: construct the RCSE v2 policy-ready evaluation dataset by joining frozen action probabilities, dedicated t0 safety probabilities, Evidence Gate v2 state, post-GATHER safety information, exposure, and benchmark outcomes. Then run the frozen v2 policy simulation without changing this specification."
}"""

def main():
    out = Path("09b2l_rcse_v2_policy_spec.json").resolve()
    out.write_text(SPEC_JSON, encoding="utf-8")
    print("=" * 80)
    print("RCSE STEP 09b.2l - V2 POLICY SPECIFICATION FROZEN")
    print("=" * 80)
    print(f"Output : {out}")
    print("Status : FROZEN_BEFORE_RCSE_V2_SIMULATION")
    print("Gate delta : 0.05")
    print("Primary t0 safety belief : p_safe_t0_cal")
    print("Reference tau_safe_t0/beta0 : 0.95 / 0.05")
    print("Reference tau_safe_t1/beta1 : 0.95 / 0.05")
    print("Primary consequence : LOG_TRAIN_P90")
    print("No R4-specific override.")
    print("\nNext: Step 09b.2m policy-ready dataset construction.")

if __name__ == "__main__":
    main()
