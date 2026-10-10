# Step 09b.2a — Frozen RCSE Policy Specification

**Status:** Frozen before Step 09b.2b policy simulation.

This specification fixes the comparator definitions, loss functions, consequence
scaling, action-cost ranges, execution-risk budgets, and post-GATHER continuation
rule before the RCSE test-set policy simulation is run.

## 1. Comparator set

| Policy | Consequence-aware | GATHER | Risk budget |
|---|---:|---:|---:|
| ARGMAX | No | Predictive action only | No |
| CONFIDENCE_THRESHOLD | No | No | Fixed `p_execute` threshold |
| COST_AWARE_BLIND | No (`C(v)=1`) | No | No |
| VOI_BLIND | No (`C(v)=1`) | Yes | Yes |
| RCSE_NO_GATHER | Yes | No | Yes |
| RCSE_FULL | Yes | Yes | Yes |

### ARGMAX

Choose the highest calibrated Step-08c action probability.

### CONFIDENCE_THRESHOLD

\[
EXECUTE \iff p_E\ge\tau_0
\]

otherwise `ESCALATE`.

Predeclared threshold grid:

\[
\tau_0\in\{0.50,0.60,0.70,0.75,0.80,0.85,0.90,0.92,0.94,
0.95,0.96,0.97,0.98,0.99,0.995\}.
\]

### COST_AWARE_BLIND

Uses the same action-loss formulation as RCSE but sets:

\[
C(v)=1
\]

and does not allow GATHER.

### VOI_BLIND

Consequence-blind one-step information-acquisition baseline. `GATHER` is
available only for the natural delayed-evidence episodes marked
`gather_policy_eligible=True`.

### RCSE_NO_GATHER

Consequence-aware RCSE ablation with GATHER disabled.

### RCSE_FULL

Full consequence-aware policy with EXECUTE, GATHER, ESCALATE, and ABSTAIN.

---

## 2. Primary consequence function

The primary consequence function is:

\[
C(v)=\frac{\log(1+|v|)}{\log(1+v_{ref})}.
\]

For each frozen evaluation regime, \(v_{ref}\) is the **90th percentile of
absolute transaction exposure in that regime's TRAIN partition**, never the
test partition.

This is named `LOG_TRAIN_P90`.

Sensitivity analyses also use:

- `UNIFORM`: \(C(v)=1\);
- `LOG_TRAIN_P50`;
- `LOG_TRAIN_P99`.

No raw test-set percentile is used for normalization.

---

## 3. Action losses

### EXECUTE

\[
L_E=(1-p_E)C(v).
\]

### ESCALATE

\[
L_H=c_H+\rho_HC(v).
\]

Reference:

\[
\rho_H=0.
\]

Sensitivity:

\[
\rho_H\in\{0,0.02,0.05\}.
\]

### ABSTAIN

\[
L_A=c_A.
\]

### GATHER

For eligible natural delayed-evidence cases:

\[
L_G=c_G+\min(L_{E,t_1},L_H,L_A).
\]

where:

\[
L_{E,t_1}=(1-q_{t_1})C(v).
\]

However, post-GATHER autonomous execution is feasible only when both:

\[
q_{t_1}\ge\tau_1
\]

and

\[
(1-q_{t_1})C(v)\le\beta_1.
\]

Otherwise the continuation falls back to the cheaper of ESCALATE and ABSTAIN.

`q_t1` is therefore used as an **auxiliary selective score**, not assumed to be
universally calibrated under OOD.

No R4-specific override is introduced in the primary policy.

---

## 4. Reference cost schedule

Normalized experimental loss units:

\[
c_G=0.02,\qquad c_H=0.10,\qquad c_A=0.50.
\]

The ordering is:

\[
c_G<c_H<c_A.
\]

These values are not claimed enterprise costs.

Sensitivity grid:

\[
c_G\in\{0.01,0.02,0.05,0.10\}
\]

\[
c_H\in\{0.05,0.10,0.20\}
\]

\[
c_A\in\{0.20,0.50,1.00\}.
\]

Only combinations satisfying:

\[
c_G<c_H<c_A
\]

are evaluated.

---

## 5. Consequence-weighted execution-risk budget

Rather than arbitrary risk-stratum confidence multipliers, the primary policy
uses the continuous constraint:

\[
(1-p_E)C(v)\le\beta_0.
\]

Predeclared grid:

\[
\beta_0\in\{0.01,0.02,0.05,0.10,0.20\}.
\]

Reference:

\[
\beta_0=0.05.
\]

This automatically makes the permissible raw prediction error smaller as
business consequence increases.

---

## 6. Post-GATHER gate

Predeclared safety-score thresholds:

\[
\tau_1\in\{0.90,0.95,0.97,0.99\}.
\]

Reference:

\[
\tau_1=0.95.
\]

Post-GATHER consequence-risk budget:

\[
\beta_1\in\{0.01,0.02,0.05,0.10\}.
\]

Reference:

\[
\beta_1=0.05.
\]

The primary post-GATHER rule is:

\[
EXECUTE_{t_1}
\iff
q_{t_1}\ge\tau_1
\land
(1-q_{t_1})C(v)\le\beta_1.
\]

Otherwise use ESCALATE/ABSTAIN fallback.

---

## 7. Frozen reference configuration

For illustrative/reference tables:

- consequence: `LOG_TRAIN_P90`;
- \(c_G=0.02\);
- \(c_H=0.10\);
- \(c_A=0.50\);
- \(\rho_H=0\);
- \(\beta_0=0.05\);
- \(\tau_1=0.95\);
- \(\beta_1=0.05\).

This is **not** the sole basis for empirical claims. Main claims must come from
the predeclared grids/frontiers and sensitivity analyses.

---

## 8. Evaluation metrics

Step 09b.2b must report:

- autonomous execution coverage;
- unsafe execution rate;
- exposure-weighted unsafe execution rate;
- expected normalized business loss;
- human escalation rate;
- GATHER rate;
- abstention rate;
- action distribution;
- risk–coverage frontier.

---

## 9. Test-set governance

1. Do not choose a single threshold/cost because it performs best on the test set.
2. Report the predeclared parameter grids/frontiers and the frozen reference point.
3. Consequence normalization can use TRAIN-only exposure statistics.
4. Step 09a.5 risk-cap operating points remain descriptive only.
5. No special R4-only rule is allowed in the primary analysis after observing the Step 09a.5 R4 behavior.
6. Policy parameters must not be redefined after Step 09b.2b results are produced unless a genuine implementation defect is found.

---

## 10. Next step

**Step 09b.2b — run the frozen comparator and RCSE policy simulation.**

The simulator must read this JSON specification rather than hard-code or
interactively alter policy parameters.
