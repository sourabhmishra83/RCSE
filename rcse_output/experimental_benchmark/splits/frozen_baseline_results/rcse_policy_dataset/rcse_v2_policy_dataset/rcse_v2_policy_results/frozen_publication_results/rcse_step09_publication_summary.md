# RCSE Step 09 Frozen Publication Summary

## Primary RCSE v2 reference results

| Regime | Coverage | Unsafe execution | Exposure-weighted unsafe | End-to-end loss | Escalation | GATHER |
|---|---:|---:|---:|---:|---:|---:|
| Grouped IID | 17.27% [16.90%, 17.63%] | 16.89% [16.03%, 17.80%] | 20.06% [17.01%, 23.77%] | 0.1009 [0.0998, 0.1022] | 78.63% [78.22%, 79.03%] | 4.10% [3.90%, 4.31%] |
| Temporal | 21.91% [21.52%, 22.28%] | 17.21% [16.39%, 18.00%] | 17.44% [15.16%, 19.91%] | 0.1047 [0.1034, 0.1061] | 76.29% [75.89%, 76.68%] | 1.80% [1.67%, 1.94%] |
| OOD exposure | 0.91% [0.55%, 1.31%] | 21.74% [4.35%, 41.67%] | 29.62% [6.34%, 51.75%] | 0.1059 [0.1021, 0.1104] | 96.69% [95.95%, 97.38%] | 2.41% [1.82%, 3.06%] |

## Natural-only robustness

| Regime | Natural execution coverage | Natural unsafe execution | Contradiction block | Missingness block | Severe evidence-loss block |
|---|---:|---:|---:|---:|---:|
| Grouped IID | 53.44% [52.50%, 54.41%] | 1.89% [1.53%, 2.26%] | 92.55% [92.18%, 92.92%] | 100.00% [100.00%, 100.00%] | 100.00% [100.00%, 100.00%] |
| Temporal | 72.89% [71.98%, 73.79%] | 0.97% [0.76%, 1.21%] | 92.13% [91.75%, 92.53%] | 100.00% [100.00%, 100.00%] | 100.00% [100.00%, 100.00%] |
| OOD exposure | 3.10% [1.86%, 4.50%] | 14.29% [0.00%, 31.25%] | 92.72% [91.28%, 94.11%] | 100.00% [100.00%, 100.00%] | 100.00% [100.00%, 100.00%] |

## Publication-safe interpretation

- The evidence-validity gate is a necessary architectural component: paired ablation shows large, statistically stable reductions in unsafe execution and realized loss.
- Natural-case execution risk is substantially lower than full-benchmark risk; the full benchmark intentionally includes controlled synthetic interventions.
- Residual unsafe execution is dominated by near-boundary contradiction cases at the frozen 5% consistency threshold.
- RCSE behaves conservatively under OOD exposure shift, primarily through escalation rather than maintaining autonomous coverage.
- GATHER improves end-to-end loss in grouped-IID and temporal settings but does not show robust benefit under OOD exposure shift.
- Do not claim universal OOD superiority, universal low-risk autonomy, or production readiness from the current reference point.

## OOD reporting notes

- **Autonomous execution coverage**: 0.91% [0.55%, 1.31%] — Very low autonomous coverage under OOD exposure shift.
- **Unsafe execution rate**: 21.74% [4.35%, 41.67%] — Wide CI because very few OOD cases are autonomously executed.
- **Human escalation rate**: 96.69% [95.95%, 97.38%] — RCSE behaves conservatively under OOD shift.
- **Natural unsafe execution rate**: 14.29% [0.00%, 31.25%] — Natural OOD execution remains uncertain and materially weaker than IID/temporal.
- **Contradiction block rate**: 92.72% [91.28%, 94.11%] — Evidence gate remains effective under OOD.
- **CONSEQUENCE_AWARE_VS_VOI_BLIND**: -0.0026 [-0.0070, 0.0014] — CI includes zero; do not claim OOD loss superiority.
- **V2_VS_V1**: -0.0014 [-0.0056, 0.0023] — CI includes zero; do not claim OOD loss superiority.
- **GATHER_CONTRIBUTION**: 0.0042 [0.0011, 0.0078] — Statistically stable difference.
