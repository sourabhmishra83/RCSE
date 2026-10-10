"""Validate included scientific inputs/results without retraining or document files."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'revision_complete'
REGIMES=['grouped_iid','temporal','ood_exposure']
def main():
    checks=[]
    def check(value,label):
        if not bool(value):raise AssertionError(label)
        checks.append(label)
    for relative in ['rcse_output/rcse_base_v2.parquet','rcse_output/experimental_benchmark/splits/rcse_experimental_with_splits.parquet','09b2l_rcse_v2_policy_spec.json','revision_complete/analysis_protocol.json']:
        check((ROOT/relative).is_file(),'input present: '+relative)
    for regime in REGIMES:
        for relative in [f'rq2_revision/action_model_without_future_count_{regime}.joblib',f'safety_mismatch_revision/mismatch_model_{regime}.joblib',f'revision_complete/separate_calibration_safety_{regime}.joblib',f'rcse_output/experimental_benchmark/splits/t0_execution_safety_results/t0_safety_predictions_{regime}.csv']:
            check((ROOT/relative).is_file(),'artifact present: '+relative)
    a=pd.read_csv(OUT/'all_policy_absolute_results.csv')
    for (regime,subset),d in a.groupby(['regime','subset']):
        tag=regime+'/'+subset
        check(len(d)==9,tag+': complete policy set')
        check(np.allclose(d.coverage+d.escalation+d.gather+d.abstention,1),tag+': actions partition episodes')
        check(np.allclose(d.coverage,d.executed/d.episodes),tag+': coverage arithmetic')
        selected=d[d.executed.gt(0)]
        check(np.allclose(selected.uer,selected.unsafe_executed/selected.executed),tag+': UER arithmetic')
        esc=d[d.policy.eq('ALWAYS_ESCALATE')].iloc[0]
        check(esc.executed==0 and np.isclose(esc.loss,.1),tag+': escalation reference')
    for directory in ['rq2_revision','safety_mismatch_revision']:
        audit=pd.read_csv(ROOT/directory/'training_audit.csv')
        check(audit.converged.all(),directory+': converged fits')
    audit=pd.read_csv(OUT/'separate_calibration_training_audit.csv')
    check(audit.converged.all() and not audit.convergence_warning.any(),'independent calibration fits converged')
    for regime in REGIMES:
        selection=pd.read_parquet(OUT/f'independent_selection_cases_{regime}.parquet')
        test=pd.read_parquet(OUT/f'separate_calibration_natural_test_{regime}.parquet')
        check(selection.source_case_id.is_unique and test.source_case_id.is_unique,regime+': natural cases unique')
        check(not(set(selection.source_case_id.astype(str))&set(test.source_case_id.astype(str))),regime+': selection/test disjoint')
        paired=pd.read_parquet(ROOT/'rq2_revision'/f'paired_predictions_{regime}.parquet')
        check(paired.source_case_id.is_unique,regime+': paired evaluation cases unique')
    fm=json.loads((ROOT/'rq2_revision/experiment_manifest.json').read_text())
    check(len(fm['numeric_features'])==21 and len(fm['categorical_features'])==19,'corrected 40-feature set')
    check('events_total_case' not in fm['numeric_features'],'future trace-count exclusion')
    check(fm['replicates']==2000,'paired bootstrap count')
    stress=pd.read_csv(OUT/'independent_gate_stress.csv')
    d=stress[stress.subset.eq('original_gate_valid')&stress.corruption.ne('ORIGINAL')]
    check(len(d)==6 and d.gate_blocked.eq(0).all(),'independent gate failures retained')
    frontier=pd.read_csv(OUT/'calibration_selected_frontier.csv')
    d=frontier[frontier.population.eq('full_benchmark_resolved')&frontier.risk_cap.le(.16)]
    check(len(d)>0 and d.calibration_selected.eq(0).all(),'no benchmark calibration point through 16 percent')
    horizon=pd.read_csv(OUT/'fixed_horizon_safety_sensitivity.csv')
    check(len(horizon)==9,'three regimes by three horizons')
    d=horizon[horizon.regime.eq('temporal')&horizon.horizon_days.isin([60,90])]
    check(len(d)==2 and d.ambiguous.eq(14285).all(),'unresolved temporal follow-up retained')
    check(len(pd.read_csv(OUT/'v2_human_cost_sensitivity.csv'))==300,'complete 25-setting cost grid')
    cal=json.loads((OUT/'calibration_manifest.json').read_text())
    check(cal['selection_uses_test_labels'] is False,'threshold selection excludes test labels')
    check(cal['gather'] is False,'new calibration comparisons disable GATHER')
    report=dict(status='passed',checks=len(checks),check_labels=checks,scope='Included inputs and numerical consistency; no retraining, manuscript, or reviewer files required',raw_log_required_for='run_followup_windows.py only in the direct additional-experiment rerun sequence')
    (OUT/'reproduction_validation.json').write_text(json.dumps(report,indent=2))
    print(f'Passed {len(checks)} scientific reproduction checks.')
if __name__=='__main__':main()
