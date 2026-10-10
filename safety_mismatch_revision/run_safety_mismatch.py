"""Post-review mismatch-feature sensitivity; preserves all frozen artifacts."""
from pathlib import Path
import importlib.util, sys, json, hashlib, time
from types import SimpleNamespace
sys.dont_write_bytecode = True
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
S = ROOT/'rcse_output/experimental_benchmark/splits'
def load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT/file)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
def log(s): print(time.strftime('%H:%M:%S'), s, flush=True)
def markdown(df):
    rows = [[str(v) for v in df.columns]] + [[f'{v:.6g}' if isinstance(v, float) else str(v) for v in row] for row in df.itertuples(index=False, name=None)]
    rows.insert(1, ['---'] * len(df.columns))
    return '\n'.join('| ' + ' | '.join(row) + ' |' for row in rows)

def main():
    h = load('robustness', '09b2h_full_benchmark_safety_robustness.py')
    g = load('gate', '09b2k_evidence_gate_v2_audit.py')
    base_path = ROOT/'rcse_output/rcse_base_v2.parquet'
    exp_path = S/'rcse_experimental_with_splits.parquet'
    base = pd.read_parquet(base_path); exp = pd.read_parquet(exp_path)
    base['source_case_id'] = base.case_id.astype(str)
    exp['source_case_id'] = exp.source_case_id.astype(str)
    assert base.source_case_id.is_unique and exp.benchmark_episode_id.is_unique
    base = base.merge(h.build_split_map(exp), on='source_case_id', validate='one_to_one')
    base['y_t0_safety_d1'] = h.build_d1_target(base)
    targets = base.set_index('source_case_id').y_t0_safety_d1
    resolved = base[base.y_t0_safety_d1.isin([h.SAFE_LABEL,h.UNSAFE_LABEL])].copy()
    fm = json.loads((S/'t0_execution_safety_results/t0_safety_feature_manifest.json').read_text())
    numeric = fm['numeric_features'] + ['inv_po_abs_rel_diff','inv_gr_abs_rel_diff']
    cats = fm['categorical_features']; features = numeric+cats
    args = SimpleNamespace(C=1.,max_iter=5000,tol=1e-4)
    metrics, gates, fits, cis = [], [], [], []
    for regime, cfg in h.SPLIT_CONFIG.items():
        train, cal, test = [resolved[resolved[cfg['column']]==cfg[k]].copy() for k in ['train','calibration','test']]
        sets = [set(x.source_case_id) for x in [train,cal,test]]
        assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
        log(f'{regime}: fitting on {len(train):,} natural resolved cases')
        pipe, iso, audit = h.fit_model(train,cal,numeric,cats,args)
        fits.append(dict(regime=regime,train_cases=len(train),calibration_cases=len(cal),test_cases=len(test),**audit))
        joblib.dump(dict(pipeline=pipe,calibrator=iso,features=features),OUT/f'mismatch_model_{regime}.joblib')
        test['mismatch_raw'],test['mismatch_cal'] = h.score(pipe,iso,test,features)
        frozen = pd.read_csv(S/f't0_execution_safety_results/t0_safety_predictions_{regime}.csv',dtype={'source_case_id':str})
        test = test.merge(frozen[['source_case_id','p_safe_t0_cal']],on='source_case_id',validate='one_to_one')
        assert len(test)==len(frozen)
        test['evidence_complete'] = ~h.as_bool(test.goods_receipt_required) | h.as_bool(test.gr_available_at_t0)
        test.to_parquet(OUT/f'natural_predictions_{regime}.parquet',index=False)
        for subset, df in [('natural_resolved',test),('evidence_complete',test[test.evidence_complete])]:
            y = (df.y_t0_safety_d1==h.SAFE_LABEL).to_numpy(int)
            scores = [df.p_safe_t0_cal.to_numpy(),df.mismatch_cal.to_numpy()]
            for name, p in zip(['original','mismatch'],scores):
                metrics.append(dict(regime=regime,subset=subset,model=name,cases=len(df),unsafe=int((1-y).sum()),auroc=roc_auc_score(y,p),ap_unsafe=average_precision_score(1-y,1-p)))
            rng=np.random.default_rng(20261003); deltas=[]
            for _ in range(2000):
                ix=rng.integers(0,len(y),len(y))
                if len(np.unique(y[ix]))==2:
                    deltas.append(roc_auc_score(y[ix],scores[1][ix])-roc_auc_score(y[ix],scores[0][ix]))
            lo,hi=np.quantile(deltas,[.025,.975])
            cis.append(dict(regime=regime,subset=subset,metric='mismatch_minus_original_auroc',estimate=roc_auc_score(y,scores[1])-roc_auc_score(y,scores[0]),lower=lo,upper=hi,replicates=len(deltas)))
        full = exp[exp[cfg['column']]==cfg['test']].copy()
        full['mismatch_raw'],full['mismatch_cal']=h.score(pipe,iso,full,features)
        full['target']=h.build_robustness_target(full,targets)
        full['family']=h.intervention_family(full)
        full['gate_state']=g.classify_gate(full,g.build_fixed_rules(full),g.build_value_rule(full,.05))[0]
        for family, df in [('ALL_RESOLVED',full)]+list(full.groupby('family')):
            df=df[df.target.isin([h.SAFE_LABEL,h.UNSAFE_LABEL])]
            for threshold in [.90,.95,.99]:
                for gated in [False,True]:
                    selected=(df.mismatch_cal>=threshold)&((df.gate_state=='VALID') if gated else True)
                    n=int(selected.sum()); u=int((selected&(df.target==h.UNSAFE_LABEL)).sum())
                    gates.append(dict(regime=regime,family=family,threshold=threshold,gate=gated,resolved_episodes=len(df),executed=n,unsafe=u,coverage=n/len(df) if len(df) else np.nan,uer=u/n if n else np.nan))
        full[['benchmark_episode_id','source_case_id','family','target','mismatch_raw','mismatch_cal','gate_state']].to_parquet(OUT/f'benchmark_predictions_{regime}.parquet',index=False)
        pd.DataFrame(metrics).to_csv(OUT/'natural_metrics.csv',index=False)
        pd.DataFrame(cis).to_csv(OUT/'paired_auroc_ci.csv',index=False)
        pd.DataFrame(gates).to_csv(OUT/'gate_increment.csv',index=False)
        pd.DataFrame(fits).to_csv(OUT/'training_audit.csv',index=False)
        log(f'{regime}: results saved; converged in {audit["iterations_required"]} iterations')
    inputs=[base_path,exp_path,ROOT/'09b2h_full_benchmark_safety_robustness.py',ROOT/'09b2k_evidence_gate_v2_audit.py']
    manifest=dict(status='complete',analysis='post-review sensitivity',training='natural D1-resolved only; original frozen splits',numeric_features=numeric,categorical_features=cats,gate_delta=.05,score_thresholds=[.90,.95,.99],threshold_selection='fixed diagnostic thresholds; no test optimization',bootstrap='2000 paired natural source-case resamples; fixed models',inputs_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    (OUT/'experiment_manifest.json').write_text(json.dumps(manifest,indent=2))
    write_report(metrics, cis, gates)
    log('Complete')
def write_report(metrics, cis, gates):
    report='# Mismatch-feature safety sensitivity\n\nPost-review analysis. Added only inv_po_abs_rel_diff and inv_gr_abs_rel_diff to the original natural-only safety model. Original splits, logistic regression, and isotonic calibration are preserved. Original artifacts remain unchanged.\n\n'+markdown(pd.DataFrame(metrics))+'\n\nPaired AUROC differences (mismatch minus original):\n\n'+markdown(pd.DataFrame(cis))+'\n\nGate increment at fixed score threshold 0.95 and delta 0.05:\n\n'+markdown(pd.DataFrame(gates).query("family == 'ALL_RESOLVED' and threshold == .95"))+'\n\nLimits: gate results are score-filter diagnostics, not a rerun of RCSE expected-loss/GATHER policies. Synthetic cases are evaluation-only NOT_SAFE under the existing robustness convention; natural ambiguous cases are excluded. Gate can lower unsafe counts by lowering coverage; this is not a matched-coverage comparison. Existing intervention families are not an independent gate stress test. Natural-only training cannot establish robustness to previously unseen corruption. Bootstrap captures held-out sampling, not training variability.\n'
    (OUT/'Safety_Mismatch_Results.md').write_text(report,encoding='utf-8')
if __name__=='__main__':
    if '--report-only' in sys.argv:
        write_report(pd.read_csv(OUT/'natural_metrics.csv'), pd.read_csv(OUT/'paired_auroc_ci.csv'), pd.read_csv(OUT/'gate_increment.csv'))
        log('Report generated from completed outputs')
    else:
        main()
