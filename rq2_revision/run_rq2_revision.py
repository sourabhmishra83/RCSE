"""Post-review RQ2 validation; source Paper2 artifacts are read-only.

Run with the existing Miniconda interpreter. Preserves original model specification,
training episodes and calibration split; removes only events_total_case.
Safety scores remain frozen. Bootstrap resamples paired natural source cases.
"""
from pathlib import Path
import importlib.util, json, hashlib, warnings, time, sys
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score, average_precision_score
import joblib

SOURCE = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
SPLITS = SOURCE / 'rcse_output/experimental_benchmark/splits'
REPS = 2000
SEED = 20261003

def log(message):
    print(time.strftime('%H:%M:%S'), message, flush=True)

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def stats(y, score):
    return np.array([roc_auc_score(y, score), average_precision_score(y, score),
                     average_precision_score(1-y, 1-score)])

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model_module = load_module('original_action', SOURCE / '08c_generate_frozen_baseline_predictions.py')
    manifest = json.loads((SPLITS/'frozen_baseline_results/frozen_baseline_feature_manifest.json').read_text())
    numeric = [c for c in manifest['numeric_features'] if c != 'events_total_case']
    categorical = manifest['categorical_features']
    features = numeric + categorical
    split_cols = ['split_'+r for r in model_module.SPLIT_CONFIG]
    log('Loading frozen benchmark and natural decision-time states')
    benchmark = pd.read_parquet(SPLITS/'rcse_experimental_with_splits.parquet',
        columns=list(dict.fromkeys(features+split_cols+['source_case_id','expected_action_reference'])))
    benchmark['source_case_id'] = benchmark.source_case_id.astype(str)
    for c in split_cols:
        assert benchmark.groupby('source_case_id')[c].nunique(dropna=False).max() == 1
    split_map = benchmark[['source_case_id']+split_cols].drop_duplicates('source_case_id')
    base = pd.read_parquet(SOURCE/'rcse_output/rcse_base_v2.parquet',
                          columns=list(dict.fromkeys(['case_id']+features)))
    base['source_case_id'] = base.case_id.astype(str)
    assert base.source_case_id.is_unique
    natural = base.merge(split_map, on='source_case_id', how='inner', validate='one_to_one')
    results, intervals, fits = [], [], []
    metric_names = ['auroc','average_precision_safe','average_precision_unsafe']
    inputs = [SOURCE/'08c_generate_frozen_baseline_predictions.py',
              SPLITS/'rcse_experimental_with_splits.parquet',SOURCE/'rcse_output/rcse_base_v2.parquet']
    for regime, cfg in model_module.SPLIT_CONFIG.items():
        col = cfg['column']
        train = benchmark.loc[benchmark[col] == cfg['train']]
        cal = benchmark.loc[benchmark[col] == cfg['calibration']]
        test = natural.loc[natural[col] == cfg['test']].copy()
        train_ids, cal_ids, test_ids = [set(x.source_case_id) for x in [train,cal,test]]
        assert not(train_ids & cal_ids or train_ids & test_ids or cal_ids & test_ids)
        pipe = model_module.build_pipeline(numeric,categorical,C=1.0,max_iter=5000,tol=1e-4)
        log(f'{regime}: fitting action model on {len(train):,} episodes')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always',ConvergenceWarning)
            pipe.fit(train[features],train.expected_action_reference.astype(str))
        iterations = int(pipe.named_steps['model'].n_iter_.max())
        converged = iterations < 5000 and not any(issubclass(w.category,ConvergenceWarning) for w in caught)
        if not converged:
            raise RuntimeError(f'{regime}: convergence failed; iterations={iterations}')
        order = [list(pipe.named_steps['model'].classes_).index(c) for c in model_module.CLASSES]
        calibrators = model_module.fit_isotonic_calibrators(cal.expected_action_reference.astype(str).to_numpy(),
                         pipe.predict_proba(cal[features])[:,order])
        raw = pipe.predict_proba(test[features])[:,order]
        calibrated = model_module.apply_isotonic(raw,calibrators)
        execute_idx = model_module.CLASSES.index('EXECUTE')
        test['p_execute_corrected_raw'] = raw[:,execute_idx]
        test['p_execute_corrected_cal'] = calibrated[:,execute_idx]
        safety_path = SPLITS/f't0_execution_safety_results/t0_safety_predictions_{regime}.csv'
        inputs.append(safety_path)
        safety = pd.read_csv(safety_path,dtype={'source_case_id':str})
        assert safety.source_case_id.is_unique
        scored = test.merge(safety[['source_case_id','y_t0_safety_d1','p_safe_t0_raw','p_safe_t0_cal']],
                            on='source_case_id',how='inner',validate='one_to_one')
        assert len(scored) == len(safety), 'Missing cases from full frozen safety population'
        truth = lambda s: s.astype(str).str.lower().isin(['true','1','yes'])
        scored['evidence_complete'] = ~truth(scored.goods_receipt_required) | truth(scored.gr_available_at_t0)
        scored.to_parquet(OUT/f'paired_predictions_{regime}.parquet',index=False)
        joblib.dump({'pipeline':pipe,'calibrators':calibrators,'class_order':model_module.CLASSES},
                    OUT/f'action_model_without_future_count_{regime}.joblib')
        fits.append({'regime':regime,'train_episodes':len(train),'calibration_episodes':len(cal),
                     'resolved_test_cases':len(scored),'iterations':iterations,'converged':converged})
        for subset, rows in [('natural_d1_resolved',scored),('evidence_complete',scored.loc[scored.evidence_complete])]:
            y = (rows.y_t0_safety_d1 == 'SAFE_TO_EXECUTE').to_numpy(dtype=int)
            scores = {name:rows[key].to_numpy(dtype=float) for name,key in {
                'action_raw':'p_execute_corrected_raw','action_calibrated':'p_execute_corrected_cal',
                'safety_raw':'p_safe_t0_raw','safety_calibrated':'p_safe_t0_cal'}.items()}
            for name,score in scores.items():
                point = stats(y,score)
                results.append({'regime':regime,'subset':subset,'score':name,'cases':len(y),
                                'safe_cases':int(y.sum()),'unsafe_cases':int((1-y).sum()),
                                **dict(zip(metric_names,point))})
            log(f'{regime}/{subset}: paired bootstrap ({REPS} source-case resamples)')
            rng=np.random.default_rng(SEED)
            boot=[]
            for rep in range(REPS):
                ix=rng.integers(0,len(y),len(y))
                if len(np.unique(y[ix]))<2: continue
                a=stats(y[ix],scores['action_calibrated'][ix])
                s=stats(y[ix],scores['safety_calibrated'][ix])
                boot.append(np.concatenate([a,s,s-a]))
            boot=np.array(boot)
            points=np.concatenate([stats(y,scores['action_calibrated']),stats(y,scores['safety_calibrated']),
                stats(y,scores['safety_calibrated'])-stats(y,scores['action_calibrated'])])
            for j, name in enumerate(['action_calibrated','safety_calibrated','safety_minus_action']):
                for k,metric in enumerate(metric_names):
                    v=boot[:,j*3+k]
                    intervals.append({'regime':regime,'subset':subset,'comparison':name,'metric':metric,
                        'point_estimate':points[j*3+k],'ci_lower':np.quantile(v,.025),'ci_upper':np.quantile(v,.975),
                        'valid_replicates':len(v),'bootstrap_seed':SEED})
            pd.DataFrame(results).to_csv(OUT/'rq2_point_estimates.csv',index=False)
            pd.DataFrame(intervals).to_csv(OUT/'rq2_paired_bootstrap_ci.csv',index=False)
        pd.DataFrame(fits).to_csv(OUT/'training_audit.csv',index=False)
    log('Hashing input artifacts for provenance')
    hashes={}
    for path in inputs:
        h=hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
        hashes[str(path)]=h.hexdigest()
    (OUT/'experiment_manifest.json').write_text(json.dumps({'status':'complete','analysis':'post-review sensitivity',
        'removed_feature':'events_total_case','numeric_features':numeric,'categorical_features':categorical,
        'safety_model':'existing frozen predictions, not retrained','training':'original full benchmark episodes',
        'calibration':'original dedicated calibration partition; one-vs-rest isotonic, renormalized',
        'bootstrap':'paired natural source-case percentile bootstrap; one row per source case',
        'replicates':REPS,'seed':SEED,'sklearn_version':sklearn.__version__,'inputs_sha256':hashes},indent=2))
    log('Experiment complete')

if __name__=='__main__':main()
