"""Independent calibration selection and risk-control baselines; no GATHER."""
from pathlib import Path
import sys,json,hashlib,time,importlib.util
from types import SimpleNamespace
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import joblib
from scipy.stats import binom
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score,average_precision_score
ROOT=Path(__file__).resolve().parents[1]; OUT=Path(__file__).resolve().parent
S=ROOT/'rcse_output/experimental_benchmark/splits'
def load(n,f):
    s=importlib.util.spec_from_file_location(n,ROOT/f); m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def log(s):print(time.strftime('%H:%M:%S'),s,flush=True)
def fixed_half(s):return s.map(lambda x:int(hashlib.sha256(('20261004'+str(x)).encode()).hexdigest()[:8],16)%2)
def main():
    h=load('safety','09b2h_full_benchmark_safety_robustness.py');g=load('gate','09b2k_evidence_gate_v2_audit.py');p=load('policy','09b2n_run_rcse_v2_policy_simulation.py')
    a=load('action','08c_generate_frozen_baseline_predictions.py')
    protocol=json.loads((OUT/'analysis_protocol.json').read_text()); spec=json.loads((ROOT/'09b2l_rcse_v2_policy_spec.json').read_text());params=p.get_reference_parameters(spec)
    fm=json.loads((S/'t0_execution_safety_results/t0_safety_feature_manifest.json').read_text());num=fm['numeric_features'];cat=fm['categorical_features'];features=num+cat
    base=pd.read_parquet(ROOT/'rcse_output/rcse_base_v2.parquet');base['source_case_id']=base.case_id.astype(str)
    exp=pd.read_parquet(S/'rcse_experimental_with_splits.parquet');exp['source_case_id']=exp.source_case_id.astype(str)
    base=base.merge(h.build_split_map(exp),on='source_case_id',validate='one_to_one');base['target']=h.build_d1_target(base)
    targetmap=base.set_index('source_case_id').target; resolved=base[base.target.isin([h.SAFE_LABEL,h.UNSAFE_LABEL])].copy(); resolved['y_t0_safety_d1']=resolved.target
    allrows=[]; curves=[]; frontiers=[]; audits=[]; candidates=[]; rawaction=[]
    for regime,cfg in h.SPLIT_CONFIG.items():
        train=resolved[resolved[cfg['column']].eq(cfg['train'])];cal=resolved[resolved[cfg['column']].eq(cfg['calibration'])]
        ca=cal[fixed_half(cal.source_case_id).eq(0)].copy(); cb=cal[fixed_half(cal.source_case_id).eq(1)].copy();test=resolved[resolved[cfg['column']].eq(cfg['test'])].copy()
        assert not(set(train.source_case_id)&set(cal.source_case_id) or set(cal.source_case_id)&set(test.source_case_id))
        log(f'{regime}: fit natural safety model; isotonic={len(ca)}, independent selection={len(cb)}')
        pipe,iso,audit=h.fit_model(train,ca,num,cat,SimpleNamespace(C=1.,max_iter=5000,tol=1e-4))
        joblib.dump(dict(pipeline=pipe,calibrator=iso,features=features),OUT/f'separate_calibration_safety_{regime}.joblib')
        audits.append(dict(regime=regime,training=len(train),isotonic=len(ca),selection=len(cb),test=len(test),**audit))
        action=joblib.load(ROOT/f'rq2_revision/action_model_without_future_count_{regime}.joblib')
        afeatures=action['pipeline'].feature_names_in_.tolist();idx=list(action['pipeline'].named_steps['model'].classes_).index('EXECUTE')
        # Raw multiclass scores use train-fitted model only; they avoid reusing original isotonic calibration labels.
        for d in [cb,test]:
            d['sraw'],d['scal']=h.score(pipe,iso,d,features)
            d['araw']=action['pipeline'].predict_proba(d[afeatures])[:,idx]
            d['valid']=g.classify_gate(d,g.build_fixed_rules(d),g.build_value_rule(d,.05))[0].eq('VALID')
        cb['unsafe']=cb.target.eq(h.UNSAFE_LABEL);test['unsafe']=test.target.eq(h.UNSAFE_LABEL)
        fullcal=exp[exp[cfg['column']].eq(cfg['calibration'])&fixed_half(exp.source_case_id).eq(1)].copy()
        fulltest=exp[exp[cfg['column']].eq(cfg['test'])].copy()
        refs={f'p{int(q*100)}':float(pd.to_numeric(exp.loc[exp[cfg['column']].eq(cfg['train']),'transaction_exposure_eur']).abs().quantile(q)) for q in [.5,.9,.99]}
        for d in [fullcal,fulltest]:
            d['sraw'],d['scal']=h.score(pipe,iso,d,features);d['araw']=action['pipeline'].predict_proba(d[afeatures])[:,idx]
            d['valid']=g.classify_gate(d,g.build_fixed_rules(d),g.build_value_rule(d,.05))[0].eq('VALID')
            d['target']=h.build_robustness_target(d,targetmap);d['unsafe']=d.target.eq(h.UNSAFE_LABEL)
        def record(policy,threshold,cap,selected,df,scope,**extra):
            known=df.target.isin([h.SAFE_LABEL,h.UNSAFE_LABEL]);sel=selected&known;n=int(sel.sum());u=int((sel&df.unsafe).sum())
            C=p.consequence_from_exposure(df.transaction_exposure_eur,'LOG_TRAIN_P90',refs)
            loss=np.where(selected,df.unsafe.astype(float)*C,params['c_H'])
            allrows.append(dict(regime=regime,population=scope,policy=policy,threshold=threshold,risk_cap=cap,cases=int(known.sum()),executed=n,unsafe=u,coverage=n/known.sum() if known.any() else np.nan,uer=u/n if n else np.nan,unsafe_fraction=u/known.sum() if known.any() else np.nan,resolved_mean_loss=float(np.asarray(loss)[known].mean()),ambiguous_executed=int((selected&~known).sum()),**extra))
        for name,col,gated in [('ACTION_RAW','araw',False),('SAFETY_RAW','sraw',False),('SAFETY_RAW_GATE','sraw',True)]:
            for threshold in protocol['threshold_grid']:
                for scope,d in [('natural_resolved',test),('full_benchmark_resolved',fulltest)]:
                    selected=d[col].ge(threshold)&(d.valid if gated else True);n=int(selected.sum());u=int((selected&d.unsafe).sum())
                    known=d.target.isin([h.SAFE_LABEL,h.UNSAFE_LABEL]);selected &= known;n=int(selected.sum());u=int((selected&d.unsafe).sum())
                    curves.append(dict(regime=regime,population=scope,policy=name,threshold=threshold,coverage=n/known.sum(),uer=u/n if n else np.nan,executed=n,unsafe=u))
            for cap in protocol['risk_caps']:
                options=[]
                for threshold in protocol['threshold_grid']:
                    sel=cb[col].ge(threshold)&(cb.valid if gated else True);n=int(sel.sum());u=int((sel&cb.unsafe).sum())
                    pv=float(binom.cdf(u,n,cap)) if n else 1.
                    certified=n>0 and pv<=.05/len(protocol['threshold_grid'])
                    candidates.append(dict(regime=regime,policy=name,cap=cap,threshold=threshold,selection_cases=len(cb),selected=n,unsafe=u,pvalue=pv,certified=certified))
                    if certified:options.append((n,threshold,u,pv))
                chosen=sorted(options,key=lambda x:(-x[0],x[1]))[0] if options else (0,np.inf,0,1.)
                for scope,d in [('natural_resolved',test),('full_benchmark_resolved',fulltest)]:
                    record('LTT_'+name,chosen[1],cap,d[col].ge(chosen[1])&(d.valid if gated else True),d,scope,calibration_selected=chosen[0],calibration_unsafe=chosen[2],certified_threshold=bool(options),guarantee_scope='IID natural cases under exchangeability only')
        # CRC controls monotone unsafe-indicator averaged across all cases, NOT selective UER.
        for cap in [.01,.05,.10]:
            options=[]
            for threshold in protocol['threshold_grid']:
                sel=cb.sraw.ge(threshold);u=int((sel&cb.unsafe).sum()); corrected=(u+1)/(len(cb)+1)
                if corrected<=cap: options.append((int(sel.sum()),threshold,corrected))
            chosen=sorted(options,key=lambda x:(-x[0],x[1]))[0] if options else (0,np.inf,np.nan)
            for scope,d in [('natural_resolved',test),('full_benchmark_resolved',fulltest)]:
                record('CRC_UNSAFE_PER_CASE',chosen[1],cap,d.sraw.ge(chosen[1]),d,scope,calibration_corrected_loss=chosen[2],guarantee_scope='expected unsafe indicator per natural case under exchangeability; not conditional UER')
        # Calibration-selected no-GATHER RCSE grid on benchmark calibration episodes, grouped by source split.
        def selected(d,tau,beta):
            C=p.consequence_from_exposure(d.transaction_exposure_eur,'LOG_TRAIN_P90',refs)
            risk=(1-d.scal)*C
            return d.valid&d.scal.ge(tau)&risk.le(beta)&risk.le(params['c_H'])
        for cap in protocol['risk_caps']:
            options=[]
            for tau in spec['risk_and_selectivity']['tau_safe_t0_grid']:
                for beta in spec['risk_and_selectivity']['beta0_grid']:
                    sel=selected(fullcal,tau,beta)&fullcal.target.isin([h.SAFE_LABEL,h.UNSAFE_LABEL]);n=int(sel.sum());u=int((sel&fullcal.unsafe).sum())
                    if n and u/n<=cap:options.append((n,tau,beta,u))
            chosen=sorted(options,key=lambda x:(-x[0],x[1],x[2]))[0] if options else (0,np.inf,0.,0)
            for scope,d in [('natural_resolved',test),('full_benchmark_resolved',fulltest)]:
                sel=selected(d,chosen[1],chosen[2]);record('RCSE_CALIBRATION_SELECTED_NO_GATHER',chosen[1],cap,sel,d,scope,beta0=chosen[2],calibration_selected=chosen[0],calibration_unsafe=chosen[3],guarantee_scope='empirical calibration selection; clustered synthetic episodes; no finite sample guarantee')
                frontiers.append(allrows[-1])
        for scope,d in [('natural_resolved',test),('full_benchmark_resolved',fulltest)]:
            record('GATE_ONLY',np.nan,np.nan,d.valid,d,scope)
            record('SAFETY_ONLY_095',.95,np.nan,d.scal.ge(.95),d,scope)
            record('GATE_SAFETY_095',.95,np.nan,d.valid&d.scal.ge(.95),d,scope)
        # Corrected multiclass natural reference-action accuracy and EXECUTE-v-rest AUROC.
        nat=exp[exp[cfg['column']].eq(cfg['test'])&~h.synthetic_flag(exp)].copy()
        probs=action['pipeline'].predict_proba(nat[afeatures]);cls=action['pipeline'].named_steps['model'].classes_;y=nat.expected_action_reference.eq('EXECUTE')
        rawaction.append(dict(regime=regime,natural_episodes=len(nat),execute_prevalence=float(y.mean()),raw_execute_min=float(probs[:,idx].min()),raw_execute_max=float(probs[:,idx].max()),raw_execute_auroc=roc_auc_score(y,probs[:,idx]),argmax_accuracy=float((cls[np.argmax(probs,axis=1)]==nat.expected_action_reference).mean()),label='expected_action_reference EXECUTE versus rest',feature_count=len(afeatures),removed_feature='events_total_case'))
        for rows,name in [(allrows,'calibration_selected_baselines.csv'),(curves,'fair_threshold_sweeps.csv'),(frontiers,'calibration_selected_frontier.csv'),(candidates,'ltt_candidate_audit.csv'),(audits,'separate_calibration_training_audit.csv'),(rawaction,'corrected_action_natural_performance.csv')]:pd.DataFrame(rows).to_csv(OUT/name,index=False)
        test[['source_case_id','target','sraw','scal','araw','valid']].to_parquet(OUT/f'separate_calibration_natural_test_{regime}.parquet',index=False)
        cb[['source_case_id','target','sraw','scal','araw','valid']].to_parquet(OUT/f'independent_selection_cases_{regime}.parquet',index=False)
        log(f'{regime}: selected on calibration and evaluated on untouched test')
    (OUT/'calibration_manifest.json').write_text(json.dumps(dict(status='complete',protocol='analysis_protocol.json',gather=False,selection_uses_test_labels=False,calibration_halves='SHA256 of seed concatenated with source-case ID mod 2',ltt='exact binomial Bonferroni over fixed threshold grid for each score family and risk cap',crc='unsafe indicator per case; not conditional UER',limitation='no exchangeability guarantee under temporal or exposure OOD, or synthetic-cluster benchmark'),indent=2))
    log('Calibration analyses complete')
if __name__=='__main__':main()
