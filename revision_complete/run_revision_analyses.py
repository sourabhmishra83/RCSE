"""Post-review evidence pack. Read originals; write only revision_complete."""
from pathlib import Path
import sys, importlib.util, json, hashlib, time
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT=Path(__file__).resolve().parents[1]; OUT=Path(__file__).resolve().parent
S=ROOT/'rcse_output/experimental_benchmark/splits'
V=S/'frozen_baseline_results/rcse_policy_dataset/rcse_v2_policy_dataset'
R=V/'rcse_v2_policy_results'
def mod(name,file):
    sp=importlib.util.spec_from_file_location(name,ROOT/file); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
def log(s): print(time.strftime('%H:%M:%S'),s,flush=True)
def save(rows,name): pd.DataFrame(rows).to_csv(OUT/name,index=False)
def gate(g,df): return g.classify_gate(df,g.build_fixed_rules(df),g.build_value_rule(df,.05))[0]
def measure(df,actions,C,params,unsafe,postloss=None):
    exe=actions.eq('EXECUTE'); u=exe&unsafe; e=df.transaction_exposure_eur.abs().fillna(0)
    loss=pd.Series(0.,index=df.index)
    loss[exe]=unsafe[exe].astype(float)*C[exe]
    loss[actions.eq('ESCALATE')]=params['c_H']+params['rho_H']*C[actions.eq('ESCALATE')]
    loss[actions.eq('ABSTAIN')]=params['c_A']
    if actions.eq('GATHER').any():
        assert postloss is not None
        loss[actions.eq('GATHER')]=params['c_G']+postloss[actions.eq('GATHER')]
    return dict(episodes=len(df),executed=int(exe.sum()),unsafe_executed=int(u.sum()),coverage=exe.mean(),uer=u.sum()/exe.sum() if exe.any() else np.nan,loss=loss.mean(),escalation=actions.eq('ESCALATE').mean(),gather=actions.eq('GATHER').mean(),abstention=actions.eq('ABSTAIN').mean(),exposure_weighted_uer=e[u].sum()/e[exe].sum() if e[exe].sum()>0 else np.nan),loss
def cluster_mean_ci(df,values,seed=20261004):
    # Ratio of source-case cluster totals, paired against common baseline.
    d=pd.DataFrame(dict(id=df.source_case_id.to_numpy(),v=np.asarray(values)))
    groups=d.groupby('id').agg(total=('v','sum'),n=('v','size'))
    a=groups.total.to_numpy(); n=groups.n.to_numpy(); rng=np.random.default_rng(seed); reps=[]
    for _ in range(2000):
        ix=rng.integers(0,len(a),len(a)); reps.append(a[ix].sum()/n[ix].sum())
    return dict(estimate=float(d.v.mean()),lower=float(np.quantile(reps,.025)),upper=float(np.quantile(reps,.975)),replicates=2000,clusters=len(a))
def main():
    p=mod('policy','09b2n_run_rcse_v2_policy_simulation.py'); b=mod('boot','09b2r_bootstrap_rcse_v2_uncertainty.py')
    h=mod('safety','09b2h_full_benchmark_safety_robustness.py'); g=mod('gate','09b2k_evidence_gate_v2_audit.py')
    spec=json.loads((ROOT/'09b2l_rcse_v2_policy_spec.json').read_text()); params=p.get_reference_parameters(spec)
    protocol=json.loads((OUT/'analysis_protocol.json').read_text())
    metrics=[]; comparisons=[]; costs=[]; ambiguity=[]; cons=[]; stress=[]; maturity=[]; boundaries=[]
    base=pd.read_parquet(ROOT/'rcse_output/rcse_base_v2.parquet'); base['source_case_id']=base.case_id.astype(str)
    exp=pd.read_parquet(S/'rcse_experimental_with_splits.parquet'); exp['source_case_id']=exp.source_case_id.astype(str)
    assert base.source_case_id.is_unique
    base['target']=h.build_d1_target(base)
    splitmap=h.build_split_map(exp); natural=base.merge(splitmap,on='source_case_id',validate='one_to_one')
    # Last event is an observation-boundary proxy, not proof of case completeness.
    log_end=pd.to_datetime(base.case_end_time,utc=True).max()
    for regime,cfg in h.SPLIT_CONFIG.items():
        log(f'{regime}: reporting all policies, losses, costs and sensitivities')
        df=b.prepare_regime_data(V,R,regime,params)
        C=df._consequence; unsafe=df._unsafe_t0
        for subset, d in [('full_benchmark',df),('natural_only',df[df._intervention_family_r.eq('NATURAL')])]:
            for name in list(b.POLICIES)+['ALWAYS_ESCALATE']:
                a=d[f'action__{name}'] if name!='ALWAYS_ESCALATE' else pd.Series('ESCALATE',index=d.index)
                m,loss=measure(d,a,C.loc[d.index],params,unsafe.loc[d.index],d._post_gather_realized_loss)
                metrics.append(dict(regime=regime,subset=subset,policy=name,scope='retrospective GATHER' if m['gather'] else 't0 only',**m))
                if name in ['RCSE_V2_FULL','RCSE_V2_NO_GATHER']:
                    baseline=params['c_H']+params['rho_H']*C.loc[d.index]
                    comparisons.append(dict(regime=regime,subset=subset,comparison=name+' minus ALWAYS_ESCALATE',**cluster_mean_ci(d,loss-baseline)))
        # All four ambiguity conventions on direct execution; original fallback retained as one sensitivity.
        raw=df.t0_safety_robustness_target.fillna('AMBIGUOUS'); known=raw.isin([h.SAFE_LABEL,h.UNSAFE_LABEL])
        for name in b.POLICIES:
            ex=df[f'action__{name}'].eq('EXECUTE')
            for mode in ['original_fallback','resolved_only','all_ambiguous_safe','all_ambiguous_unsafe']:
                u=unsafe.copy(); keep=pd.Series(True,index=df.index)
                if mode=='resolved_only': keep=known
                elif mode=='all_ambiguous_safe': u.loc[~known]=False
                elif mode=='all_ambiguous_unsafe': u.loc[~known]=True
                selected=ex&keep; n=int(selected.sum()); un=int((selected&u).sum())
                ambiguity.append(dict(regime=regime,policy=name,convention=mode,executed=n,unsafe=un,ambiguous_executed=int((ex&~known).sum()),uer=un/n if n else np.nan))
        # Rerun frozen decision rules for human cost/residual-risk grid.
        for cH in protocol['cost_grid_c_H']:
            for rho in protocol['cost_grid_rho_H']:
                q=dict(params,c_H=cH,rho_H=rho)
                for name, gather in [('RCSE_V2_NO_GATHER',False),('RCSE_V2_FULL_RETROSPECTIVE',True)]:
                    action=p.v2_policy(df,C,q,True,True,gather)
                    post=b.reconstruct_post_gather_action(df,C,q)
                    postloss=b.realized_post_gather_continuation_loss(df,post,C,q)
                    for subset, ix in [('full_benchmark',df.index),('natural_only',df[df._intervention_family_r.eq('NATURAL')].index)]:
                        m,_=measure(df.loc[ix],action.loc[ix],C.loc[ix],q,unsafe.loc[ix],postloss.loc[ix])
                        baseline=float((cH+rho*C.loc[ix]).mean())
                        costs.append(dict(regime=regime,subset=subset,policy=name,c_H=cH,rho_H=rho,always_escalate_loss=baseline,loss_minus_always=m['loss']-baseline,**m))
        # Consequence-only removal: uniform decision costs, SAME log-consequence evaluation costs.
        for name,gather in [('RCSE_V2_NO_GATHER',False),('RCSE_V2_FULL_RETROSPECTIVE',True)]:
            for decision_mode in ['LOG_TRAIN_P90','UNIFORM']:
                decisionC=C if decision_mode=='LOG_TRAIN_P90' else pd.Series(1.,index=df.index)
                action=p.v2_policy(df,decisionC,params,True,True,gather)
                post=b.reconstruct_post_gather_action(df,decisionC,params)
                postloss=b.realized_post_gather_continuation_loss(df,post,C,params)
                m,_=measure(df,action,C,params,unsafe,postloss)
                cons.append(dict(regime=regime,policy=name,decision_consequence=decision_mode,evaluation_consequence='LOG_TRAIN_P90',**m))
        # Test fixed gate on unseen corruption mechanisms, paired with held-out natural states.
        test=natural[natural[cfg['column']].eq(cfg['test'])].copy()
        model=joblib.load(ROOT/f'safety_mismatch_revision/mismatch_model_{regime}.joblib')
        for corruption in ['ORIGINAL','NEGATIVE_PO_AGE','COHERENT_VALUE_SUBSTITUTION']:
            d=test.copy()
            if corruption=='NEGATIVE_PO_AGE': d['po_to_invoice_days']=-7.
            if corruption=='COHERENT_VALUE_SUBSTITUTION':
                for field in ['po_value_initial_eur','invoice_value_t0_eur','latest_gr_value_before_t0_eur']: d[field]*=10.
                d['inv_po_abs_rel_diff']=(d.invoice_value_t0_eur-d.po_value_initial_eur).abs()/d.po_value_initial_eur.abs().where(d.po_value_initial_eur.abs()>1e-12)
                d['inv_gr_abs_rel_diff']=(d.invoice_value_t0_eur-d.latest_gr_value_before_t0_eur).abs()/d.latest_gr_value_before_t0_eur.abs().where(d.latest_gr_value_before_t0_eur.abs()>1e-12)
            states=gate(g,d); _,score=h.score(model['pipeline'],model['calibrator'],d,model['features'])
            for subset,mask in [('all_natural',pd.Series(True,index=d.index)),('original_gate_valid',gate(g,test).eq('VALID'))]:
                stress.append(dict(regime=regime,corruption=corruption,subset=subset,cases=int(mask.sum()),gate_blocked=int((mask&states.ne('VALID')).sum()),gate_block_rate=float(states[mask].ne('VALID').mean()),score_ge_095=int((score[mask]>=.95).sum()),score_and_gate_selected=int(((score>=.95)&states.eq('VALID')&mask).sum())))
            pd.DataFrame(dict(source_case_id=d.source_case_id,corruption=corruption,gate_state=states,p_safe=score)).to_parquet(OUT/f'stress_{regime}_{corruption}.parquet',index=False)
        # Existing future price changes are outcome-defined subgroups, not observed corruption truth.
        for signal in ['outcome_future_price_change','outcome_future_quantity_change']:
            flag=h.as_bool(test[signal]); states=gate(g,test)
            stress.append(dict(regime=regime,corruption='NATURAL_SUBGROUP_'+signal,subset='outcome_defined_not_corruption',cases=int(flag.sum()),gate_blocked=int((flag&states.ne('VALID')).sum()),gate_block_rate=float(states[flag].ne('VALID').mean())))
        # Maturity restriction: retain original D1, restrict by available calendar observation.
        predictions=pd.read_csv(S/f't0_execution_safety_results/t0_safety_predictions_{regime}.csv',dtype={'source_case_id':str})
        d=test.merge(predictions[['source_case_id','p_safe_t0_cal']],on='source_case_id',validate='one_to_one')
        follow=(log_end-pd.to_datetime(d.decision_time,utc=True)).dt.total_seconds()/86400.
        for days in [0]+protocol['follow_up_days']:
            selected=d[follow>=days]; y=selected.target.eq(h.SAFE_LABEL).astype(int)
            maturity.append(dict(regime=regime,minimum_calendar_followup_days=days,cases=len(selected),unsafe=int((1-y).sum()),auroc=roc_auc_score(y,selected.p_safe_t0_cal) if y.nunique()==2 else np.nan,ap_unsafe=average_precision_score(1-y,1-selected.p_safe_t0_cal) if y.nunique()==2 else np.nan,observation_end_proxy=str(log_end)))
        for partition in ['train','calibration','test']:
            d=natural[natural[cfg['column']].eq(cfg[partition])]
            times=pd.to_datetime(d.decision_time,utc=True)
            boundaries.append(dict(regime=regime,partition=partition,cases=len(d),decision_min=str(times.min()),decision_max=str(times.max()),ambiguous=int(d.target.eq('AMBIGUOUS').sum()),median_calendar_followup_days=float((log_end-times).dt.total_seconds().median()/86400)))
        for rows,name in [(metrics,'all_policy_absolute_results.csv'),(comparisons,'paired_loss_vs_always_escalate_ci.csv'),(costs,'v2_human_cost_sensitivity.csv'),(ambiguity,'ambiguous_uer_sensitivity.csv'),(cons,'consequence_only_ablation.csv'),(stress,'independent_gate_stress.csv'),(maturity,'followup_maturity_sensitivity.csv'),(boundaries,'split_followup_details.csv')]: save(rows,name)
    # Preserve exact existing CI and decomposition evidence locally for report authoring.
    import shutil
    for folder,pattern in [('bootstrap_uncertainty','*.csv'),('unsafe_error_decomposition','*.csv'),('residual_contradiction_audit','*.csv')]:
        target=OUT/'frozen_evidence'/folder; target.mkdir(parents=True,exist_ok=True)
        for file in (R/folder).glob(pattern): shutil.copy2(file,target/file.name)
    pd.read_csv(R/'rcse_v2_policy_results_summary.csv').to_csv(OUT/'four_consequence_modes_frozen.csv',index=False)
    exp.groupby(['benchmark_arm','intervention_type','intervention_strength'],dropna=False).size().rename('episodes').reset_index().to_csv(OUT/'benchmark_composition.csv',index=False)
    inputs=[ROOT/'09b2l_rcse_v2_policy_spec.json',ROOT/'09b2n_run_rcse_v2_policy_simulation.py',ROOT/'09b2r_bootstrap_rcse_v2_uncertainty.py',ROOT/'rcse_output/rcse_base_v2.parquet',S/'rcse_experimental_with_splits.parquet',OUT/'analysis_protocol.json']
    manifest=dict(status='complete',protocol='analysis_protocol.json',bootstrap='paired source-case cluster percentile intervals; fixed policies',censoring_limit='maturity restrictions are not fixed-horizon relabeling or censoring correction',gather_limit='uses retrospective observed t1 scores in original simulation; prospective analyses disable GATHER',inputs_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in inputs})
    (OUT/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2))
    log('Evidence analyses complete')
if __name__=='__main__': main()
