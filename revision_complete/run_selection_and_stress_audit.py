"""Recheck the old 2% headline and probe fixed full-population models."""
from pathlib import Path
import importlib.util,sys,json
sys.dont_write_bytecode=True
import pandas as pd
import numpy as np
import joblib
from scipy.stats import binom
from sklearn.isotonic import IsotonicRegression
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent/'robustness_extension'
def load(name,file):
 s=importlib.util.spec_from_file_location(name,ROOT/file);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def main():
 e=load('ext','revision_complete/run_robustness_extension.py');h=load('legacy_h','09b2h_full_benchmark_safety_robustness.py');g=load('stress_g','09b2k_evidence_gate_v2_audit.py')
 base=pd.read_parquet(ROOT/'rcse_output/rcse_base_v2.parquet');base['source_case_id']=base.case_id.astype(str)
 exp=pd.read_parquet(ROOT/'rcse_output/experimental_benchmark/splits/rcse_experimental_with_splits.parquet')
 old=base.merge(h.build_split_map(exp),on='source_case_id',validate='one_to_one');old['target']=h.build_d1_target(old)
 old=old[old.target.isin([h.SAFE_LABEL,h.UNSAFE_LABEL])].copy();rows=[];stress=[];legacy_cost=[]
 grid=[0.,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,.96,.97,.98,.99,1.]
 spec=json.loads((ROOT/'09b2l_rcse_v2_policy_spec.json').read_text());ta=spec['risk_and_selectivity']['tau_safe_t0_grid'];be=spec['risk_and_selectivity']['beta0_grid']
 for reg,cfg in h.SPLIT_CONFIG.items():
  model=joblib.load(OUT.parent/f'separate_calibration_safety_{reg}.joblib');cal=old[old[cfg['column']].eq(cfg['calibration'])].copy();test=old[old[cfg['column']].eq(cfg['test'])].copy()
  cr=model['pipeline'].predict_proba(cal[model['features']])[:,1];tr=model['pipeline'].predict_proba(test[model['features']])[:,1]
  cg=g.classify_gate(cal,g.build_fixed_rules(cal),g.build_value_rule(cal,.05))[0].eq('VALID').to_numpy();tg=g.classify_gate(test,g.build_fixed_rules(test),g.build_value_rule(test,.05))[0].eq('VALID').to_numpy()
  ref=float(pd.read_parquet(ROOT/f'rcse_output/experimental_benchmark/splits/frozen_baseline_results/rcse_policy_dataset/rcse_v2_policy_dataset/rcse_v2_policy_dataset_{reg}.parquet',columns=['train_exposure_p90_eur']).iloc[0,0])
  cc=np.log1p(cal.transaction_exposure_eur.abs().to_numpy())/np.log1p(ref);tc=np.log1p(test.transaction_exposure_eur.abs().to_numpy())/np.log1p(ref)
  test['unsafe']=test.target.eq(h.UNSAFE_LABEL)
  for seed in e.SEEDS:
   a=e.half(cal.source_case_id,seed)==0;b=~a
   iso=IsotonicRegression(y_min=0,y_max=1,out_of_bounds='clip').fit(cr[a],cal.loc[a,'target'].eq(h.SAFE_LABEL).astype(int));cs=iso.predict(cr[b]);ts=iso.predict(tr);u=cal.loc[b,'target'].eq(h.UNSAFE_LABEL).to_numpy()
   if seed==e.SEEDS[0]:
    comp=(~h.as_bool(test.goods_receipt_required)|h.as_bool(test.gr_available_at_t0)).to_numpy();safe=ts>=.95;rc=e.chosen_rc(ts,tg,tc,.95,.05)
    for name,left,right in [('RCSE minus Safety 0.95',rc,safe),('Gate minus completeness',tg,comp),('RCSE minus Gate safety 0.95',rc,tg&safe)]:
     delta=left.astype(int)-right.astype(int);removed=-int(delta.sum());avoided=-float(np.sum(delta*test.unsafe.to_numpy()*tc))
     legacy_cost.append(dict(regime=reg,comparison=name,removed=removed,unsafe_removed=-int(np.sum(delta*test.unsafe.to_numpy())),break_even_cH=avoided/removed if removed else np.nan,population='earlier intervention-source natural resolved sample'))
   for cap in e.CAPS:
    for name,gate0,gate1,gatea in [('Safety FST',np.ones(b.sum(),bool),np.ones(len(test),bool),np.ones(a.sum(),bool)),('Gate safety FST',cg[b],tg,cg[a]),('Safety split FST',np.ones(b.sum(),bool),np.ones(len(test),bool),np.ones(a.sum(),bool)),('Gate safety split FST',cg[b],tg,cg[a])]:
     candidates=np.unique(iso.y_thresholds_)[::-1]
     if 'split' in name:candidates=e.learned_candidates(iso.predict(cr[a]),cal.loc[a,'target'].eq(h.UNSAFE_LABEL).to_numpy(),gatea,candidates,cap)
     best,_=e.sequence(cs,u,gate0,candidates,cap,.05);n,t,uu=best
     rows.append(dict(regime=reg,seed=seed,cap=cap,policy=name,certified=n>0,**e.metrics(test,gate1&(ts>=t),tc)))
    options=[]
    for t in grid:
     sel=cr[b]>=t;n=int(sel.sum());uu=int((sel&u).sum())
     if n and binom.cdf(uu,n,cap)<=.05/25:options.append((n,t))
    n,t=sorted(options,key=lambda x:(-x[0],x[1]))[0] if options else (0,np.inf)
    rows.append(dict(regime=reg,seed=seed,cap=cap,policy='Safety old Bonferroni',certified=n>0,**e.metrics(test,tr>=t,tc)))
    options=[]
    for t in ta:
     for beta in be:
      sel=e.chosen_rc(cs,cg[b],cc[b],t,beta);n=int(sel.sum());uu=int((sel&u).sum())
      if n and binom.cdf(uu,n,cap)<=.05/(len(ta)*len(be)):options.append((n,t,beta))
    n,t,beta=sorted(options,key=lambda x:(-x[0],x[1],x[2]))[0] if options else (0,np.inf,0.)
    rows.append(dict(regime=reg,seed=seed,cap=cap,policy='RCSE old Bonferroni',certified=n>0,**e.metrics(test,e.chosen_rc(ts,tg,tc,t,beta),tc)))
  print('Legacy selection reanalysis',reg,flush=True)
 pd.DataFrame(rows).to_csv(OUT/'legacy_selection_reanalysis.csv',index=False)
 pd.DataFrame(legacy_cost).to_csv(OUT/'legacy_break_even.csv',index=False)
 assignments=pd.read_parquet(OUT/'all_eligible_split_assignments.parquet')
 for reg in ['iid','late_temporal','early_temporal90','ood']:
  ids=assignments.loc[assignments[reg].eq('test'),'source_case_id'];d=base.set_index('source_case_id').loc[ids].copy();origvalid=g.classify_gate(d,g.build_fixed_rules(d),g.build_value_rule(d,.05))[0].eq('VALID').to_numpy()
  for model,label in [('logistic','Logistic'),('histboost','HistBoost')]:
   fitted=joblib.load(OUT/f'{model}_{reg}.joblib');cal=joblib.load(OUT/f'calibration_{reg}_{label}.joblib')
   def score(x):
    raw=fitted.predict_proba(x[cal['features']])[:,1] if model=='logistic' else fitted['model'].predict_proba(e.nonlinear_matrix(x,fitted))[:,1]
    return cal['calibrator'].predict(raw)
   before=score(d)
   for probe in ['Negative PO age','Coherent amount substitution']:
    changed=d.copy()
    if probe=='Negative PO age':changed['po_to_invoice_days']=-7.
    else:
     for c in ['po_value_initial_eur','invoice_value_t0_eur','latest_gr_value_before_t0_eur']:changed[c]=changed[c]*10
    after=score(changed);newvalid=g.classify_gate(changed,g.build_fixed_rules(changed),g.build_value_rule(changed,.05))[0].eq('VALID').to_numpy()
    stress.append(dict(regime=reg,model=label,probe=probe,test_cases=len(d),original_gate_valid=int(origvalid.sum()),newly_gate_blocked=int((origvalid&~newvalid).sum()),original_high_safety=int((origvalid&(before>=.95)).sum()),perturbed_high_safety=int((origvalid&(after>=.95)).sum()),mean_score_change=float((after[origvalid]-before[origvalid]).mean()),scope='specification-gap and score-sensitivity probe; no downstream harm label assigned'))
 pd.DataFrame(stress).to_csv(OUT/'score_integrity_probes.csv',index=False)
 part=base[['source_case_id']].merge(h.build_split_map(exp),on='source_case_id').merge(old[['source_case_id','target']],on='source_case_id',how='left')
 part.groupby('split_ood_exposure').agg(total=('source_case_id','size'),resolved=('target','count')).reset_index().to_csv(OUT/'legacy_ood_partition_reconciliation.csv',index=False)
 print('Selection and score-integrity audit complete',flush=True)
if __name__=='__main__':
 with threadpool_limits(limits=4):main()
