"""Grouped holdouts, matched eight-chain selectors and component uncertainty.

Exploratory sensitivity on the previously inspected BPIC 2019 population.
Run with the numerical environment documented in requirements-reproduction.txt.
"""
from pathlib import Path
import sys,json,importlib.util,warnings
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import joblib
from scipy.stats import binom
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'revision_complete/robustness_extension'
OUT=ROOT/'revision_complete/grouped_component_extension'
spec=importlib.util.spec_from_file_location('robust',ROOT/'revision_complete/run_robustness_extension.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
STARTS=[.80,.85,.90,.92,.94,.95,.97,.99]
REGS=['iid','early_temporal90','ood','late_temporal']

def bootmean(delta,groups,reps=2000):
 delta=np.asarray(delta,float);codes,keys=pd.factorize(groups);tot=np.bincount(codes,weights=delta);counts=np.bincount(codes)
 rng=np.random.default_rng(20261023);boots=[]
 for _ in range(reps):
  ix=rng.integers(0,len(keys),len(keys));boots.append(tot[ix].sum()/counts[ix].sum())
 return dict(estimate=float(delta.mean()),lower=float(np.quantile(boots,.025)),upper=float(np.quantile(boots,.975)),clusters=len(keys),replicates=reps)

def aucboot(d,group):
 y=(~d.unsafe).to_numpy(int);a=d.lr_raw.to_numpy();b=d.gb_raw.to_numpy();codes,keys=pd.factorize(d[group]);rng=np.random.default_rng(20261023);boots=[]
 for _ in range(1000):
  weights=np.bincount(rng.integers(0,len(keys),len(keys)),minlength=len(keys))[codes]
  if np.unique(y[weights>0]).size==2:boots.append(roc_auc_score(y,b,sample_weight=weights)-roc_auc_score(y,a,sample_weight=weights))
 return dict(estimate=float(roc_auc_score(y,b)-roc_auc_score(y,a)),lower=float(np.quantile(boots,.025)),upper=float(np.quantile(boots,.975)),clusters=len(keys),replicates=len(boots))

def safetygraph(scores,unsafe,gate,values,cap):
 passed=[];audit=[]
 for start in STARTS:
  candidates=np.unique(np.r_[start,values[values<start]])[::-1]
  for threshold in candidates:
   ex=gate&(scores>=threshold);n=int(ex.sum());u=int((ex&unsafe).sum());p=float(binom.cdf(u,n,cap)) if n else 1.
   audit.append((start,float(threshold),n,u,p))
   if p>.05/8:break
   passed.append((n,float(threshold),start,u))
 return (sorted(passed,key=lambda x:(-x[0],x[1],x[2]))[0] if passed else (0,np.inf,np.nan,0)),audit

def masks(d,s,C):
 g=d.valid.to_numpy(bool)
 return {'Evidence complete':d.complete.to_numpy(bool),'Gate only':g,'Safety 0.95':s>=.95,'Gate safety 0.95':g&(s>=.95),'RCSE reference':m.chosen_rc(s,g,C,.95,.05),'Plug-in cost':m.chosen_rc(s,g,C,0.,.1),'Escalate all':np.zeros(len(d),bool)}

def evaluate(reg,d,ref,group,policyrows,contrasts,concentration):
 u=d.unsafe.to_numpy(bool);v=d.transaction_exposure_eur.abs().to_numpy(float);N=len(d)
 for model,col in [('Logistic','lr_raw'),('HistBoost','gb_raw')]:
  fit=joblib.load(OUT/f'calibration_{reg}_{model}.joblib' if reg.startswith('group_') else R/f'calibration_{reg}_{model}.joblib')
  s=fit['calibrator'].predict(d[col])
  for mode,C in [('Log',np.log1p(v)/np.log1p(ref)),('Linear',v/ref),('Capped linear',np.minimum(v/ref,10))]:
   rules=masks(d,s,C)
   for name,ex in rules.items():
    row=dict(regime=reg,model=model,consequence=mode,policy=name,**m.metrics(d,ex,C))
    row.update(unsafe_eur_per_1000=float(v[ex&u].sum()*1000/N),escalated_eur_per_1000=float(v[~ex].sum()*1000/N),executed_eur_per_1000=float(v[ex].sum()*1000/N))
    policyrows.append(row)
   pairs=[('Gate only','Evidence complete'),('Safety 0.95','Evidence complete'),('Gate safety 0.95','Safety 0.95'),('RCSE reference','Gate safety 0.95'),('RCSE reference','Safety 0.95'),('Plug-in cost','RCSE reference'),('Plug-in cost','Escalate all')]
   for left,right in pairs:
    delta=np.where(rules[left],u*C,.1)-np.where(rules[right],u*C,.1)
    for sampling,groups in [('Vendor cluster' if group=='vendor_id' else 'PO cluster',d[group]),('case',d.source_case_id)]:
     contrasts.append(dict(regime=reg,model=model,consequence=mode,comparison=left+' minus '+right,sampling=sampling,**bootmean(delta,groups)))
    nonzero=np.flatnonzero(np.abs(delta)>1e-14);order=nonzero[np.argsort(np.abs(delta[nonzero]))[::-1]]
    benefit=np.where(delta<0,-delta,0);cost=np.where(delta>0,delta,0)
    concentration.append(dict(regime=reg,model=model,consequence=mode,comparison=left+' minus '+right,changed_cases=len(nonzero),total_difference=float(delta.sum()),largest_abs_case=float(np.abs(delta).max()),benefit_top1_share=float(np.sort(benefit)[-1]/benefit.sum()) if benefit.sum() else np.nan,benefit_top5_share=float(np.sort(benefit)[-5:].sum()/benefit.sum()) if benefit.sum() else np.nan,estimate_without_largest_abs_case=float(np.delete(delta,order[0]).mean()) if len(order) else 0.,estimate_without_largest5_abs_cases=float(np.delete(delta,order[:5]).mean()) if len(order) else 0.))

def main():
 OUT.mkdir(exist_ok=True)
 (OUT/'protocol.json').write_text(json.dumps(dict(status='exploratory; design fixed before new grouped results',groups=['purchasing_document','vendor_id'],allocation='SHA256 all-eligible-20261023 plus group ID, 70/15/15; no group crosses train/calibration/test',calibration='20261004 hash of group ID, two halves; no group crosses calibration halves',selection_safety_starts=STARTS,selection='eight fixed starts; each descends distinct first-half isotonic values; delta .05/8 each; first nonrejection stops that chain; maximum selection coverage among rejected nodes',uncertainty='2000 paired cluster and case bootstrap loss contrasts, 1000 cluster AUROC differences; models fixed; no cross-family/seed guarantee; item dependence invalidates an unqualified item-binomial certificate',grouped_feature_set='same 40 predictors; group identifiers excluded',cost_optimal='plug-in VALID and (1-s)C <= cH=.10, tau0 beta.10; only optimal for the assumed calibrated-score model and fixed costs'),indent=2))
 h=m.load('group_h','09b2h_full_benchmark_safety_robustness.py');g=m.load('group_g','09b2k_evidence_gate_v2_audit.py')
 b=pd.read_parquet(ROOT/'rcse_output/rcse_base_v2.parquet');b['source_case_id']=b.case_id.astype(str);b['target']=m.target(b,h);b['known']=b.target.isin([h.SAFE_LABEL,h.UNSAFE_LABEL]);b['unsafe']=b.target.eq(h.UNSAFE_LABEL)
 b['complete']=~h.as_bool(b.goods_receipt_required)|h.as_bool(b.gr_available_at_t0);b['valid']=g.classify_gate(b,g.build_fixed_rules(b),g.build_value_rule(b,.05))[0].eq('VALID')
 for group in ['purchasing_document','vendor_id']:
  assert b[group].notna().all(),group
  b[group]=b[group].astype(str)
 fm=json.loads((ROOT/'rcse_output/experimental_benchmark/splits/t0_execution_safety_results/t0_safety_feature_manifest.json').read_text());num=fm['numeric_features'];cat=fm['categorical_features'];features=num+cat
 policyrows=[];contrasts=[];concentration=[];capacity=[];cohorts=[];overlaps=[];assign=[];counts=[];selection=[];selection_audit=[]
 # Measure existing sibling overlap without using it to tune the new splits.
 parts=pd.read_parquet(R/'all_eligible_split_assignments.parquet').merge(b[['source_case_id','purchasing_document','vendor_id']],on='source_case_id',validate='one_to_one')
 for group in ['purchasing_document','vendor_id']:
  tr=set(parts.loc[parts.iid.eq('train'),group]);te=parts[parts.iid.eq('test')]
  overlaps.append(dict(group=group,test_cases=len(te),test_cases_with_group_in_train=int(te[group].isin(tr).sum()),test_groups=te[group].nunique(),shared_train_test_groups=len(set(te[group])&tr)))
 for reg,group in [('group_po','purchasing_document'),('group_vendor','vendor_id')]:
  z=b.copy();u=m.split_hash(z[group]);z['part']=np.where(u<.7,'train',np.where(u<.85,'calibration','test'))
  assert z.groupby(group).part.nunique().max()==1
  assign.append(z[['source_case_id',group,'part']].assign(regime=reg))
  for p,d0 in z.groupby('part'):cohorts.append(dict(regime=reg,part=p,total=len(d0),resolved=int(d0.known.sum()),ambiguous=int((~d0.known).sum()),groups=d0[group].nunique()))
  tr=z[z.part.eq('train')&z.known];cal=z[z.part.eq('calibration')&z.known];ref=float(z[z.part.eq('train')].transaction_exposure_eur.abs().quantile(.9))
  print('Fitting',reg,len(tr),'resolved training cases',flush=True)
  lf=OUT/f'logistic_{reg}.joblib';gf=OUT/f'histboost_{reg}.joblib'
  if lf.exists():lr=joblib.load(lf)
  else:
   lr=h.build_pipeline(num,cat,C=1.,max_iter=5000,tol=1e-4)
   with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always',ConvergenceWarning);lr.fit(tr[features],(~tr.unsafe).astype(int))
   assert not any(issubclass(w.category,ConvergenceWarning) for w in caught);joblib.dump(lr,lf)
  if gf.exists():gb=joblib.load(gf)
  else:gb=m.nonlinear_fit(tr,num,cat,(~tr.unsafe).astype(int));joblib.dump(gb,gf)
  sd=z[z.part.ne('train')][['source_case_id',group,'part','target','known','unsafe','complete','valid','transaction_exposure_eur']].copy();src=z.set_index('source_case_id').loc[sd.source_case_id]
  sd['lr_raw']=lr.predict_proba(src[features])[:,1];sd['gb_raw']=gb['model'].predict_proba(m.nonlinear_matrix(src,gb))[:,1];sd.to_parquet(OUT/f'raw_scores_{reg}.parquet',index=False)
  cal=sd[sd.part.eq('calibration')&sd.known];ca=m.half(cal[group],20261004)==0
  assert not(set(cal.loc[ca,group])&set(cal.loc[~ca,group]))
  for model,col in [('Logistic','lr_raw'),('HistBoost','gb_raw')]:
   iso=IsotonicRegression(y_min=0,y_max=1,out_of_bounds='clip').fit(cal.loc[ca,col],(~cal.loc[ca,'unsafe']).astype(int));joblib.dump(dict(calibrator=iso,train_P90=ref,model=model),OUT/f'calibration_{reg}_{model}.joblib')
  te=sd[sd.part.eq('test')&sd.known].copy();sub=te[te.complete]
  capacity.append(dict(regime=reg,scope='evidence_complete',cases=len(sub),logistic_auc=roc_auc_score(~sub.unsafe,sub.lr_raw),boosted_auc=roc_auc_score(~sub.unsafe,sub.gb_raw),**aucboot(sub,group)))
  evaluate(reg,te,ref,group,policyrows,contrasts,concentration)
  print('Grouped evaluation complete',reg,flush=True)
 # Fixed original models: new plug-in row, exposure totals and all contrasts.
 for reg in REGS:
  sd=pd.read_parquet(R/f'raw_scores_{reg}.parquet').merge(b[['source_case_id','purchasing_document']],on='source_case_id',validate='one_to_one');te=sd[sd.part.eq('test')&sd.known].copy();cal=sd[sd.part.eq('calibration')&sd.known].copy()
  ref=joblib.load(R/f'calibration_{reg}_Logistic.joblib')['train_P90']
  evaluate(reg,te,ref,'purchasing_document',policyrows,contrasts,concentration)
  sub=te[te.complete];capacity.append(dict(regime=reg,scope='evidence_complete',cases=len(sub),logistic_auc=roc_auc_score(~sub.unsafe,sub.lr_raw),boosted_auc=roc_auc_score(~sub.unsafe,sub.gb_raw),**aucboot(sub,'purchasing_document')))
  for model,col in [('Logistic','lr_raw'),('HistBoost','gb_raw')]:
   for seed in m.SEEDS:
    ca=m.half(cal.source_case_id,seed)==0;cb=~ca
    iso=IsotonicRegression(y_min=0,y_max=1,out_of_bounds='clip').fit(cal.loc[ca,col],(~cal.loc[ca,'unsafe']).astype(int));sc=iso.predict(cal.loc[cb,col]);ts=iso.predict(te[col]);values=np.unique(iso.y_thresholds_)
    C=np.log1p(te.transaction_exposure_eur.abs().to_numpy())/np.log1p(ref)
    counts.append(dict(regime=reg,model=model,seed=seed,distinct_isotonic_values=len(values),test_distinct_values=len(np.unique(ts)),above_095_distinct=len(np.unique(ts[ts>=.95])),gated_above_p90=int((te.valid.to_numpy()&(ts>=.95)&(te.transaction_exposure_eur.abs().to_numpy()>ref)).sum()),extra_log_budget_removals=int((te.valid.to_numpy()&(ts>=.95)&((1-ts)*C>.05)).sum())))
    for cap in m.CAPS:
     for name,cg,tg in [('Safety graph',np.ones(cb.sum(),bool),np.ones(len(te),bool)),('Gate safety graph',cal.loc[cb,'valid'].to_numpy(bool),te.valid.to_numpy(bool))]:
      best,audit=safetygraph(sc,cal.loc[cb,'unsafe'].to_numpy(bool),cg,values,cap);n,t,start,u=best
      selection.append(dict(regime=reg,model=model,seed=seed,policy=name,cap=cap,selection_executed=n,selection_unsafe=u,certified=n>0,threshold=t,start=start,**m.metrics(te,tg&(ts>=t),C)))
      if seed==m.SEEDS[0]:selection_audit.extend(dict(regime=reg,model=model,policy=name,cap=cap,start=st,threshold=th,n=nn,u=uu,pvalue=p,delta=.05/8) for st,th,nn,uu,p in audit)
  print('Original regime component and graph audit complete',reg,flush=True)
 for data,name in [(policyrows,'fixed_policies'),(contrasts,'loss_contrasts'),(concentration,'loss_concentration'),(capacity,'capacity_auc'),(cohorts,'grouped_cohorts'),(overlaps,'original_group_overlap'),(counts,'isotonic_steps'),(selection,'safety_graph_results'),(selection_audit,'safety_graph_audit')]:pd.DataFrame(data).to_csv(OUT/f'{name}.csv',index=False)
 pd.concat(assign,ignore_index=True).to_parquet(OUT/'grouped_split_assignments.parquet',index=False)
 q=pd.DataFrame(selection);q.groupby(['regime','model','policy','cap']).agg(repetitions=('seed','size'),certification_fraction=('certified','mean'),coverage_median=('coverage','median'),uer_median=('uer','median'),loss_median=('mean_loss','median')).reset_index().to_csv(OUT/'safety_graph_summary.csv',index=False)
 (OUT/'completion.json').write_text(json.dumps(dict(status='complete',grouped_fits=4,graph_rows=len(selection),fixed_rows=len(policyrows),contrast_rows=len(contrasts),all_group_partitions_disjoint=True),indent=2))
 print('All grouped and component analyses complete',flush=True)

if __name__=='__main__':
 with threadpool_limits(limits=4):main()
