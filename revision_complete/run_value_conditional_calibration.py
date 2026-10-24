"""Independent value-bin calibration and held-out calibration diagnostics."""
from pathlib import Path
import importlib.util,sys,json
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import joblib
from sklearn.isotonic import IsotonicRegression
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'revision_complete/robustness_extension';G=ROOT/'revision_complete/grouped_component_extension';A=ROOT/'revision_complete/alignment_extension';OUT=ROOT/'revision_complete/value_calibration_extension'
spec=importlib.util.spec_from_file_location('vc_component',ROOT/'revision_complete/run_grouped_component_extension.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
REGS=['iid','group_po','early_temporal90','ood']

def rate_diff(d,score,ref):
 selected=d.valid.to_numpy(bool)&(score>=.95);high=d.transaction_exposure_eur.abs().to_numpy()>ref;u=d.unsafe.to_numpy(bool)
 codes,keys=pd.factorize(d.purchasing_document);nlow=np.bincount(codes,weights=selected&~high);nhigh=np.bincount(codes,weights=selected&high);ulow=np.bincount(codes,weights=selected&~high&u);uhigh=np.bincount(codes,weights=selected&high&u)
 rng=np.random.default_rng(20261024);boot=[]
 for _ in range(2000):
  ix=rng.integers(0,len(keys),len(keys));lo=nlow[ix].sum();hi=nhigh[ix].sum()
  if lo and hi:boot.append(uhigh[ix].sum()/hi-ulow[ix].sum()/lo)
 return dict(estimate=float(uhigh.sum()/nhigh.sum()-ulow.sum()/nlow.sum()) if nhigh.sum() and nlow.sum() else np.nan,lower=float(np.quantile(boot,.025)) if boot else np.nan,upper=float(np.quantile(boot,.975)) if boot else np.nan,replicates=len(boot),clusters=len(keys))

def main():
 OUT.mkdir(exist_ok=True)
 (OUT/'protocol.json').write_text(json.dumps(dict(status='exploratory mechanism check; schemes fixed before new result inspection',regimes=REGS,seeds=m.m.SEEDS,schemes={'P90 bins':'<= training P90 versus > training P90','Quartile bins':'training P25 P50 P75 cutpoints'},calibration='Fit separate isotonic models on raw scores in each value bin using only the first calibration half; predict independent second half and fixed test. One pooled first-half fit is a fallback only if a bin is empty.',graph='same eight scalar starts and delta/8; descending union of first-half fitted bin anchor values; all four caps; gated and ungated variants',diagnostic='Independent selection-half UER within gate and globally calibrated score >=.95, split at training P90, plus score bands [.95,.97), [.97,.99), [.99,1]. First-seed PO-cluster bootstrap high-minus-low difference, 2000 reps, seed20261024.',limits='Risk differences within a broad score range do not isolate conditional miscalibration. Bin schemes are finite tests of one mechanism, not exhaustive calibration models. Certification remains nominal under item dependence; no shift guarantee.'),indent=2))
 base=pd.read_parquet(ROOT/'rcse_output/rcse_base_v2.parquet',columns=['case_id','purchasing_document','transaction_exposure_eur']);base['source_case_id']=base.case_id.astype(str)
 assignments=pd.read_parquet(R/'all_eligible_split_assignments.parquet');po=pd.read_parquet(G/'grouped_split_assignments.parquet');rows=[];diagnostics=[];diffs=[];fits=[];bins=[]
 for reg in REGS:
  src=G if reg=='group_po' else R;sd=pd.read_parquet(src/f'raw_scores_{reg}.parquet')
  if 'purchasing_document' not in sd:sd=sd.merge(base[['source_case_id','purchasing_document']],on='source_case_id',validate='one_to_one')
  cal=sd[sd.part.eq('calibration')&sd.known].copy();test=sd[sd.part.eq('test')&sd.known].copy();ref=joblib.load(src/f'calibration_{reg}_Logistic.joblib')['train_P90']
  ids=set(po.loc[po.regime.eq('group_po')&po.part.eq('train'),'source_case_id']) if reg=='group_po' else set(assignments.loc[assignments[reg].eq('train'),'source_case_id'])
  trainv=base.loc[base.source_case_id.isin(ids),'transaction_exposure_eur'].abs();assert np.isclose(trainv.quantile(.9),ref)
  schemes={'P90 bins':np.array([ref]),'Quartile bins':np.unique(trainv.quantile([.25,.5,.75]).to_numpy())}
  for scheme,cuts in schemes.items():bins.append(dict(regime=reg,scheme=scheme,cutpoints_eur=json.dumps(cuts.tolist())))
  cv=cal.transaction_exposure_eur.abs().to_numpy();tv=test.transaction_exposure_eur.abs().to_numpy();C=np.log1p(tv)/np.log1p(ref)
  for model,col in [('Logistic','lr_raw'),('HistBoost','gb_raw')]:
   for seed in m.m.SEEDS:
    ca=m.m.half(cal.purchasing_document if reg=='group_po' else cal.source_case_id,seed)==0;cb=~ca
    pooled=IsotonicRegression(y_min=0,y_max=1,out_of_bounds='clip').fit(cal.loc[ca,col],(~cal.loc[ca,'unsafe']).astype(int));pooledcal=pooled.predict(cal.loc[cb,col]);pooledtest=pooled.predict(test[col]);sel=cal.loc[cb].copy();cu=sel.unsafe.to_numpy(bool);cg=sel.valid.to_numpy(bool);tg=test.valid.to_numpy(bool)
    for band,lo,hi in [('s>=.95',.95,1.01),('[.95,.97)',.95,.97),('[.97,.99)',.97,.99),('[.99,1]',.99,1.01)]:
     for high in [False,True]:
      mask=cg&(pooledcal>=lo)&(pooledcal<hi)&((cv[cb]>ref)==high);n=int(mask.sum());bad=int((mask&cu).sum())
      diagnostics.append(dict(regime=reg,model=model,seed=seed,score_band=band,value_bin='Above P90' if high else 'At most P90',selected=n,unsafe=bad,uer=bad/n if n else np.nan,mean_score=float(pooledcal[mask].mean()) if n else np.nan,expected_uer=float((1-pooledcal[mask]).mean()) if n else np.nan))
    if seed==m.m.SEEDS[0]:diffs.append(dict(regime=reg,model=model,**rate_diff(sel,pooledcal,ref)))
    for scheme,cuts in [('Pooled',None)]+list(schemes.items()):
     if cuts is None:sc=pooledcal;ts=pooledtest;values=np.unique(pooled.y_thresholds_)
     else:
      group=np.searchsorted(cuts,cv,side='left');testgroup=np.searchsorted(cuts,tv,side='left');sc=np.empty(cb.sum());ts=np.empty(len(test));values=[];models=[]
      for bi in range(len(cuts)+1):
       fitmask=ca&(group==bi);calmask=group[cb]==bi;testmask=testgroup==bi
       iso=IsotonicRegression(y_min=0,y_max=1,out_of_bounds='clip').fit(cal.loc[fitmask,col],(~cal.loc[fitmask,'unsafe']).astype(int)) if fitmask.any() else pooled
       if calmask.any():sc[calmask]=iso.predict(sel.loc[calmask,col])
       if testmask.any():ts[testmask]=iso.predict(test.loc[testmask,col])
       values.extend(iso.y_thresholds_);models.append(iso)
       fits.append(dict(regime=reg,model=model,seed=seed,scheme=scheme,bin=bi,fit_cases=int(fitmask.sum()),selection_cases=int(calmask.sum()),test_cases=int(testmask.sum()),fallback=not fitmask.any()))
      values=np.unique(values)
      if seed==m.m.SEEDS[0]:joblib.dump(dict(calibrators=models,cutpoints=cuts,train_P90=ref),OUT/f'{reg}_{model}_{scheme.replace(" ","_")}.joblib')
     for cap in m.m.CAPS:
      for policy,cg0,tg0 in [('Safety graph',np.ones(cb.sum(),bool),np.ones(len(test),bool)),('Gate safety graph',cg,tg)]:
       best,_=m.safetygraph(sc,cu,cg0,values,cap);n,t,start,u=best
       rows.append(dict(regime=reg,model=model,seed=seed,scheme=scheme,policy=policy,cap=cap,certified=n>0,selection_executed=n,selection_unsafe=u,threshold=t,**m.m.metrics(test,tg0&(ts>=t),C)))
   print('Value-conditional calibration complete',reg,model,flush=True)
  for data,name in [(rows,'selection_results'),(diagnostics,'conditional_calibration_rates'),(diffs,'p90_risk_difference'),(fits,'calibration_fit_audit'),(bins,'value_bin_boundaries')]:pd.DataFrame(data).to_csv(OUT/f'{name}.csv',index=False)
 q=pd.DataFrame(rows);q.groupby(['regime','model','scheme','policy','cap']).agg(repetitions=('seed','size'),certification_fraction=('certified','mean'),coverage_median=('coverage','median'),uer_median=('uer','median'),loss_median=('mean_loss','median')).reset_index().to_csv(OUT/'selection_summary.csv',index=False)
 assert len(rows)==3840 and q.groupby(['regime','model','scheme','policy','cap']).seed.nunique().eq(20).all()
 old=pd.concat([pd.read_csv(G/'safety_graph_results.csv'),pd.read_csv(A/'po_grouped_safety_graph.csv')]);new=q[q.scheme.eq('Pooled')];joined=new.merge(old,on=['regime','model','seed','policy','cap'],suffixes=('_new','_old'),validate='one_to_one');assert len(joined)==1280
 for col in ['executed','unsafe','mean_loss']:assert np.allclose(joined[col+'_new'],joined[col+'_old'])
 (OUT/'validation.json').write_text(json.dumps(dict(status='passed',rows=len(rows),twenty_calibration_splits=True,pooled_baseline_matches_archive=True,first_half_only_fits=True,train_only_value_bins=True,empty_bin_fallbacks=sum(x['fallback'] for x in fits)),indent=2));print('Value calibration checks passed',flush=True)

if __name__=='__main__':main()
