"""Signed component tables and an independently refreshed tail diagnostic."""
from pathlib import Path
import json,importlib.util,sys
sys.dont_write_bytecode=True
import pandas as pd
import numpy as np
import joblib
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'revision_complete/alignment_extension';G=ROOT/'revision_complete/grouped_component_extension';R=ROOT/'revision_complete/robustness_extension'
spec=importlib.util.spec_from_file_location('gaudit',ROOT/'revision_complete/run_grouped_component_extension.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
lc=pd.read_csv(G/'loss_contrasts.csv');rows=[]
for reg in ['iid','group_po','early_temporal90','ood']:
 for component,left,right in [('Gate','Gate only','Evidence complete'),('Safety','Safety 0.95','Evidence complete'),('Consequence constraint','RCSE reference','Gate safety 0.95'),('Reference constraints vs plug-in','RCSE reference','Plug-in cost')]:
  for model in (['Logistic'] if component=='Gate' else ['Logistic','HistBoost']):
   reverse=right=='Plug-in cost';comparison=(right+' minus '+left) if reverse else (left+' minus '+right)
   x=lc[lc.regime.eq(reg)&lc.model.eq(model)&lc.consequence.eq('Log')&lc.comparison.eq(comparison)&lc.sampling.eq('PO cluster')].iloc[0]
   rows.append(dict(regime=reg,component=component,model=model,comparison=left+' minus '+right,estimate=-x.estimate if reverse else x.estimate,lower=-x.upper if reverse else x.lower,upper=-x.lower if reverse else x.upper))
pd.DataFrame(rows).to_csv(OUT/'component_effects.csv',index=False)
d=pd.read_parquet(R/'raw_scores_early_temporal90.parquet');d=d[d.part.eq('test')&d.known].copy();groups=pd.read_parquet(ROOT/'rcse_output/rcse_base_v2.parquet',columns=['case_id','purchasing_document']);groups['source_case_id']=groups.case_id.astype(str);d=d.merge(groups[['source_case_id','purchasing_document']],on='source_case_id',validate='one_to_one')
fit=joblib.load(R/'calibration_early_temporal90_HistBoost.joblib');s=fit['calibrator'].predict(d.gb_raw);C=d.transaction_exposure_eur.abs().to_numpy()/fit['train_P90'];u=d.unsafe.to_numpy(bool)
rc=m.m.chosen_rc(s,d.valid.to_numpy(bool),C,.95,.05);safety=s>=.95;delta=np.where(rc,u*C,.1)-np.where(safety,u*C,.1);order=np.argsort(np.abs(delta))[::-1];benefit=np.maximum(-delta,0)
audit=dict(model='HistBoost',regime='early_temporal90',consequence='Linear',comparison='RCSE reference minus Safety 0.95',**m.bootmean(delta,d.purchasing_document),benefit_top1_share=float(benefit.max()/benefit.sum()),benefit_top5_share=float(np.sort(benefit)[-5:].sum()/benefit.sum()),estimate_without_largest_abs_case=float(np.delete(delta,order[0]).mean()),estimate_without_largest5_abs_cases=float(np.delete(delta,order[:5]).mean()))
old=pd.read_csv(G/'loss_concentration.csv');old=old[old.model.eq('HistBoost')&old.regime.eq('early_temporal90')&old.consequence.eq('Linear')&old.comparison.eq(audit['comparison'])].iloc[0]
for key in ['benefit_top1_share','benefit_top5_share','estimate_without_largest_abs_case','estimate_without_largest5_abs_cases']:assert np.isclose(audit[key],old[key])
(OUT/'boosted_linear_temporal_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit))
