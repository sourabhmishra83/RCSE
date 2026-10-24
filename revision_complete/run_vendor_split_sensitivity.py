"""Ten vendor holdouts with frozen hyperparameters and cluster diagnostics."""
from pathlib import Path
import sys,json,hashlib,importlib.util,warnings
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score
from sklearn.isotonic import IsotonicRegression
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];G=ROOT/'revision_complete/grouped_component_extension';OUT=ROOT/'revision_complete/alignment_extension'
spec=importlib.util.spec_from_file_location('groupaudit',ROOT/'revision_complete/run_grouped_component_extension.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
SEEDS=[20261023]+list(range(20261024,20261033))

def concentration(d):
 share=d.vendor_id.value_counts(normalize=True)
 return dict(vendors=len(share),top_vendor_share=float(share.iloc[0]),top5_vendor_share=float(share.head(5).sum()),effective_vendors=float(1/(share**2).sum()))

def main():
 OUT.mkdir(exist_ok=True)
 (OUT/'vendor_protocol.json').write_text(json.dumps(dict(status='exploratory split sensitivity',seeds=SEEDS,first_seed='exact original hash of all-eligible-20261023 plus vendor ID',other_seeds='SHA256 vendor-heldout-{seed}: plus vendor ID, 70/15/15; vendor sizes not balanced',calibration='group-disjoint halves from 20261004 plus vendor ID',features='same 40 predictors; vendor_id and vendor_name_anon excluded; vendor-invoice availability and time intervals are process features',inference='1000 vendor-cluster AUROC replicates per split, fixed fitted model; ten split estimates are descriptive sensitivity, not independent external replications',models='original fixed logistic and HistGradientBoosting hyperparameters; no test tuning'),indent=2))
 h=m.m.load('vendor_h','09b2h_full_benchmark_safety_robustness.py');g=m.m.load('vendor_g','09b2k_evidence_gate_v2_audit.py');b=pd.read_parquet(ROOT/'rcse_output/rcse_base_v2.parquet');b['source_case_id']=b.case_id.astype(str);b['vendor_id']=b.vendor_id.astype(str)
 b['target']=m.m.target(b,h);b['known']=b.target.isin([h.SAFE_LABEL,h.UNSAFE_LABEL]);b['unsafe']=b.target.eq(h.UNSAFE_LABEL);b['complete']=~h.as_bool(b.goods_receipt_required)|h.as_bool(b.gr_available_at_t0);b['valid']=g.classify_gate(b,g.build_fixed_rules(b),g.build_value_rule(b,.05))[0].eq('VALID')
 fm=json.loads((ROOT/'rcse_output/experimental_benchmark/splits/t0_execution_safety_results/t0_safety_feature_manifest.json').read_text());num=fm['numeric_features'];cat=fm['categorical_features'];features=num+cat
 assert not set(features)&{'vendor_id','vendor_name_anon','purchasing_document','case_id'}
 rows=[];coh=[];parts=[]
 for seed in SEEDS:
  u=m.m.split_hash(b.vendor_id) if seed==SEEDS[0] else b.vendor_id.map(lambda x:int(hashlib.sha256((f'vendor-heldout-{seed}:'+x).encode()).hexdigest()[:12],16)/(16**12)).to_numpy()
  z=b.copy();z['part']=np.where(u<.7,'train',np.where(u<.85,'calibration','test'));assert z.groupby('vendor_id').part.nunique().max()==1
  parts.append(z[['source_case_id','vendor_id','part']].assign(seed=seed))
  for part,d in z.groupby('part'):coh.append(dict(seed=seed,part=part,cases=len(d),resolved=int(d.known.sum()),**concentration(d)))
  tr=z[z.part.eq('train')&z.known];cal=z[z.part.eq('calibration')&z.known];te=z[z.part.eq('test')&z.known].copy();ref=float(z[z.part.eq('train')].transaction_exposure_eur.abs().quantile(.9))
  if seed==SEEDS[0]:
   sd=pd.read_parquet(G/'raw_scores_group_vendor.parquet');te=sd[sd.part.eq('test')&sd.known].copy();cal=sd[sd.part.eq('calibration')&sd.known].copy()
  else:
   lf=OUT/f'vendor_{seed}_logistic.joblib';gf=OUT/f'vendor_{seed}_histboost.joblib'
   if lf.exists():lr=joblib.load(lf)
   else:
    lr=h.build_pipeline(num,cat,C=1.,max_iter=5000,tol=1e-4)
    with warnings.catch_warnings(record=True) as caught:
     warnings.simplefilter('always',ConvergenceWarning);lr.fit(tr[features],(~tr.unsafe).astype(int))
    assert not any(issubclass(w.category,ConvergenceWarning) for w in caught);joblib.dump(lr,lf)
   if gf.exists():gb=joblib.load(gf)
   else:gb=m.m.nonlinear_fit(tr,num,cat,(~tr.unsafe).astype(int));joblib.dump(gb,gf)
   score=z[z.part.ne('train')][['source_case_id','vendor_id','part','known','unsafe','complete','valid','transaction_exposure_eur']].copy();src=z.set_index('source_case_id').loc[score.source_case_id]
   score['lr_raw']=lr.predict_proba(src[features])[:,1];score['gb_raw']=gb['model'].predict_proba(m.m.nonlinear_matrix(src,gb))[:,1];score.to_parquet(OUT/f'vendor_{seed}_scores.parquet',index=False)
   te=score[score.part.eq('test')&score.known].copy();cal=score[score.part.eq('calibration')&score.known].copy()
  sub=te[te.complete];diff=m.aucboot(sub,'vendor_id')
  rows.append(dict(seed=seed,cases=len(sub),logistic_auc=roc_auc_score(~sub.unsafe,sub.lr_raw),boosted_auc=roc_auc_score(~sub.unsafe,sub.gb_raw),**diff,**concentration(sub)))
  # Fixed-policy results on these split-specific calibrated scores.
  ca=m.m.half(cal.vendor_id,20261004)==0;assert not(set(cal.loc[ca,'vendor_id'])&set(cal.loc[~ca,'vendor_id']))
  fixedrows=[]
  for model,col in [('Logistic','lr_raw'),('HistBoost','gb_raw')]:
   iso=IsotonicRegression(y_min=0,y_max=1,out_of_bounds='clip').fit(cal.loc[ca,col],(~cal.loc[ca,'unsafe']).astype(int));s=iso.predict(te[col]);C=np.log1p(te.transaction_exposure_eur.abs().to_numpy())/np.log1p(ref)
   joblib.dump(dict(calibrator=iso,train_P90=ref,model=model),OUT/f'vendor_{seed}_{model}_calibration.joblib')
   for policy,ex in m.masks(te,s,C).items():fixedrows.append(dict(seed=seed,model=model,policy=policy,**m.m.metrics(te,ex,C)))
  pd.DataFrame(fixedrows).to_csv(OUT/f'vendor_{seed}_fixed_policies.csv',index=False)
  pd.DataFrame(rows).to_csv(OUT/'vendor_repeated_auc.csv',index=False);pd.DataFrame(coh).to_csv(OUT/'vendor_repeated_cohorts.csv',index=False)
  print('Vendor split complete',seed,'AUC gap',round(diff['estimate'],4),flush=True)
 pd.concat(parts,ignore_index=True).to_parquet(OUT/'vendor_repeated_assignments.parquet',index=False)
 summarize(b)

def summarize(b):
 po=pd.read_parquet(G/'grouped_split_assignments.parquet');po=po.loc[po.regime.eq('group_po'),['source_case_id','part']].merge(b[['source_case_id','vendor_id']],on='source_case_id',validate='one_to_one');trainvendors=set(po.loc[po.part.eq('train'),'vendor_id']);test=po[po.part.eq('test')]
 q=pd.read_csv(OUT/'vendor_repeated_auc.csv');result=dict(splits=len(SEEDS),auc_gap_median=float(q.estimate.median()),auc_gap_min=float(q.estimate.min()),auc_gap_max=float(q.estimate.max()),positive_gap_splits=int(q.estimate.gt(0).sum()),cluster_intervals_positive=int(q.lower.gt(0).sum()),cluster_intervals_negative=int(q.upper.lt(0).sum()),test_top_vendor_share_min=float(q.top_vendor_share.min()),test_top_vendor_share_max=float(q.top_vendor_share.max()),test_effective_vendors_min=float(q.effective_vendors.min()),test_effective_vendors_max=float(q.effective_vendors.max()),po_test_cases=len(test),po_test_vendor_seen_train_cases=int(test.vendor_id.isin(trainvendors).sum()))
 (OUT/'vendor_repeated_summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':
 with threadpool_limits(limits=4):main()
