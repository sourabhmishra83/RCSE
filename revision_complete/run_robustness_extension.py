"""All-eligible study, independent repeated calibration, sequential selection,
nonlinear comparison, fixed-action break-even and consequence sensitivities.

Original analyses are preserved. No test outcome selects a rule or model setting.
"""
from pathlib import Path
import importlib.util,json,hashlib,sys,time,warnings
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import joblib
from scipy.stats import binom
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent/'robustness_extension'
SEEDS=[20261004]+list(range(20261024,20261043))
CAPS=[.02,.03,.05,.10]
TAUS=[0.,.5,.7,.8,.9,.95,.97,.99]
BETAS=[.01,.02,.03,.05,.075,.10]

def load(name,file):
 s=importlib.util.spec_from_file_location(name,ROOT/file);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def log(s):print(time.strftime('%H:%M:%S'),s,flush=True)
def half(ids,seed):return ids.map(lambda x:int(hashlib.sha256((str(seed)+str(x)).encode()).hexdigest()[:8],16)%2).to_numpy()
def split_hash(ids):return ids.map(lambda x:int(hashlib.sha256(('all-eligible-20261023:'+str(x)).encode()).hexdigest()[:12],16)/(16**12)).to_numpy()
def target(df,h):return h.build_d1_target(df)
def chosen_rc(s,g,C,tau,beta,cH=.1):return g & (s>=tau) & ((1-s)*C<=min(beta,cH))
def metrics(d,sel,C):
 sel=np.asarray(sel,bool);u=d.unsafe.to_numpy(bool);v=d.transaction_exposure_eur.abs().to_numpy(float)
 n=int(sel.sum());bad=int((sel&u).sum());den=float(v[sel].sum())
 return dict(cases=len(d),executed=n,unsafe=bad,coverage=n/len(d),uer=bad/n if n else np.nan,euro_weighted_uer=float(v[sel&u].sum()/den) if den else np.nan,mean_loss=float(np.where(sel,u*C,.1).mean()))
def sequence(cal_scores,unsafe,gate,candidates,alpha,delta):
 # Candidate values and their order derive only from isotonic-fitting data.
 tested=[];passed=[]
 for t in candidates:
  sel=gate&(cal_scores>=t);n=int(sel.sum());u=int((sel&unsafe).sum());pv=float(binom.cdf(u,n,alpha)) if n else 1.
  tested.append((float(t),n,u,pv))
  if pv>delta:break
  passed.append((n,float(t),u))
 best=sorted(passed,key=lambda x:(-x[0],x[1]))[0] if passed else (0,np.inf,0)
 return best,tested
def graph(cal_scores,unsafe,gate,C,alpha):
 # Eight disjoint chains, each initially assigned delta/8. Within a chain,
 # increasing budget expands the execution set; stop at the first nonrejection.
 passed=[];tested=[]
 for tau in TAUS:
  for beta in BETAS:
   sel=chosen_rc(cal_scores,gate,C,tau,beta);n=int(sel.sum());u=int((sel&unsafe).sum());pv=float(binom.cdf(u,n,alpha)) if n else 1.
   tested.append((tau,beta,n,u,pv))
   if pv>.05/len(TAUS):break
   passed.append((n,tau,beta,u))
 best=sorted(passed,key=lambda x:(-x[0],x[1],x[2]))[0] if passed else (0,np.inf,0.,0)
 return best,tested
def learned_candidates(scores,unsafe,gate,candidates,alpha):
 # The first-half binomial calculation is a start-point heuristic only,
 # NOT a certificate, because the same labels fitted the calibrator.
 # The independent second half performs every reported certification test.
 opts=[]
 for t in candidates:
  sel=gate&(scores>=t);n=int(sel.sum());u=int((sel&unsafe).sum())
  if n and binom.cdf(u,n,alpha)<=.05:opts.append((n,float(t)))
 if not opts:return candidates
 start=sorted(opts,key=lambda x:(-x[0],x[1]))[0][1]
 return candidates[candidates<=start]
def nonlinear_fit(train,num,cat,y):
 med=train[num].apply(pd.to_numeric,errors='coerce').median().fillna(0)
 maps={}
 for c in cat:
  vals=train[c].fillna('__MISSING__').astype(str).value_counts().head(253).index.tolist()
  maps[c]={v:i for i,v in enumerate(vals)}
 model=HistGradientBoostingClassifier(learning_rate=.05,max_iter=180,max_leaf_nodes=31,min_samples_leaf=30,l2_regularization=1.,early_stopping=False,random_state=20261023,categorical_features=[False]*len(num)+[True]*len(cat))
 bundle=dict(medians=med,maps=maps,numeric=num,categorical=cat)
 model.fit(nonlinear_matrix(train,bundle),y);bundle['model']=model
 return bundle
def nonlinear_matrix(d,b):
 numeric=d[b['numeric']].apply(pd.to_numeric,errors='coerce').fillna(b['medians']).to_numpy(float)
 categorical=np.column_stack([d[c].fillna('__MISSING__').astype(str).map(b['maps'][c]).fillna(253).to_numpy(float) for c in b['categorical']])
 return np.column_stack([numeric,categorical])

def main():
 OUT.mkdir(exist_ok=True)
 protocol=dict(status='exploratory reanalysis of previously inspected data',seeds=SEEDS,risk_caps=CAPS,source_population='all 210692 eligible natural cases; no intervention-source sampling',iid_split='SHA256 uniform source hash, 70/15/15',late_temporal='70/15/15 sorted decision_time and source ID; original-D1 target',early_temporal='90-day target; training Jan-Mar 2018, calibration Apr-May 2018, test Sep 2018; latest calibration horizon ends before test begins',ood_split='top 1% absolute PO exposure held out; lower 99% hash split 70/15/15 including ID test',selection='Refit isotonic for each of 20 calibration-half seeds; train model fixed. Candidates are distinct fitted isotonic y thresholds, independent of selection labels; descending fixed sequence stops at first failure.',graph=dict(tau=TAUS,beta=BETAS,initial_error_per_chain=.05/len(TAUS),stop='first failed exact binomial test in each increasing-budget chain'),costs='dimensionless; cH=.10, rhoH=0, cA=.50; fixed-decision repricing distinguished from RCSE retuning',consequences=['log1p(v)/log1p(train_P90)','v/train_P90','min(v/train_P90,10)'],nonlinear='sklearn histogram gradient boosting: 180 trees, learning_rate=.05, max_leaf_nodes=31, min_samples_leaf=30, L2=1; no hyperparameter search; train-only category vocabulary, top 253 plus other',guarantee='Within each reported family and seed only, under IID natural-case assumptions. No simultaneous guarantee over seeds/families, no shift guarantee, no capture-completeness claim.')
 protocol['learned_start']='Safety split FST chooses its starting candidate on the isotonic-fitting half using an in-sample binomial heuristic; that heuristic is not a certificate. The independent selection half tests the learned sequence at delta=.05. Strictest-start FST is retained separately.'
 (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
 h=load('robust_h','09b2h_full_benchmark_safety_robustness.py');g=load('robust_g','09b2k_evidence_gate_v2_audit.py')
 b=pd.read_parquet(ROOT/'rcse_output/rcse_base_v2.parquet');b['source_case_id']=b.case_id.astype(str)
 b['target']=target(b,h);b['complete']=~h.as_bool(b.goods_receipt_required)|h.as_bool(b.gr_available_at_t0)
 b['valid']=g.classify_gate(b,g.build_fixed_rules(b),g.build_value_rule(b,.05))[0].eq('VALID')
 follow=pd.read_parquet(OUT/'all_eligible_followup.parquet');b=b.merge(follow[['source_case_id','target_90','observed_span_days']],on='source_case_id',validate='one_to_one')
 u=split_hash(b.source_case_id);b['iid']=np.where(u<.7,'train',np.where(u<.85,'calibration','test'))
 ordered=b.sort_values(['decision_time','source_case_id']).index.to_numpy();b['late_temporal']='test';b.loc[ordered[:int(.7*len(b))],'late_temporal']='train';b.loc[ordered[int(.7*len(b)):int(.85*len(b))],'late_temporal']='calibration'
 times=pd.to_datetime(b.decision_time,utc=True)
 b['early_temporal90']=np.where(times.lt('2018-04-01'),'train',np.where(times.lt('2018-06-01'),'calibration',np.where(times.ge('2018-09-01')&times.lt('2018-10-01'),'test','unused')))
 exposure=b.transaction_exposure_eur.abs();q99=float(exposure.quantile(.99));b['ood']=np.where(exposure.gt(q99),'test',np.where(u<.7,'train',np.where(u<.85,'calibration','id_test')))
 b[['source_case_id','iid','late_temporal','early_temporal90','ood','target','target_90']].to_parquet(OUT/'all_eligible_split_assignments.parquet',index=False)
 fm=json.loads((ROOT/'rcse_output/experimental_benchmark/splits/t0_execution_safety_results/t0_safety_feature_manifest.json').read_text());num=fm['numeric_features'];cat=fm['categorical_features'];features=num+cat
 allresults=[];cohorts=[];discrimination=[];selection_audit=[];costrows=[];breakeven=[];sens=[];training=[];ambiguity=[]
 labels={'iid':'IID','late_temporal':'Late temporal','early_temporal90':'Earlier temporal 90d','ood':'Exposure OOD'}
 for regime in labels:
  z=b.copy();z['part']=z[regime]
  if regime=='early_temporal90':z['target']=z.target_90
  z['known']=z.target.isin([h.SAFE_LABEL,h.UNSAFE_LABEL]);z['unsafe']=z.target.eq(h.UNSAFE_LABEL)
  for part,d in z[z.part.ne('unused')].groupby('part'):
   cohorts.append(dict(regime=regime,part=part,total=len(d),resolved=int(d.known.sum()),ambiguous=int((~d.known).sum()),complete=int(d.complete.sum()),decision_min=str(pd.to_datetime(d.decision_time,utc=True).min()),decision_max=str(pd.to_datetime(d.decision_time,utc=True).max()),observed_span_ge90=int(d.observed_span_days.ge(90).sum())))
  tr=z[z.part.eq('train')&z.known];cal=z[z.part.eq('calibration')&z.known].copy();testall=z[z.part.eq('test')].copy();test=testall[testall.known].copy()
  assert not(set(tr.source_case_id)&set(cal.source_case_id) or set(tr.source_case_id)&set(testall.source_case_id) or set(cal.source_case_id)&set(testall.source_case_id))
  ref=float(z[z.part.eq('train')].transaction_exposure_eur.abs().quantile(.9))
  C=np.log1p(test.transaction_exposure_eur.abs().to_numpy())/np.log1p(ref);calC=np.log1p(cal.transaction_exposure_eur.abs().to_numpy())/np.log1p(ref)
  scorefile=OUT/f'raw_scores_{regime}.parquet';lrfile=OUT/f'logistic_{regime}.joblib';gbfile=OUT/f'histboost_{regime}.joblib'
  log(regime+': fitting or loading train-only models')
  y=tr.target.eq(h.SAFE_LABEL).to_numpy(int)
  if lrfile.exists():lr=joblib.load(lrfile)
  else:
   lr=h.build_pipeline(num,cat,C=1.,max_iter=5000,tol=1e-4)
   with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always',ConvergenceWarning);lr.fit(tr[features],y)
   assert not any(issubclass(w.category,ConvergenceWarning) for w in caught)
   joblib.dump(lr,lrfile)
  if gbfile.exists():gb=joblib.load(gbfile)
  else:gb=nonlinear_fit(tr,num,cat,y);joblib.dump(gb,gbfile)
  training.append(dict(regime=regime,training_resolved=len(tr),train_P90=ref,lr_iterations=int(lr.named_steps['model'].n_iter_.max()),histboost_iterations=int(gb['model'].n_iter_),features=len(features)))
  scoredata=z[z.part.isin(['calibration','test','id_test'])][['source_case_id','part','target','known','unsafe','complete','valid','transaction_exposure_eur']].copy()
  source=z.set_index('source_case_id').loc[scoredata.source_case_id]
  scoredata['lr_raw']=lr.predict_proba(source[features])[:,list(lr.named_steps['model'].classes_).index(1)]
  scoredata['gb_raw']=gb['model'].predict_proba(nonlinear_matrix(source,gb))[:,1]
  scoredata.to_parquet(scorefile,index=False)
  cal=scoredata[scoredata.part.eq('calibration')&scoredata.known].copy();test=scoredata[scoredata.part.eq('test')&scoredata.known].copy();testall=scoredata[scoredata.part.eq('test')].copy()
  C=np.log1p(test.transaction_exposure_eur.abs().to_numpy())/np.log1p(ref);calC=np.log1p(cal.transaction_exposure_eur.abs().to_numpy())/np.log1p(ref)
  for model,col in [('Logistic','lr_raw'),('HistBoost','gb_raw')]:
   for scope,mask in [('all_resolved',np.ones(len(test),bool)),('evidence_complete',test.complete.to_numpy(bool))]:
    d=test.loc[mask];discrimination.append(dict(regime=regime,model=model,scope=scope,cases=len(d),unsafe=int(d.unsafe.sum()),auroc=roc_auc_score(~d.unsafe,d[col])))
   for seed in SEEDS:
    ca=half(cal.source_case_id,seed)==0;cb=~ca
    assert ca.any() and cb.any()
    iso=IsotonicRegression(y_min=0,y_max=1,out_of_bounds='clip').fit(cal.loc[ca,col],(~cal.loc[ca,'unsafe']).astype(int))
    cals=iso.predict(cal.loc[cb,col]);ts=iso.predict(test[col]);allts=iso.predict(testall[col]);cc=calC[cb]
    cu=cal.loc[cb,'unsafe'].to_numpy(bool);cg=cal.loc[cb,'valid'].to_numpy(bool);tg=test.valid.to_numpy(bool)
    candidates=np.unique(iso.y_thresholds_)[::-1]
    def rec(policy,selected,cap=np.nan,**extra):
     allresults.append(dict(regime=regime,model=model,seed=seed,policy=policy,cap=cap,**metrics(test,selected,C),**extra))
    fixed={'Evidence complete':test.complete.to_numpy(bool),'Gate only':tg,'Safety 0.95':ts>=.95,'Gate safety 0.95':tg&(ts>=.95),'RCSE reference':chosen_rc(ts,tg,C,.95,.05),'Escalate all':np.zeros(len(test),bool)}
    for name,selected in fixed.items():rec(name,selected)
    for cap in CAPS:
     for name,cgate,tgate,ag in [('Safety FST',np.ones(cb.sum(),bool),np.ones(len(test),bool),np.ones(ca.sum(),bool)),('Gate safety FST',cg,tg,cal.loc[ca,'valid'].to_numpy(bool)),('Safety split FST',np.ones(cb.sum(),bool),np.ones(len(test),bool),np.ones(ca.sum(),bool)),('Gate safety split FST',cg,tg,cal.loc[ca,'valid'].to_numpy(bool))]:
      cand=learned_candidates(iso.predict(cal.loc[ca,col]),cal.loc[ca,'unsafe'].to_numpy(bool),ag,candidates,cap) if 'split' in name else candidates
      best,tests=sequence(cals,cu,cgate,cand,cap,.05);n,t,u=best
      rec(name,tgate&(ts>=t),cap,selection_executed=n,selection_unsafe=u,certified=n>0,threshold=t,candidate_count=len(candidates),tests_performed=len(tests))
      if seed==SEEDS[0]:
       selection_audit.extend(dict(regime=regime,model=model,policy=name,cap=cap,threshold=t0,selected=n0,unsafe=u0,pvalue=pv,delta=.05) for t0,n0,u0,pv in tests)
     best,tests=graph(cals,cu,cg,cc,cap);n,t,beta,u=best
     rec('RCSE graph',chosen_rc(ts,tg,C,t,beta),cap,selection_executed=n,selection_unsafe=u,certified=n>0,threshold=t,beta=beta,candidate_count=len(TAUS)*len(BETAS),tests_performed=len(tests))
     if seed==SEEDS[0]:selection_audit.extend(dict(regime=regime,model=model,policy='RCSE graph',cap=cap,threshold=t0,beta=beta0,selected=n0,unsafe=u0,pvalue=pv,delta=.05/len(TAUS)) for t0,beta0,n0,u0,pv in tests)
    if seed==SEEDS[0]:
     joblib.dump(dict(calibrator=iso,features=features,train_P90=ref,model=model),OUT/f'calibration_{regime}_{model}.joblib')
     if model=='Logistic':
      pred=test.copy();pred['score']=ts;pred['C_log']=C;pred.to_parquet(OUT/f'primary_test_predictions_{regime}.parquet',index=False)
     for policy in ['Safety 0.95','Gate safety 0.95','RCSE reference']:
      ex=chosen_rc(allts,testall.valid.to_numpy(bool),np.log1p(testall.transaction_exposure_eur.abs())/np.log1p(ref),.95,.05) if policy=='RCSE reference' else (allts>=.95)&(testall.valid.to_numpy(bool) if policy=='Gate safety 0.95' else True)
      ac=np.log1p(testall.transaction_exposure_eur.abs().to_numpy())/np.log1p(ref);amb=~testall.known.to_numpy(bool);unsafe=testall.unsafe.to_numpy(bool)
      for convention in ['ambiguous_safe','ambiguous_unsafe']:
       lab=unsafe|(amb if convention=='ambiguous_unsafe' else False)
       ambiguity.append(dict(regime=regime,model=model,policy=policy,convention=convention,cases=len(testall),ambiguous=int(amb.sum()),ambiguous_executed=int((amb&ex).sum()),coverage=float(np.mean(ex)),uer=float(lab[ex].mean()) if ex.any() else np.nan,mean_loss=float(np.where(ex,lab*ac,.1).mean())))
     for left,right in [('RCSE reference','Safety 0.95'),('Gate only','Evidence complete'),('RCSE reference','Gate safety 0.95')]:
      a=fixed[left].astype(int);bb=fixed[right].astype(int);delta=a-bb;removed=-int(delta.sum());avoided=-float(np.sum(delta*test.unsafe.to_numpy()*C))
      breakeven.append(dict(regime=regime,model=model,comparison=left+' minus '+right,removed=removed,unsafe_removed=-int(np.sum(delta*test.unsafe.to_numpy())),avoided_consequence=avoided,break_even_cH=avoided/removed if removed else np.nan))
      for cH in np.linspace(0,.25,126):
       fixed_delta=float(np.mean(delta*(test.unsafe.to_numpy()*C-cH)))
       retuned=chosen_rc(ts,tg,C,.95,.05,cH)
       other=fixed[right]
       retuned_delta=float((np.where(retuned,test.unsafe.to_numpy()*C,cH)-np.where(other,test.unsafe.to_numpy()*C,cH)).mean()) if left=='RCSE reference' else np.nan
       costrows.append(dict(regime=regime,model=model,comparison=left+' minus '+right,c_H=cH,fixed_action_loss_difference=fixed_delta,retuned_rcse_loss_difference=retuned_delta))
     value=test.transaction_exposure_eur.abs().to_numpy()
     for mode,cs in [('Log',C),('Linear',value/ref),('Capped linear',np.minimum(value/ref,10))]:
      for name,ex in [('Safety 0.95',ts>=.95),('Gate safety 0.95',tg&(ts>=.95)),('RCSE reference',chosen_rc(ts,tg,cs,.95,.05)),('Evidence complete',test.complete.to_numpy(bool)),('Escalate all',np.zeros(len(test),bool))]:sens.append(dict(regime=regime,model=model,consequence=mode,policy=name,**metrics(test,ex,cs)))
   log(regime+' '+model+': twenty independent calibration-half repetitions complete')
  # Checkpoint after each regime, allowing safe restart using the fitted model cache.
  for data,name in [(allresults,'repeated_policy_results'),(cohorts,'cohorts'),(discrimination,'discrimination'),(selection_audit,'sequential_selection_audit'),(costrows,'cost_curves'),(breakeven,'break_even'),(sens,'consequence_sensitivity'),(training,'training_audit'),(ambiguity,'ambiguity_bounds')]:pd.DataFrame(data).to_csv(OUT/f'{name}.csv',index=False)
 r=pd.DataFrame(allresults);selected=r[r.policy.isin(['Safety FST','Gate safety FST','Safety split FST','Gate safety split FST','RCSE graph'])]
 summary=selected.groupby(['regime','model','policy','cap']).agg(repetitions=('seed','count'),certification_fraction=('certified','mean'),coverage_median=('coverage','median'),coverage_min=('coverage','min'),coverage_max=('coverage','max'),uer_median=('uer','median'),loss_median=('mean_loss','median')).reset_index()
 summary.to_csv(OUT/'repeated_summary.csv',index=False)
 fig,axs=plt.subplots(1,3,figsize=(12,3.7));reglist=['iid','early_temporal90','ood']
 c=pd.DataFrame(costrows)
 for ax,reg in zip(axs,reglist):
  for compare in ['RCSE reference minus Safety 0.95','Gate only minus Evidence complete']:
   x=c[c.regime.eq(reg)&c.model.eq('Logistic')&c.comparison.eq(compare)]
   ax.plot(x.c_H,x.fixed_action_loss_difference,label=compare.replace('reference','').replace('Evidence complete','completeness'))
  ax.axhline(0,color='black',lw=.8);ax.set(title=labels[reg],xlabel='Escalation cost cH');ax.grid(alpha=.2)
 axs[0].set_ylabel('Mean modeled loss difference');fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,-.12));fig.tight_layout()
 for ext in ['png','svg']:fig.savefig(OUT/f'break_even_curves.{ext}',dpi=200,bbox_inches='tight')
 assert summary.repetitions.eq(20).all()
 (OUT/'completion.json').write_text(json.dumps(dict(status='complete',model_fits=8,repetitions=20,eligible=len(b),q99_exposure=q99,rows=len(r),scope='all eligible cases; resolved outcomes condition the primary comparison; earlier-window fixed-90-day refitting included'),indent=2))
 log('All robustness extension analyses complete')

if __name__=='__main__':
 with threadpool_limits(limits=4):main()
