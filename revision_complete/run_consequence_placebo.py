"""Consequence-alignment placebo and PO-grouped selection sensitivity.

Uses frozen models. Permutation choices never depend on outcomes or test risk.
"""
from pathlib import Path
import importlib.util,sys,json
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import joblib
from scipy.stats import binom
from sklearn.isotonic import IsotonicRegression
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'revision_complete/robustness_extension';G=ROOT/'revision_complete/grouped_component_extension';OUT=ROOT/'revision_complete/alignment_extension'
spec=importlib.util.spec_from_file_location('component',ROOT/'revision_complete/run_grouped_component_extension.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
REGS=['iid','group_po','early_temporal90','ood'];PERMS=list(range(20))

def graph_nodes(s,u,g,C):
 n=[];bad=[]
 for tau in m.m.TAUS:
  mask=g&(s>=tau)
  risk=(1-s)*C
  for beta in m.m.BETAS:
   ex=mask&(risk<=beta);n.append(int(ex.sum()));bad.append(int((ex&u).sum()))
 return np.array(n).reshape(8,6),np.array(bad).reshape(8,6)

def choose(n,u,cap):
 passed=(binom.cdf(u,n,cap)<=.05/8)&(n>0);passed=np.cumprod(passed,axis=1).astype(bool)
 if not passed.any():return 0,np.inf,0.,0
 candidates=[(int(n[i,j]),m.m.TAUS[i],m.m.BETAS[j],int(u[i,j])) for i,j in zip(*np.where(passed))]
 return sorted(candidates,key=lambda x:(-x[0],x[1],x[2]))[0]

def main():
 OUT.mkdir(exist_ok=True)
 (OUT/'placebo_protocol.json').write_text(json.dumps(dict(status='exploratory sensitivity; fixed before inspecting new results',regimes=REGS,calibration_seeds=m.m.SEEDS,permutations=20,permutation_seed=20261024,assignment='Separate uniform permutations of actual log consequence among all resolved calibration cases and among resolved test cases; fixed within permutation ID across calibration-half seeds and models',labels='never used to assign permutations; isotonic fitted only on its first half',model='frozen original models; PO-grouped calibration halves hash purchasing_document, other halves hash source_case_id',selection='identical 8 by 6 RCSE graph and alpha allocation; all four caps; stop first nonrejection per chain',loss='evaluate each placebo decision on actual unpermuted consequence, not shuffled consequence; this is a placebo decision family, not a feasible economic constraint',limits='Tests are nominal under item independence; cluster dependence and shift remain. A similar certification frequency does not prove mechanisms equivalent; a gap does not prove economic benefit.'),indent=2))
 rows=[];grouped=[];audits=[]
 for ri,reg in enumerate(REGS):
  source=G if reg=='group_po' else R
  sd=pd.read_parquet(source/f'raw_scores_{reg}.parquet');cal=sd[sd.part.eq('calibration')&sd.known].copy();test=sd[sd.part.eq('test')&sd.known].copy()
  ref=joblib.load(source/f'calibration_{reg}_Logistic.joblib')['train_P90'];Ccal=np.log1p(cal.transaction_exposure_eur.abs().to_numpy())/np.log1p(ref);Ctest=np.log1p(test.transaction_exposure_eur.abs().to_numpy())/np.log1p(ref)
  permcal=[];permtest=[]
  for j in PERMS:
   rng=np.random.default_rng(np.random.SeedSequence([20261024,ri,j]));permcal.append(rng.permutation(Ccal));permtest.append(rng.permutation(Ctest))
   assert np.array_equal(np.sort(permcal[-1]),np.sort(Ccal));assert np.array_equal(np.sort(permtest[-1]),np.sort(Ctest))
  for model,col in [('Logistic','lr_raw'),('HistBoost','gb_raw')]:
   for seed in m.m.SEEDS:
    ca=m.m.half(cal.purchasing_document if reg=='group_po' else cal.source_case_id,seed)==0;cb=~ca
    if reg=='group_po':assert not(set(cal.loc[ca,'purchasing_document'])&set(cal.loc[cb,'purchasing_document']))
    iso=IsotonicRegression(y_min=0,y_max=1,out_of_bounds='clip').fit(cal.loc[ca,col],(~cal.loc[ca,'unsafe']).astype(int));s=iso.predict(cal.loc[cb,col]);ts=iso.predict(test[col]);cu=cal.loc[cb,'unsafe'].to_numpy(bool);cg=cal.loc[cb,'valid'].to_numpy(bool);tg=test.valid.to_numpy(bool)
    for j in [-1]+PERMS:
     cc=Ccal[cb] if j==-1 else permcal[j][cb];tc=Ctest if j==-1 else permtest[j];n,u=graph_nodes(s,cu,cg,cc)
     for cap in m.m.CAPS:
      nn,t,beta,uu=choose(n,u,cap);ex=m.m.chosen_rc(ts,tg,tc,t,beta)
      rows.append(dict(regime=reg,model=model,seed=seed,permutation=j,cap=cap,certified=nn>0,selection_executed=nn,selection_unsafe=uu,tau=t,beta=beta,**m.m.metrics(test,ex,Ctest)))
    if reg=='group_po':
     for cap in m.m.CAPS:
      for policy,cg0,tg0 in [('Safety graph',np.ones(cb.sum(),bool),np.ones(len(test),bool)),('Gate safety graph',cg,tg)]:
       best,_=m.safetygraph(s,cu,cg0,np.unique(iso.y_thresholds_),cap);nn,t,start,uu=best
       grouped.append(dict(regime=reg,model=model,seed=seed,policy=policy,cap=cap,certified=nn>0,threshold=t,**m.m.metrics(test,tg0&(ts>=t),Ctest)))
   print('Placebo complete',reg,model,flush=True)
  # Checkpoints preserve long-running analysis progress.
  pd.DataFrame(rows).to_csv(OUT/'placebo_results.csv',index=False);pd.DataFrame(grouped).to_csv(OUT/'po_grouped_safety_graph.csv',index=False)
 q=pd.DataFrame(rows)
 freq=q.groupby(['regime','model','cap','permutation']).agg(certifications=('certified','sum'),repetitions=('seed','size'),coverage_median=('coverage','median'),uer_median=('uer','median'),loss_median=('mean_loss','median')).reset_index();freq.to_csv(OUT/'placebo_permutation_summary.csv',index=False)
 aggregate=[]
 for key,z in freq.groupby(['regime','model','cap']):
  a=z[z.permutation.eq(-1)].iloc[0];b=z[z.permutation.ge(0)]
  aggregate.append(dict(regime=key[0],model=key[1],cap=key[2],actual_certifications=int(a.certifications),placebo_certifications_median=float(b.certifications.median()),placebo_certifications_min=int(b.certifications.min()),placebo_certifications_max=int(b.certifications.max()),placebo_certifications_q025=float(b.certifications.quantile(.025)),placebo_certifications_q975=float(b.certifications.quantile(.975)),placebo_equal_or_exceed_fraction=float(b.certifications.ge(a.certifications).mean()),actual_coverage_median=a.coverage_median,placebo_coverage_median=float(b.coverage_median.median()),actual_loss_median=a.loss_median,placebo_loss_median=float(b.loss_median.median())))
 pd.DataFrame(aggregate).to_csv(OUT/'placebo_summary.csv',index=False)
 grouped=pd.DataFrame(grouped)
 actual=q[q.regime.eq('group_po')&q.permutation.eq(-1)].assign(policy='RCSE graph')
 both=pd.concat([grouped,actual],ignore_index=True);both.to_csv(OUT/'po_grouped_selection_results.csv',index=False)
 both.groupby(['regime','model','policy','cap']).agg(repetitions=('seed','size'),certification_fraction=('certified','mean'),coverage_median=('coverage','median'),uer_median=('uer','median'),loss_median=('mean_loss','median')).reset_index().to_csv(OUT/'po_grouped_selection_summary.csv',index=False)
 assert len(rows)==13440 and freq.repetitions.eq(20).all()
 # Vectorized decisions must agree exactly with the released graph implementation.
 for cap in m.m.CAPS:
  n,u=graph_nodes(s,cu,cg,Ccal[cb]);a=choose(n,u,cap);b,_=m.m.graph(s,cu,cg,Ccal[cb],cap);assert a==b,(a,b)
 (OUT/'placebo_validation.json').write_text(json.dumps(dict(status='passed',rows=len(rows),permutations_per_calibration=20,calibration_repetitions=20,graph_matches_reference=True,permutation_marginals_preserved=True,grouped_calibration_halves_disjoint=True),indent=2))
 print('Placebo analysis complete',flush=True)

if __name__=='__main__':main()
