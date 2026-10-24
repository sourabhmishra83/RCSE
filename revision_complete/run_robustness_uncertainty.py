"""Paired natural-case uncertainty and figures for the all-eligible study."""
from pathlib import Path
import json,sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent/'robustness_extension'
REGS=['iid','early_temporal90','ood','late_temporal']
def cost_figure():
 plt.rcParams.update({'font.size':16,'axes.titlesize':18,'axes.labelsize':17,'xtick.labelsize':15,'ytick.labelsize':14,'legend.fontsize':15})
 c=pd.read_csv(OUT/'cost_curves.csv');fig,axes=plt.subplots(1,3,figsize=(13,4.6))
 for ax,reg,label in zip(axes,REGS[:3],['IID','Earlier temporal 90d','Exposure OOD']):
  for comp,title in [('RCSE reference minus Safety 0.95','RCSE minus Safety 0.95'),('Gate only minus Evidence complete','Gate minus completeness')]:
   x=c[c.regime.eq(reg)&c.model.eq('Logistic')&c.comparison.eq(comp)];ax.plot(x.c_H,x.fixed_action_loss_difference,label=title)
  ax.axhline(0,color='black',lw=.8);ax.set(title=label,xlabel='Escalation cost cH');ax.grid(alpha=.2);ax.ticklabel_format(axis='y',style='sci',scilimits=(-2,2))
 axes[0].set_ylabel('Mean modeled loss difference');fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,-.09));fig.tight_layout(rect=(0,.08,1,1))
 for ext in ['png','svg']:fig.savefig(OUT/f'break_even_curves.{ext}',dpi=200,bbox_inches='tight')
def main():
 rows=[];auc=[]
 for reg in REGS:
  d=pd.read_parquet(OUT/f'raw_scores_{reg}.parquet');d=d[d.part.eq('test')&d.known].copy();u=d.unsafe.to_numpy(bool);gate=d.valid.to_numpy(bool)
  for model,col in [('Logistic','lr_raw'),('HistBoost','gb_raw')]:
   fit=joblib.load(OUT/f'calibration_{reg}_{model}.joblib');s=fit['calibrator'].predict(d[col]);C=np.log1p(d.transaction_exposure_eur.abs())/np.log1p(fit['train_P90'])
   rc=gate&(s>=.95)&((1-s)*C<=.05)
   for name,other in [('Safety 0.95',s>=.95),('Gate safety 0.95',gate&(s>=.95)),('Evidence complete',d.complete.to_numpy(bool))]:
    delta=np.where(rc,u*C,.1)-np.where(other,u*C,.1);delta=np.asarray(delta)
    rng=np.random.default_rng(20261023);boot=np.array([delta[rng.integers(0,len(d),len(d))].mean() for _ in range(2000)])
    rows.append(dict(regime=reg,model=model,comparison='RCSE reference minus '+name,estimate=float(delta.mean()),lower=float(np.quantile(boot,.025)),upper=float(np.quantile(boot,.975)),replicates=2000,cases=len(d)))
  sub=d[d.complete];y=(~sub.unsafe).to_numpy(int);lr=sub.lr_raw.to_numpy();gb=sub.gb_raw.to_numpy();rng=np.random.default_rng(20261023);boot=[]
  for _ in range(1000):
   ix=rng.integers(0,len(sub),len(sub))
   if len(np.unique(y[ix]))==2:boot.append(roc_auc_score(y[ix],gb[ix])-roc_auc_score(y[ix],lr[ix]))
  auc.append(dict(regime=reg,scope='evidence_complete',estimate=roc_auc_score(y,gb)-roc_auc_score(y,lr),lower=np.quantile(boot,.025),upper=np.quantile(boot,.975),replicates=len(boot),cases=len(sub)))
  print('Paired uncertainty complete:',reg,flush=True)
 pd.DataFrame(rows).to_csv(OUT/'paired_loss_uncertainty.csv',index=False);pd.DataFrame(auc).to_csv(OUT/'paired_capacity_auc.csv',index=False)
 # Descriptive risk-coverage sweeps on the same independently calibrated scores.
 plt.rcParams.update({'font.size':16,'axes.titlesize':18,'axes.labelsize':18,'xtick.labelsize':16,'ytick.labelsize':16,'legend.fontsize':16})
 fig,axes=plt.subplots(1,3,figsize=(13,4.8),sharey=True);sweeps=[]
 for ax,reg,label in zip(axes,REGS[:3],['IID','Earlier temporal 90d','Exposure OOD']):
  d=pd.read_parquet(OUT/f'raw_scores_{reg}.parquet');d=d[d.part.eq('test')&d.known].copy();u=d.unsafe.to_numpy(bool);g=d.valid.to_numpy(bool)
  for model,col,linestyle in [('Logistic','lr_raw','-'),('HistBoost','gb_raw','--')]:
   fit=joblib.load(OUT/f'calibration_{reg}_{model}.joblib');s=fit['calibrator'].predict(d[col]);C=np.log1p(d.transaction_exposure_eur.abs().to_numpy())/np.log1p(fit['train_P90'])
   for fam,color in [('Safety threshold','C0'),('RCSE budget','C1')]:
    cov=[];risk=[]
    for t in (np.linspace(0,1,201) if fam=='Safety threshold' else np.linspace(0,.1,201)):
     ex=(s>=t) if fam=='Safety threshold' else g&(s>=.95)&((1-s)*C<=t)
     cov.append(ex.mean());risk.append(u[ex].mean() if ex.any() else np.nan)
     sweeps.append(dict(regime=reg,model=model,family=fam,parameter=t,coverage=cov[-1],uer=risk[-1]))
    ix=np.argsort(cov);ax.plot(100*np.array(cov)[ix],100*np.array(risk)[ix],linestyle,color=color,label=model+' '+fam)
  ex=d.complete.to_numpy(bool);ax.scatter(100*ex.mean(),100*u[ex].mean(),marker='s',color='black',s=35,label='Evidence complete')
  ax.set(title=label,xlabel='Execution coverage (%)',xlim=(0,102),ylim=(0,40));ax.grid(alpha=.2)
 axes[0].set_ylabel('Unsafe execution rate (%)');fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,-.16));fig.tight_layout(rect=(0,.12,1,1))
 for ext in ['png','svg']:fig.savefig(OUT/f'full_population_risk_coverage.{ext}',dpi=200,bbox_inches='tight')
 pd.DataFrame(sweeps).to_csv(OUT/'descriptive_risk_coverage.csv',index=False)
 cost_figure()
 print('Uncertainty and publication figures complete',flush=True)
if __name__=='__main__':main()
