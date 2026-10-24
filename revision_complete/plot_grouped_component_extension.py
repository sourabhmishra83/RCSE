"""Publication figures with regime-specific axes for the component extension."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'revision_complete/robustness_extension';OUT=ROOT/'revision_complete/grouped_component_extension'
REGS=['iid','early_temporal90','ood'];LABELS=['Item-hash IID','Fixed-horizon temporal\nSep 2018','Exposure OOD']
plt.rcParams.update({'font.size':15,'axes.titlesize':17,'axes.labelsize':17,'xtick.labelsize':14,'ytick.labelsize':14,'legend.fontsize':14})
q=pd.read_csv(R/'descriptive_risk_coverage.csv');fig,axes=plt.subplots(1,3,figsize=(13,5.2))
for ax,reg,label,limit in zip(axes,REGS,LABELS,[15,10,40]):
 for model,ls in [('Logistic','-'),('HistBoost','--')]:
  for family,color in [('Safety threshold','C0'),('RCSE budget','C1')]:
   z=q[q.regime.eq(reg)&q.model.eq(model)&q.family.eq(family)].sort_values('coverage')
   ax.plot(100*z.coverage,100*z.uer,ls,color=color,label=('Boosted trees' if model=='HistBoost' else model)+' '+family)
 d=pd.read_parquet(R/f'raw_scores_{reg}.parquet');d=d[d.part.eq('test')&d.known];ex=d.complete.to_numpy(bool)
 ax.scatter(100*ex.mean(),100*d.loc[ex,'unsafe'].mean(),marker='s',color='black',s=35,label='Evidence complete')
 ax.set(title=label,xlabel='Execution coverage (%)',xlim=(0,102),ylim=(0,limit));ax.grid(alpha=.2)
axes[0].set_ylabel('Unsafe execution rate (%)');fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,-.12));fig.tight_layout(rect=(0,.12,1,1))
for ext in ['png','svg']:fig.savefig(OUT/f'risk_coverage.{ext}',dpi=200,bbox_inches='tight')
plt.close(fig)
c=pd.read_csv(R/'cost_curves.csv');fig,axes=plt.subplots(1,3,figsize=(13,5.1))
for ax,reg,label in zip(axes,REGS,LABELS):
 for comp,title in [('RCSE reference minus Safety 0.95','RCSE minus Safety 0.95'),('Gate only minus Evidence complete','Gate minus completeness')]:
  z=c[c.regime.eq(reg)&c.model.eq('Logistic')&c.comparison.eq(comp)];ax.plot(z.c_H,z.fixed_action_loss_difference,label=title)
 ax.axhline(0,color='black',lw=.8);ax.set(title=label,xlabel='Escalation cost cH');ax.grid(alpha=.2);ax.ticklabel_format(axis='y',style='sci',scilimits=(-2,2))
axes[0].set_ylabel('Mean modeled loss difference');fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,-.06));fig.tight_layout(rect=(0,.08,1,1))
for ext in ['png','svg']:fig.savefig(OUT/f'break_even.{ext}',dpi=200,bbox_inches='tight')
print('Component figures complete')
