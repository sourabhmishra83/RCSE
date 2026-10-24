"""Cost figure with scientific multipliers in axis labels rather than offsets."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'revision_complete/alignment_extension'
c=pd.read_csv(ROOT/'revision_complete/robustness_extension/cost_curves.csv')
plt.rcParams.update({'font.size':15,'axes.titlesize':17,'axes.labelsize':16,'xtick.labelsize':14,'ytick.labelsize':14,'legend.fontsize':14})
fig,axes=plt.subplots(1,3,figsize=(13,5.1))
for ax,reg,title,power in zip(axes,['iid','early_temporal90','ood'],['Item-hash IID','Fixed-horizon temporal\nSep 2018','Exposure OOD'],[3,3,2]):
 for comparison,label in [('RCSE reference minus Safety 0.95','RCSE minus Safety 0.95'),('Gate only minus Evidence complete','Gate minus completeness')]:
  x=c[c.regime.eq(reg)&c.model.eq('Logistic')&c.comparison.eq(comparison)]
  ax.plot(x.c_H,x.fixed_action_loss_difference*10**power,label=label)
 ax.axhline(0,color='black',lw=.8);ax.set(title=title,xlabel='Escalation cost cH',ylabel=r'Loss difference ($\times 10^{-'+str(power)+'}$)');ax.grid(alpha=.2)
fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,-.06));fig.tight_layout(rect=(0,.08,1,1))
for ext in ['png','svg']:fig.savefig(OUT/f'break_even.{ext}',dpi=200,bbox_inches='tight')
svg=OUT/'break_even.svg';svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
print('Cost figure complete')
