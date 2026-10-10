"""Publication figures from saved evidence; no training or selection."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
OUT=Path(__file__).resolve().parent;F=OUT/'figures';F.mkdir(exist_ok=True)
regimes=['grouped_iid','temporal','ood_exposure'];names=['IID','Temporal','Exposure OOD']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
def save(fig,name):
    fig.savefig(F/(name+'.png'),dpi=240,bbox_inches='tight');fig.savefig(F/(name+'.svg'),bbox_inches='tight');plt.close(fig)
ci=pd.read_csv(OUT/'frozen_evidence/bootstrap_uncertainty/rcse_v2r_bootstrap_diagnostic_ci.csv')
fig,ax=plt.subplots(figsize=(6.5,3.6));x=np.arange(3)
for j,(metric,label,color) in enumerate([('contradiction_gate_block_rate','Contradiction','#265c85'),('missingness_gate_block_rate','Missingness','#6298aa'),('severe_evidence_loss_gate_block_rate','Severe loss','#afc8c7')]):
    rows=ci[ci.metric.eq(metric)].set_index('regime').reindex(regimes)
    if rows.point_estimate.isna().all():continue
    y=rows.point_estimate.to_numpy()*100;lo=rows.ci_lower.to_numpy()*100;hi=rows.ci_upper.to_numpy()*100
    ax.bar(x+(j-1)*.22,y,width=.22,label=label,color=color,yerr=np.vstack([y-lo,hi-y]),capsize=3)
ax.set(xticks=x,xticklabels=names,ylabel='Blocked episodes (%)',ylim=(0,105));ax.legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncol=3,frameon=False);fig.tight_layout();save(fig,'gate_with_ci')
pol=pd.read_csv(OUT/'frozen_evidence/bootstrap_uncertainty/rcse_v2r_bootstrap_policy_ci.csv')
fig,ax=plt.subplots(figsize=(6.5,3.6))
for j,(data,metric,label,color) in enumerate([(pol[pol.policy.eq('RCSE_V2_NO_GATHER')],'unsafe_execution_rate','Full benchmark','#265c85'),(ci,'natural_unsafe_execution_rate','Natural only','#6298aa')]):
    rows=data[data.metric.eq(metric)].set_index('regime').reindex(regimes);y=rows.point_estimate.to_numpy()*100;lo=rows.ci_lower.to_numpy()*100;hi=rows.ci_upper.to_numpy()*100
    ax.errorbar(x+(j-.5)*.12,y,yerr=np.vstack([y-lo,hi-y]),fmt='o',capsize=4,label=label,color=color)
ax.set(xticks=x,xticklabels=names,ylabel='Unsafe execution (%)',ylim=(0,48));ax.legend(loc='upper left',frameon=False);fig.tight_layout();save(fig,'natural_full_risk_with_ci')
d=pd.read_csv(OUT/'calibration_selected_frontier.csv');d=d[d.population.eq('full_benchmark_resolved')]
fig,axs=plt.subplots(1,2,figsize=(8,3.6))
for r,label in zip(regimes,names):
    row=d[d.regime.eq(r)].sort_values('risk_cap');axs[0].plot(row.risk_cap*100,row.coverage*100,'o-',label=label);valid=row[row.coverage.gt(0)];axs[1].plot(valid.risk_cap*100,valid.uer*100,'o-',label=label)
axs[0].set(xlabel='Calibration risk cap (%)',ylabel='Test coverage (%)');axs[1].set(xlabel='Calibration risk cap (%)',ylabel='Observed test unsafe rate (%)');axs[0].legend(frameon=False,fontsize=8);fig.tight_layout();save(fig,'calibration_frontier')
d=pd.read_csv(OUT/'consequence_only_ablation.csv');d=d[d.policy.eq('RCSE_V2_NO_GATHER')]
fig,axs=plt.subplots(1,2,figsize=(8,3.5))
for j,(mode,label,color) in enumerate([('LOG_TRAIN_P90','Consequence aware','#265c85'),('UNIFORM','Uniform decision cost','#6298aa')]):
    row=d[d.decision_consequence.eq(mode)].set_index('regime').reindex(regimes)
    axs[0].bar(x+(j-.5)*.28,row.coverage*100,.28,label=label,color=color);axs[1].bar(x+(j-.5)*.28,row.loss,.28,label=label,color=color)
for ax in axs:ax.set_xticks(x,['IID','Temporal','Exposure\nOOD']);ax.tick_params(axis='x',labelsize=9)
axs[0].set_ylabel('Coverage (%)');axs[1].set_ylabel('Loss on same log consequence scale');axs[0].legend(frameon=False,fontsize=8,loc='upper center',bbox_to_anchor=(1.08,-.20),ncol=2);fig.tight_layout();save(fig,'consequence_only_ablation')
fig,ax=plt.subplots(figsize=(8,5));ax.set(xlim=(0,10),ylim=(0,7));ax.axis('off')
def box(x,y,w,h,text,color='#eaf0f4'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.10',facecolor=color,edgecolor='#354e63',linewidth=1.2));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10)
def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='#354e63',lw=1.4))
box(3.15,5.75,3.7,.8,'Observed evidence at t0')
box(.3,3.95,2.8,1.,'Evidence gate\nValid / incomplete /\ncontradictory');box(3.6,3.95,2.8,1.,'Dedicated safety score\nTrain-fitted model');box(6.9,3.95,2.8,1.,'Transaction consequence\nTraining-only reference')
for to in [(1.7,4.95),(5,4.95),(8.3,4.95)]:arrow((5,5.75),to)
box(2.7,2.05,4.6,1.,'Feasibility and expected loss\nSafety threshold + risk budget + gate')
for frm in [(1.7,3.95),(5,3.95),(8.3,3.95)]:arrow(frm,(5,3.05))
box(2.7,.5,4.6,.8,'Execute / escalate / abstain',color='#dce9e0');arrow((5,2.05),(5,1.3))
ax.text(5,.03,'GATHER requires an ex ante transition model; existing t1 replay is retrospective.',ha='center',va='bottom',fontsize=9)
save(fig,'architecture')
print('Saved five PNG and SVG figures')
