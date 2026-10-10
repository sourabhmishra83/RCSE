"""Evaluate saved no-GATHER decisions under explicit natural-label conventions.

No model fitting, threshold selection, or policy retuning is performed. Resolved-only
uses a different episode denominator and is a descriptive conditional subset.
"""
from pathlib import Path
import sys,json,importlib.util
import numpy as np
import pandas as pd
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]; OUT=Path(__file__).resolve().parent

def load(name,path):
 spec=importlib.util.spec_from_file_location(name,ROOT/path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def main():
 a=load('revision_analysis','revision_complete/run_revision_analyses.py')
 b=load('saved_bootstrap','09b2r_bootstrap_rcse_v2_uncertainty.py');p=load('saved_policy','09b2n_run_rcse_v2_policy_simulation.py')
 params=p.get_reference_parameters(json.loads((ROOT/'09b2l_rcse_v2_policy_spec.json').read_text()))
 rows=[];meta=[]
 for regime in ['grouped_iid','temporal','ood_exposure']:
  d=b.prepare_regime_data(a.V,a.R,regime,params);d=d[d._intervention_family_r.eq('NATURAL')].copy()
  known=d.t0_safety_robustness_target.isin(['SAFE_TO_EXECUTE','NOT_SAFE_TO_EXECUTE']);ex=d.action__RCSE_V2_NO_GATHER.eq('EXECUTE')
  for convention in ['original_fallback','resolved_only','all_ambiguous_safe','all_ambiguous_unsafe']:
   x=d[known].copy() if convention=='resolved_only' else d.copy();u=x._unsafe_t0.copy()
   if convention=='all_ambiguous_safe':u.loc[~known]=False
   if convention=='all_ambiguous_unsafe':u.loc[~known]=True
   m,loss=a.measure(x,x.action__RCSE_V2_NO_GATHER,x._consequence,params,u)
   baseline=params['c_H']+params['rho_H']*x._consequence
   ci=a.cluster_mean_ci(x,loss-baseline)
   rows.append(dict(regime=regime,policy='RCSE_V2_NO_GATHER',convention=convention,episodes=len(x),executed=m['executed'],unsafe_executed=m['unsafe_executed'],ambiguous_executed=int((ex&~known).sum()),mean_loss=m['loss'],loss_minus_escalate=ci['estimate'],ci_lower=ci['lower'],ci_upper=ci['upper'],bootstrap_replicates=ci['replicates'],coverage=m['coverage'],uer=m['uer']))
  amb=ex&~known
  meta.append(dict(regime=regime,ambiguous_executed=int(amb.sum()),mean_consequence_ambiguous_executed=float(d.loc[amb,'_consequence'].mean()),min_consequence=float(d._consequence.min()),max_consequence=float(d._consequence.max())))
 df=pd.DataFrame(rows);df.to_csv(OUT/'natural_loss_label_sensitivity.csv',index=False)
 assert np.allclose(df[df.convention.eq('original_fallback')].mean_loss,[.05473579538825172,.032871325267522865,.10279433046285494])
 (OUT/'natural_loss_label_sensitivity_manifest.json').write_text(json.dumps(dict(seed=20261004,bootstrap_replicates=2000,models_fixed=True,policies_fixed=True,resolved_denominator='all D1-resolved natural benchmark episodes',ambiguous_count='all unresolved direct executions before subset restriction',consequence_diagnostics=meta),indent=2))
 print(df[['regime','convention','mean_loss','loss_minus_escalate','ci_lower','ci_upper']].to_string(index=False))
if __name__=='__main__':main()
