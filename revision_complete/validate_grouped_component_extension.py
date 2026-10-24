"""Scientific consistency checks for grouped holdouts and component outputs."""
from pathlib import Path
import json
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'revision_complete/grouped_component_extension'
b=pd.read_parquet(ROOT/'rcse_output/rcse_base_v2.parquet',columns=['case_id','purchasing_document','vendor_id']);b['source_case_id']=b.case_id.astype(str)
a=pd.read_parquet(OUT/'grouped_split_assignments.parquet');coh=pd.read_csv(OUT/'grouped_cohorts.csv')
for reg,group in [('group_po','purchasing_document'),('group_vendor','vendor_id')]:
 z=a[a.regime.eq(reg)].merge(b[['source_case_id',group]],on='source_case_id',suffixes=('_saved',''),validate='one_to_one')
 assert len(z)==210692 and z.source_case_id.is_unique
 assert z.groupby(group).part.nunique().max()==1
 assert coh[coh.regime.eq(reg)].total.sum()==210692
f=pd.read_csv(OUT/'fixed_policies.csv')
assert np.allclose(f.uer.dropna(),(f.unsafe/f.executed).dropna())
assert np.allclose(f.coverage,f.executed/f.cases)
for key,z in f.groupby(['regime','model','consequence']):
 z=z.set_index('policy')
 for restricted,other in [('RCSE reference','Gate safety 0.95'),('Gate safety 0.95','Safety 0.95')]:
  assert z.loc[restricted,'executed']<=z.loc[other,'executed']
  assert z.loc[restricted,'unsafe_eur_per_1000']<=z.loc[other,'unsafe_eur_per_1000']+1e-7
  assert z.loc[restricted,'escalated_eur_per_1000']+1e-7>=z.loc[other,'escalated_eur_per_1000']
 assert np.isclose(z.loc['Escalate all','mean_loss'],.1)
 assert z.loc['Plug-in cost','executed']>=z.loc['RCSE reference','executed']
s=pd.read_csv(OUT/'safety_graph_results.csv');assert len(s)==1280
assert s.groupby(['regime','model','policy','cap']).seed.nunique().eq(20).all()
ci=pd.read_csv(OUT/'loss_contrasts.csv');assert ci.lower.le(ci.upper).all()
assert set(ci.sampling)=={'PO cluster','Vendor cluster','case'}
for _,x in ci.iterrows():
 left,right=x.comparison.split(' minus ');z=f[f.regime.eq(x.regime)&f.model.eq(x.model)&f.consequence.eq(x.consequence)].set_index('policy')
 assert np.isclose(x.estimate,z.loc[left,'mean_loss']-z.loc[right,'mean_loss'])
receipt=dict(status='passed',group_isolation=True,all_eligible_accounted=True,policy_identities=True,restriction_exposure_monotonicity=True,twenty_seeds_all_targets=True,contrast_identities=True,graph_rows=len(s),fixed_rows=len(f))
(OUT/'validation.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
