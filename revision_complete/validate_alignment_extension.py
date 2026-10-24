"""Check permutation, grouped selection and repeated vendor evidence."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'revision_complete/alignment_extension';G=ROOT/'revision_complete/grouped_component_extension';R=ROOT/'revision_complete/robustness_extension'
p=pd.read_csv(OUT/'placebo_results.csv');assert len(p)==13440
assert p.groupby(['regime','model','permutation','cap']).seed.nunique().eq(20).all()
actual=p[p.permutation.eq(-1)]
old=pd.read_csv(R/'repeated_policy_results.csv');old=old[old.policy.eq('RCSE graph')]
j=actual[actual.regime.ne('group_po')].merge(old,on=['regime','model','seed','cap'],suffixes=('_new','_old'),validate='one_to_one')
for col in ['executed','unsafe','coverage','mean_loss']:assert np.allclose(j[col+'_new'],j[col+'_old'])
v=pd.read_csv(OUT/'vendor_repeated_auc.csv');assert len(v)==10 and v.seed.nunique()==10
a=pd.read_parquet(OUT/'vendor_repeated_assignments.parquet');assert len(a)==2106920
assert a.groupby(['seed','vendor_id']).part.nunique().eq(1).all()
assert a.groupby('seed').source_case_id.nunique().eq(210692).all()
original=pd.read_csv(G/'capacity_auc.csv');original=original[original.regime.eq('group_vendor')].iloc[0];first=v[v.seed.eq(20261023)].iloc[0]
assert np.isclose(original.estimate,first.estimate)
for col in ['top_vendor_share','top5_vendor_share']:assert v[col].between(0,1).all()
assert v.effective_vendors.le(v.vendors+1e-9).all()
out=dict(status='passed',actual_graph_matches_release=True,permutation_calibration_counts=True,ten_vendor_splits=True,all_vendor_splits_group_disjoint=True,all_eligible_accounted=True,concentration_identities=True)
(OUT/'validation.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
