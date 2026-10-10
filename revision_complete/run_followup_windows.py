"""Fixed-horizon D1 sensitivity from source events; scores are fixed, not retrained."""
from pathlib import Path
import sys,time,importlib.util,json,hashlib
sys.dont_write_bytecode=True
from datetime import datetime
from lxml import etree
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import roc_auc_score,average_precision_score
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
def log(s):print(time.strftime('%H:%M:%S'),s,flush=True)
def main():
    spec=importlib.util.spec_from_file_location('h',ROOT/'09b2h_full_benchmark_safety_robustness.py');h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
    base=pd.read_parquet(ROOT/'rcse_output/rcse_base_v2.parquet');base['source_case_id']=base.case_id.astype(str)
    exp=pd.read_parquet(ROOT/'rcse_output/experimental_benchmark/splits/rcse_experimental_with_splits.parquet')
    base=base.merge(h.build_split_map(exp),on='source_case_id',validate='one_to_one')
    times=dict(zip(base.source_case_id,pd.to_datetime(base.decision_time,utc=True).map(lambda x:x.to_pydatetime())))
    adverse={'Cancel Invoice Receipt','Set Payment Block','Cancel Goods Receipt','Change Price','Change Quantity','Record Subsequent Invoice'}
    windows=[30,60,90];records=[];n=0;maximum=None
    source=ROOT/'BPI_Challenge_2019.xes'
    log('Streaming source traces to construct fixed follow-up outcomes')
    for _,trace in etree.iterparse(str(source),events=('end',),tag='{*}trace'):
        cid=next((c.get('value') for c in trace if c.get('key')=='concept:name'),None)
        if cid in times:
            origin=times[cid]; flags={w:[False,False] for w in windows}
            for event in trace:
                if etree.QName(event).localname!='event':continue
                attrs={c.get('key'):c.get('value') for c in event}
                timestamp=attrs.get('time:timestamp'); activity=attrs.get('concept:name')
                if not timestamp:continue
                dt=datetime.fromisoformat(timestamp.replace('Z','+00:00'))
                maximum=dt if maximum is None or dt>maximum else maximum
                delta=(dt-origin).total_seconds()/86400
                if delta<=0:continue
                for w in windows:
                    if delta<=w:
                        if activity in adverse:flags[w][0]=True
                        if activity=='Clear Invoice':flags[w][1]=True
            records.append(dict(source_case_id=cid,**{f'adverse_{w}':flags[w][0] for w in windows},**{f'cleared_{w}':flags[w][1] for w in windows}))
        trace.clear()
        while trace.getprevious() is not None:del trace.getparent()[0]
        n+=1
        if n%50000==0:log(f'{n:,} traces inspected')
    follow=pd.DataFrame(records); assert follow.source_case_id.is_unique
    base=base.merge(follow,on='source_case_id',validate='one_to_one');assert len(base)==len(times)
    missing=h.as_bool(base.goods_receipt_required)&~h.as_bool(base.gr_available_at_t0)
    fulltarget=h.build_d1_target(base)
    for w in windows:
        base[f'target_{w}']=np.where(missing|base[f'adverse_{w}'],h.UNSAFE_LABEL,np.where(base[f'cleared_{w}'],h.SAFE_LABEL,'AMBIGUOUS'))
    base[['source_case_id']+[f'target_{w}' for w in windows]].to_parquet(OUT/'fixed_horizon_targets.parquet',index=False)
    rows=[];balance=[]
    for regime,cfg in h.SPLIT_CONFIG.items():
        test=base[base[cfg['column']].eq(cfg['test'])].copy()
        model=joblib.load(OUT/f'separate_calibration_safety_{regime}.joblib')
        _,score=h.score(model['pipeline'],model['calibrator'],test,model['features']);test['score']=score
        for w in windows:
            target=test[f'target_{w}'];known=target.isin([h.SAFE_LABEL,h.UNSAFE_LABEL]);y=target[known].eq(h.SAFE_LABEL).astype(int)
            rows.append(dict(regime=regime,horizon_days=w,heldout_cases=len(test),resolved=int(known.sum()),ambiguous=int((~known).sum()),unsafe=int((1-y).sum()),safe=int(y.sum()),auroc=roc_auc_score(y,test.loc[known,'score']) if y.nunique()==2 else np.nan,ap_unsafe=average_precision_score(1-y,1-test.loc[known,'score']) if y.nunique()==2 else np.nan,score_model='fixed original-D1 train model with separate isotonic calibration half'))
            cross=pd.crosstab(fulltarget.loc[test.index],target)
            for original in cross.index:
                for horizon in cross.columns:balance.append(dict(regime=regime,horizon_days=w,original_label=original,horizon_label=horizon,cases=int(cross.loc[original,horizon])))
    pd.DataFrame(rows).to_csv(OUT/'fixed_horizon_safety_sensitivity.csv',index=False)
    pd.DataFrame(balance).to_csv(OUT/'fixed_horizon_label_transitions.csv',index=False)
    (OUT/'followup_window_manifest.json').write_text(json.dumps(dict(status='complete',horizons_days=windows,event_window='strictly after t0, through t0+h inclusive',target='missing required GR OR adverse within horizon => unsafe; cleared within horizon with complete evidence and no adverse => safe; otherwise ambiguous',model='fixed original D1 training; no horizon-specific retraining',observed_event_max=str(maximum),censoring_limit='maximum timestamp is only an observation-boundary proxy. No external capture-completeness indicator is available; unresolved cases remain ambiguous. These analyses do not establish censoring-free outcomes.',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()),indent=2))
    log('Fixed-horizon sensitivity complete')
if __name__=='__main__':main()
