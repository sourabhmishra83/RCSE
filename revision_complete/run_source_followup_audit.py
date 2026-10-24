"""Audit raw timestamps and reconstruct 90-day outcomes for all eligible cases."""
from pathlib import Path
from datetime import datetime
from collections import Counter
import json,hashlib,importlib.util,sys
sys.dont_write_bytecode=True
import pandas as pd
import numpy as np
from lxml import etree
ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent/'robustness_extension'

def main():
 OUT.mkdir(exist_ok=True)
 base=pd.read_parquet(ROOT/'rcse_output/rcse_base_v2.parquet')
 times=dict(zip(base.case_id.astype(str),pd.to_datetime(base.decision_time,utc=True).map(lambda x:x.to_pydatetime())))
 adverse={'Cancel Invoice Receipt','Set Payment Block','Cancel Goods Receipt','Change Price','Change Quantity','Record Subsequent Invoice'}
 years=Counter();acts=Counter();records=[];ntrace=nevent=invalid=inversions=0;minimum=maximum=None;after_po=0
 for _,trace in etree.iterparse(str(ROOT/'BPI_Challenge_2019.xes'),events=('end',),tag='{*}trace'):
  cid=next((c.get('value') for c in trace if c.get('key')=='concept:name'),None)
  origin=times.get(cid);adv=clear=False;last=None;latest=None;first_po=first_inv=None;bad=0;eligible_events=0
  for event in trace:
   if etree.QName(event).localname!='event':continue
   attrs={c.get('key'):c.get('value') for c in event};stamp=attrs.get('time:timestamp');activity=attrs.get('concept:name','')
   nevent+=1
   try:dt=datetime.fromisoformat(stamp.replace('Z','+00:00'))
   except Exception:invalid+=1;continue
   years[dt.year]+=1;minimum=dt if minimum is None or dt<minimum else minimum;maximum=dt if maximum is None or dt>maximum else maximum
   if last is not None and dt<last:inversions+=1
   last=dt
   if origin is None:continue
   eligible_events+=1;latest=dt if latest is None or dt>latest else latest
   if dt.year<2018 or dt.year>2019:bad+=1;acts[activity]+=1
   if activity=='Create Purchase Order Item':first_po=dt if first_po is None or dt<first_po else first_po
   if activity=='Record Invoice Receipt':first_inv=dt if first_inv is None or dt<first_inv else first_inv
   days=(dt-origin).total_seconds()/86400
   if 0<days<=90:
    adv |= activity in adverse;clear |= activity=='Clear Invoice'
  if origin is not None:
   po_late=first_po is not None and first_inv is not None and first_po>first_inv
   after_po+=int(po_late)
   records.append(dict(source_case_id=cid,adverse_90=adv,cleared_90=clear,events_outside_2018_2019=bad,raw_po_after_first_invoice=po_late,observed_span_days=(latest-origin).total_seconds()/86400 if latest else np.nan))
  trace.clear()
  while trace.getprevious() is not None:del trace.getparent()[0]
  ntrace+=1
  if ntrace%100000==0:print('Audited traces',ntrace,flush=True)
 follow=pd.DataFrame(records);assert follow.source_case_id.is_unique and len(follow)==len(base)
 base['source_case_id']=base.case_id.astype(str)
 joined=base.merge(follow,on='source_case_id',validate='one_to_one')
 missing=joined.goods_receipt_required.fillna(False).astype(bool)&~joined.gr_available_at_t0.fillna(False).astype(bool)
 targets=pd.Series(np.where(missing|joined.adverse_90,'NOT_SAFE_TO_EXECUTE',np.where(joined.cleared_90,'SAFE_TO_EXECUTE','AMBIGUOUS')),index=joined.source_case_id)
 follow['target_90']=follow.source_case_id.map(targets)
 follow.to_parquet(OUT/'all_eligible_followup.parquet',index=False)
 report=dict(raw_traces=ntrace,raw_events=nevent,invalid_timestamp_events=invalid,original_order_time_decreases=inversions,raw_min=str(minimum),raw_max=str(maximum),events_by_year=dict(years),eligible_cases=len(base),eligible_cases_with_events_outside_2018_2019=int(follow.events_outside_2018_2019.gt(0).sum()),eligible_outside_year_events_by_activity=dict(acts),raw_po_after_first_invoice_cases=after_po,derived_negative_po_age_cases=int(base.po_to_invoice_days.lt(0).sum()),handling='Source timestamps retained. Derived construction sorts UTC timestamp then original event order. Temporal splits use decision_time, not maximum trace time. The 90-day outcome sensitivity ignores events outside the decision-relative window; outside-2018/2019 is an audit category, not automatic evidence of invalidity.',followup_limit='Observed last event is not proof of complete capture. Remaining outcomes stay ambiguous.',raw_sha256=hashlib.file_digest((ROOT/'BPI_Challenge_2019.xes').open('rb'),'sha256').hexdigest())
 (OUT/'source_timestamp_audit.json').write_text(json.dumps(report,indent=2))
 print(json.dumps({k:v for k,v in report.items() if k not in ['eligible_outside_year_events_by_activity','raw_sha256']},indent=2),flush=True)
if __name__=='__main__':main()
