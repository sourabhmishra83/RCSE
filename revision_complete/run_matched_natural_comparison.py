"""Matched natural-case comparisons using frozen independent calibration scores.

No fitting or threshold selection uses test labels. Original outputs are retained.
Run from the repository root with the reproduction environment.
"""
from pathlib import Path
import importlib.util
import hashlib
import json
import sys
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from scipy.stats import binom, binomtest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / 'matched_natural'
S = ROOT / 'rcse_output/experimental_benchmark/splits'
V = S / 'frozen_baseline_results/rcse_policy_dataset/rcse_v2_policy_dataset'
CAPS = [.02, .03, .05, .10]
GRID = [0., .05, .10, .15, .20, .25, .30, .35, .40, .45, .50,
        .55, .60, .65, .70, .75, .80, .85, .90, .95, .96, .97, .98, .99, 1.]
SEED = 20261023

def load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main():
    OUT.mkdir(exist_ok=True)
    protocol = dict(seed=SEED, risk_caps=CAPS, score_grid=GRID,
        populations='same natural D1-resolved source cases for every policy',
        scores='frozen separate-calibration safety raw/isotonic and corrected action raw',
        selection='saved independent selection half; no test-label selection',
        consequence='frozen source-case TRAIN P90 from original policy metadata',
        costs=dict(c_H=.1, c_A=.5, rho_H=0),
        ltt='one-sided exact binomial; Bonferroni within each fixed family and cap',
        guarantee='IID/exchangeability assumptions only; no joint guarantee across reported families/caps',
        crc='marginal unsafe incidence, not conditional UER',
        bootstrap='2000 paired natural source-case replicates; fitted models held fixed')
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    h = load('matched_safety', '09b2h_full_benchmark_safety_robustness.py')
    p = load('matched_policy', '09b2n_run_rcse_v2_policy_simulation.py')
    b = load('matched_boot', '09b2r_bootstrap_rcse_v2_uncertainty.py')
    base = pd.read_parquet(ROOT / 'rcse_output/rcse_base_v2.parquet')
    base['source_case_id'] = base.case_id.astype(str)
    base['evidence_complete'] = ~base.goods_receipt_required.fillna(False).astype(bool) | base.gr_available_at_t0.fillna(False).astype(bool)
    spec = json.loads((ROOT / '09b2l_rcse_v2_policy_spec.json').read_text())
    params = p.get_reference_parameters(spec)
    rcgrid = [(t, v) for t in spec['risk_and_selectivity']['tau_safe_t0_grid']
              for v in spec['risk_and_selectivity']['beta0_grid']]
    rows, sweeps, audits, retention, paired, snapshots, losses = [], [], [], [], [], [], []
    plt.rcParams.update({'font.size':16,'axes.titlesize':18,'axes.labelsize':18,'xtick.labelsize':16,'ytick.labelsize':16,'legend.fontsize':16})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), sharey=True)
    labels = {'grouped_iid':'IID', 'temporal':'Temporal', 'ood_exposure':'Exposure OOD'}
    for ax, (regime, cfg) in zip(axes, h.SPLIT_CONFIG.items()):
        policydata = pd.read_parquet(V / f'rcse_v2_policy_dataset_{regime}.parquet')
        refs = {f'p{int(q*100)}':p.compute_train_reference_exposure(policydata, regime, q) for q in [.5,.9,.99]}
        datasets = []
        for prefix in ['independent_selection_cases', 'separate_calibration_natural_test']:
            path = OUT.parent / f'{prefix}_{regime}.parquet'
            snapshots.append(dict(file=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            d = pd.read_parquet(path)
            d['source_case_id'] = d.source_case_id.astype(str)
            d = d.merge(base[['source_case_id','transaction_exposure_eur','evidence_complete']], on='source_case_id', validate='one_to_one')
            d['unsafe'] = d.target.eq(h.UNSAFE_LABEL)
            assert d.target.isin([h.SAFE_LABEL,h.UNSAFE_LABEL]).all()
            d['C'] = p.consequence_from_exposure(d.transaction_exposure_eur, 'LOG_TRAIN_P90', refs)
            datasets.append(d)
        cal, test = datasets
        assert not set(cal.source_case_id) & set(test.source_case_id)
        def rc(d, tau, beta):
            risk = (1-d.scal)*d.C
            return d.valid & d.scal.ge(tau) & risk.le(beta) & risk.le(.1)
        def loss(d, selected):
            return np.where(selected, d.unsafe*d.C, .1)
        def record(name, selected, cap=np.nan, **extra):
            selected = np.asarray(selected, dtype=bool)
            n = int(selected.sum()); u = int(test.unsafe.to_numpy()[selected].sum())
            ci = binomtest(u,n).proportion_ci() if n else None
            row = dict(regime=regime, policy=name, risk_cap=cap, cases=len(test),
                executed=n, unsafe=u, coverage=n/len(test), uer=u/n if n else np.nan,
                uer_lower=ci.low if ci else np.nan, uer_upper=ci.high if ci else np.nan,
                modeled_loss=float(loss(test, selected).mean()),
                unsafe_incidence=u/len(test), **extra)
            rows.append(row)
            return row
        fixed = {'ALWAYS_ESCALATE':np.zeros(len(test),bool),
                 'EVIDENCE_COMPLETE':test.evidence_complete,
                 'GATE_ONLY':test.valid,
                 'SAFETY_095':test.scal.ge(.95),
                 'GATE_SAFETY_095':test.valid & test.scal.ge(.95),
                 'RCSE_REFERENCE':rc(test,.95,.05)}
        for name, selected in fixed.items(): record(name,selected)
        for comparator in ['EVIDENCE_COMPLETE','SAFETY_095','GATE_SAFETY_095']:
            delta = loss(test,fixed['RCSE_REFERENCE'])-loss(test,fixed[comparator])
            rng=np.random.default_rng(SEED); boot=[]
            for _ in range(2000): boot.append(delta[rng.integers(0,len(delta),len(delta))].mean())
            paired.append(dict(regime=regime,comparison='RCSE_REFERENCE minus '+comparator,estimate=delta.mean(),lower=np.quantile(boot,.025),upper=np.quantile(boot,.975),replicates=2000,cases=len(test)))
        families=[('ACTION_RAW','araw',False),('SAFETY_RAW','sraw',False),('SAFETY_RAW_GATE','sraw',True),('SAFETY_CAL','scal',False),('SAFETY_CAL_GATE','scal',True)]
        for name,col,gated in families:
            for threshold in GRID:
                chosen=test[col].ge(threshold)&(test.valid if gated else True)
                n=int(chosen.sum());u=int((chosen & test.unsafe).sum())
                sweeps.append(dict(regime=regime,policy=name,threshold=threshold,coverage=n/len(test),uer=u/n if n else np.nan,executed=n,unsafe=u))
            for cap in CAPS:
                options=[]
                for threshold in GRID:
                    chosen=cal[col].ge(threshold)&(cal.valid if gated else True)
                    n=int(chosen.sum());u=int((chosen&cal.unsafe).sum())
                    pv=float(binom.cdf(u,n,cap)) if n else 1.
                    ok=n>0 and pv<=.05/len(GRID)
                    audits.append(dict(regime=regime,policy=name,cap=cap,threshold=threshold,selected=n,unsafe=u,pvalue=pv,certified=ok))
                    if ok:options.append((n,threshold,u))
                n,t,u=sorted(options,key=lambda x:(-x[0],x[1]))[0] if options else (0,np.inf,0)
                record('LTT_'+name,test[col].ge(t)&(test.valid if gated else True),cap,threshold=t,selection_executed=n,selection_unsafe=u,certified=bool(options),controlled_quantity='conditional UER')
        for cap in CAPS:
            options=[];certified=[]
            for tau,beta in rcgrid:
                chosen=rc(cal,tau,beta);n=int(chosen.sum());u=int((chosen&cal.unsafe).sum())
                if n and u/n<=cap:options.append((n,tau,beta,u))
                pv=float(binom.cdf(u,n,cap)) if n else 1.
                if n and pv<=.05/len(rcgrid):certified.append((n,tau,beta,u))
            for name, opts in [('RCSE_EMPIRICAL',options),('RCSE_LTT',certified)]:
                n,tau,beta,u=sorted(opts,key=lambda x:(-x[0],x[1],x[2]))[0] if opts else (0,np.inf,0,0)
                record(name,rc(test,tau,beta),cap,threshold=tau,beta0=beta,selection_executed=n,selection_unsafe=u,certified=bool(opts) if name=='RCSE_LTT' else False,controlled_quantity='conditional UER' if name=='RCSE_LTT' else 'empirical calibration UER only')
            opts=[]
            for t in GRID:
                chosen=cal.sraw.ge(t);u=int((chosen&cal.unsafe).sum())
                if (u+1)/(len(cal)+1)<=cap:opts.append((int(chosen.sum()),t,u))
            n,t,u=sorted(opts,key=lambda x:(-x[0],x[1]))[0] if opts else (0,np.inf,0)
            record('CRC_UNSAFE_INCIDENCE',test.sraw.ge(t),cap,threshold=t,selection_executed=n,selection_unsafe=u,controlled_quantity='marginal unsafe incidence; not UER')
        for beta in np.linspace(0,.2,101):
            chosen=rc(test,.95,beta);n=int(chosen.sum());u=int((chosen&test.unsafe).sum())
            sweeps.append(dict(regime=regime,policy='RCSE_TAU095_BETA_SWEEP',threshold=beta,coverage=n/len(test),uer=u/n if n else np.nan,executed=n,unsafe=u))
        curves=pd.DataFrame(sweeps);curves=curves[curves.regime.eq(regime)]
        for name,label in [('ACTION_RAW','Action threshold'),('SAFETY_CAL','Calibrated safety'),('SAFETY_CAL_GATE','Gate + calibrated safety'),('RCSE_TAU095_BETA_SWEEP','RCSE budget sweep')]:
            d=curves[curves.policy.eq(name)].sort_values('coverage');ax.plot(100*d.coverage,100*d.uer,label=label,linewidth=1.4)
        for name,marker,label in [('EVIDENCE_COMPLETE','s','Evidence complete'),('RCSE_REFERENCE','*','RCSE reference')]:
            row=next(x for x in rows if x['regime']==regime and x['policy']==name)
            ax.scatter(100*row['coverage'],100*row['uer'],marker=marker,s=70,label=label,zorder=5)
        ax.set(title=f'{labels[regime]} (n={len(test):,})',xlabel='Execution coverage (%)',xlim=(0,102),ylim=(0,35));ax.grid(alpha=.2)
        original=b.prepare_regime_data(V,V/'rcse_v2_policy_results',regime,params)
        natural=original._intervention_family_r.eq('NATURAL')
        ref=natural & original.expected_action_reference.eq('EXECUTE')
        exe=original.action__RCSE_V2_NO_GATHER.eq('EXECUTE')
        expected_exec=(1-original.p_safe_t0_cal)*original._consequence
        observed=np.where(exe,original._unsafe_t0*original._consequence,.1)
        predicted=np.where(exe,expected_exec,.1)
        assert expected_exec[exe].le(.05+1e-10).all()
        losses.append(dict(regime=regime,episodes=len(original),executed=int(exe.sum()),mean_predicted_cost=float(predicted.mean()),mean_outcome_based_loss=float(observed.mean()),budget_based_predicted_upper_bound=float(exe.mean()*.05+(1-exe.mean())*.1),max_predicted_execution_loss=float(expected_exec[exe].max())))
        gatevalid=original.gate_state_v2.fillna('').astype(str).str.upper().eq('VALID')
        retention.append(dict(regime=regime,natural_cases=int(natural.sum()),reference_execute=int(ref.sum()),reference_execute_gate_valid=int((ref&gatevalid).sum()),reference_execute_policy_execute=int((ref&exe).sum()),all_natural_policy_execute=int((natural&exe).sum()),gate_pass_retention=(ref&gatevalid).sum()/ref.sum(),policy_execution_retention=(ref&exe).sum()/ref.sum(),natural_coverage=(natural&exe).sum()/natural.sum()))
        print(regime, 'matched cases',len(test),flush=True)
    axes[0].set_ylabel('Unsafe execution rate (%)')
    handles,legends=axes[0].get_legend_handles_labels();fig.legend(handles,legends,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,-.08));fig.tight_layout(rect=(0,.1,1,1))
    for ext in ['png','svg']:fig.savefig(OUT/f'matched_risk_coverage.{ext}',dpi=200,bbox_inches='tight')
    for data,name in [(rows,'policy_results'),(sweeps,'descriptive_sweeps'),(audits,'ltt_selection_audit'),(retention,'retention_audit'),(paired,'paired_loss_difference'),(losses,'reference_loss_audit')]:pd.DataFrame(data).to_csv(OUT/f'{name}.csv',index=False)
    (OUT/'manifest.json').write_text(json.dumps(dict(status='complete',protocol=protocol,score_inputs=snapshots,limitations=['D1-resolved cohort excludes ambiguous cases','No independent new dataset','No training variability in bootstrap','Risk-coverage curves descriptive; operating points selected on calibration only']),indent=2))

if __name__ == '__main__':main()
