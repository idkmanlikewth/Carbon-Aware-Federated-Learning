"""Generate a report using completed, checksum-verified studies only."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from .verify import verify
from .export import export
from .status import status

ROOT=Path(__file__).resolve().parents[1]


def report(campaign=None):
    campaign=Path(campaign or ROOT/'carma_runs/ccgrid2027')
    evidence=ROOT/'carma_evidence'; evidence.mkdir(exist_ok=True)
    snapshots=evidence/'convergence-gates'; snapshots.mkdir(exist_ok=True)
    gates=[]
    for name in ('gate-mnist','gate-fashion','gate-cifar'):
        path=ROOT/'carma_runs'/name/'diagnostic.json'
        if path.exists():
            raw=json.loads(path.read_text())
            if raw['status']=='complete':
                shutil.copyfile(path,snapshots/(name+'.json')); gates.append((name,raw))
    diag=ROOT/'carma_runs/learning-diagnostics/diagnostics.json'
    if diag.exists() and json.loads(diag.read_text())['status']=='complete':
        shutil.copyfile(diag,snapshots/'controlled-validation-diagnostics.json')
    checksum={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshots.glob('*.json') if p.name!='checksums.json'}
    (snapshots/'checksums.json').write_text(json.dumps(checksum,indent=2),encoding='utf-8')
    completed=[]
    for path in sorted(campaign.glob('*/manifest.json')):
        m=json.loads(path.read_text())
        if m['status']!='complete':continue
        verify(path.parent,require_checkpoints=True)
        target=evidence/path.parent.name
        if not target.exists():export(path.parent,target)
        verify(target)
        completed.append((target,m,json.loads((target/'analysis.json').read_text())))
    current=status(campaign)
    (evidence/'campaign-status.json').write_text(json.dumps(current,indent=2),encoding='utf-8')
    if (campaign/'plan.json').exists():shutil.copyfile(campaign/'plan.json',evidence/'plan.json')
    lines=['# CARMA-FL executed research report','',
           '**A Multi-Objective Reinforcement Learning Model for Optimizing the Latency–Carbon Trade-off in Federated Computing.**','',
           'CARMA-FL is a candidate method. Novelty, superiority and publication readiness are not assumed. Infrastructure is simulated; classifier outcomes and wall overhead are observed. The user-reported PC1 meter reference remains an unverified input, not a new measurement.','',
           f"Campaign status: **{current['status']}**. **{len(completed)} / {current['planned_studies']}** studies are completed and checksum-verified. **{current['completed_evaluations']}** final policy evaluations have actually been recorded, including interim studies. Statistical tables below include completed studies only; pending work is never represented as a result.",'',
           '## Conventional FedAvg learning gates','',
           'These CPU development gates preceded the scheduler campaign. They are not ten-seed scheduler comparisons. Validation is drawn from official training data; test is held out until final evaluation. They predate the strict source-snapshot campaign protocol and are not retroactively relabeled as source-pinned confirmatory evidence.','',
           '| Dataset | Train images | Validation / test | Rounds × local epochs | Seed | Final validation | Final test |',
           '|---|---:|---|---|---|---:|---:|']
    for name,m in gates:
        c=m['config']
        for r in m['results']:
            lines.append(f"| {c['dataset']} | {c['train_samples']} | {c['validation_samples']} / {c['test_samples']} | {c['rounds']} × {c['local_epochs']} | {r['seed']} | {r['validation_history'][-1]['validation_accuracy']:.2%} | {r['test_accuracy']:.2%} |")
    if (snapshots/'controlled-validation-diagnostics.json').exists():
        d=json.loads((snapshots/'controlled-validation-diagnostics.json').read_text())
        lines+=['','## Controlled failure diagnosis','','All cases use the same 6,000-image IID MNIST pool, disjoint validation set, local batches, ten rounds and two local epochs per seed. Six architecture/optimizer/standardization cases use seeds 11, 22 and 33. Test accuracy is not inspected. These are development diagnostics with three seeds, not a causal explanation for every difference from the original four-client protocol.','',
                '| Case | Mean final validation accuracy | Mean final validation loss |','|---|---:|---:|']
        for case in sorted({r['case'] for r in d['results']}):
            rows=[r['history'][-1] for r in d['results'] if r['case']==case]
            lines.append(f"| {case} | {np.mean([r['validation_accuracy'] for r in rows]):.2%} | {np.mean([r['validation_loss'] for r in rows]):.4f} |")
        lines+=['','The original chance-level results were not an adequate basis for scheduler claims. The controlled diagnostics separate architecture, optimizer and input-standardization effects at a common training budget. Data size, local epochs, participation and communication budgets also changed in the revised protocol; they are not falsely attributed to architecture alone.']
    lines+=['','## Completed independent-seed comparisons','',
            'Primary endpoint: two-dimensional normalized latency/carbon hypervolume, with final validation-accuracy feasibility required. Infeasible points contribute zero. Accuracy thresholds and cost reference bounds were saved before execution. Time/carbon to target are censored when never reached; conditional means must be interpreted with hit counts.','',
            '| Study | Policy | Seeds | Test accuracy mean | Final feasible | Target hit | Feasible HV mean [95% CI] | Policy training seconds mean |',
            '|---|---|---:|---:|---:|---:|---|---:|']
    for folder,m,a in completed:
        for p,s in a['summary'].items():
            hv=s['feasible_hypervolume']; ci=hv['ci95']; interval=f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci else 'not estimated'
            lines.append(f"| {folder.name} | {p} | {s['test_accuracy']['n']} | {s['test_accuracy']['mean']:.2%} | {s['final_accuracy_feasible']['mean']:.0%} | {s['target_hit']['mean']:.0%} | {hv['mean']:.4f} {interval} | {s['policy_training_wall_s']['mean']:.2f} |")
    lines+=['','## Statistical claims and limits','','Per-study analyses include Student-t confidence intervals and paired sign-flip permutation tests, Holm-corrected within the study baseline family. All seeds are independent model/policy retrainings. There is no campaign-wide multiplicity correction; exploratory scenario claims must be labeled accordingly. Raw test accuracy, every failed target, total modeled energy/latency/carbon, both constraint-violation counts, tail CVaR, fairness, per-round prediction-interval coverage and wall overhead remain in the raw JSON.','',
            'No unconditional carbon-saving claim is made from an accuracy-infeasible run. Nominal 90% marginal prediction intervals are empirically audited, not guaranteed under adaptive selection or unseen shifts. The risk reward uses a marginal-CVaR cohort surrogate, not exact joint straggler/idle CVaR. Measured policy training/inference wall time is contention-dependent because independent studies share one GPU. Coordinator-only overhead carbon uses an assumed 30 W; it excludes measured GPU energy and must not be called the full physical training footprint. Separate virtual training-rollout scenario costs are retained.','',
            'The Oort-style control is simplified. No published-method name is attached to a purported faithful reproduction. FedCAMO already contains dual carbon pricing and holistic accounting; CEFL already connects cost-aware utility to critical learning periods. Deep Sets, CVaR and constrained RL are established techniques. The candidate contribution needs method-level review and robust held-out evidence, not a novelty claim from combining them. See CARMA_LITERATURE.md and CARMA_PROTOCOL.md.','',
            'A confounded development ablation protocol was stopped and preserved outside the locked campaign. Those partial records are excluded from the tables. The locked no-constraint ablation removes only its penalty; the no-calibrated-prediction ablation retains the separate empirical CVaR mechanism.','',
            '## Reproduce and verify','','Run `python -m carma.campaign --output carma_runs/ccgrid2027 --device cuda --workers 3`. Inspect first with `--plan-only`. Resume skips completed seed-policy pairs and rejects changed runtime/config/input hashes. The exact 24 configurations are in carma_evidence/plan.json. Run `python -m carma.verify carma_evidence` for published raw evidence; classifier checkpoint hashes are retained but those large states remain in original local runs/full bundle. Use `--require-checkpoints` on the originals. Policy checkpoints and source snapshots are included in published evidence. Historical UK carbon observations and raw API response are under carma_inputs; they are grid estimates, not physical measurements of the virtual clients.','',
            'Regenerate this report only from actual files: `python -m carma.report`. Completed-study plots are generated from the same verified data. Pending studies are not extrapolated.']
    target=ROOT/'docs/CARMA_RESULTS.md'; target.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    make_figures(gates,completed,evidence)
    print(json.dumps({'report':str(target),'completed_verified_studies':len(completed),'campaign_status':current['status']}))


def make_figures(gates,completed,evidence):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures=evidence/'figures'; figures.mkdir(exist_ok=True)
    if gates:
        fig,axes=plt.subplots(1,len(gates),figsize=(12,3.6),squeeze=False)
        for ax,(name,m) in zip(axes[0],gates):
            for r in m['results']:
                h=r['validation_history']; ax.plot([v['round'] for v in h],[v['validation_accuracy'] for v in h],label=f"seed {r['seed']}")
            ax.set_title(m['config']['dataset']); ax.set_xlabel('Federated round'); ax.set_ylabel('Validation accuracy'); ax.set_ylim(0,1); ax.grid(alpha=.2); ax.legend()
        fig.suptitle('Conventional FedAvg learning gates · CPU development runs'); fig.tight_layout(); fig.savefig(figures/'fedavg_learning_gates.png',dpi=180); plt.close(fig)
    main=[(f,m,a) for f,m,a in completed if f.name.startswith('main-')]
    if main:
        fig,axes=plt.subplots(1,len(main),figsize=(6*len(main),5),squeeze=False)
        for ax,(folder,m,a) in zip(axes[0],main):
            names=list(a['summary']); values=[a['summary'][p]['test_accuracy']['mean'] for p in names]
            error=[(a['summary'][p]['test_accuracy']['ci95'][1]-a['summary'][p]['test_accuracy']['ci95'][0])/2 if a['summary'][p]['test_accuracy']['ci95'] else 0 for p in names]
            ax.bar(range(len(names)),values,yerr=error,color=['#C82D14' if p=='carma' else '#6b7b87' for p in names],capsize=2)
            ax.set_xticks(range(len(names)),names,rotation=70,ha='right',fontsize=8); ax.set_ylim(0,1); ax.set_title(folder.name); ax.set_ylabel('Held-out test accuracy · mean ± 95% t CI'); ax.grid(axis='y',alpha=.2)
        fig.tight_layout(); fig.savefig(figures/'independent_seed_accuracy.png',dpi=180); plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--campaign'); args=p.parse_args(); report(args.campaign)
