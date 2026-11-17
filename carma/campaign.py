"""Resumable GPU campaign; runs independently of the chat and never invents results.

python -m carma.campaign --output carma_runs/ccgrid2027 --device cuda
Use --plan-only to inspect all exact configurations before execution.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from dataclasses import replace
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time
import threading
from .config import Config,BASELINES,ABLATIONS
from .run import write_json


def plan(device='cuda',seeds=tuple(range(101,111))):
    base=Config(dataset='mnist',rounds=15,episodes=8,train_samples=6000,accuracy_target=.9,
                partition='dirichlet',alpha=.5,seeds=seeds,policies=BASELINES+ABLATIONS,device=device)
    jobs=[('main-mnist',base),('main-fashion',replace(base,dataset='fashion-mnist',rounds=20,accuracy_target=.8)),
          ('main-cifar',replace(base,dataset='cifar10',rounds=30,train_samples=10000,accuracy_target=.6,carbon_budget_g=60,time_budget_s=2400))]
    compare=replace(base,policies=('random','pc_ddqn','carma'))
    scenarios=[('iid',{'partition':'iid'}),('noniid-01',{'alpha':.1}),('noniid-10',{'alpha':10.}),
               ('static-carbon',{'dynamic_carbon':False}),('homogeneous',{'heterogeneity':0.}),('heterogeneous-2',{'heterogeneity':2.}),
               ('availability-06',{'availability':.6}),('uncertainty-0',{'uncertainty':0.}),('uncertainty-05',{'uncertainty':.5}),
               ('unseen-shift-2',{'scenario_shift':2.}),('latency-preference',{'preference':(.8,.2)}),
               ('carbon-preference',{'preference':(.2,.8)}),('risk-low',{'risk_weight':.05}),('risk-high',{'risk_weight':.5}),
               ('historical-carbon',{'historical_trace':'carma_inputs/gb-2025-01-01.csv'})]
    jobs += [(name,replace(compare,**values)) for name,values in scenarios]
    jobs += [(f'scale-{n}',replace(compare,clients=n,rounds=30)) for n in (4,10,20,50,100)]
    jobs += [('rl-convergence',replace(compare,policies=('dqn','pc_ddqn','carma'),episodes=24))]
    return jobs


def process_exists(pid):
    if os.name=='nt':
        # Avoid os.kill(pid,0), which is unsafe on Windows.
        out=subprocess.run(['tasklist','/FI',f'PID eq {pid}','/FO','CSV','/NH'],capture_output=True,text=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)).stdout
        return str(pid) in out and Path(sys.executable).name.lower() in out.lower()
    try:os.kill(pid,0); return True
    except ProcessLookupError:return False


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True); parser.add_argument('--device',choices=('cuda','cpu','auto'),default='cuda')
    parser.add_argument('--plan-only',action='store_true'); parser.add_argument('--only',nargs='*')
    parser.add_argument('--workers',type=int,choices=(1,2,3),default=1)
    parser.add_argument('--extend-plan',action='store_true')
    args=parser.parse_args(); root=Path(args.output).resolve(); root.mkdir(parents=True,exist_ok=True)
    jobs=plan(args.device)
    if args.only:jobs=[(name,c) for name,c in jobs if name in args.only]
    if not jobs:parser.error('No selected jobs')
    plan_json=[{'name':name,'config':c.to_dict()} for name,c in jobs]
    plan_path=root/'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text())!=json.loads(json.dumps(plan_json)):
        old={r['name']:r['config'] for r in json.loads(plan_path.read_text())}
        new={r['name']:r['config'] for r in json.loads(json.dumps(plan_json))}
        if not args.extend_plan or not old.keys()<=new.keys() or any(old[k]!=new[k] for k in old):
            raise ValueError('Campaign plan differs; existing configurations may not change. Use a new directory, or --extend-plan for additional jobs only.')
    write_json(plan_path,plan_json)
    if args.plan_only:
        print(json.dumps({'jobs':len(jobs),'policy_evaluations':sum(len(c.seeds)*len(c.policies) for _,c in jobs),'plan':str(plan_path)})); return
    prior=json.loads((root/'campaign.json').read_text()) if (root/'campaign.json').exists() else {}
    active={e['name']:e for e in prior.get('jobs',[]) if e.get('status')=='running' and e.get('pid')}
    state={'status':'running','started_utc':prior.get('started_utc',datetime.now(timezone.utc).isoformat()),'workers':args.workers,
           'timing_context':'Independent study processes may share the GPU; measured wall overhead is contention-dependent, not isolated hardware timing.',
           'jobs':[{'name':name,'status':'pending'} for name,_ in jobs]}
    lock=threading.Lock()
    def save():
        with lock:write_json(root/'campaign.json',state)
    write_json(root/'campaign.json',state)
    def execute(job):
        name,c=job
        entry=next(e for e in state['jobs'] if e['name']==name)
        out=root/name; cfg_path=root/(name+'.config.json'); write_json(cfg_path,c.to_dict())
        manifest=out/'manifest.json'
        if manifest.exists() and json.loads(manifest.read_text())['status']=='complete':
            entry.update(status='complete',resumed_skip=True); save(); return
        previous=active.get(name)
        if previous and process_exists(previous['pid']):
            entry.update(previous,adopted_existing_process=True); save()
            while process_exists(previous['pid']):time.sleep(2)
            complete=manifest.exists() and json.loads(manifest.read_text())['status']=='complete'
            entry.update(status='complete' if complete else 'failed'); save()
            if not complete:raise RuntimeError(name+' existing process did not complete; inspect its log and resume')
            return
        command=[sys.executable,'-u','-m','carma.run','--config',str(cfg_path),'--output',str(out)]
        if manifest.exists():command.append('--resume')
        entry.update(status='running',started_utc=datetime.now(timezone.utc).isoformat())
        save(); print(f'START {name}',flush=True)
        started=time.perf_counter()
        with (root/(name+'.log')).open('a',encoding='utf-8') as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            entry['pid']=process.pid; save()
            code=process.wait()
        entry.update(status='complete' if code==0 else 'failed',returncode=code,wall_s=time.perf_counter()-started)
        save()
        print(f'{entry["status"].upper()} {name} {entry["wall_s"]:.1f}s',flush=True)
        if code:
            raise RuntimeError(f'{name} failed; inspect its log. Re-run this command to resume after correcting the cause.')
    failures=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in [pool.submit(execute,job) for job in jobs]:
            try:future.result()
            except Exception as exc:failures.append(str(exc))
    if failures:
        state.update(status='failed',errors=failures); save(); raise SystemExit('; '.join(failures))
    state.update(status='complete',completed_utc=datetime.now(timezone.utc).isoformat()); write_json(root/'campaign.json',state)


if __name__=='__main__':main()
