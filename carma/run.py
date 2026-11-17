"""Auditable CARMA-FL runner: python -m carma.run --config study.json --output carma_runs/study"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
from dataclasses import replace
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time
import zipfile
import numpy as np
import torch
from research.run import write_json
from .config import Config,PROBLEM
from .data import load
from .experiment import train,episode,metrics
from .analysis import analyze

ROOT=Path(__file__).resolve().parents[1]


def config_from(raw):
    raw=dict(raw)
    for key in ('seeds','policies','preference'):
        if key in raw:raw[key]=tuple(raw[key])
    return Config(**raw)


def run(c,output,resume=False):
    out=Path(output)
    if out.exists() and not resume:raise FileExistsError('Use a new output directory or --resume')
    out.mkdir(parents=True,exist_ok=resume)
    device='cuda' if c.device=='auto' and torch.cuda.is_available() else 'cpu' if c.device=='auto' else c.device
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
    torch.set_num_threads(c.threads); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    runtime=('__init__','config','data','learning','infrastructure','policy','environment','experiment','analysis','run')
    paths=[ROOT/'carma'/(name+'.py') for name in runtime]+[ROOT/'fl_pipeline/data.py',ROOT/'config.json']
    hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    source_hash=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    inputs={'profiles':hashlib.sha256(Path(c.profiles).read_bytes()).hexdigest()}
    if c.historical_trace:inputs['historical_trace']=hashlib.sha256(Path(c.historical_trace).read_bytes()).hexdigest()
    if c.observation_history:inputs['observation_history']=hashlib.sha256(Path(c.observation_history).read_bytes()).hexdigest()
    if resume:
        manifest=json.loads((out/'manifest.json').read_text())
        if manifest['config']!=json.loads(json.dumps(c.to_dict())) or manifest['source_hash']!=source_hash or manifest.get('input_hashes')!=inputs:
            raise ValueError('Resume requires identical config and source; start a new study otherwise')
        if manifest['status']=='complete':return manifest
        results=json.loads((out/'results.json').read_text()) if (out/'results.json').exists() else []
        trajectories=json.loads((out/'trajectories.json').read_text()) if (out/'trajectories.json').exists() else []
        checkpoint=out/'checkpoint.json'
        if checkpoint.exists():
            saved=json.loads(checkpoint.read_text()); results=saved['results']; trajectories=saved['trajectories']; manifest['training']=saved['training']
    else:
        results=[]; trajectories=[]
        manifest={'schema_version':2,'algorithm':'CARMA-FL candidate; novelty and benefit not established','problem_statement':PROBLEM,
                  'status':'running','config':c.to_dict(),'source_hash':source_hash,'source_files':hashes,'started_utc':datetime.now(timezone.utc).isoformat(),
                  'device':device,'gpu':torch.cuda.get_device_name() if device=='cuda' else None,'python':platform.python_version(),'input_hashes':inputs,
                  'versions':{p:importlib.metadata.version(p) for p in ('torch','torchvision','numpy','scipy')},
                  'git_commit':subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,cwd=ROOT).stdout.strip(),
                  'git_dirty':bool(subprocess.run(['git','status','--porcelain'],capture_output=True,text=True,cwd=ROOT).stdout.strip()),
                  'measurement_type':'SIMULATED_INFRASTRUCTURE; observed classifier accuracy and perf_counter wall overhead. No new physical energy measurement.',
                  'hardware_profile':json.loads(Path(c.profiles).read_text()),'datasets':{},'training':[],
                  'baseline_fidelity':'fedavg/random/round-robin/fastest/carbon-greedy/weighted are implemented controls. Oort-style is simplified, not a faithful reproduction. No result is labeled FedCAMO/GreenFL/CEFL/FedMarl/MODRL-FL reproduction.'}
        with zipfile.ZipFile(out/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
            for p in paths:archive.write(p,p.relative_to(ROOT).as_posix())
        write_json(out/'manifest.json',manifest)
    started=time.perf_counter()
    try:
        done={(r['seed'],r['policy']) for r in results}
        for seed in c.seeds:
            if all((seed,p) in done for p in c.policies):continue
            data=load(c,seed).to(device)
            manifest['datasets'][str(seed)]=data.provenance
            for policy in c.policies:
                if (seed,policy) in done:continue
                print(f'{out.name} dataset={c.dataset} clients={c.clients} seed={seed} policy={policy} training',flush=True)
                agent,training=train(c,data,seed,policy,device)
                if agent is not None:
                    torch.save({'model':agent.online.state_dict(),'config':c.to_dict(),'seed':seed,'policy':policy},out/f'policy_{seed}_{policy}.pt')
                env=episode(c,data,seed+100000,policy,agent,device=device)
                row=metrics(env,training)
                inference=sum(r['inference_wall_s'] for r in env.rows)
                overhead_wh=30*(training['wall_s']+inference)/3600
                row.update(seed=seed,policy=policy,preference_index=0,inference_wall_s=inference,
                           practical_carbon_with_training_g=row['carbon_g']+overhead_wh*.4,
                           practical_energy_with_training_wh=row['energy_wh']+overhead_wh,
                           training_rollout_scenario_carbon_g=training['rollout_scenario_carbon_g'],
                           training_rollout_scenario_energy_wh=training['rollout_scenario_energy_wh'],
                           overhead_carbon_scope='Coordinator-only 30W assumption; excludes GPU energy. Separate virtual training rollout costs reported; neither is physical metering.',
                           practical_latency_with_training_s=row['latency_s']+training['wall_s']+inference,
                           interval_coverage=(env.infra.coverage_hits/max(1,env.infra.coverage_count)).tolist(),
                           interval_coverage_n=int(env.infra.coverage_count),profile_sha256=env.infra.profile_sha256,
                           historical_trace=env.infra.trace_provenance)
                row['observation_history']=env.infra.observation_provenance
                results.append(row)
                trajectories.extend({'seed':seed,'policy':policy,'preference_index':0,**r} for r in env.rows)
                manifest['training'].append({'seed':seed,'policy':policy,**training})
                torch.save(env.model.state_dict(),out/f'classifier_{seed}_{policy}.pt')
                write_json(out/'checkpoint.json',{'results':results,'trajectories':trajectories,'training':manifest['training']})
                write_json(out/'results.json',results); write_json(out/'trajectories.json',trajectories); write_json(out/'manifest.json',manifest)
                print(f'{out.name} seed={seed} policy={policy} test={row["test_accuracy"]:.4f} val={row["validation_accuracy"]:.4f} target={row["target_hit"]} updates={training["updates"]}',flush=True)
            del data
            if device=='cuda':torch.cuda.empty_cache()
        write_json(out/'analysis.json',analyze(results,c))
        manifest.update(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),wall_s=manifest.get('wall_s',0)+time.perf_counter()-started)
        manifest['artifact_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file() and p.name!='manifest.json'}
        write_json(out/'manifest.json',manifest)
        return manifest
    except BaseException as exc:
        manifest.update(status='failed',error=f'{type(exc).__name__}: {exc}',wall_s=manifest.get('wall_s',0)+time.perf_counter()-started)
        write_json(out/'manifest.json',manifest)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True); parser.add_argument('--output',required=True); parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(); run(config_from(json.loads(Path(args.config).read_text(encoding='utf-8-sig'))),args.output,args.resume)


if __name__=='__main__':main()
