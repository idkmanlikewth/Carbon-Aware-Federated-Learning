"""Local HTTP adapter for the original React workbench's research panel."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import uuid
from .config import Config

ROOT = Path(__file__).resolve().parents[1]
LOCK = threading.Lock()
PROCESS = None
ACTIVE = None
ACTIVE_CONFIG = None


def state():
    with LOCK:
        if ACTIVE is None:
            return {'id': None, 'status': 'idle'}
        status = 'running' if PROCESS is not None and PROCESS.poll() is None else 'failed'
        path = folder_for(ACTIVE)/'manifest.json'
        if path.exists():
            status = json.loads(path.read_text(encoding='utf-8'))['status']
            if status == 'running' and (PROCESS is None or PROCESS.poll() is not None):
                status = 'failed'
        return {'id': ACTIVE, 'status': status}


def folder_for(identifier):
    if not re.fullmatch(r'(evidence|research_runs|carma_runs|carma_evidence)/[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)?', identifier):
        raise ValueError('Invalid research run identifier')
    return ROOT / identifier


def list_runs():
    rows = []
    for base in ('carma_evidence','carma_runs','evidence', 'research_runs'):
        paths=list((ROOT/base).glob('*/manifest.json'))+list((ROOT/base).glob('*/*/manifest.json'))
        for path in sorted(paths):
            m = json.loads(path.read_text(encoding='utf-8'))
            rows.append({'id': path.parent.relative_to(ROOT).as_posix(), 'status': m['status'], 'config': m['config'],'schema_version':m.get('schema_version',1)})
    return rows


def detail(identifier):
    folder = folder_for(identifier)
    if not (folder/'manifest.json').exists() and identifier == ACTIVE:
        log = folder.parent/(folder.name+'.log')
        return {'manifest': {'status': state()['status'], 'config': ACTIVE_CONFIG, 'training': [], 'datasets': {},
                             'measurement_type': 'SIMULATED_INFRASTRUCTURE'},
                'log': log.read_text(encoding='utf-8', errors='replace')[-12000:] if log.exists() else ''}
    data = {'manifest': json.loads((folder/'manifest.json').read_text(encoding='utf-8'))}
    if data['manifest']['status'] == 'complete' or data['manifest'].get('schema_version')==2:
        for name in ('results', 'analysis', 'trajectories'):
            path=folder/(name+'.json')
            if path.exists():data[name] = json.loads(path.read_text(encoding='utf-8'))
    if data['manifest'].get('schema_version')==2 and 'analysis' in data:
        data['analysis']['summary_metrics']=data['analysis']['summary']
        data['analysis']['summary']={p:metrics['feasible_hypervolume'] for p,metrics in data['analysis']['summary_metrics'].items()}
    log = folder.parent/(folder.name+'.log')
    data['log'] = log.read_text(encoding='utf-8', errors='replace')[-12000:] if log.exists() else ''
    return data


def start(raw):
    global PROCESS, ACTIVE, ACTIVE_CONFIG
    allowed = {'dataset', 'clients', 'rounds', 'episodes', 'train_samples', 'eval_samples', 'seeds', 'policies', 'download','engine',
               'partition','alpha','local_epochs','learning_rate','optimizer','participation','accuracy_target','carbon_budget_g','time_budget_s',
               'risk_alpha','risk_weight','uncertainty','heterogeneity','availability','scenario_shift','dynamic_carbon','device','preference'}
    if not isinstance(raw, dict) or set(raw)-allowed:
        raise ValueError('Unsupported research settings')
    cfg = dict(raw)
    engine=cfg.pop('engine','legacy')
    if engine not in ('legacy','carma'):raise ValueError('Unsupported engine')
    if engine=='carma':
        from carma.run import config_from
        count=cfg.pop('eval_samples',1000)
        cfg.update(validation_samples=count,test_samples=count)
        c=config_from(cfg)
        if c.rounds>200 or c.episodes>500 or c.train_samples>50000 or count>10000 or len(c.seeds)>20 or type(c.download) is not bool or type(c.dynamic_carbon) is not bool:
            raise ValueError('Research settings exceed supported limits')
        with LOCK:
            if PROCESS is not None and PROCESS.poll() is None:raise ValueError('A research experiment is already running')
            ACTIVE='carma_runs/'+uuid.uuid4().hex[:12]; ACTIVE_CONFIG=c.to_dict()
            folder=folder_for(ACTIVE); folder.parent.mkdir(exist_ok=True)
            config_path=folder.parent/(folder.name+'.config.json')
            config_path.write_text(json.dumps(c.to_dict()),encoding='utf-8')
            executable=os.environ.get('CARMA_PYTHON',sys.executable)
            command=[executable,'-u','-m','carma.run','--config',str(config_path),'--output',str(folder)]
            with (folder.parent/(folder.name+'.log')).open('w',encoding='utf-8') as log:
                PROCESS=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            return {'id':ACTIVE,'status':'running'}
    for key in ('seeds', 'policies'):
        if key in cfg:
            cfg[key] = tuple(cfg[key])
    c = Config(**cfg)
    if type(c.clients) is not int or c.rounds > 200 or c.episodes > 500 or c.train_samples > 50000 or c.eval_samples > 10000 or len(c.seeds) > 20 or type(c.download) is not bool:
        raise ValueError('Research settings exceed supported limits')
    with LOCK:
        if PROCESS is not None and PROCESS.poll() is None:
            raise ValueError('A research experiment is already running')
        ACTIVE = 'research_runs/'+uuid.uuid4().hex[:12]
        ACTIVE_CONFIG = c.to_dict()
        folder = folder_for(ACTIVE)
        folder.parent.mkdir(exist_ok=True)
        command = [sys.executable, '-u', '-m', 'research.run']
        for key in ('dataset', 'clients', 'rounds', 'episodes', 'train_samples', 'eval_samples'):
            command += ['--'+key.replace('_', '-'), str(getattr(c, key))]
        command += ['--seeds', *map(str, c.seeds), '--policies', *c.policies, '--output', str(folder)]
        if c.download:
            command.append('--download')
        with (folder.parent/(folder.name+'.log')).open('w', encoding='utf-8') as log:
            PROCESS = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        return {'id': ACTIVE, 'status': 'running'}


def cancel():
    with LOCK:
        if PROCESS is not None and PROCESS.poll() is None:
            PROCESS.terminate()
            PROCESS.wait(timeout=10)
            path = folder_for(ACTIVE)/'manifest.json'
            if path.exists():
                data = json.loads(path.read_text(encoding='utf-8'))
                if data['status'] == 'running':
                    from .run import write_json
                    data['status'] = 'cancelled'
                    write_json(path, data)
    return {'ok': True}
