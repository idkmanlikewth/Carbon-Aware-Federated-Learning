"""Local HTTP adapter for the original React workbench's research panel."""
import json
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
    if not re.fullmatch(r'(evidence|research_runs)/[A-Za-z0-9_-]+', identifier):
        raise ValueError('Invalid research run identifier')
    return ROOT / identifier


def list_runs():
    rows = []
    for base in ('evidence', 'research_runs'):
        for path in sorted((ROOT/base).glob('*/manifest.json')):
            m = json.loads(path.read_text(encoding='utf-8'))
            rows.append({'id': path.parent.relative_to(ROOT).as_posix(), 'status': m['status'], 'config': m['config']})
    return rows


def detail(identifier):
    folder = folder_for(identifier)
    if not (folder/'manifest.json').exists() and identifier == ACTIVE:
        log = folder.parent/(folder.name+'.log')
        return {'manifest': {'status': state()['status'], 'config': ACTIVE_CONFIG, 'training': [], 'datasets': {},
                             'measurement_type': 'SIMULATED_INFRASTRUCTURE'},
                'log': log.read_text(encoding='utf-8', errors='replace')[-12000:] if log.exists() else ''}
    data = {'manifest': json.loads((folder/'manifest.json').read_text(encoding='utf-8'))}
    if data['manifest']['status'] == 'complete':
        for name in ('results', 'analysis', 'trajectories'):
            data[name] = json.loads((folder/(name+'.json')).read_text(encoding='utf-8'))
    log = folder.parent/(folder.name+'.log')
    data['log'] = log.read_text(encoding='utf-8', errors='replace')[-12000:] if log.exists() else ''
    return data


def start(raw):
    global PROCESS, ACTIVE, ACTIVE_CONFIG
    allowed = {'dataset', 'clients', 'rounds', 'episodes', 'train_samples', 'eval_samples', 'seeds', 'policies', 'download'}
    if not isinstance(raw, dict) or set(raw)-allowed:
        raise ValueError('Unsupported research settings')
    cfg = dict(raw)
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
