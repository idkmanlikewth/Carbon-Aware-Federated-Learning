"""Audit CARMA-FL records, checksums, split provenance and constraint metrics."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
from .analysis import analyze
from .run import config_from


def equivalent(a,b):
    if isinstance(a,dict) and isinstance(b,dict):return a.keys()==b.keys() and all(equivalent(a[k],b[k]) for k in a)
    if isinstance(a,list) and isinstance(b,list):return len(a)==len(b) and all(equivalent(x,y) for x,y in zip(a,b))
    if isinstance(a,(float,int)) and isinstance(b,(float,int)):return bool(np.isclose(a,b,rtol=1e-8,atol=1e-10))
    return a==b


def verify(folder,require_checkpoints=False):
    folder=Path(folder); m=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    if m.get('schema_version')!=2 or m['status']!='complete':raise ValueError('Require a completed schema-2 study')
    omitted=m.get('omitted_checkpoints',{})
    for name,digest in m['artifact_sha256'].items():
        if Path(name).name!=name:raise ValueError('Unsafe artifact filename')
        path=folder/name
        if not path.exists() and name in omitted and name.endswith('.pt') and name.startswith('classifier_') and not require_checkpoints:
            if omitted[name]!=digest:raise ValueError('Omitted checkpoint checksum differs')
            continue
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('Artifact checksum mismatch: '+name)
    with zipfile.ZipFile(folder/'source_snapshot.zip') as archive:
        for name,digest in m['source_files'].items():
            if hashlib.sha256(archive.read(name)).hexdigest()!=digest:raise ValueError('Source checksum mismatch: '+name)
    if hashlib.sha256(json.dumps(m['source_files'],sort_keys=True).encode()).hexdigest()!=m['source_hash']:
        raise ValueError('Source mapping digest differs')
    c=config_from(m['config']); rows=json.loads((folder/'results.json').read_text()); trace=json.loads((folder/'trajectories.json').read_text())
    expected={(s,p) for s in c.seeds for p in c.policies}
    if len(rows)!=len(expected) or {(r['seed'],r['policy']) for r in rows}!=expected:raise ValueError('Incomplete or duplicate policy evaluations')
    if len(trace)!=len(rows)*c.rounds:raise ValueError('Wrong evaluation-round count')
    if len(m['datasets'])!=len(c.seeds) or len({v['subset_sha256'] for v in m['datasets'].values()})!=len(c.seeds):raise ValueError('Dataset subsets reused across seeds')
    for row in rows:
        rs=[r for r in trace if r['seed']==row['seed'] and r['policy']==row['policy']]
        if sorted(r['round'] for r in rs)!=list(range(1,c.rounds+1)):raise ValueError('Round sequence mismatch')
        for r in rs:
            if not r['selected'] or len(set(r['selected']))!=len(r['selected']) or not set(r['selected'])<=set(r['available']):raise ValueError('Invalid client selection')
            if any(type(i) is not int or not 0<=i<c.clients for i in r['available']):raise ValueError('Client ID out of bounds')
            if not np.isclose(sum(r['preference'][:2]),1) or min(r['preference'])<0:raise ValueError('Invalid preferences')
            for key in ('latency_s','energy_wh','carbon_g','validation_loss'):
                if not np.isfinite(r[key]) or r[key]<0:raise ValueError('Invalid cost/loss')
        for key in ('latency_s','energy_wh','carbon_g'):
            if not np.isclose(sum(r[key] for r in rs),row[key]):raise ValueError('Trajectory total mismatch: '+key)
        hits=[i for i,r in enumerate(rs) if r['validation_accuracy']>=c.accuracy_target]
        if row['target_hit']!=bool(hits):raise ValueError('Target hit status mismatch')
        if not hits:
            if row['time_to_target_s'] is not None or row['carbon_to_target_g'] is not None:raise ValueError('Unreached targets must be censored')
        else:
            for target_key,key in (('time_to_target_s','latency_s'),('carbon_to_target_g','carbon_g')):
                if not np.isclose(sum(r[key] for r in rs[:hits[0]+1]),row[target_key]):raise ValueError('Target cost mismatch')
        if row['final_accuracy_feasible']!=(rs[-1]['validation_accuracy']>=c.accuracy_target):raise ValueError('Final accuracy feasibility mismatch')
        if row['accuracy_constraint_violations']!=sum(r['accuracy_constraint_violation']>0 for r in rs):raise ValueError('Constraint count mismatch')
        coverage=np.sum([r['coverage_hits'] for r in rs],axis=0)/sum(r['coverage_n'] for r in rs)
        if not np.allclose(coverage,row['interval_coverage']):raise ValueError('Interval coverage mismatch')
        if not 0<=row['test_accuracy']<=1:raise ValueError('Invalid final test accuracy')
    stored=json.loads((folder/'analysis.json').read_text())
    if not equivalent(analyze(rows,c),stored):raise ValueError('Analysis does not reproduce')
    return {'run':folder.name,'status':'verified','seeds':len(c.seeds),'evaluations':len(rows),'rounds':len(trace),
            'omitted_classifier_checkpoints':len(omitted)}


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('folders',nargs='+'); p.add_argument('--require-checkpoints',action='store_true'); a=p.parse_args()
    for item in a.folders:
        path=Path(item); folders=[path] if (path/'manifest.json').exists() else [v.parent for v in path.glob('*/manifest.json')]
        if not folders:raise ValueError('No study manifests found: '+item)
        for folder in folders:print(json.dumps(verify(folder,a.require_checkpoints)))


if __name__=='__main__':main()
