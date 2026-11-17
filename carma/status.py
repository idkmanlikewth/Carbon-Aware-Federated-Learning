"""Compact campaign status from actual files: python -m carma.status"""
import json
from pathlib import Path


def status(root='carma_runs/ccgrid2027'):
    root=Path(root)
    state=json.loads((root/'campaign.json').read_text()) if (root/'campaign.json').exists() else {'status':'not_started','jobs':[]}
    rows=[]
    for job in state['jobs']:
        folder=root/job['name']; manifest=folder/'manifest.json'; result=folder/'results.json'
        count=len(json.loads(result.read_text())) if result.exists() else 0
        m=json.loads(manifest.read_text()) if manifest.exists() else {}
        rows.append({'name':job['name'],'status':m.get('status',job['status']),'evaluations':count,
                     'expected':len(m.get('config',{}).get('seeds',[]))*len(m.get('config',{}).get('policies',[]))})
    return {'status':state['status'],'workers':state.get('workers',1),'completed_studies':sum(r['status']=='complete' for r in rows),
            'planned_studies':len(rows),'completed_evaluations':sum(r['evaluations'] for r in rows),'studies':rows}


if __name__=='__main__':print(json.dumps(status(),indent=2))
