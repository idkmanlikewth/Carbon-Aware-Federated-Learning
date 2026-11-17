import itertools
import numpy as np
from research.analysis import hypervolume,interval,pareto_mask


def paired_permutation(differences):
    x=np.asarray(differences,float)
    if len(x)<2:return None
    observed=abs(x.mean())
    if len(x)<=15:
        means=np.array([abs(np.mean(x*np.asarray(signs))) for signs in itertools.product((-1,1),repeat=len(x))])
        return float(np.mean(means>=observed-1e-14))
    rng=np.random.default_rng(9271)
    means=np.abs((rng.choice([-1,1],size=(20000,len(x)))*x).mean(1))
    return float((1+np.sum(means>=observed-1e-14))/(len(means)+1))


def analyze(results,c):
    scored=[]
    for row in results:
        # Accuracy is a validation constraint; failed points contribute ZERO HV.
        point=[-row['latency_s']/c.time_budget_s,-row['carbon_g']/c.carbon_budget_g]
        hv=hypervolume([point],reference=(-1.,-1.)) if row['final_accuracy_feasible'] else 0.
        scored.append({**row,'feasible_hypervolume':hv})
    summary={}
    keys=('test_accuracy','latency_s','carbon_g','energy_wh','gini','tail_latency_cvar','tail_carbon_cvar',
          'policy_training_wall_s','inference_wall_s','practical_carbon_with_training_g','feasible_hypervolume',
          'accuracy_constraint_violations','carbon_budget_violations','final_accuracy_feasible','target_hit')
    for policy in c.policies:
        rows=[r for r in scored if r['policy']==policy]
        summary[policy]={k:interval([r[k] for r in rows]) for k in keys}
        for k in ('time_to_target_s','carbon_to_target_g'):
            values=[r[k] for r in rows if r[k] is not None]
            summary[policy][k]=interval(values) if values else {'n':0,'mean':None,'ci95':None}
    frontiers=[]
    for seed in c.seeds:
        rows=[r for r in scored if r['seed']==seed and r['final_accuracy_feasible']]
        if rows:
            mask=pareto_mask([[-r['latency_s'],-r['carbon_g']] for r in rows])
            frontiers.extend({'seed':seed,'policy':r['policy']} for r,keep in zip(rows,mask) if keep)
    tests=[]
    reference={r['seed']:r for r in scored if r['policy']=='carma'}
    for policy in c.policies:
        if policy=='carma':continue
        pairs=[(reference[r['seed']],r) for r in scored if r['policy']==policy and r['seed'] in reference]
        if len(pairs)<2:continue
        diffs=[a['feasible_hypervolume']-b['feasible_hypervolume'] for a,b in pairs]
        tests.append({'baseline':policy,'n':len(pairs),'mean_difference':float(np.mean(diffs)),
                      'ci95':interval(diffs)['ci95'],'p_exact_or_mc':paired_permutation(diffs),'p_holm':None})
    ordered=sorted(tests,key=lambda r:r['p_exact_or_mc']); previous=0.
    for i,row in enumerate(ordered):
        previous=max(previous,min(1.,row['p_exact_or_mc']*(len(ordered)-i))); row['p_holm']=previous
    return {'schema_version':2,'primary_endpoint':'validation-accuracy-feasible 2D cost hypervolume',
            'reference':[-1,-1],'normalization':{'latency_s':c.time_budget_s,'carbon_g':c.carbon_budget_g},
            'accuracy_target':c.accuracy_target,'summary':summary,'pareto':frontiers,'paired_tests':tests,
            'inference':'Independent seeds are the units. Student-t 95% intervals; two-sided paired sign-flip permutation tests on feasible HV, Holm-corrected across baselines in this run. Target costs are conditional on reaching the validation target, with n/hit rate reported; failures remain censored, never zero-filled.',
            'results':scored}
