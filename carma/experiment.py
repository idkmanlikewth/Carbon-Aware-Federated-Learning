"""Actual classifier rollouts; no simulator-accuracy surrogate is used for RL."""
import time
from dataclasses import replace
import numpy as np
import torch
from .environment import Environment
from .infrastructure import cvar
from .learning import evaluate
from .policy import Agent

LEARNED={'carma','dqn','pc_ddqn','no_uncertainty','no_risk','no_adaptation','no_constraint','no_conditioning'}


def heuristic(env,policy,rng):
    state=env.state(); x,g,p,mask=state
    ids=np.flatnonzero(mask); k=env.choose_count(policy)
    if policy=='fedavg':return ids.tolist()
    if policy=='random':return rng.choice(ids,k,replace=False).tolist()
    if policy=='round_robin':
        start=(env.round*k)%env.c.clients
        ordered=sorted(ids,key=lambda i:(int(i)-start)%env.c.clients)
        return [int(i) for i in ordered[:k]]
    if policy=='fastest':score=-x[:,1]
    elif policy=='carbon_greedy':score=-x[:,2]*x[:,3]
    elif policy=='oort_style':
        # Inspired by utility/straggler/exploration concepts, not a faithful Oort reproduction.
        score=np.sqrt(np.maximum(x[:,0],1e-9))*np.maximum(x[:,10],.01)/(np.maximum(x[:,1],.01)**.5)+.05*x[:,9]
    else:
        risk=env.c.risk_weight if policy=='no_learned_selection' else 0
        urgency=p[2]*env.dual if policy=='no_learned_selection' else 1.
        score=urgency*(x[:,0]+.3*x[:,11]+.05*x[:,9]) - p[0]*x[:,1]-p[1]*x[:,2]*x[:,3]-risk*(x[:,12]+x[:,13])-.05*x[:,8]
    return ids[np.argsort(-score[ids],kind='stable')[:k]].tolist()


def episode(c,data,seed,policy,agent=None,training=False,epsilon=0,device='cpu'):
    env=Environment(c,data,seed,policy,evaluation=not training,device=device)
    rng=np.random.default_rng(seed+555)
    for _ in range(c.rounds):
        before=time.perf_counter()
        if agent is not None:
            picks=[]; transitions=[]
            for _ in range(env.choose_count(policy)):
                state=env.state(); action=agent.act(state,epsilon if training else 0)
                env.picked.append(action); picks.append(action)
                transitions.append((state,action,env.state()))
        else:picks=heuristic(env,policy,rng)
        overhead=time.perf_counter()-before
        reward,done,row=env.step(picks,overhead)
        if training and agent is not None:
            for index,(state,action,intermediate) in enumerate(transitions):
                last=index==len(transitions)-1
                agent.learn(state,action,reward if last else 0.,env.state() if last else intermediate,done if last else False)
    return env


def train(c,data,seed,policy,device):
    if policy not in LEARNED:return None,{'wall_s':0.,'updates':0,'episodes':[],'rollout_scenario_carbon_g':0.,'rollout_scenario_energy_wh':0.}
    started=time.perf_counter()
    agent=Agent(c,seed+100,conditioned=policy not in ('dqn','no_conditioning'),double=policy!='dqn')
    history=[]
    rollout_carbon=rollout_energy=0.
    for ep in range(c.episodes):
        anchors=(c.preference,(.8,.2),(.2,.8))
        weights=anchors[ep%3] if ep<3 else tuple(agent.rng.dirichlet([1.,1.]))
        episode_config=c if policy=='dqn' else replace(c,preference=weights)
        env=episode(episode_config,data,seed+10000+ep,policy,agent,True,max(.05,.8*(1-ep/c.episodes)),device)
        rollout_carbon+=env.total_carbon; rollout_energy+=env.total_wh
        history.append({'episode':ep+1,'validation_accuracy':env.accuracy,'validation_loss':env.loss,
                        'return':sum(r['reward'] for r in env.rows),'dqn_loss_mean':float(np.mean(agent.losses[-c.rounds:])) if agent.losses else None,
                        'updates':agent.steps,'model_training_wall_s':sum(r['classifier_training_wall_s'] for r in env.rows)})
    return agent,{'wall_s':time.perf_counter()-started,'updates':agent.steps,'episodes':history,'losses':agent.losses,
                  'rollout_scenario_carbon_g':rollout_carbon,'rollout_scenario_energy_wh':rollout_energy}


def metrics(env,training):
    c=env.c
    accuracy,test_loss=evaluate(env.model,env.data.test)
    hit=None; elapsed=carbon=0.
    for r in env.rows:
        elapsed+=r['latency_s']; carbon+=r['carbon_g']
        if hit is None and r['validation_accuracy']>=c.accuracy_target:
            hit={'round':r['round'],'latency_s':elapsed,'carbon_g':carbon}
    inference=sum(r['inference_wall_s'] for r in env.rows)
    # Scenario conversion of measured wall overhead using declared coordinator W
    # and g/kWh. This is an ESTIMATE, never an energy-meter observation.
    overhead_wh=30*(inference+training['wall_s'])/3600
    return {'test_accuracy':accuracy,'test_loss':test_loss,'final_accuracy':accuracy,
            'validation_accuracy':env.accuracy,'validation_loss':env.loss,
            'target_hit':hit is not None,'time_to_target_s':hit['latency_s'] if hit else None,
            'carbon_to_target_g':hit['carbon_g'] if hit else None,'round_to_target':hit['round'] if hit else None,
            'latency_s':env.total_latency,'carbon_g':env.total_carbon,'energy_wh':env.total_wh,
            'accuracy_constraint_violations':sum(r['accuracy_constraint_violation']>0 for r in env.rows),
            'final_accuracy_feasible':env.accuracy>=c.accuracy_target,
            'carbon_budget_violations':sum(r['carbon_budget_violation']>0 for r in env.rows),
            'tail_latency_cvar':cvar([r['latency_s'] for r in env.rows],c.risk_alpha),
            'tail_carbon_cvar':cvar([r['carbon_g'] for r in env.rows],c.risk_alpha),
            'gini':env.rows[-1]['gini'],'policy_training_wall_s':training['wall_s'],'inference_wall_s':inference,
            'estimated_policy_overhead_wh':overhead_wh,'practical_energy_wh':env.total_wh+overhead_wh,
            'practical_carbon_g':env.total_carbon+overhead_wh*.4,
            'practical_latency_s':env.total_latency+inference+training['wall_s'],
            'prediction_interval_coverage':(env.infra.coverage_hits/max(env.infra.coverage_count,1)).tolist(),
            'coverage_client_observations':env.infra.coverage_count,
            'availability_probability_mean':float(env.infra.history.availability_probability().mean()),
            'provenance_note':'Infrastructure modeled; wall overhead measured by perf_counter; overhead Wh derived from assumed 30W coordinator, not measured energy'}
