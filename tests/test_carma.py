import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
import torch
from carma.config import Config
from carma.data import load
from carma.environment import Environment
from carma.infrastructure import CostHistory,cvar
from carma.policy import Agent,ClientQ
from carma.experiment import episode,train,metrics
from carma.analysis import paired_permutation,analyze
from carma.run import run
from carma.verify import verify
from carma.export import export


@pytest.fixture
def cfg():
    torch.set_num_threads(1)
    return Config(dataset='synthetic',clients=4,rounds=3,episodes=2,train_samples=80,validation_samples=40,test_samples=40,
                  policies=('fedavg','random','carma'),seeds=(1,2),replay_batch=2,threads=1,device='cpu')


def test_equivariance_and_action_mask(cfg):
    torch.manual_seed(1); net=ClientQ()
    x=torch.randn(7,16); g=torch.randn(10); p=torch.randn(3); perm=torch.tensor([5,1,2,0,6,4,3])
    torch.testing.assert_close(net(x[perm],g,p),net(x,g,p)[perm])
    agent=Agent(cfg,1)
    state=(x.numpy(),g.numpy(),p.numpy(),np.array([False,False,True,False,False,False,False]))
    assert agent.act(state)==2
    assert agent.act(state,epsilon=1)==2


def test_cvar_fractional_mass():
    assert cvar([1,2,3,4],.5)==3.5
    assert cvar([1,2,3,4],.9)==4
    assert cvar([1,2,3,4],.625)==pytest.approx((4+.5*3)/1.5)
    assert paired_permutation([1,1,1])==pytest.approx(.25)


def test_calibration_uses_preceding_samples():
    rng=np.random.default_rng(1); ratios=rng.lognormal(0,.1,(4,60,4))
    h=CostHistory(ratios); nominal=np.ones((4,4))
    mean,lo,hi=h.interval(nominal)
    assert (lo<=mean).all() and (mean<=hi).all()
    before=h.means.copy(); h.observe(nominal,np.ones((4,4))*1.5,[0],np.ones(4))
    assert not np.array_equal(before[0],h.means[0])
    np.testing.assert_array_equal(before[1:],h.means[1:])


def test_strict_test_isolation_and_pairing(cfg):
    data=load(cfg,1)
    altered=replace(data,test=(data.test[0],(data.test[1]+1)%10))
    a=episode(cfg,data,101,'random'); b=episode(cfg,altered,101,'random')
    assert [r['validation_accuracy'] for r in a.rows]==[r['validation_accuracy'] for r in b.rows]
    assert [r['carbon_g'] for r in a.rows]==[r['carbon_g'] for r in b.rows]
    for r in a.rows:assert set(r['selected'])<=set(r['available'])
    assert data.provenance['source']=='EXPLICIT_SYNTHETIC_FIXTURE'


def test_100_client_features_and_linear_parameter_count(cfg):
    c=replace(cfg,clients=100,train_samples=200)
    env=Environment(c,load(c,1),1)
    x,g,p,mask=env.state()
    assert x.shape==(100,16) and mask.any()
    assert torch.isfinite(ClientQ()(torch.tensor(x),torch.tensor(g),torch.tensor(p))).all()
    base=ClientQ(); assert sum(p.numel() for p in base.parameters())<100000


def test_actual_learning_and_dqn(cfg):
    data=load(cfg,1)
    agent,training=train(cfg,data,1,'carma','cpu')
    assert training['updates']>0 and np.isfinite(training['losses']).all()
    assert training['rollout_scenario_carbon_g']>0
    env=episode(cfg,data,100001,'carma',agent)
    assert env.accuracy>.1
    assert metrics(env,training)['policy_training_wall_s']>0


def test_ablations_are_independent_switches(cfg):
    data=load(cfg,1)
    risk=Environment(cfg,data,1,'carma'); off=Environment(cfg,data,1,'no_risk')
    assert risk.constrained and off.constrained
    no_constraint=Environment(cfg,data,1,'no_constraint'); assert not no_constraint.constrained
    fixed=Environment(cfg,data,1,'no_adaptation')
    np.testing.assert_array_equal(fixed.preference(),[.5,.5,1.])
    no_unc=Environment(cfg,data,1,'no_uncertainty')
    x,*_=no_unc.state(); np.testing.assert_array_equal(x[:,5:7],np.zeros((4,2)))
    np.testing.assert_allclose(x[:,12:14],risk.state()[0][:,12:14])
    np.testing.assert_array_equal(off.state()[0][:,12:14],np.zeros((4,2)))


def test_no_constraint_removes_only_the_penalty(cfg):
    c=replace(cfg,rounds=1,accuracy_target=.99)
    env=Environment(c,load(c,1),1,'no_constraint')
    p=env.preference(); selected=[0]
    reward,_,row=env.step(selected)
    tail_d,tail_c=env.infra.tail_estimates(0)
    expected=-p[0]*row['latency_s']/120-p[1]*row['carbon_g']/5-c.risk_weight*(p[0]*tail_d[selected].max()/120+p[1]*tail_c[selected].sum()/5)
    assert reward==pytest.approx(expected)


def test_feasibility_blocks_artificial_carbon_savings(cfg):
    from carma.experiment import metrics
    rows=[]
    for seed in cfg.seeds:
        env=episode(cfg,load(cfg,seed),seed+100,'random')
        row=metrics(env,{'wall_s':0})
        row.update(seed=seed,policy='random',inference_wall_s=0.,practical_carbon_with_training_g=row['carbon_g'],final_accuracy_feasible=False)
        rows.append(row)
    analysis=analyze(rows,replace(cfg,policies=('random',)))
    assert analysis['summary']['random']['feasible_hypervolume']['mean']==0


def test_runner_and_source_pinned_resume(cfg,tmp_path):
    result=run(cfg,tmp_path/'study')
    assert result['status']=='complete'
    assert len(result['training'])==6
    assert (tmp_path/'study'/'checkpoint.json').exists()
    assert run(cfg,tmp_path/'study',resume=True)['status']=='complete'
    with pytest.raises(ValueError,match='identical'):
        run(replace(cfg,rounds=4),tmp_path/'study',resume=True)
    assert result['datasets']['1']['subset_sha256']!=result['datasets']['2']['subset_sha256']
    assert verify(tmp_path/'study',require_checkpoints=True)['status']=='verified'
    assert export(tmp_path/'study',tmp_path/'published')['status']=='verified'
    with pytest.raises(ValueError,match='checksum'):
        verify(tmp_path/'published',require_checkpoints=True)
    path=tmp_path/'published'/'results.json'
    path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError,match='checksum'):
        verify(tmp_path/'published')
