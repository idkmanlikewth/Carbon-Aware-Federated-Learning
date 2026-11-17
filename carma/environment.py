import numpy as np
import torch
from .infrastructure import Infrastructure
from .learning import Classifier,evaluate,train_round
from research.environment import gini


class Environment:
    def __init__(self,c,data,seed,policy='carma',evaluation=False,device='cpu'):
        self.c,self.data,self.policy=c,data,policy
        torch.manual_seed(seed); self.model=Classifier(data.channels).to(device)
        self.infra=Infrastructure(c,data,seed,evaluation,model_bytes=sum(p.numel()*p.element_size() for p in self.model.parameters()))
        self.round=0; self.counts=np.zeros(c.clients); self.picked=[]
        self.accuracy,self.loss=evaluate(self.model,data.validation)
        self.initial_loss=self.loss; self.rate=0.; self.previous_loss=self.loss
        self.total_carbon=self.total_latency=self.total_wh=0.; self.dual=1.
        self.rows=[]; self.seed=seed
        self._state_cache=None
        self.entropy=np.array([float(-(p[p>0]*np.log(p[p>0])).sum()/np.log(10)) for _,y in data.clients for p in [np.bincount(y.detach().cpu().numpy(),minlength=10)/len(y)]])

    @property
    def constrained(self):return self.policy not in ('no_constraint','dqn','pc_ddqn')

    def preference(self):
        if self.policy in ('no_adaptation','pc_ddqn','dqn'):return np.array([*self.c.preference,1.],dtype='float32')
        progress=self.round/self.c.rounds
        gap=max(0,self.c.accuracy_target-self.accuracy)
        urgency=.25+2*gap+max(0,self.loss/max(self.initial_loss,.001)-.2)*(1-progress)+max(0,-self.rate)
        carbon_pressure=self.c.preference[1]+max(0,self.total_carbon/self.c.carbon_budget_g-progress)
        time_pressure=self.c.preference[0]+max(0,self.total_latency/self.c.time_budget_s-progress)
        return np.array([time_pressure/(time_pressure+carbon_pressure),carbon_pressure/(time_pressure+carbon_pressure),urgency],dtype='float32')

    def state(self):
        if self._state_cache is not None:
            x,g,p,available,k=self._state_cache
            x=x.copy(); g=g.copy(); x[:,15]=0; x[self.picked,15]=1
            g[7]=(k-len(self.picked))/self.c.clients
            return x,g,p.copy(),available & (x[:,15]==0)
        nominal=self.infra.nominal(self.round)
        uncertainty=self.policy!='no_uncertainty'
        mean,low,high=self.infra.history.interval(nominal) if uncertainty else (nominal,nominal,nominal)
        duration,wh,ci,bandwidth=mean.T
        if self.policy=='no_risk':tail_d,tail_c=np.zeros(self.c.clients),np.zeros(self.c.clients)
        else:tail_d,tail_c=self.infra.tail_estimates(self.round)
        n=self.c.clients
        sizes=np.array([len(y) for _,y in self.data.clients])
        picked=np.zeros(n); picked[self.picked]=1
        current_available=self.infra.available[self.round]
        availability=self.infra.history.available_success/self.infra.history.available_trials if uncertainty else np.full(n,self.c.availability)
        x=np.stack([sizes/sizes.sum(),duration/120,wh/5,ci/1000,bandwidth/100,
                    (high[:,0]-low[:,0])/120,(high[:,2]-low[:,2])/1000,availability,
                    self.counts/max(1,self.round), (self.round-self.infra.last_seen)/self.c.rounds,
                    self.infra.last_loss/3,self.entropy,tail_d/120,tail_c/5,
                    current_available.astype(float),picked],axis=1).astype('float32')
        k=min(int(current_available.sum()),max(1,int(np.ceil(n*self.c.participation))))
        global_state=np.array([self.round/self.c.rounds,self.accuracy,self.loss/3,self.rate,
                               self.total_carbon/self.c.carbon_budget_g,self.total_latency/self.c.time_budget_s,
                               self.dual/10,(k-len(self.picked))/n,
                               1-self.total_carbon/self.c.carbon_budget_g,1-self.total_latency/self.c.time_budget_s],dtype='float32')
        self._state_cache=(x.copy(),global_state.copy(),self.preference(),current_available.copy(),k)
        return x,global_state,self.preference(),current_available & (picked==0)

    def choose_count(self,policy):
        available=int(self.infra.available[self.round].sum())
        return available if policy=='fedavg' else min(available,max(1,int(np.ceil(self.c.clients*self.c.participation))))

    def step(self,selected,inference_s=0.):
        if self.round>=self.c.rounds:raise RuntimeError('Episode complete')
        selected=np.asarray(selected,dtype=int)
        if not len(selected) or len(set(selected))!=len(selected) or (selected<0).any() or (selected>=self.c.clients).any() or not self.infra.available[self.round,selected].all():
            raise ValueError('Selection must contain distinct available clients')
        previous=self.accuracy
        p=self.preference()
        losses,train_wall=train_round(self.model,self.data,selected,self.c,self.seed+self.round*100003)
        self.accuracy,self.loss=evaluate(self.model,self.data.validation)
        self.rate=.7*self.rate+.3*(self.previous_loss-self.loss)/max(self.previous_loss,.001)
        self.previous_loss=self.loss
        cost=self.infra.cost(self.round,selected,0.)
        for i,v in losses.items():self.infra.last_loss[i]=v; self.infra.last_seen[i]=self.round+1
        self.counts[selected]+=1
        self.total_carbon+=cost['carbon_g']; self.total_latency+=cost['latency_s']; self.total_wh+=cost['energy_wh']
        target=self.c.accuracy_target*min(1,(self.round+1)/max(1,.6*self.c.rounds))
        violation=max(0,target-self.accuracy)
        reward=-p[0]*cost['latency_s']/120-p[1]*cost['carbon_g']/5
        if self.constrained:
            reward-=self.dual*p[2]*violation
            self.dual=float(np.clip(self.dual+.5*(target-self.accuracy),0,20))
        elif self.policy in ('dqn','pc_ddqn'):
            reward+=float(np.clip((self.accuracy-previous)/.1,-1,1))
        if self.policy not in ('no_risk','dqn','pc_ddqn'):
            tail_d,tail_c=self.infra.tail_estimates(self.round)
            reward-=self.c.risk_weight*(p[0]*float(tail_d[selected].max())/120+p[1]*float(tail_c[selected].sum())/5)
        self.round+=1; self.picked=[]
        self._state_cache=None
        reward=float(reward)
        row={'round':self.round,'selected':selected.tolist(),'available':np.flatnonzero(self.infra.available[self.round-1]).tolist(),
             'validation_accuracy':self.accuracy,'validation_loss':self.loss,'loss_improvement_ema':self.rate,
             'preference':p.tolist(),'lagrange_multiplier':self.dual,'accuracy_constraint_target':target,
             'accuracy_constraint_violation':violation,'carbon_budget_violation':max(0,self.total_carbon-self.c.carbon_budget_g),
             'inference_wall_s':inference_s,'classifier_training_wall_s':train_wall,'reward':reward,'gini':gini(self.counts),
             'intensity_g_kwh':self.infra.nominal(self.round-1)[:,2].tolist(),**cost}
        self.rows.append(row)
        return float(reward),self.round==self.c.rounds,row
