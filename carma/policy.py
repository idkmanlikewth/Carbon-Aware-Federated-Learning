"""Permutation-equivariant client-value network and sequential masked Double DQN.

Each pick is a genuine MDP transition. Intermediate picks receive zero reward;
the final cohort pick receives the observed round reward. No independent-client
additivity assumption or exponential enumeration of client subsets is used.
"""
from collections import deque
import copy
import numpy as np
import torch
from torch import nn

CLIENT_DIM=16
GLOBAL_DIM=10


class ClientQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding=nn.Sequential(nn.Linear(CLIENT_DIM,32),nn.ReLU(),nn.Linear(32,32),nn.ReLU())
        self.head=nn.Sequential(nn.Linear(32+32+GLOBAL_DIM+3,64),nn.ReLU(),nn.Linear(64,1))

    def forward(self,x,g,p):
        single=x.ndim==2
        if single:x,g,p=x[None],g[None],p[None]
        embed=self.embedding(x)
        context=embed.mean(1,keepdim=True).expand_as(embed)
        shared=torch.cat((g,p),-1)[:,None].expand(-1,x.shape[1],-1)
        out=self.head(torch.cat((embed,context,shared),-1)).squeeze(-1)
        return out[0] if single else out


class Agent:
    def __init__(self,c,seed,conditioned=True,double=True):
        torch.manual_seed(seed)
        self.c=c; self.conditioned=conditioned; self.double=double
        self.rng=np.random.default_rng(seed)
        self.online=ClientQ(); self.target=copy.deepcopy(self.online).eval()
        self.optimizer=torch.optim.Adam(self.online.parameters(),lr=1e-3)
        self.replay=deque(maxlen=20000); self.steps=0; self.losses=[]

    def prepared(self,state):
        x,g,p,mask=state
        return (np.array(x,dtype='float32',copy=True),np.array(g,dtype='float32',copy=True),
                np.array(p,dtype='float32',copy=True) if self.conditioned else np.zeros(3,dtype='float32'),np.array(mask,dtype=bool,copy=True))

    def act(self,state,epsilon=0):
        x,g,p,mask=self.prepared(state)
        ids=np.flatnonzero(mask)
        if not len(ids):raise ValueError('No available action')
        if self.rng.random()<epsilon:return int(self.rng.choice(ids))
        with torch.no_grad():
            q=self.online(*[torch.from_numpy(a) for a in (x,g,p)])
            return int(q.masked_fill(~torch.from_numpy(mask),-torch.inf).argmax())

    def learn(self,state,action,reward,next_state,done):
        self.replay.append((self.prepared(state),int(action),float(reward),self.prepared(next_state),bool(done)))
        if len(self.replay)<self.c.replay_batch:return
        idx=self.rng.choice(len(self.replay),self.c.replay_batch,replace=False)
        transitions=[self.replay[i] for i in idx]
        s,a,r,ns,d=zip(*transitions)
        def stack(states):return [torch.from_numpy(np.stack(v)) for v in zip(*states)]
        x,g,p,mask=stack(s); nx,ng,npref,nmask=stack(ns)
        a=torch.tensor(a,dtype=torch.long); r=torch.tensor(r,dtype=torch.float32); d=torch.tensor(d,dtype=torch.bool)
        predicted=self.online(x,g,p).gather(1,a[:,None]).squeeze(1)
        with torch.no_grad():
            target_values=self.target(nx,ng,npref).masked_fill(~nmask,-torch.inf)
            if self.double:
                best=self.online(nx,ng,npref).masked_fill(~nmask,-torch.inf).argmax(1)
                bootstrap=target_values.gather(1,best[:,None]).squeeze(1)
            else:bootstrap=target_values.max(1).values
            # torch.where avoids 0 * -inf for terminal/no-action transitions.
            bootstrap=torch.where(d,torch.zeros_like(bootstrap),bootstrap)
            target=r+self.c.gamma*bootstrap
        loss=nn.functional.smooth_l1_loss(predicted,target)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite DQN loss')
        self.optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.online.parameters(),5.); self.optimizer.step()
        self.steps+=1; self.losses.append(float(loss.detach()))
        if self.steps%self.c.target_interval==0:self.target.load_state_dict(self.online.state_dict())
