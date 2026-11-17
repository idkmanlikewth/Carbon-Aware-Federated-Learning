"""Preference-conditioned Double DQN over a bounded catalogue of dynamic selectors.

Actions choose ranking rules, NOT all 2**N subsets. Each rule sees current client
availability, intensity and participation; network size is independent of N.
"""
from collections import deque
import copy
import numpy as np
import torch
from torch import nn

OBS_DIM = 10
N_ACTIONS = 6


class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(OBS_DIM + 4, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, N_ACTIONS))

    def forward(self, state, preference):
        return self.layers(torch.cat((state, preference), dim=-1))


def adapt_preference(base, accuracy, progress):
    """Declared feedback controller: emphasize accuracy when behind a rising target.

    Adaptation occurs between rounds. Preferences are observed by the agent and
    saved in transitions, including next-round preferences for Bellman targets.
    """
    p = np.asarray(base, dtype=np.float32).copy()
    p[0] += max(0., .2 + .6 * progress - accuracy)
    return p / p.sum()


class Agent:
    def __init__(self, c, seed, conditioned=True):
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.c = c
        self.conditioned = conditioned
        self.online = QNetwork()
        self.target = copy.deepcopy(self.online).eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=1e-3)
        self.replay = deque(maxlen=c.replay_capacity)
        self.steps = 0
        self.losses = []

    def pref(self, p):
        return np.asarray(p, dtype=np.float32) if self.conditioned else np.zeros(4, dtype=np.float32)

    def act(self, state, preference, epsilon=0.):
        if self.rng.random() < epsilon:
            return int(self.rng.integers(N_ACTIONS))
        with torch.no_grad():
            return int(self.online(torch.tensor(state), torch.tensor(self.pref(preference))).argmax())

    def learn(self, state, p, action, vector, next_state, next_p, done):
        self.replay.append((state.copy(), self.pref(p), action, float(np.dot(p, vector)), next_state.copy(), self.pref(next_p), done))
        if len(self.replay) < self.c.replay_batch:
            return
        idx = self.rng.choice(len(self.replay), self.c.replay_batch, replace=False)
        s, p, a, r, ns, npref, d = zip(*(self.replay[i] for i in idx))
        s, p, ns, npref = [torch.tensor(np.stack(v), dtype=torch.float32) for v in (s, p, ns, npref)]
        a = torch.tensor(a, dtype=torch.long)
        r, d = [torch.tensor(v, dtype=torch.float32) for v in (r, d)]
        q = self.online(s, p).gather(1, a[:, None]).squeeze(1)
        with torch.no_grad():
            best = self.online(ns, npref).argmax(1)
            target = r + self.c.gamma * (1-d) * self.target(ns, npref).gather(1, best[:, None]).squeeze(1)
        loss = nn.functional.smooth_l1_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 5.)
        self.optimizer.step()
        self.steps += 1
        self.losses.append(float(loss.detach()))
        if self.steps % self.c.target_interval == 0:
            self.target.load_state_dict(self.online.state_dict())
