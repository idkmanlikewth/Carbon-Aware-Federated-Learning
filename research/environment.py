import copy
import numpy as np
import torch
from torch import nn
from .agent import OBS_DIM


class Classifier(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(channels, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                 nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)),
                                 nn.Flatten(), nn.Linear(256, 10))

    def forward(self, x):
        return self.net(x)


def gini(counts):
    a = np.sort(np.asarray(counts, dtype=float))
    if a.sum() == 0:
        return 0.
    return float(np.dot(2*np.arange(1, len(a)+1)-len(a)-1, a) / (len(a)*a.sum()))


def normalized_reward(gain, latency, carbon, inequality, c):
    return np.array([np.clip(gain/c.accuracy_gain_scale, -1, 1),
                     -np.clip(latency/c.latency_scale_s, 0, 1),
                     -np.clip(carbon/c.carbon_scale_g, 0, 1), -np.clip(inequality, 0, 1)], dtype=np.float32)


class Environment:
    """Real CPU FedAvg with explicitly modeled infrastructure (no physical measurements).

    Power (W) * modeled duration (s) / 3600 = Wh;
    Wh * intensity (g/kWh) / 1000 = g CO2e. Includes idle and coordinator energy.
    Exogenous traces are drawn for all clients before any selection, preserving
    paired comparisons regardless of the policy's action sequence.
    """
    def __init__(self, c, data, seed, static_carbon=False, raw_reward=False):
        self.c, self.data = c, data
        self.raw_reward = raw_reward
        rng = np.random.default_rng(seed)
        n, r = c.clients, c.rounds
        self.speed = rng.uniform(.5, 2., n)
        self.power = rng.uniform(10, 80, n)
        self.idle = rng.uniform(1, 5, n)
        base = rng.uniform(100, 800, n)
        phase = rng.uniform(0, 2*np.pi, n)
        traces = base[None] * (1 + .4*np.sin(np.arange(r+1)[:, None]*.7 + phase))
        self.intensities = np.tile(base, (r+1, 1)) if static_carbon else traces
        self.available = rng.random((r+1, n)) > .15
        # Explicit always-online coordinator-associated virtual client.
        self.available[:, 0] = True
        self.jitter = rng.uniform(.9, 1.1, (r+1, n))
        self.random_rank = rng.random((r+1, n))
        self.counts = np.zeros(n)
        self.round = 0
        torch.manual_seed(seed)
        self.model = Classifier(data.channels)
        self.accuracy = self.evaluate(data.validation)
        self.total_carbon = self.total_latency = self.total_wh = 0.

    def evaluate(self, pair):
        self.model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in zip(pair[0].split(256), pair[1].split(256)):
                correct += int((self.model(x).argmax(1) == y).sum())
        return correct / len(pair[1])

    def durations(self):
        sizes = np.array([len(y) for _, y in self.data.clients])
        return (2 + sizes * self.c.local_epochs * .03) / self.speed * self.jitter[self.round]

    def observation(self):
        active = self.available[self.round]
        ci = self.intensities[self.round]
        return np.asarray([self.round/self.c.rounds, self.accuracy, gini(self.counts), active.mean(),
                           ci[active].mean()/1200, ci[active].min()/1200, ci[active].max()/1200,
                           self.durations()[active].mean()/self.c.latency_scale_s,
                           min(1, self.total_carbon/(self.c.carbon_scale_g*self.c.rounds)),
                           min(1, self.total_latency/(self.c.latency_scale_s*self.c.rounds))], dtype=np.float32)

    def select(self, action):
        if action not in range(6):
            raise ValueError("Unknown selector action")
        ids = np.flatnonzero(self.available[self.round])
        duration = self.durations()
        carbon = duration * self.power * self.intensities[self.round] / 3_600_000
        sizes = np.array([len(y) for _, y in self.data.clients])
        scores = (self.random_rank[self.round], duration, carbon, self.counts,
                  -sizes, duration/duration.max() + carbon/max(carbon.max(), 1e-12) + self.counts/max(1, self.counts.max()))
        k = min(len(ids), max(1, int(np.ceil(self.c.clients*self.c.participation))))
        return ids[np.argsort(scores[action][ids], kind="stable")[:k]]

    def step(self, action):
        if self.round >= self.c.rounds:
            raise RuntimeError("Episode finished")
        selected = self.select(action)
        # Stream weighted states; never retain one model per virtual client.
        aggregate = {k: torch.zeros_like(v) for k, v in self.model.state_dict().items()}
        total = sum(len(self.data.clients[i][1]) for i in selected)
        for i in selected:
            local = copy.deepcopy(self.model).train()
            optimizer = torch.optim.SGD(local.parameters(), lr=self.c.learning_rate)
            x, y = self.data.clients[i]
            for _ in range(self.c.local_epochs):
                for start in range(0, len(y), self.c.batch_size):
                    optimizer.zero_grad()
                    loss = nn.functional.cross_entropy(local(x[start:start+self.c.batch_size]), y[start:start+self.c.batch_size])
                    loss.backward()
                    optimizer.step()
            for k, v in local.state_dict().items():
                aggregate[k].add_(v, alpha=len(y)/total)
        self.model.load_state_dict(aggregate)
        previous = self.accuracy
        self.accuracy = self.evaluate(self.data.validation)
        duration = self.durations()
        latency = float(duration[selected].max() + .45)
        # All nodes remain idle outside their active interval.
        wh = self.idle*latency/3600
        wh[selected] += (self.power[selected]-self.idle[selected])*duration[selected]/3600
        coordinator_wh = 30*latency/3600
        carbon = float(np.dot(wh, self.intensities[self.round])/1000 + coordinator_wh*400/1000)
        energy = float(wh.sum()+coordinator_wh)
        self.counts[selected] += 1
        inequality = gini(self.counts)
        gain = self.accuracy - previous
        vector = normalized_reward(gain, latency, carbon, inequality, self.c)
        if self.raw_reward:
            vector = np.asarray([gain, -latency, -carbon, -inequality], dtype=np.float32)
        self.total_latency += latency
        self.total_carbon += carbon
        self.total_wh += energy
        info = {"round": self.round+1, "selected": selected.tolist(), "available": np.flatnonzero(self.available[self.round]).tolist(),
                "intensity_g_kwh": self.intensities[self.round].tolist(), "validation_accuracy": self.accuracy,
                "latency_s": latency, "carbon_g": carbon, "energy_wh": energy,
                "gini": inequality, "reward_vector": vector.tolist(), "action": action,
                "reward_clipped": bool(abs(gain) > self.c.accuracy_gain_scale or latency > self.c.latency_scale_s or carbon > self.c.carbon_scale_g)}
        self.round += 1
        return self.observation(), vector, self.round == self.c.rounds, info
