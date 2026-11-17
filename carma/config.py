from dataclasses import asdict, dataclass
import math

PROBLEM = 'A Multi-Objective Reinforcement Learning Model for Optimizing the Latency–Carbon Trade-off in Federated Computing.'
BASELINES = ('fedavg', 'random', 'round_robin', 'fastest', 'carbon_greedy', 'weighted', 'dqn', 'pc_ddqn', 'oort_style', 'carma')
ABLATIONS = ('no_uncertainty', 'no_risk', 'no_adaptation', 'no_constraint', 'no_conditioning', 'no_learned_selection')


@dataclass(frozen=True)
class Config:
    dataset: str = 'mnist'
    clients: int = 10
    rounds: int = 20
    local_epochs: int = 2
    batch_size: int = 64
    learning_rate: float = .001
    optimizer: str = 'adam'
    partition: str = 'iid'
    alpha: float = .5
    train_samples: int = 6000
    validation_samples: int = 1000
    test_samples: int = 1000
    participation: float = .5
    preference: tuple[float,float] = (.5,.5)
    seeds: tuple[int, ...] = (11, 22, 33)
    policies: tuple[str, ...] = ('fedavg',)
    episodes: int = 4
    replay_batch: int = 32
    target_interval: int = 20
    gamma: float = .95
    accuracy_target: float = .85
    carbon_budget_g: float = 20.
    time_budget_s: float = 1200.
    risk_alpha: float = .9
    risk_weight: float = .2
    uncertainty: float = .2
    heterogeneity: float = 1.
    availability: float = .95
    dynamic_carbon: bool = True
    scenario_shift: float = 1.
    historical_trace: str | None = None
    observation_history: str | None = None
    profiles: str = 'config.json'
    download: bool = False
    data_dir: str = 'data'
    threads: int = 2
    device: str = 'auto'

    def __post_init__(self):
        if self.dataset not in ('mnist', 'fashion-mnist', 'cifar10', 'synthetic'):
            raise ValueError('Unsupported dataset')
        if self.device not in ('auto','cpu','cuda'):
            raise ValueError('device must be auto, cpu or cuda')
        for k in ('clients', 'rounds', 'local_epochs', 'batch_size', 'train_samples', 'validation_samples', 'test_samples', 'episodes', 'replay_batch', 'target_interval', 'threads'):
            if type(getattr(self,k)) is not int or getattr(self,k) < 1:
                raise ValueError(k+' must be a positive integer')
        if not 2 <= self.clients <= 100 or self.train_samples < self.clients:
            raise ValueError('Require 2–100 clients and at least one sample per client')
        if self.validation_samples < 20 or self.test_samples < 20:
            raise ValueError('Validation and test require at least 20 samples')
        if self.partition not in ('iid', 'dirichlet') or self.optimizer not in ('adam', 'sgd'):
            raise ValueError('Invalid partition or optimizer')
        for k in ('learning_rate', 'alpha', 'carbon_budget_g', 'time_budget_s', 'scenario_shift'):
            if not math.isfinite(getattr(self,k)) or getattr(self,k) <= 0:
                raise ValueError(k+' must be finite and positive')
        for k in ('risk_weight', 'uncertainty', 'heterogeneity'):
            if not math.isfinite(getattr(self,k)) or getattr(self,k) < 0:
                raise ValueError(k+' must be finite and nonnegative')
        if not 0 < self.participation <= 1 or not 0 < self.risk_alpha < 1 or not 0 < self.availability <= 1 or not 0 < self.accuracy_target < 1 or not 0 <= self.gamma < 1:
            raise ValueError('Invalid probability, target or discount')
        if len(self.preference)!=2 or any(not math.isfinite(v) or v<0 for v in self.preference) or not math.isclose(sum(self.preference),1,abs_tol=1e-6):
            raise ValueError('Latency/carbon preferences must sum to one')
        if not self.seeds or len(set(self.seeds)) != len(self.seeds) or any(type(s) is not int or not 0 <= s < 2**31 for s in self.seeds):
            raise ValueError('Require distinct nonnegative integer seeds below 2**31')
        if not self.policies or len(set(self.policies)) != len(self.policies) or set(self.policies)-set(BASELINES+ABLATIONS):
            raise ValueError('Invalid policies')

    def to_dict(self):
        return asdict(self)
