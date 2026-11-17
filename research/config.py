from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class Config:
    dataset: str = "mnist"
    clients: int = 10
    rounds: int = 10
    episodes: int = 10
    train_samples: int = 2000
    eval_samples: int = 500
    local_epochs: int = 1
    batch_size: int = 64
    learning_rate: float = 0.05
    alpha: float = 0.5
    participation: float = 0.2
    seeds: tuple[int, ...] = (11, 22, 33)
    preferences: tuple[tuple[float, ...], ...] = (
        (0.7, 0.1, 0.1, 0.1), (0.25, 0.25, 0.25, 0.25), (0.1, 0.1, 0.7, 0.1))
    policies: tuple[str, ...] = ("adaptive", "fixed", "unconditioned", "static_carbon", "raw_reward", "random", "carbon", "fair")
    download: bool = False
    data_dir: str = "data"
    # Fixed, declared engineering bounds; never fitted on evaluation results.
    latency_scale_s: float = 120.0
    carbon_scale_g: float = 5.0
    accuracy_gain_scale: float = 0.1
    gamma: float = 0.95
    replay_capacity: int = 10000
    replay_batch: int = 32
    target_interval: int = 25

    def __post_init__(self):
        if self.dataset not in ("mnist", "fashion-mnist", "cifar10", "synthetic"):
            raise ValueError("Unsupported dataset")
        if not 2 <= self.clients <= 100:
            raise ValueError("clients must be between 2 and 100")
        for name in ("rounds", "episodes", "train_samples", "eval_samples", "local_epochs", "batch_size", "replay_capacity", "replay_batch", "target_interval"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.train_samples < self.clients or self.eval_samples < 20:
            raise ValueError("Insufficient samples for clients or validation/test split")
        for name in ("learning_rate", "alpha", "latency_scale_s", "carbon_scale_g", "accuracy_gain_scale"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 < self.participation <= 1 or not 0 <= self.gamma < 1:
            raise ValueError("Invalid participation or discount")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds) or any(type(s) is not int or not 0 <= s < 2**31 for s in self.seeds):
            raise ValueError("Seeds must be distinct nonnegative integers below 2**31")
        if not self.preferences:
            raise ValueError("At least one preference required")
        for p in self.preferences:
            if len(p) != 4 or any(not math.isfinite(v) or v < 0 for v in p) or not math.isclose(sum(p), 1, abs_tol=1e-6):
                raise ValueError("Preferences must be four nonnegative weights summing to one")
        allowed = {"adaptive", "fixed", "unconditioned", "static_carbon", "raw_reward", "random", "carbon", "fair"}
        if not self.policies or len(set(self.policies)) != len(self.policies) or set(self.policies) - allowed:
            raise ValueError("Unknown or duplicated policy")

    def to_dict(self):
        return asdict(self)
