"""Benchmark loading with no silent synthetic fallback; train/validation/test separation."""
from dataclasses import dataclass
import hashlib
import numpy as np
import torch
from torchvision import datasets
from fl_pipeline.data import partition_dirichlet


@dataclass
class Data:
    clients: list
    validation: tuple
    test: tuple
    provenance: dict
    channels: int


def load_data(c, seed):
    rng = np.random.default_rng(seed)
    if c.dataset == "synthetic":
        # Explicit smoke fixture, never presented as a benchmark.
        templates = rng.uniform(0, 1, (10, 1, 28, 28)).astype("float32")
        def fixture(n):
            y = rng.integers(0, 10, n)
            x = np.clip(templates[y] + rng.normal(0, .15, (n, 1, 28, 28)), 0, 1).astype("float32")
            return x, y
        x, y = fixture(c.train_samples)
        xe, ye = fixture(c.eval_samples)
        source = "EXPLICIT_SYNTHETIC_SMOKE_FIXTURE"
    else:
        cls = {"mnist": datasets.MNIST, "fashion-mnist": datasets.FashionMNIST, "cifar10": datasets.CIFAR10}[c.dataset]
        try:
            tr = cls(c.data_dir, train=True, download=c.download)
            te = cls(c.data_dir, train=False, download=c.download)
        except Exception as exc:
            raise RuntimeError(f"Cannot load {c.dataset}; use --download or populate {c.data_dir}. No synthetic substitution.") from exc
        if c.train_samples > len(tr) or c.eval_samples > len(te):
            raise ValueError("Requested subset exceeds benchmark size")
        def subset(ds, n):
            idx = rng.permutation(len(ds))[:n]
            a = np.asarray(ds.data)[idx]
            a = a.transpose(0, 3, 1, 2) if a.ndim == 4 else a[:, None]
            return a.astype("float32") / 255., np.asarray(ds.targets, dtype="int64")[idx]
        x, y = subset(tr, c.train_samples)
        xe, ye = subset(te, c.eval_samples)
        source = f"torchvision.datasets.{cls.__name__}:official-train-and-test"
    # Validation drives policy reward. Final test is untouched during RL training.
    split = len(ye) // 2
    parts = partition_dirichlet(x, y, c.clients, c.alpha, min_samples_per_client=1, seed=seed + 1)
    def pair(a, b):
        return torch.from_numpy(a), torch.from_numpy(b.astype("int64"))
    digest = hashlib.sha256(x.tobytes() + y.tobytes() + xe.tobytes() + ye.tobytes()).hexdigest()
    return Data([pair(a, b) for a, b in parts], pair(xe[:split], ye[:split]), pair(xe[split:], ye[split:]),
                {"source": source, "subset_sha256": digest, "train_samples": len(y), "validation_samples": split,
                 "test_samples": len(ye)-split, "partition_sizes": [len(b) for _, b in parts]}, x.shape[1])
