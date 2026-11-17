from dataclasses import dataclass
import hashlib
import numpy as np
import torch
from torchvision import datasets
from fl_pipeline.data import partition_iid, partition_dirichlet


@dataclass
class Data:
    clients: list
    validation: tuple
    test: tuple
    provenance: dict
    channels: int

    def to(self,device):
        self.clients=[(x.to(device),y.to(device)) for x,y in self.clients]
        self.validation=tuple(v.to(device) for v in self.validation)
        self.test=tuple(v.to(device) for v in self.test)
        return self


def load(c, seed):
    rng = np.random.default_rng(seed)
    if c.dataset == 'synthetic':
        # Explicit high-contrast learnability fixture, never a benchmark substitute.
        def fixture(n):
            y = np.arange(n)%10; rng.shuffle(y)
            x = rng.normal(0, .05, (n, 1, 28, 28)).astype('float32')
            for i, label in enumerate(y):
                x[i, 0, (label//5)*14:(label//5)*14+10, (label%5)*5:(label%5)*5+4] += 1
            return x, y.astype('int64')
        pool, labels = fixture(c.train_samples+c.validation_samples)
        test_x, test_y = fixture(c.test_samples)
        source = 'EXPLICIT_SYNTHETIC_FIXTURE'
    else:
        cls = {'mnist': datasets.MNIST, 'fashion-mnist': datasets.FashionMNIST, 'cifar10': datasets.CIFAR10}[c.dataset]
        train = cls(c.data_dir, train=True, download=c.download)
        test = cls(c.data_dir, train=False, download=c.download)
        def images(a):
            a = np.asarray(a)
            return (a.transpose(0,3,1,2) if a.ndim == 4 else a[:,None]).astype('float32')/255
        if c.train_samples+c.validation_samples > len(train) or c.test_samples > len(test):
            raise ValueError('Requested subset exceeds official dataset size')
        train_ids = rng.permutation(len(train))[:c.train_samples+c.validation_samples]
        test_ids = rng.permutation(len(test))[:c.test_samples]
        pool, labels = images(np.asarray(train.data)[train_ids]), np.asarray(train.targets, dtype='int64')[train_ids]
        test_x, test_y = images(np.asarray(test.data)[test_ids]), np.asarray(test.targets, dtype='int64')[test_ids]
        source = f'torchvision.datasets.{cls.__name__}'
    x, y = pool[:c.train_samples], labels[:c.train_samples]
    vx, vy = pool[c.train_samples:], labels[c.train_samples:]
    mean = x.mean(axis=(0,2,3), keepdims=True)
    std = np.maximum(x.std(axis=(0,2,3), keepdims=True), .01)
    x, vx, test_x = [(a-mean)/std for a in (x, vx, test_x)]
    # Partition unique indices, then retain tensors. Validation never comes from test.
    ids = np.arange(len(y))
    parts = partition_iid(ids, y, c.clients, seed+1) if c.partition == 'iid' else partition_dirichlet(ids, y, c.clients, c.alpha, 1, seed+1)
    def pair(a,b):
        return torch.from_numpy(np.ascontiguousarray(a)), torch.from_numpy(np.ascontiguousarray(b))
    digest = hashlib.sha256(x.tobytes()+y.tobytes()+vx.tobytes()+vy.tobytes()+test_x.tobytes()+test_y.tobytes()).hexdigest()
    return Data([pair(x[idx], y[idx]) for idx,_ in parts], pair(vx,vy), pair(test_x,test_y),
                {'source': source, 'subset_sha256': digest, 'split': 'disjoint official-training subsets for train/validation; official test used only for final evaluation',
                 'train_indices_sha256': hashlib.sha256(ids.tobytes() if c.dataset == 'synthetic' else train_ids.tobytes()).hexdigest(),
                 'normalization_mean': mean.flatten().tolist(), 'normalization_std': std.flatten().tolist(),
                 'partition_sizes': [len(idx) for idx,_ in parts]}, x.shape[1])
