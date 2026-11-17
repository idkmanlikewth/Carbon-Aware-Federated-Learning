"""Validation-only matched controls for architecture, optimizer and standardization.

This is a development diagnostic, not a reproduction of the old paper/demo table.
No final-test evaluation occurs in this matrix.
"""
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
import torch
from research.environment import Classifier as EarlierCNN
from .config import Config
from .data import load
from .learning import Classifier,evaluate
from research.run import write_json


def main():
    root=Path('carma_runs/learning-diagnostics'); root.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    cfg=Config(dataset='mnist',partition='iid',train_samples=6000,rounds=10,local_epochs=2,seeds=(11,22,33),device='cpu')
    cases=[('earlier_sgd',False,'sgd',True),('earlier_adam',False,'adam',True),
           ('stronger_sgd',True,'sgd',True),('stronger_adam',True,'adam',True),
           ('earlier_adam_raw',False,'adam',False),('stronger_adam_raw',True,'adam',False)]
    record={'status':'running','purpose':'Matched validation-only factorial diagnostic; no test accuracy is inspected.',
            'config':cfg.to_dict(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'results':[]}
    started=time.perf_counter()
    for seed in cfg.seeds:
        data=load(cfg,seed)
        for name,strong,opt,standardized in cases:
            torch.manual_seed(seed); model=Classifier(1) if strong else EarlierCNN(1)
            clients=data.clients; validation=data.validation
            if not standardized:
                mean=torch.tensor(data.provenance['normalization_mean']).view(1,-1,1,1)
                std=torch.tensor(data.provenance['normalization_std']).view(1,-1,1,1)
                clients=[(x*std+mean,y) for x,y in clients]
                validation=(validation[0]*std+mean,validation[1])
            history=[]
            for r in range(cfg.rounds):
                accumulator={k:torch.zeros_like(v) for k,v in model.state_dict().items()}
                for i,(x,y) in enumerate(clients):
                    import copy
                    local=copy.deepcopy(model).train()
                    optimizer=torch.optim.Adam(local.parameters(),lr=.001,weight_decay=1e-4) if opt=='adam' else torch.optim.SGD(local.parameters(),lr=.05,weight_decay=1e-4)
                    generator=torch.Generator().manual_seed(seed+r*100003+i*1009)
                    for _ in range(cfg.local_epochs):
                        for idx in torch.randperm(len(y),generator=generator).split(cfg.batch_size):
                            optimizer.zero_grad(); loss=torch.nn.functional.cross_entropy(local(x[idx]),y[idx]); loss.backward()
                            torch.nn.utils.clip_grad_norm_(local.parameters(),10.); optimizer.step()
                    for key,value in local.state_dict().items():accumulator[key].add_(value,alpha=len(y)/cfg.train_samples)
                model.load_state_dict(accumulator)
                accuracy,loss=evaluate(model,validation)
                history.append({'round':r+1,'validation_accuracy':accuracy,'validation_loss':loss})
            record['results'].append({'seed':seed,'case':name,'architecture':'stronger' if strong else 'earlier',
                                      'optimizer':opt,'standardized':standardized,'history':history})
            record['wall_s']=time.perf_counter()-started
            write_json(root/'diagnostics.json',record)
            print(f'diagnostic seed={seed} case={name} validation={accuracy:.4f} loss={loss:.4f}',flush=True)
    record['status']='complete'; write_json(root/'diagnostics.json',record)


if __name__=='__main__':main()
