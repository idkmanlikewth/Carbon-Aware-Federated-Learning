import copy
import time
import numpy as np
import torch
from torch import nn


class Classifier(nn.Module):
    def __init__(self, channels):
        super().__init__()
        if channels == 1:
            self.net = nn.Sequential(nn.Conv2d(1,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                     nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                     nn.Flatten(), nn.Linear(32*7*7,64), nn.ReLU(), nn.Linear(64,10))
        else:
            layers = []
            for cin, cout in ((3,16),(16,32),(32,64)):
                layers += [nn.Conv2d(cin,cout,3,padding=1), nn.GroupNorm(4,cout), nn.ReLU(),
                           nn.Conv2d(cout,cout,3,padding=1), nn.ReLU(), nn.MaxPool2d(2)]
            self.net = nn.Sequential(*layers, nn.Flatten(), nn.Linear(64*4*4,128), nn.ReLU(), nn.Linear(128,10))

    def forward(self,x):
        return self.net(x)


def evaluate(model, pair):
    model.eval(); correct = 0; loss = 0.
    with torch.no_grad():
        for x,y in zip(pair[0].split(256),pair[1].split(256)):
            logits = model(x)
            correct += int((logits.argmax(1)==y).sum())
            loss += float(nn.functional.cross_entropy(logits,y,reduction='sum'))
    return correct/len(pair[1]), loss/len(pair[1])


def train_round(model, data, selected, c, seed):
    started = time.perf_counter()
    accumulator = {k: torch.zeros_like(v) for k,v in model.state_dict().items()}
    total = sum(len(data.clients[i][1]) for i in selected)
    local_losses = {}
    for i in selected:
        local = copy.deepcopy(model).train()
        optimizer = torch.optim.Adam(local.parameters(),lr=c.learning_rate,weight_decay=1e-4) if c.optimizer == 'adam' else torch.optim.SGD(local.parameters(),lr=c.learning_rate,momentum=.9,weight_decay=1e-4)
        x,y = data.clients[i]
        generator = torch.Generator(device=x.device).manual_seed(seed+int(i)*1009)
        losses = []
        for _ in range(c.local_epochs):
            order = torch.randperm(len(y),generator=generator,device=x.device)
            for idx in order.split(c.batch_size):
                bx = x[idx]
                if data.channels == 3:
                    flip = torch.rand(len(idx),generator=generator,device=x.device) < .5
                    bx[flip] = bx[flip].flip(-1)
                optimizer.zero_grad()
                loss = nn.functional.cross_entropy(local(bx),y[idx])
                loss.backward(); nn.utils.clip_grad_norm_(local.parameters(),10.); optimizer.step()
                losses.append(loss.detach())
        local_losses[int(i)] = float(torch.stack(losses).mean())
        for key,value in local.state_dict().items():
            accumulator[key].add_(value,alpha=len(y)/total)
    model.load_state_dict(accumulator)
    return local_losses, time.perf_counter()-started
