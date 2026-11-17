"""Convergence gate: train conventional all-client FedAvg before scheduler studies."""
import argparse
import json
from pathlib import Path
import time
import torch
from .config import Config
from .data import load
from .learning import Classifier,evaluate,train_round


def run(c,output):
    output = Path(output); output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(c.threads); torch.use_deterministic_algorithms(True)
    started=time.perf_counter(); results=[]
    for seed in c.seeds:
        data=load(c,seed); torch.manual_seed(seed); model=Classifier(data.channels)
        accuracy,loss=evaluate(model,data.validation)
        history=[{'round':0,'validation_accuracy':accuracy,'validation_loss':loss}]
        for r in range(c.rounds):
            _,wall=train_round(model,data,list(range(c.clients)),c,seed*1000+r)
            accuracy,loss=evaluate(model,data.validation)
            history.append({'round':r+1,'validation_accuracy':accuracy,'validation_loss':loss,'training_wall_s':wall})
            print(f'{c.dataset} seed={seed} round={r+1} validation={accuracy:.4f} loss={loss:.4f}',flush=True)
        test_acc,test_loss=evaluate(model,data.test)
        results.append({'seed':seed,'test_accuracy':test_acc,'test_loss':test_loss,'validation_history':history,'data':data.provenance})
        torch.save(model.state_dict(),output/f'classifier_{seed}.pt')
        record={'status':'complete' if len(results)==len(c.seeds) else 'running','config':c.to_dict(),'results':results,'wall_s':time.perf_counter()-started,
                'purpose':'All-client conventional FedAvg convergence diagnostic, not a scheduler comparison or infrastructure measurement'}
        (output/'diagnostic.json').write_text(json.dumps(record,indent=2,allow_nan=False),encoding='utf-8')
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',default='mnist'); parser.add_argument('--rounds',type=int,default=10)
    parser.add_argument('--train-samples',type=int,default=6000); parser.add_argument('--clients',type=int,default=10)
    parser.add_argument('--local-epochs',type=int,default=2); parser.add_argument('--learning-rate',type=float,default=.001)
    parser.add_argument('--optimizer',default='adam'); parser.add_argument('--partition',default='iid'); parser.add_argument('--alpha',type=float,default=.5)
    parser.add_argument('--seeds',type=int,nargs='+',default=[11]); parser.add_argument('--output',required=True)
    args=vars(parser.parse_args()); output=args.pop('output'); args['seeds']=tuple(args['seeds'])
    run(Config(**args),output)


if __name__=='__main__':main()
