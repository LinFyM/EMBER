"""One resident source per matched arm through fixed64 and all fixed consumers."""
import argparse
import gc
import torch
from ember.pi05_source_checkpoint import write_json_atomic
from .common import ROOT, ARMS, initialize
from .train import learn
from .readout import materialize, readout, remove_training_adapter
from .evaluation import evaluate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=ARMS,required=True)
    arm=parser.parse_args().arm
    if (ROOT/'evaluation'/arm/'consumer_completion.json').exists():
        raise ValueError('arm already complete; no duplicate execution')
    runtime,spec,identity,affinity=initialize()
    binding=learn(runtime,spec,identity,affinity,arm)
    torch.set_grad_enabled(False)
    bank=materialize(runtime,spec,binding,arm,identity)
    del binding;gc.collect();torch.cuda.empty_cache()
    readout(runtime,spec,arm,bank)
    remove_training_adapter(runtime)
    rows=evaluate(runtime,bank,arm,identity)
    write_json_atomic(ROOT/'evaluation'/arm/'consumer_completion.json',
        dict(complete=True,arm=arm,reading_git=identity,Writer_removed=True,
             rows=len(rows),full=31,B20_queries=320,source_loads=1,closed=True))


if __name__=='__main__':
    main()
