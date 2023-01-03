import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer
import torch.distributed as dist
import os
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
def cleanup():
    dist.destroy_process_group()

def main(rank,world_size):
    global model
    setup(rank,world_size)
    if checkpoint.ok:
        loader = data.Data(rank, args)
        model = model.Model(args, checkpoint,rank)
        print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
        _loss = loss.Loss(args, checkpoint, rank) if not args.test_only else None
        t = Trainer(args, loader, model, _loss, checkpoint,rank)
        while not t.terminate():
            t.train()
            if rank==0:
                t.test()

        checkpoint.done()
        cleanup()

if __name__ == '__main__':
    world_size = args.n_GPUs
    mp.spawn(main,args=(world_size,),nprocs=world_size)
