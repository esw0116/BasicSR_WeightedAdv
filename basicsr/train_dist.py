import os
import torch
import argparse

parser = argparse.ArgumentParser(
    description='run multiple gpu setting')
parser.add_argument('--conf', default='',
                    type=str)
args = parser.parse_args()
conf = args.conf

n = torch.cuda.device_count()
print(n)

command = 'python -m torch.distributed.launch --nproc_per_node=%d train.py %s  --launcher pytorch' % (n, conf)
print (command)
os.system(command)