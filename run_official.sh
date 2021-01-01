#!/bin/bash
#nohup time python project_pytorch/train.py -a resnet50 --dist-url 'tcp://127.0.0.1:7999' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 /media/niu/niu_g/data/imagenet/



# TODO
source activate torch16
nohup time python -m torch.distributed.launch --nproc_per_node=4 /media/niu/niu_d/dpj_workspace/mixed_precision_pytorch/project_pytorch/apex_train.py -a resnet50 --opt-level O2 /media/niu/niu_g/data/imagenet/
nohup time python -m torch.distributed.launch --nproc_per_node=4 /media/niu/niu_d/dpj_workspace/mixed_precision_pytorch/project_pytorch/apex_train.py -a resnet50 --opt-level O3 /media/niu/niu_g/data/imagenet/