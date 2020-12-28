nohup time python project_pytorch/train.py -a resnet50 --dist-url 'tcp://127.0.0.1:7999' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 /media/niu/niu_g/data/imagenet/
