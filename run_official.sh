nohup time python project/pytorch_official_imagenet.py -a resnet50 --dist-url 'tcp://127.0.0.1:7999' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 /media/niu/niu_d/data/imagenet/
