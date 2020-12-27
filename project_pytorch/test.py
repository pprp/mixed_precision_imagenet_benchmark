import torch
import torchvision
from torch.utils.data import DataLoader
import models
from loader import get_val_dataloader
import argparse
from utils import accuracy
import torch.nn as nn


def get_parser():
    parser = argparse.ArgumentParser('testing')
    parser.add_argument('--rootdir', type=str,
                        default="E:/imagenet_data", help='root dir of imagenet')
    parser.add_argument('-b', '--batch_size', type=int,
                        default=64, help='batch size')
    parser.add_argument('--arch', type='str',
                        default='resnet50', help='CNN architecture')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU id to use')
    parser.add_argument('--distributed', type=bool,
                        default=False, help='multi gpu inference')
    parser.add_argument('-j', '--workers', type=int,
                        default=4, help='num of workers')
    parser.add_argument('--load_path', type=str,
                        default='checkpoints', help='weights path')
    args = parser.parse_args()
    return args


def main():
    args = get_parser()

    val_loader = get_val_dataloader(
        args.rootdir, args.distributed, args.workers, args.batch_size)

    # model new
    model = models.resnet50()

    # checkpoint load
    if args.gpu is None:
        # cpu
        checkpoint = torch.load(args.load_path)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        # gpu
        torch.cuda.set_device(args.gpu)
        checkpoint = torch.load(
            args.load_path, map_location='cuda:{}'.format(args.gpu))
        model.load_state_dict(checkpoint['state_dict'])
        model = nn.DataParallel(model)  # 数据并行
        model = model.cuda(args.gpu)

    print(model)
    model.eval()

    with torch.no_grad():
        total = len(val_loader)
        correct_1 = 0
        correct_5 = 0

        for i, (images, targets) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                targets = images.cuda(args.gpu, non_blocking=True)

            output = model(images)

            _, pred = output.topk(5, 1, largest=True, sorted=True)

            targets = targets.view(targets.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()

            # top 5
            correct_5 += correct[:, :5].sum()

            # top 1
            correct_1 += correct_1[:, :1].sum()

        print("TOP 1:", correct_1/total)
        print("TOP 5", correct_5/total)    
        print("Parameter numbers: {}".format(sum(p.numel() for p in model.parameters())))


        # acc1, acc5 = accuracy(output, target, topk=(1, 5))
