# DataLoader
# Author: pprp
# Date: 2020-11-28

import argparse
import datetime
import os
import time
from argparse import Namespace

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.profiler import AdvancedProfiler
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.mnist import MNIST
from torchvision.models import resnet18, resnet50
from torchvision.models.resnet import Bottleneck, BasicBlock
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.WarmUp import WarmUpLR
from utils.LabelSmoothing import LSR
from mix_pil_dataloader import get_mix_train_dataloader, get_mix_val_dataloader
# from mix_dataloader import get_mix_train_dataloader, get_mix_val_dataloader



class MixClassifier(pl.LightningModule):
    def __init__(
        self,
        pretrained: bool,
        learning_rate: float,
        momentum: float,
        weight_decay: float,
        root_path: str,
        batch_size: int,
        workers: int,
        **kwargs,
    ):
        super(MixClassifier, self).__init__()

        # args
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.root_path = root_path
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.workers = workers
        self.zero_init_residual = True

        # model
        self.resnet50 = resnet50(pretrained=pretrained)
        self.lsr_loss = LSR()

        # Built-in API for metrics
        self.train_accuracy = pl.metrics.Accuracy()
        self.val_accuracy = pl.metrics.Accuracy()
        self.test_accuracy = pl.metrics.Accuracy()

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        return self.resnet50(x)

    @staticmethod
    def custom_accuracy(output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""

        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0,keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def configure_optimizers(self):
        # 修改优化器
        optimizer = torch.optim.SGD(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda epoch: 0.1 ** (epoch // 30)
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # 每一个循环内部执行
        x_image, y_true = batch

        # feed the model and catch the prediction
        y_pred = self.forward(x_image)

        # loss calculation
        # loss_train = F.cross_entropy(y_pred, y_true)
        # label smoothing
        loss_train = self.lsr_loss(y_pred, y_true)

        # train accuracy calculation
        acc1, acc5 = self.custom_accuracy(y_pred, y_true, topk=(1, 5))

        # Save metrics for current batch
        self.log("train_loss", loss_train, on_step=True,
                 on_epoch=True, logger=True)
        self.log("train_acc1", acc1, on_step=True,
                 on_epoch=True, logger=True, prog_bar=True)
        self.log("train_acc5", acc5, on_step=True,
                 on_epoch=True, logger=True, prog_bar=True)

        # TODO 这种返回猜测应该是会输出到屏幕的内容，所以key的值可自定义
        return loss_train

    def validation_step(self, batch, batch_idx):
        # 一个epoch结束以后，进行验证集测试
        x_image, y_true = batch
        y_pred = self.forward(x_image)

        # compute loss
        # loss_valid = F.cross_entropy(y_pred, y_true)
        # label smoothing
        loss_valid = self.lsr_loss(y_pred, y_true)

        # compute accuracy
        acc1, acc5 = self.custom_accuracy(y_pred, y_true, topk=(1, 5))

        self.log('val_loss', loss_valid, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc1', acc1, on_step=True, prog_bar=True, on_epoch=True, logger=True)
        self.log('val_acc5', acc5, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    def train_dataloader(self):
        # dataloader = get_train_dataloader(
        #     self.root_path, batch_size=self.batch_size, workers=self.workers)
        dataloader = get_mix_train_dataloader(
            self.root_path, batch_size=self.batch_size, workers=self.workers)
        return dataloader

    def val_dataloader(self):
        # dataloader = get_val_dataloader(
        #     self.root_path, batch_size=self.batch_size, workers=self.workers)
        dataloader = get_mix_val_dataloader(
            self.root_path, batch_size=self.batch_size, workers=self.workers)
        return dataloader

    def test_dataloader(self):
        dataloader = get_mix_val_dataloader(
            self.root_path, batch_size=self.batch_size, workers=self.workers)
        return dataloader

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('-j', '--workers', default=4,
                            type=int, metavar="N")
        parser.add_argument('-l', '--learning_rate', type=float,
                            default=0.1, dest="learning_rate")
        parser.add_argument('-b', '--batch_size', type=int,
                            default=64, dest="batch_size")
        parser.add_argument('--momentum', default=0.9,
                            type=float, metavar='M', dest="momentum")
        parser.add_argument('--wd', '--weight_decay', default=1e-4,
                            type=float, metavar="W", dest="weight_decay")
        parser.add_argument(
            '--pretrained', dest="pretrained", action='store_true')

        return parser


def process_args():
    ######################
    # args
    ######################
    parent_parser = argparse.ArgumentParser()
    parent_parser = pl.Trainer.add_argparse_args(parent_parser)
    parent_parser.add_argument(
        '--root_path', type=str, default="/media/niu/niu_g/data/imagenet", metavar="DIR", dest="root_path")
    parent_parser.add_argument('--seed', type=int, default=1234)
    parser = MixClassifier.add_model_specific_args(parent_parser)
    parser.set_defaults(
        profile=True,
        deterministic=True,
        max_epochs=90,
    )
    args = parser.parse_args()
    return args


def mix_main(args: Namespace) -> None:
    ######################
    # seed
    ######################
    if args.seed is not None:
        pl.seed_everything(args.seed)
    ######################
    # modify distributed args
    ######################
    if args.distributed_backend == 'ddp':
        args.batch_size = int(args.batch_size/max(1, args.gpus))
        args.workers = int(args.workers/max(1, args.gpus))

    ######################
    # model trainer
    ######################

    model = MixClassifier(**vars(args))

    # profiler = AdvancedProfiler
    checkpoint_callback = ModelCheckpoint(
        # save_best_only=False,
        verbose=True,
        monitor='val_loss',
        mode='min',
        filename='imagenet_1k-{epoch:02d}-{val_loss:.2f}')

    trainer = pl.Trainer(max_epochs=args.max_epochs,
                        #  amp_level='01',
                        #  amp_backend='apex',
                         progress_bar_refresh_rate=1,
                         gpus='-1',
                         deterministic=True,
                         profiler='advanced',
                         checkpoint_callback=checkpoint_callback,
                         accumulate_grad_batches=4,
                         distributed_backend="ddp")

    # lr_finder = trainer.tuner.lr_find(
    #     model, min_lr=5e-5, max_lr=5e-2, mode='linear')

    # fig = lr_finder.plot(suggest=True)
    # fig.savefig('./lr_finder.png')

    # model.learning_rate = lr_finder.suggestion()

    # find the largest optimal batch size
    # trainer.tune(model)

    # train
    trainer.fit(model)

    # test
    test_dataloader = model.test_dataloader()
    results = trainer.test(
        model,
        test_dataloaders=test_dataloader)
    print("Results:", results)


if __name__ == "__main__":
    args = process_args()
    f = open("time_record.txt", "w")
    print(time.asctime(time.localtime(time.time())), file=f)
    mix_main(args)
    print(time.asctime(time.localtime(time.time())), file=f)
    f.close()
