# DataLoader
# Author: pprp
# Date: 2020-11-28

from mix_dataloader import get_train_dataloader, get_val_dataloader
from torchvision.models import resnet18, resnet50
from torchvision.datasets.mnist import MNIST
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import functional as F
import torch.nn as nn
import torch
import pytorch_lightning as pl
import time
import os
import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"


class MixClassifier(pl.LightningModule):
    def __init__(self, learning_rate=0.1, root_path="../data", batch_size=3):
        super(MixClassifier, self).__init__()
        # self.resnet50 = resnet50(pretrained=False)
        # self.resnet50.fc = nn.Linear(2048, 10)
        self.resnet50 = resnet50(pretrained=False)
        # self.resnet18.conv1 = nn.Conv2d(
        #     1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet50.fc = nn.Linear(2048, 1000)

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.root_path = root_path

        self.save_hyperparameters()

        # Built-in API for metrics
        self.train_accuracy = pl.metrics.Accuracy()
        self.val_accuracy = pl.metrics.Accuracy()
        self.test_accuracy = pl.metrics.Accuracy()

    def forward(self, x):
        # return self.resnet50(x)
        return self.resnet50(x)

    def configure_optimizers(self):
        # 修改优化器
        # optimizer = torch.optim.SGD(
        #     self.parameters(), lr=self.learning_rate, weight_decay=1e-4, momentum=0.9)
        # scheduler = ReduceLROnPlateau(
        #     optimizer, mode="min", patience=10, factor=0.1, verbose=False, threshold=0.0001, eps=1e-08)
        # return {
        #     'optimizer': optimizer,
        #     'lr_scheduler': scheduler,
        #     'monitor': "loss"
        # }
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        # 每一个循环内部执行
        x, y = batch

        # feed the model and catch the prediction
        y_pred = self.forward(x)

        # loss calculation
        loss = F.cross_entropy(y_pred, y)

        # train accuracy calculation
        train_acc_batch = self.train_accuracy(y_pred, y)

        # Save metrics for current batch
        self.log("train_acc_batch", train_acc_batch)
        self.log("train_loss_batch", loss)
        self.log("training_time", time.time())

        # TODO 这种返回猜测应该是会输出到屏幕的内容，所以key的值可自定义
        return {"loss": loss, "y_pred": y_pred, "y_true": y}

    # def training_step_end(self, outputs):
    #     # 一个epoch结束
    #     accuracy_epoch = self.train_accuracy.compute()

    #     # save the metrics for current epoch
    #     self.log('train_acc_epoch', accuracy_epoch)
    #     loss = F.cross_entropy()

    def validation_step(self, batch, batch_idx):
        # 一个epoch结束以后，进行验证集测试
        x, y = batch
        y_pred = self.forward(x)

        # compute loss
        loss_valid = F.cross_entropy(y_pred, y)

        # compute accuracy
        accuracy_valid = self.val_accuracy(y_pred, y)

        self.log('valid_acc_batch', accuracy_valid)
        self.log('valid_loss_batch', loss_valid)

        return {'loss': loss_valid, 'y_pred': y_pred, 'target': y}

    def validation_epoch_end(self, outputs):
        # outputs是validation_step输出的一系列结果
        accuracy_valid_epoch = self.val_accuracy.compute()

        # save the metric
        self.log('val_acc_epoch', accuracy_valid_epoch)

    def test_step(self, batch, batch_idx):
        x, y = batch

        y_pred = self.forward(x)

        test_acc_batch = self.test_accuracy(y_pred, y)

        test_loss_batch = F.cross_entropy(y_pred, y)

        self.log("test_loss_batch", test_loss_batch)
        self.log("test_acc_batch", test_acc_batch)
        # 这个函数可以返回任何一个值
        return test_acc_batch


    def train_dataloader(self):
        # return DataLoader(self.mnist_train, batch_size=self.batch_size)
        dataloader = get_train_dataloader(
            self.root_path, batch_size=self.batch_size)
        return dataloader

    def val_dataloader(self):
        # return DataLoader(self.mnist_val, batch_size=self.batch_size)
        dataloader = get_val_dataloader(
            self.root_path, batch_size=self.batch_size)
        return dataloader

    def test_dataloader(self):
        # set same as valid
        # return DataLoader(self.mnist_test, batch_size=self.batch_size)
        dataloader = get_val_dataloader(
            self.root_path, batch_size=self.batch_size)
        return dataloader


def mix_main():
    pl.seed_everything(1234)
    model = MixClassifier(
        batch_size=64, root_path="/media/niu/niu_d/data/imagenet")

    trainer = pl.Trainer(max_epochs=200, check_val_every_n_epoch=5, precision=32,
                         weights_summary=None, progress_bar_refresh_rate=1,
                         auto_scale_batch_size='binsearch', gpus='0,1')

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
    results = trainer.test(test_dataloaders=test_dataloader)
    print("Results:", results)


if __name__ == "__main__":
    f = open("time_record.txt", "w")
    print(time.asctime(time.localtime(time.time())), file=f)
    mix_main()
    print(time.asctime(time.localtime(time.time())), file=f)
    f.close()
