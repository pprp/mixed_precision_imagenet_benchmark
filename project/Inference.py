import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.metrics import functional as FM
from torch import nn

from tiny_imagenet_classifier import MixClassifier


class ClassificationTask(pl.LightningModule):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        acc = FM.accuracy(y_hat, y)

        # loss is tensor. The Checkpoint Callback is monitoring 'checkpoint_on'
        metrics = {'val_acc': acc, 'val_loss': loss}
        self.log_dict(metrics)
        return metrics

    def test_step(self, batch, batch_idx):
        metrics = self.validation_step(batch, batch_idx)
        metrics = {'test_acc': metrics['val_acc'],
                   'test_loss': metrics['val_loss']}
        self.log_dict(metrics)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.02)


if __name__ == "__main__":
    model = MixClassifier(batch_size=1, root_path="D:\imagenet_data", pretrained=False,
                          learning_rate=1e-4, momentum=0.9, weight_decay=1e-4, workers=32)

    model.eval()
    test_dataloader = model.test_dataloader()

    trainer = pl.Trainer(max_epochs=20, check_val_every_n_epoch=5,
                         precision=32,
                         weights_summary=None,
                         progress_bar_refresh_rate=1,
                         auto_scale_batch_size='binsearch',
                         gpus='-1',
                         deterministic=True,
                         profiler='advanced')
    trainer.test(model, ckpt_path="./lightning_logs/version_3/checkpoints/epoch=14.ckpt",
                 test_dataloaders=test_dataloader)
