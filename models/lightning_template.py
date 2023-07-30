from torch import nn
from torch import optim
from pytorch_lightning import LightningModule
from torchmetrics import MeanMetric
from torch_lr_finder import LRFinder

from utils.metrics import RunningAccuracy


class ModelTemplate(LightningModule):
    def __init__(self, dataset):
        super(ModelTemplate, self).__init__()

        self.dataset = dataset
        self.criterion = nn.CrossEntropyLoss()
        self.train_accuracy = RunningAccuracy()
        self.val_accuracy = RunningAccuracy()
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

    def common_step(self, batch, acc_metric, loss_metric):
        x, y = batch
        batch_len = y.numel()
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        loss_metric.update(loss * batch_len, batch_len)
        acc_metric.update(logits, y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, self.train_accuracy, self.train_loss)
        self.log("train_step_loss", loss, prog_bar=True)
        self.log("train_step_acc", self.accuracy, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.log("train_loss", self.train_loss, logger=True)
        self.log("train_acc", self.train_accuracy, logger=True)
        self.train_loss.reset()
        self.train_accuracy.reset()

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, self.val_accuracy, self.val_loss)
        self.log("val_step_loss", loss, prog_bar=True)
        self.log("val_step_acc", self.accuracy, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        self.log("val_loss", self.val_loss, logger=True)
        self.log("val_acc", self.val_accuracy, logger=True)
        self.val_loss.reset()
        self.val_accuracy.reset()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        self.on_validation_epoch_end()

    def find_lr(self, optimizer):
        lr_finder = LRFinder(self, optimizer, self.criterion)
        lr_finder.range_test(self.dataset.train_loader, start_lr=1e-5, end_lr=0.1, num_iter=100, step_mode='exp')
        _, best_lr = lr_finder.plot()
        lr_finder.reset()
        return best_lr

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), momentum=0.9, lr=0.01)
        best_lr = self.find_lr(optimizer)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=best_lr,
            steps_per_epoch=len(self.dataset.train_loader),
            epochs=20,
            pct_start=0.25,
            div_factor=100,
            three_phase=False,
            final_div_factor=100,
            anneal_strategy='linear'
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                "scheduler": scheduler,
                "interval": "step",
            }
        }

    def prepare_data(self):
        self.dataset.download()

    def train_dataloader(self):
        return self.dataset.get_train_loader()

    def val_dataloader(self):
        return self.dataset.get_test_loader()

    def test_dataloader(self):
        return self.val_dataloader()
