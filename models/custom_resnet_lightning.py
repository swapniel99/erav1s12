from torch import nn
from torch import optim
from pytorch_lightning import LightningModule
from torchmetrics import MeanMetric
from torch_lr_finder import LRFinder

from utils.metrics import RunningAccuracy


class ConvLayer(nn.Module):
    def __init__(self, input_c, output_c, bias=False, stride=1, padding=1, pool=False, dropout=0):
        super(ConvLayer, self).__init__()

        layers = list()
        layers.append(
            nn.Conv2d(input_c, output_c, kernel_size=3, bias=bias, stride=stride, padding=padding,
                      padding_mode='replicate')
        )
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        layers.append(nn.BatchNorm2d(output_c))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        self.all_layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.all_layers(x)


class CustomLayer(nn.Module):
    def __init__(self, input_c, output_c, pool=True, residue=2, dropout=0):
        super(CustomLayer, self).__init__()

        self.pool_block = ConvLayer(input_c, output_c, pool=pool, dropout=dropout)
        self.res_block = None
        if residue > 0:
            layers = list()
            for i in range(0, residue):
                layers.append(ConvLayer(output_c, output_c, pool=False, dropout=dropout))
            self.res_block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.pool_block(x)
        if self.res_block is not None:
            x_ = x
            x = self.res_block(x)
            # += operator causes inplace errors in pytorch if done right after relu.
            x = x + x_
        return x


class Model(LightningModule):
    def __init__(self, dataset, dropout=0):
        super(Model, self).__init__()

        self.dataset = dataset

        self.network = nn.Sequential(
            CustomLayer(3, 64, pool=False, residue=0, dropout=dropout),
            CustomLayer(64, 128, pool=True, residue=2, dropout=dropout),
            CustomLayer(128, 256, pool=True, residue=0, dropout=dropout),
            CustomLayer(256, 512, pool=True, residue=2, dropout=dropout),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Flatten(),
            nn.Linear(512, 10)
        )

        self.criterion = nn.CrossEntropyLoss()
        self.train_accuracy = RunningAccuracy()
        self.val_accuracy = RunningAccuracy()
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        self.epoch = 1

    def forward(self, x):
        return self.network(x)

    def common_step(self, batch, loss_metric, acc_metric):
        x, y = batch
        batch_len = y.numel()
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        loss_metric.update(loss, batch_len)
        acc_metric.update(logits, y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, self.train_loss, self.train_accuracy)
        self.log("train_step_loss", self.train_loss, prog_bar=True, logger=True)
        self.log("train_step_acc", self.train_accuracy, prog_bar=True, logger=True)
        return loss

    def on_train_epoch_end(self):
        print(f"Epoch: {self.epoch}, Train Accuracy: {self.train_accuracy.compute()}, Train Loss: "
              f"{self.train_loss.compute()}")
        self.train_loss.reset()
        self.train_accuracy.reset()

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, self.val_loss, self.val_accuracy)
        self.log("val_step_loss", self.val_loss, prog_bar=True, logger=True)
        self.log("val_step_acc", self.val_accuracy, prog_bar=True, logger=True)
        return loss

    def on_validation_epoch_end(self):
        print(f"Epoch: {self.epoch}, Test Accuracy: {self.val_accuracy.compute()}, Test Loss: "
              f"{self.val_loss.compute()}")
        self.val_loss.reset()
        self.val_accuracy.reset()
        self.epoch += 1

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
            steps_per_epoch=len(self.dataset.get_train_loader()),
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
