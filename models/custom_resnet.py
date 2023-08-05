from torch import nn
from torch import optim
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy
from torch_lr_finder import LRFinder

from utils.metrics import RunningAccuracy


class ConvLayer(nn.Module):
    def __init__(self, input_c, output_c, bias=False, stride=1, padding=1, pool=False, dropout=0.):
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
    def __init__(self, input_c, output_c, pool=True, residue=2, dropout=0.):
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
    def __init__(self, dataset, dropout=0.05, max_epochs=24):
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
        self.train_accuracy2 = Accuracy(task='multiclass', num_classes=10)
        self.val_accuracy2 = Accuracy(task='multiclass', num_classes=10)

        self.max_epochs = max_epochs

    def forward(self, x):
        return self.network(x)

    def common_step(self, batch, mode):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)

        if mode == 'train':
            acc_metric = self.train_accuracy
            acc_metric2 = self.train_accuracy2
        else:
            acc_metric = self.val_accuracy
            acc_metric2 = self.val_accuracy2
        acc_metric.update(logits, y)
        acc_metric2(logits, y)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, 'train')
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", self.train_accuracy, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc2", self.train_accuracy2, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_train_epoch_end(self):
        print(f"Train Accuracy {self.train_accuracy.compute():0.2f}")
        self.train_accuracy.reset()

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, 'val')
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", self.val_accuracy, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc2", self.val_accuracy2, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_validation_epoch_end(self):
        print(f"Val Accuracy: {self.val_accuracy.compute():0.2f}")
        self.val_accuracy.reset()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if isinstance(batch, list):
            x, _ = batch
        else:
            x = batch
        return self.forward(x)

    def find_lr(self, optimizer):
        lr_finder = LRFinder(self, optimizer, self.criterion)
        lr_finder.range_test(self.dataset.train_loader, end_lr=0.1, num_iter=100, step_mode='exp')
        _, best_lr = lr_finder.plot()
        lr_finder.reset()
        return best_lr

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-7, weight_decay=1e-2)
        best_lr = self.find_lr(optimizer)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=best_lr,
            steps_per_epoch=len(self.dataset.train_loader),
            epochs=self.max_epochs,
            pct_start=5/self.max_epochs,
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
        return self.dataset.train_loader

    def val_dataloader(self):
        return self.dataset.test_loader

    def predict_dataloader(self):
        return self.val_dataloader()
