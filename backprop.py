import torch
from torch import optim
from torch.nn import functional as F
from tqdm import tqdm
from matplotlib import pyplot as plt
from collections import defaultdict

from utils import get_device


def get_correct_count(prediction, labels):
    return prediction.argmax(dim=1).eq(labels).sum().item()


def get_incorrect_preds(prediction, labels):
    prediction = prediction.argmax(dim=1)
    indices = prediction.ne(labels).nonzero().reshape(-1).tolist()
    return indices, prediction[indices].tolist(), labels[indices].tolist()


class Train(object):
    def __init__(self, model, dataset, criterion, optimizer, l1=0):
        self.model = model
        self.device = get_device()
        self.criterion = criterion
        self.dataset = dataset
        self.optimizer = optimizer
        self.l1 = l1

        self.train_losses = list()
        self.train_accuracies = list()

    def __call__(self):
        self.model.train()
        pbar = tqdm(self.dataset.train_loader)

        train_loss = 0
        correct = 0
        processed = 0

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()

            # Predict
            pred = self.model(data)

            # Calculate loss
            loss = self.criterion(pred, target)
            if self.l1 > 0:
                loss += self.l1 * sum(p.abs().sum() for p in self.model.parameters())

            train_loss += loss.item() * len(data)

            # Backpropagation
            loss.backward()
            self.optimizer.step()

            correct += get_correct_count(pred, target)
            processed += len(data)

            pbar.set_description(
                desc=f"Train: Average Loss: {train_loss / processed:0.4f}, Accuracy: {100 * correct / processed:0.2f}"
            )

        train_acc = 100 * correct / processed
        train_loss /= processed
        self.train_accuracies.append(train_acc)
        self.train_losses.append(train_loss)

        # print(f"Train: Average loss: {train_loss:0.4f}, Accuracy: {train_acc:0.2f}")

        return train_loss, train_acc

    def plot_stats(self):
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        axs[0].plot(self.train_losses)
        axs[0].set_title("Training Loss")
        axs[1].plot(self.train_accuracies)
        axs[1].set_title("Training Accuracy")


class Test(object):
    def __init__(self, model, dataset, criterion):
        self.model = model
        self.device = get_device()
        self.criterion = criterion
        self.dataset = dataset

        self.test_losses = list()
        self.test_accuracies = list()

    def __call__(self, incorrect_preds=None):
        self.model.eval()

        test_loss = 0
        correct = 0
        processed = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.dataset.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                pred = self.model(data)

                test_loss += self.criterion(pred, target, reduction="sum").item()

                correct += get_correct_count(pred, target)
                processed += len(data)

                if incorrect_preds is not None:
                    ind, pred, truth = get_incorrect_preds(pred, target)
                    incorrect_preds["images"] += data[ind]
                    incorrect_preds["ground_truths"] += truth
                    incorrect_preds["predicted_vals"] += pred

        test_acc = 100 * correct / processed
        test_loss /= processed
        self.test_accuracies.append(test_acc)
        self.test_losses.append(test_loss)

        print(f"Test:  Average loss: {test_loss:0.4f}, Accuracy: {test_acc:0.2f}")

        return test_loss, test_acc

    def plot_stats(self):
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        axs[0].plot(self.test_losses)
        axs[0].set_title("Test Loss")
        axs[1].plot(self.test_accuracies)
        axs[1].set_title("Test Accuracy")


class Experiment(object):
    def __init__(self, model, dataset, lr=0.01, criterion=F.nll_loss):
        self.model = model.to(get_device())
        self.dataset = dataset
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=1, verbose=True, factor=0.1)
        self.train = Train(self.model, dataset, criterion, self.optimizer)
        self.test = Test(self.model, dataset, criterion)
        self.incorrect_preds = None

    def execute(self, epochs=20, target=None):
        target_count = 0
        for epoch in range(1, epochs + 1):
            print(f'Epoch {epoch}')
            self.train()
            test_loss, test_acc = self.test()
            if target is not None and test_acc >= target:
                target_count += 1
                if target_count >= 3:
                    print("Target Validation accuracy achieved thrice. Stopping Training.")
                    break
            self.scheduler.step(test_loss)

    def show_incorrect(self, denorm=True):
        self.incorrect_preds = defaultdict(list)
        self.test(self.incorrect_preds)

        _ = plt.figure(figsize=(10, 3))
        for i in range(10):
            plt.subplot(2, 5, i + 1)
            plt.tight_layout()
            image = self.incorrect_preds["images"][i].cpu()
            if denorm:
                image = self.dataset.denormalise(image)
            plt.imshow(self.dataset.show_transform(image), cmap='gray')
            pred = self.incorrect_preds["predicted_vals"][i]
            truth = self.incorrect_preds["ground_truths"][i]
            if self.dataset.classes is not None:
                pred = f'{pred}:{self.dataset.classes[pred]}'
                truth = f'{truth}:{self.dataset.classes[truth]}'
            plt.title(f'{pred}/{truth}')
            plt.xticks([])
            plt.yticks([])
