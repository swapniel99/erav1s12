from collections import defaultdict
from torch import nn, optim
from torch_lr_finder import LRFinder

from utils import get_device, plot_examples
from .backprop import Train, Test


class Experiment(object):
    criterions = {
        'nll': nn.NLLLoss,
        'crossentropy': nn.CrossEntropyLoss
    }

    def __init__(self, model, dataset, criterion='crossentropy', epochs=20, lr=0.01, scheduler='one_cycle'):
        self.device = get_device()
        self.model = model.to(self.device)
        self.dataset = dataset
        self.criterion = self.criterions.get(criterion, nn.CrossEntropyLoss)()
        self.epochs = epochs
        self.optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=lr)
        if scheduler == 'one_cycle':
            self.best_lr = self.find_lr()
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.best_lr,
                steps_per_epoch=len(self.dataset.train_loader),
                epochs=self.epochs,
                pct_start=5 / self.epochs,
                div_factor=100,
                three_phase=False,
                final_div_factor=100,
                anneal_strategy='linear'
            )
            perform_step = True
        else:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=1, verbose=True, factor=0.1)
            perform_step = False
        self.train = Train(self.model, dataset, self.criterion, self.optimizer, scheduler=self.scheduler,
                           perform_step=perform_step)
        self.test = Test(self.model, dataset, self.criterion)
        self.incorrect_preds = None

    def find_lr(self):
        lr_finder = LRFinder(self.model, self.optimizer, self.criterion, device=self.device)
        lr_finder.range_test(self.dataset.train_loader, end_lr=0.1, num_iter=100, step_mode='exp')
        _, best_lr = lr_finder.plot()  # to inspect the loss-learning rate graph
        lr_finder.reset()  # to reset the model and optimizer to their initial state
        return best_lr

    def execute(self, target=None):
        target_count = 0
        for epoch in range(1, self.epochs + 1):
            print(f'Epoch: {epoch}')
            self.train()
            test_loss, test_acc = self.test()
            if target is not None and test_acc >= target:
                target_count += 1
                if target_count >= 3:
                    print("Target Validation accuracy achieved thrice. Stopping Training.")
                    break

    def show_incorrect(self, denorm=True):
        self.incorrect_preds = defaultdict(list)
        self.test(self.incorrect_preds)

        images = list()
        labels = list()

        for i in range(len(self.incorrect_preds["images"])):
            image = self.incorrect_preds["images"][i].cpu()
            if denorm:
                image = self.dataset.denormalise(image)
            image = self.dataset.show_transform(image)

            pred = self.incorrect_preds["predicted_vals"][i]
            truth = self.incorrect_preds["ground_truths"][i]
            if self.dataset.classes is not None:
                pred = f'{pred}:{self.dataset.classes[pred]}'
                truth = f'{truth}:{self.dataset.classes[truth]}'
            label = f'{pred}/{truth}'

            images.append(image)
            labels.append(label)

        plot_examples(images, labels, figsize=(10, 6))
