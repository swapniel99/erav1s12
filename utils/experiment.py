from collections import defaultdict
from torch import nn, optim
from torch_lr_finder import LRFinder
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

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
        self.grad_cam = None

    def find_lr(self):
        lr_finder = LRFinder(self.model, self.optimizer, self.criterion, device=self.device)
        lr_finder.range_test(self.dataset.train_loader, start_lr=1e-5, end_lr=0.1, num_iter=100, step_mode='exp')
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

    def get_incorrect_preds(self):
        if self.incorrect_preds is None:
            self.incorrect_preds = defaultdict(list)
            self.test(self.incorrect_preds)

    def get_cam_visualisation(self, input_tensor, label):
        if self.grad_cam is None:
            self.grad_cam = GradCAM(model=self.model, target_layers=[self.model.layer3[-1]],
                                    use_cuda=(self.device == 'cuda'))

        targets = [ClassifierOutputTarget(label)]

        grayscale_cam = self.grad_cam(input_tensor=input_tensor.unsqueeze(0), targets=targets)
        # In this example grayscale_cam has only one image in the batch:
        grayscale_cam = grayscale_cam[0, :]

        output = show_cam_on_image(self.dataset.show_transform(input_tensor).cpu().numpy(), grayscale_cam, use_rgb=True)
        return output

    def show_incorrect(self, cams=False):
        self.get_incorrect_preds()

        images = list()
        labels = list()

        for i in range(20):
            image = self.incorrect_preds["images"][i]
            pred = self.incorrect_preds["predicted_vals"][i]
            truth = self.incorrect_preds["ground_truths"][i]

            if cams:
                image = self.get_cam_visualisation(image, pred)
            else:
                image = self.dataset.show_transform(image).cpu()

            if self.dataset.classes is not None:
                pred = f'{pred}:{self.dataset.classes[pred]}'
                truth = f'{truth}:{self.dataset.classes[truth]}'
            label = f'{pred}/{truth}'

            images.append(image)
            labels.append(label)

        plot_examples(images, labels, figsize=(10, 8))
