from collections import defaultdict
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from pytorch_lightning import Trainer

from .misc import plot_examples
from .backprop import Test


class Experiment(object):
    def __init__(self, model, max_epochs=20, precision=32):
        self.model = model
        self.dataset = model.dataset
        self.dataset.get_train_loader()
        self.dataset.get_test_loader()
        self.incorrect_preds = None
        self.grad_cam = None
        self.trainer = Trainer(max_epochs=max_epochs, precision=precision)
        self.test = Test(self.model, self.model.dataset, self.model.criterion)
        self.incorrect_preds = None
        self.grad_cam = None

    def execute(self):
        self.trainer.fit(self.model)

    def get_incorrect_preds(self):
        if self.incorrect_preds is None:
            self.incorrect_preds = defaultdict(list)
            self.test(self.incorrect_preds)

    def get_cam_visualisation(self, input_tensor, label):
        if self.grad_cam is None:
            self.grad_cam = GradCAM(model=self.model, target_layers=[self.model.layer3[-1]],
                                    use_cuda=True)

        targets = [ClassifierOutputTarget(label)]

        grayscale_cam = self.grad_cam(input_tensor=input_tensor.unsqueeze(0), targets=targets)
        # In this example grayscale_cam has only one image in the batch:
        grayscale_cam = grayscale_cam[0, :]

        output = show_cam_on_image(self.model.dataset.show_transform(input_tensor).cpu().numpy(), grayscale_cam,
                                   use_rgb=True)
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
