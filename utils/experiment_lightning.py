import pandas as pd
from collections import defaultdict
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelSummary

from .misc import plot_examples
from .backprop import Test


def get_incorrect_preds(prediction, labels):
    prediction = prediction.argmax(dim=1)
    indices = prediction.ne(labels).nonzero().reshape(-1).tolist()
    return indices, prediction[indices].tolist(), labels[indices].tolist()


class Experiment(object):
    def __init__(self, model, max_epochs=None, precision="32-true"):
        self.model = model
        self.dataset = model.dataset
        self.incorrect_preds = None
        self.grad_cam = None
        self.trainer = Trainer(callbacks=ModelSummary(max_depth=10), max_epochs=max_epochs or model.epochs,
                               precision=precision)
        self.test = Test(self.model, self.model.dataset, self.model.criterion)
        self.incorrect_preds = None
        self.incorrect_preds_pd = None
        self.grad_cam = None

    def execute(self):
        self.trainer.fit(self.model)

    def get_incorrect_preds(self):
        self.incorrect_preds = defaultdict(list)
        incorrect_images = list()
        processed = 0
        results = self.trainer.predict()
        for (data, target), pred in zip(self.model.predict_dataloader(), results):
            ind, pred_, truth = get_incorrect_preds(pred, target)
            self.incorrect_preds["indices"] += [x + processed for x in ind]
            incorrect_images += data[ind]
            self.incorrect_preds["ground_truths"] += truth
            self.incorrect_preds["predicted_vals"] += pred_
            processed += len(data)
        self.incorrect_preds_pd = pd.DataFrame(self.incorrect_preds)
        self.incorrect_preds["images"] = incorrect_images

    def get_cam_visualisation(self, input_tensor, label, target_layer):
        if self.grad_cam is None:
            self.grad_cam = GradCAM(model=self.model, target_layers=[target_layer], use_cuda=True)

        targets = [ClassifierOutputTarget(label)]

        grayscale_cam = self.grad_cam(input_tensor=input_tensor.unsqueeze(0), targets=targets)
        # In this example grayscale_cam has only one image in the batch:
        grayscale_cam = grayscale_cam[0, :]

        output = show_cam_on_image(self.model.dataset.show_transform(input_tensor).cpu().numpy(), grayscale_cam,
                                   use_rgb=True)
        return output

    def show_incorrect(self, cams=False, target_layer=None):
        if self.incorrect_preds is None:
            self.get_incorrect_preds()

        images = list()
        labels = list()

        for i in range(20):
            image = self.incorrect_preds["images"][i]
            pred = self.incorrect_preds["predicted_vals"][i]
            truth = self.incorrect_preds["ground_truths"][i]

            if cams:
                image = self.get_cam_visualisation(image, pred, target_layer)
            else:
                image = self.dataset.show_transform(image).cpu()

            if self.dataset.classes is not None:
                pred = f'{pred}:{self.dataset.classes[pred]}'
                truth = f'{truth}:{self.dataset.classes[truth]}'
            label = f'{pred}/{truth}'

            images.append(image)
            labels.append(label)

        plot_examples(images, labels, figsize=(10, 8))
