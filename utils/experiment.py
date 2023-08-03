import pandas as pd
from collections import defaultdict

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelSummary

from .misc import plot_examples, get_cam_visualisation, get_incorrect_preds


class Experiment(object):
    def __init__(self, model, max_epochs=None, precision="32-true"):
        self.model = model
        self.dataset = model.dataset
        self.incorrect_preds = None
        self.grad_cam = None
        self.trainer = Trainer(callbacks=ModelSummary(max_depth=10), max_epochs=max_epochs or model.max_epochs,
                               precision=precision)
        self.incorrect_preds = None
        self.incorrect_preds_pd = None
        self.grad_cam = None

    def execute(self):
        self.trainer.fit(self.model)

    def get_incorrect_preds(self):
        self.incorrect_preds = defaultdict(list)
        incorrect_images = list()
        processed = 0
        results = self.trainer.predict(self.model, self.model.predict_dataloader())
        for (data, target), pred in zip(self.model.predict_dataloader(), results):
            ind, pred_, truth = get_incorrect_preds(pred, target)
            self.incorrect_preds["indices"] += [x + processed for x in ind]
            incorrect_images += data[ind]
            self.incorrect_preds["ground_truths"] += truth
            self.incorrect_preds["predicted_vals"] += pred_
            processed += len(data)
        self.incorrect_preds_pd = pd.DataFrame(self.incorrect_preds)
        self.incorrect_preds["images"] = incorrect_images

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
                image = get_cam_visualisation(self.model, self.dataset, image, pred, target_layer)
            else:
                image = self.dataset.show_transform(image).cpu()

            if self.dataset.classes is not None:
                pred = f'{pred}:{self.dataset.classes[pred]}'
                truth = f'{truth}:{self.dataset.classes[truth]}'
            label = f'{pred}/{truth}'

            images.append(image)
            labels.append(label)

        plot_examples(images, labels, figsize=(10, 8))
