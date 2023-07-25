try:
    from epoch.utils import set_seed, model_summary
    from epoch.utils.experiment import Experiment
    from epoch.models.resnet import ResNet18
    from epoch.datasets import CIFAR10
except ModuleNotFoundError:
    from utils.misc import set_seed, model_summary
    from utils.experiment import Experiment
    from models import *
    from datasets import *

set_seed(42)
batch_size = 32

cifar10 = CIFAR10(batch_size)
model = ResNet18()


def print_summary(model_):
    print(model_summary(model_, input_size=(batch_size, 3, 32, 32)))


def create_experiment(model_, dataset, criterion, epochs, scheduler):
    return Experiment(model_, dataset, criterion=criterion, epochs=epochs, scheduler=scheduler)


def main(criterion='crossentropy', epochs=20, scheduler='one_cycle'):
    experiment = create_experiment(criterion=criterion, epochs=epochs, scheduler=scheduler)
    experiment.execute()
    experiment.train.plot_stats()
    experiment.test.plot_stats()
    experiment.show_incorrect()
    experiment.show_incorrect(cams=True)


if __name__ == '__main__':
    main()
