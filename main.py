import torch

from deepproblog.dataset import DataLoader as DPLDataLoader
from deepproblog.network import Network
from deepproblog.model import Model
from deepproblog.engines import ExactEngine
from deepproblog.train import TrainObject
from deepproblog.evaluate import get_confusion_matrix, get_fact_accuracy
from deepproblog.optimizer import SGD, Optimizer

from model import ProtoTSNet
from autoencoder import RegularConvEncoder
from artificial_datasets_DPL import (
    ArtificialProtosDataset,
    ArtificialProtosDatasetRandomShift,
    Queries,
    QueriesWithNegatives
)
from train import train_prototsnet_DPL, ProtoTSCoeffs


protos_per_class = 1
latent_features = 32
num_features = 5
# num_features = 3
num_classes = 4
NOISE_POWER = 0.1

print('Preparing ProtoTSNet...')
autoencoder = RegularConvEncoder(num_features=num_features, latent_features=latent_features, padding='same')
encoder = autoencoder.encoder
net = ProtoTSNet(
    cnn_base=encoder,
    for_deepproblog=True,
    num_features=num_features,
    ts_sample_len=100,
    proto_num=protos_per_class*num_classes,
    latent_features=latent_features,
    proto_len_latent=20,
    num_classes=num_classes,
)

dpl_net = Network(net, "ptsnet", batching=False)
# dpl_net.optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

class MySGD(Optimizer):
    """
    An optimizer that also optimizes the probabilistic parameters in the model using stochastic gradient descent.
    """

    def __init__(self, model: "Model", param_lr: float):
        """

        :param model: The model whose parameters will be optimized.
        :param param_lr: The learning rate for the parameters.
        """
        Optimizer.__init__(self, model)
        self.param_lr = param_lr

    def get_lr(self) -> float:
        """

        :return: The learning rate for the probabilistic parameters.
        """
        return self.param_lr

    def add_parameter_gradient(self, k, grad: torch.Tensor):
        self._params_grad[k] += grad

    def step(self):
        Optimizer.step(self)
        for k in self._params_grad:
            self.model.parameters[k] -= float(self.get_lr() * self._params_grad[k])
            # self.model.parameters[k] = max(min(self.model.parameters[k], 1.0), 0.0)
        for group in self.model.parameter_groups:
            p_sum = sum(self.model.parameters[x] for x in group)
            for param in group:
                self.model.parameters[param] /= p_sum

print('Loading logic file...')
model = Model("proto_logic.pl", [dpl_net])
model.set_engine(ExactEngine(model))
model.optimizer = MySGD(model, 1e-2)
# model.set_engine(ApproximateEngine(model, 1, ApproximateEngine.geometric_mean, exploration=False))

train_dataset = ArtificialProtosDatasetRandomShift(
    200, num_feat=num_features, classes=num_classes, feature_noise_power=NOISE_POWER
)
# train_dataset = ArtificialProtosDataset(1000)
model.add_tensor_source("train", train_dataset)
train_queries = QueriesWithNegatives(train_dataset, "train")
train_loader = DPLDataLoader(train_queries, 1, True)

test_dataset = ArtificialProtosDatasetRandomShift(
    40,
    num_feat=num_features,
    classes=num_classes,
    feature_noise_power=NOISE_POWER,
    class_specifics=train_dataset.class_specifics,
)
# test_dataset = ArtificialProtosDataset(200)
model.add_tensor_source("test", test_dataset)
test_queries = QueriesWithNegatives(test_dataset, "test")
test_loader = DPLDataLoader(test_queries, 1, False)

def log_params(trainer, _):
    trainer.log(f'############# {"Parameters":25s} #############')
    trainer.log(f'{trainer.model.parameters}')

print("Training...")
trainer = train_prototsnet_DPL(
    dpl_model=model,
    ptsnet=net,
    experiment_dir='./experiments/SimpleProtosWithShiftNoNoiseTrainableLogic4Classes',
    device=torch.device('cpu'),
    coeffs=ProtoTSCoeffs(1, clst=0.8, l1=1e-2),
    train_dataset=train_dataset,
    train_loader=train_loader,
    test_loader=test_loader,
    num_epochs=30,
    num_warm_epochs=0,
    push_start_epoch=10,
    push_epochs=range(10, 1000, 10),
    loss_uses_negatives=False,
    custom_hooks=[
        log_params
    ]
    # lr_sched_setup=(lambda opt: torch.optim.lr_scheduler.CyclicLR(opt, base_lr=1e-4, max_lr=3e-2, step_size_up=3, step_size_down=7, mode='exp_range', gamma=0.99, cycle_momentum=False)),
)
# train = TrainObject(model)
# train.train(train_loader, 20, log_iter=100, profile=0)

# model.save_state('./snapshots/initial_model.pth')
# train.logger.comment(json.dumps(model.get_hyperparameters()))
# train.logger.comment(
# "Accuracy {}".format(get_confusion_matrix(model, test_queries, verbose=1).accuracy())
# )
# get_fact_accuracy(model, train_queries, verbose=1)
# get_fact_accuracy(model, test_queries, verbose=1)
# train.logger.write_to_file("log/initial.log")
