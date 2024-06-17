import torch

from deepproblog.dataset import DataLoader as DPLDataLoader
from deepproblog.network import Network
from deepproblog.model import Model
from deepproblog.engines import ExactEngine
from deepproblog.train import TrainObject
from deepproblog.evaluate import get_confusion_matrix, get_fact_accuracy

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
num_features = 2
# num_features = 3
num_classes = 2

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

print('Loading logic file...')
model = Model("proto_logic.pl", [dpl_net])
model.set_engine(ExactEngine(model))
# model.set_engine(ApproximateEngine(model, 1, ApproximateEngine.geometric_mean, exploration=False))

train_dataset = ArtificialProtosDatasetRandomShift(1000, num_feat=num_features, classes=num_classes, feature_noise_power=0)
# train_dataset = ArtificialProtosDataset(1000)
model.add_tensor_source("train", train_dataset)
train_queries = QueriesWithNegatives(train_dataset, "train")
train_loader = DPLDataLoader(train_queries, 1, True)

test_dataset = ArtificialProtosDatasetRandomShift(200, num_feat=num_features, classes=num_classes, feature_noise_power=0)
# test_dataset = ArtificialProtosDataset(200)
model.add_tensor_source("test", test_dataset)
test_queries = QueriesWithNegatives(test_dataset, "test")
train_loader = DPLDataLoader(test_queries, 1, False)

print("Training...")
trainer = train_prototsnet_DPL(
    dpl_model=model,
    ptsnet=net,
    experiment_dir='./experiments/SimpleProtosWithShiftNoNoise',
    device=torch.device('cpu'),
    coeffs=ProtoTSCoeffs(1, 0),
    train_dataset=train_dataset,
    train_loader=train_loader,
    test_loader=train_loader,
    num_epochs=30,
    num_warm_epochs=0,
    push_start_epoch=10,
    push_epochs=range(10, 1000, 10),
    loss_uses_negatives=False,
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
