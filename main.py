import torch

from deepproblog.dataset import Dataset as DPLDataset, DataLoader as DPLDataLoader
from deepproblog.query import Query
from deepproblog.network import Network
from deepproblog.model import Model
from deepproblog.engines import ExactEngine
from deepproblog.train import TrainObject
from deepproblog.evaluate import get_confusion_matrix, get_fact_accuracy
from problog.logic import Term, Constant, list2term

from model import ProtoTSNet
from autoencoder import RegularConvEncoder
from artificial_datasets_DPL import ArtificialProtosDataset, ArtificialProtosDatasetRandomShift
from train import train_prototsnet_DPL, ProtoTSCoeffs


class ArtificialProtosQueries(DPLDataset):
    def __init__(self, dataset: ArtificialProtosDataset, phase: str):
        self.phase = phase
        self.dataset = dataset
        self.dataset_len = len(dataset)
        self.num_classes = len(set([dataset.get_label(i) for i in range(self.dataset_len)]))
        print(f'Detected number of classes in ArtificialProtosQueries: {self.num_classes}')

    def to_query(self, i: int) -> Query:
        
        ds_entry = i // self.num_classes
        cls_num = i % self.num_classes
        correct_cls = self.dataset.get_label(ds_entry)

        ts_term = Term(f'ts{ds_entry}')
        q = Query(
            Term(
                'is_class',
                ts_term,
                Term(f'c{cls_num}')
            ),
            {
                ts_term: Term(
                    "tensor",
                    Term(
                        self.phase,
                        Constant(ds_entry),
                    ),
                )
            },
            p = float(cls_num == correct_cls)
        )
        return q

    def __len__(self):
        return self.dataset_len * self.num_classes

protos_per_class = 1
latent_features = 32
# num_features = 2
num_features = 3
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

# train_dataset = ArtificialProtosDatasetRandomShift(1000, num_feat=num_features, classes=num_classes)
train_dataset = ArtificialProtosDataset(1000)
model.add_tensor_source("train", train_dataset)
train_queries = ArtificialProtosQueries(train_dataset, "train")
train_loader = DPLDataLoader(train_queries, 1, True)

# test_dataset = ArtificialProtosDatasetRandomShift(200, num_feat=num_features, classes=num_classes)
test_dataset = ArtificialProtosDataset(200)
model.add_tensor_source("test", test_dataset)
test_queries = ArtificialProtosQueries(test_dataset, "test")
train_loader = DPLDataLoader(test_queries, 1, False)

print("Training...")
trainer = train_prototsnet_DPL(
    dpl_model=model,
    ptsnet=net,
    experiment_dir='./experiments/initial',
    device=torch.device('cpu'),
    coeffs=ProtoTSCoeffs(1, 0),
    train_dataset=train_dataset,
    train_loader=train_loader,
    test_loader=train_loader,
    num_epochs=40,
    num_warm_epochs=0,
    push_start_epoch=20,
    push_epochs=range(20, 1000, 20),
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


