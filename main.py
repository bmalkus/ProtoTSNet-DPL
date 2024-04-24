from scipy import signal
import numpy as np
import torch
import json

from deepproblog.dataset import Dataset as DPLDataset, DataLoader
from deepproblog.query import Query
from deepproblog.network import Network
from deepproblog.model import Model
from deepproblog.engines import ExactEngine
from deepproblog.train import train_model
from deepproblog.evaluate import get_confusion_matrix, get_fact_accuracy
from problog.logic import Term, Constant, list2term

from model import ProtoTSNet
from autoencoder import RegularConvEncoder

class ArtificialProtosDataset():
    def __init__(self, N, feature_noise_power=0.1, randomize_right_side=False):
        self.data = []
        x = np.linspace(0, 100, 100)
        for _ in range(N):
            label = np.random.randint(0, 2)
            ts = np.zeros((3, 100))
            if label == 0:
                ts[0, :40] = signal.sawtooth(x[:40] / (1+1))
                ts[1, :40] = signal.square(x[:40] / (2+1))
            else:
                ts[0, :40] = signal.square(x[:40] / (1+1))
                ts[1, :40] = signal.sawtooth(x[:40] / (2+1))
            if np.random.choice([0, 1]) == 0:
                ts[2, :40] = signal.square(np.random.choice([-1, 1]) * x[:40] / 3)
            else:
                ts[2, :40] = signal.sawtooth(np.random.choice([-1, 1]) * x[:40] / 3)
            for i in range(3):
                if randomize_right_side:
                    ts[i, 40:] = np.sin(x[40:] / (np.random.randint(0, 4)+i+1)) / 3
                else:
                    ts[i, 40:] = np.sin(x[40:] / (i+1)) / 3
                ts[i, :] += np.random.normal(0, feature_noise_power, 100)
            self.data.append((ts.astype('float32'), label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[int(idx[0])][0])
    
    def get_label(self, idx):
        return self.data[idx][1]

class ArtificialProtosQueries(DPLDataset):
    def __init__(self, dataset: ArtificialProtosDataset, phase: str):
        self.phase = phase
        self.dataset = dataset
        self.dataset_len = len(dataset)
        self.num_classes = 2 # len(set([dataset.get_label(i) for i in range(self.dataset_len)]))

    # def to_query(self, i: int) -> Query:
    #     ds_entry = i
    #     correct_cls = self.dataset.get_label(ds_entry)

    #     ts_term = Term(f'ts{i}')
    #     q = Query(
    #         Term(
    #             'is_class',
    #             ts_term,
    #             Term(f'c{correct_cls}')
    #         ),
    #         {
    #             ts_term: Term(
    #                 "tensor",
    #                 Term(
    #                     self.phase,
    #                     Constant(ds_entry),
    #                 ),
    #             )
    #         }
    #     )
    #     return q

    # def __len__(self):
    #     return self.dataset_len

    def to_query(self, i: int) -> Query:
        
        ds_entry = i // self.num_classes
        cls_num = i % self.num_classes
        correct_cls = self.dataset.get_label(ds_entry)

        ts_term = Term(f'ts{i}')
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

print('Preparing ProtoTSNet...')
autoencoder = RegularConvEncoder(num_features=3, latent_features=latent_features, padding='same')
encoder = autoencoder.encoder
net = ProtoTSNet(
    cnn_base=encoder,
    for_deepproblog=True,
    num_features=3,
    ts_sample_len=100,
    prototype_shape=(protos_per_class*2, latent_features, 20),
    num_classes=2,
    prototype_activation_function='log'
)

dpl_net = Network(net, "ptsnet", batching=False)
dpl_net.optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

print('Loading logic file...')
model = Model("proto_logic.pl", [dpl_net])
model.set_engine(ExactEngine(model))
# model.set_engine(ApproximateEngine(model, 1, ApproximateEngine.geometric_mean, exploration=False))

train_dataset = ArtificialProtosDataset(100)
model.add_tensor_source("train", train_dataset)
train_queries = ArtificialProtosQueries(train_dataset, "train")
train_loader = DataLoader(train_queries, 1, False)

test_dataset = ArtificialProtosDataset(50)
model.add_tensor_source("test", test_dataset)
test_queries = ArtificialProtosQueries(test_dataset, "test")

print("Training...")
train = train_model(model, train_loader, 20, log_iter=100, profile=0)
# model.save_state('./snapshots/initial_model.pth')
# train.logger.comment(json.dumps(model.get_hyperparameters()))
# train.logger.comment(
    # "Accuracy {}".format(get_confusion_matrix(model, test_queries, verbose=1).accuracy())
# )
get_fact_accuracy(model, test_queries, verbose=1)
train.logger.write_to_file("log/initial.log")


