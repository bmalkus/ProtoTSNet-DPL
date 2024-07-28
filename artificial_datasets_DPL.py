import numpy as np
from scipy import signal
import torch
import random
import itertools

from deepproblog.query import Query
from deepproblog.dataset import Dataset as DPLDataset
from problog.logic import Term, Constant, And, Not

class ArtificialProtosDatasetRandomShift():
    def __init__(self, N, num_feat=5, classes=3, feature_noise_power=0.1, class_specifics=None):
        self.data = []
        x = np.linspace(0, 100, 100)
        if class_specifics is None:
            class_specifics = []
            for i in range(classes):
                features = random.sample(range(0, num_feat), np.random.randint(1, min(num_feat, 3)))
                features.sort()
                # patterns = random.choices([signal.sawtooth, signal.square], k=len(features))
                signals = [signal.sawtooth, signal.square] if i % 2 == 0 else [signal.square, signal.sawtooth]
                patterns = list(itertools.islice(itertools.cycle(signals), len(features)))
                spec = {
                    'features': features,
                    'patterns': patterns
                }
                while spec in class_specifics:
                    features = random.sample(range(0, num_feat), np.random.randint(1, min(num_feat, 3)))
                    patterns = random.choices([signal.sawtooth, signal.square], k=len(features))
                    spec = {
                        'features': features,
                        'patterns': patterns
                    }
                class_specifics.append(spec)

        self.class_specifics = class_specifics

        for _ in range(N):
            label = np.random.randint(0, classes)
            ts = np.zeros((num_feat, 100))

            for i in range(num_feat):
                ts[i, :] = np.random.normal(0, feature_noise_power, 100)

            class_spec = class_specifics[label]
            patter_start = np.random.randint(0, 70)
            for i, pattern in zip(class_spec['features'], class_spec['patterns']):
                pattern_length = 30
                pattern_end = patter_start + pattern_length
                xs = x[(0 if patter_start >= 0 else -patter_start):(pattern_length if pattern_end <= 100 else 100 - patter_start)]
                ts[i, max(patter_start, 0):min(patter_start + pattern_length, 100)] = pattern(xs / 2)
            self.data.append((ts.astype('float32'), label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return torch.tensor(self.data[idx][0])
        return torch.tensor(self.data[int(idx[0])][0])
    
    def get_label(self, idx):
        return self.data[idx][1]


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
        if isinstance(idx, int):
            return torch.tensor(self.data[idx][0])
        return torch.tensor(self.data[int(idx[0])][0])
    
    def get_label(self, idx):
        return self.data[idx][1]


class Queries(DPLDataset):
    def __init__(self, dataset: ArtificialProtosDataset, phase: str):
        self.phase = phase
        self.dataset = dataset
        self.dataset_len = len(dataset)
        self.num_classes = len(set([dataset.get_label(i) for i in range(self.dataset_len)]))
        print(f'Detected number of classes in Queries: {self.num_classes}')

    def to_query(self, i: int) -> Query:
        ds_entry = i
        cls_num = self.dataset.get_label(ds_entry)

        ts_term = Term(f'ts{ds_entry}')
        q = Query(
            Term(
                'excl_is_class',
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
            }
        )
        return q, self.dataset[ds_entry], cls_num

    def __len__(self):
        return self.dataset_len


class QueriesWithNegatives(DPLDataset):
    def __init__(self, dataset: ArtificialProtosDataset, phase: str):
        self.phase = phase
        self.dataset = dataset
        self.dataset_len = len(dataset)
        self.num_classes = len(set([dataset.get_label(i) for i in range(self.dataset_len)]))
        print(f'Detected number of classes in QueriesWithNegatives: {self.num_classes}')

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
        return q, self.dataset[ds_entry], correct_cls

    def __len__(self):
        return self.dataset_len * self.num_classes


# class QueriesWithNegatives2(DPLDataset):
#     def __init__(self, dataset: ArtificialProtosDataset, phase: str):
#         self.phase = phase
#         self.dataset = dataset
#         self.dataset_len = len(dataset)
#         self.num_classes = len(set([dataset.get_label(i) for i in range(self.dataset_len)]))
#         print(f'Detected number of classes in QueriesWithNegatives: {self.num_classes}')

#     def to_query(self, i: int) -> Query:
#         ds_entry = i
#         correct_cls = self.dataset.get_label(ds_entry)

#         t = Term(
#                 'is_class',
#                 Term(
#                     'tensor',
#                     Term(
#                         self.phase,
#                         Constant(ds_entry),
#                     ),
#                 ),
#                 Term(f'c0'),
#             )

#         if correct_cls != 0:
#             t = Not("not", t)

#         for cls_num in range(1, self.num_classes):
#             new_t = Term(
#                 'is_class',
#                 Term(
#                     'tensor',
#                     Term(
#                         self.phase,
#                         Constant(ds_entry),
#                     ),
#                 ),
#                 Term(f'c{cls_num}'),
#             )
#             if cls_num != correct_cls:
#                 t = And(t, Not("not", new_t))
#             else:
#                 t = And(t, new_t)

#         q = Query(
#             t,
#             {
#                 Term(f'ts{ds_entry}'): Term(
#                     "tensor",
#                     Term(
#                         self.phase,
#                         Constant(ds_entry),
#                     ),
#                 )
#             },
#         )
#         print(q)
#         return q, self.dataset[ds_entry]

#     def __len__(self):
#         return self.dataset_len
