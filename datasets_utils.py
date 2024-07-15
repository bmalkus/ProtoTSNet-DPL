from sklearn.preprocessing import LabelEncoder
from scipy.io.arff import loadarff
from typing import Union, List, Dict
from dataclasses import dataclass

from sklearn.model_selection import train_test_split

import numpy as np

import torch


class TSCDataset():
    X: np.ndarray
    y: np.ndarray

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    # def __getitem__(self, idx):
    #     return self.X[idx], self.y[idx]

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return torch.tensor(self.X[idx]).cuda()
        return torch.tensor(self.X[int(idx[0])]).cuda()
    
    def get_label(self, idx):
        return self.y[idx]


@dataclass
class TrainTestDS:
    name: str
    train: TSCDataset
    test: TSCDataset
    val: TSCDataset = None


def transform_ts_data(X, scaler):
    for i in range(X.shape[1]):
        X[:, i, :] = scaler.fit_transform(X[:, i, :])
    return X

def load_arff_dataset(datasets_path, dataset_name, train_size=None, scaler=None):
    train_file = datasets_path / dataset_name / f'{dataset_name}_TRAIN.arff'
    test_file = datasets_path / dataset_name / f'{dataset_name}_TEST.arff'
    label_encoder = LabelEncoder()

    def arff_to_numpy(file_path):
        data, meta = loadarff(file_path)
        X, y = [], []
        for row in data:
            x, label = row
            X.append(np.array(x.tolist(), dtype='float32'))
            y.append(label)
        return np.nan_to_num(np.stack(X, axis=0), 0), np.stack(y, axis=0)
    
    if train_size is None:
        trainX, trainy = arff_to_numpy(train_file)
        testX, testy = arff_to_numpy(test_file)
        trainy = label_encoder.fit_transform(trainy)
        testy = label_encoder.transform(testy)
    else:
        X1, y1 = arff_to_numpy(train_file)
        X2, y2 = arff_to_numpy(test_file)
        X = np.concatenate([X1, X2])
        y = np.concatenate([y1, y2])
        y = label_encoder.fit_transform(y)
        trainX, testX, trainy, testy = train_test_split(X, y, train_size=train_size)

    if scaler:
        transform_ts_data(trainX, scaler)
        transform_ts_data(testX, scaler)

    return TrainTestDS(dataset_name, TSCDataset(trainX, trainy), TSCDataset(testX, testy))


def ds_load(datasets_path, ds_or_lst, train_size=None, scaler=None) -> Union[TrainTestDS, Dict[str, TrainTestDS]]:
    # scaler is assumed to reset its state when calling fit_transform/fit
    def load_ds(ds):
        return (ds, load_arff_dataset(datasets_path, ds, train_size, scaler))

    ds_names = ds_or_lst
    if not isinstance(ds_names, (list, tuple)):
        ds_names = [ds_names]

    # with _tqdm_joblib(tqdm(desc="Loading datasets", total=len(ds_names))) as progress_bar:
    #     results = Parallel(n_jobs=32)(delayed(load_ds)(ds) for ds in ds_names)
    results = [load_ds(ds) for ds in ds_names]

    results = dict(results)

    if not isinstance(ds_or_lst, (list, tuple)):
        return results[ds_or_lst]
    else:
        return results
