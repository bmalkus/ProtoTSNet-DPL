from typing import Tuple

import numpy as np
import torch
from scipy.io.arff import loadarff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class TSCDataset():
    X: np.ndarray
    y: np.ndarray

    def __init__(self, X, y, for_torch_dl=False):
        self.X = X
        self.y = y
        self.for_torch_dl = for_torch_dl

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.for_torch_dl:
            return self.X[idx], self.y[idx]
        if isinstance(idx, int):
            return torch.tensor(self.X[idx])#.cuda()
        return torch.tensor(self.X[int(idx[0])])#.cuda()
    
    def get_label(self, idx):
        return self.y[idx]
    
    def get_for_torch_dl(self):
        return TSCDataset(self.X, self.y, for_torch_dl=True)


def transform_ts_data(X, scaler):
    for i in range(X.shape[1]):
        X[:, i, :] = scaler.fit_transform(X[:, i, :])
    return X

def ds_load(datasets_path, dataset_name, train_size=None, scaler=None) -> Tuple[TSCDataset, TSCDataset]:
    train_file = datasets_path / dataset_name / f'{dataset_name}_TRAIN.arff'
    test_file = datasets_path / dataset_name / f'{dataset_name}_TEST.arff'
    label_encoder = LabelEncoder()

    def arff_to_numpy(file_path):
        data, _ = loadarff(file_path)
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

    return TSCDataset(trainX, trainy), TSCDataset(testX, testy)
