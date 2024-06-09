import numpy as np
from scipy import signal
import torch
import random

class ArtificialProtosDatasetRandomShift():
    def __init__(self, N, num_feat=5, classes=3, feature_noise_power=0.1, class_specifics=None):
        self.data = []
        x = np.linspace(0, 100, 100)
        if class_specifics is None:
            class_specifics = []
            for _ in range(classes):
                features = random.sample(range(0, num_feat), np.random.randint(1, min(num_feat, 3)))
                patterns = random.choices([signal.sawtooth, signal.square], k=len(features))
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
                xs = x[0 if patter_start >= 0 else -patter_start:pattern_length if pattern_end <= 100 else 100 - patter_start]
                ts[i, max(patter_start, 0):min(patter_start + pattern_length, 100)] = pattern(xs / 2)
            self.data.append((ts.astype('float32'), label))
            
            

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
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
        return torch.tensor(self.data[int(idx[0])][0])
    
    def get_label(self, idx):
        return self.data[idx][1]
