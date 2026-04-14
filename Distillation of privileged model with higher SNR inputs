import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, balanced_accuracy_score, f1_score
from torchmetrics.classification import BinaryF1Score

from pnpl.datasets import LibriBrainSpeech

from collections import defaultdict
import random

# Set up dataloader

class NeuralDataset(Dataset):
    def __init__(self, X, y, n_averaging = 1, strategy = "random", seed = 42):

        self.X = X
        self.y = y
        self.n_averaging = max(1, n_averaging)
        self.strategy = strategy
        self.rng = np.random.default_rng(seed)

        self.indices_by_class = defaultdict(list)
        for i, cls in enumerate(y):
            self.indices_by_class[int(cls)].append(i)

    def __len__(self):
        return len(self.y)
    
    def _average_trials(self, idx):
        cls = int(self.y[idx])
        available_indices = self.indices_by_class[cls]

        if self.strategy == "random":
            chosen = self.rng.choice(available_indices, size = self.n_averaging, replace = False)
        elif self.strategy == "sequential":
            pos = available_indices.index(idx)
            start = pos
            end = min(len(available_indices), start + self.n_averaging)
            chosen = available_indices[start:end]

            if len(chosen) < len(self.n_averaging):
                extra = self.rng.choice(available_indices, size = len(self.n_averaging) - len(chosen), replace = False)
                np.concatenate([chosen, extra])

        else:
            raise ValueError(f"Unknown averaging strategy: {self.strategy}")
    
        averaged = np.mean(self.X[chosen], axis = 0)
        return averaged, cls

    def __getitem__(self, idx):
        if self.n_averaging == 1:
            x = self.X[idx]
            y = self.y[idx]
        else:
            x, y = self._average_trials(idx)
        return torch.tensor(x, dtype = torch.float32), torch.tensor(y, dtype = torch.long)

# Set up metrics

def compute_metrics(y_true, y_pred, task = "phoneme"):
    if task == "speech":
        metric = balanced_accuracy_score(y_true, y_pred)
    elif task == "phoneme":
        metric = f1_score(y_true, y_pred, average = "macro")
    else:
        raise ValueError(f"Unknown task: {task}")
    return metric