import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np

from datasets import load_sancscreen
from parameters import sancscreen_config
from preprocessing import evaluate
from model import SancScreenNet
from sklearn.metrics import roc_auc_score

def train_balanced():
    np.random.seed(0)
    torch.manual_seed(0)

    c = sancscreen_config
    d = load_sancscreen()

    # Original train and test set
    train_dataset = torch.utils.data.TensorDataset(d.c_train_x, d.c_train_y.reshape(-1, 1).float())
    test_dataset = torch.utils.data.TensorDataset(d.c_test_x, d.c_test_y.reshape(-1, 1).float())

    train_length = int(0.8 * len(train_dataset))
    val_length = len(train_dataset) - train_length

    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_length, val_length])

    dataloaders = {
        'train': torch.utils.data.DataLoader(train_dataset, batch_size=c.batch_size, shuffle=True),
        'val': torch.utils.data.DataLoader(val_dataset, batch_size=c.batch_size, shuffle=True),
        'test': torch.utils.data.DataLoader(test_dataset, batch_size=c.batch_size, shuffle=False),
    }

    net = SancScreenNet(c)
    net.fit(dataloaders['train'], dataloaders['val'], verbose=True, early_stopping_lag=50)
    evaluate(net, d.c_test_x.numpy(), d.c_test_y.numpy(), verbose=True)

    torch.save(net.state_dict(), c.out_path)

    rocs_train = []
    rocs_test = []
    for _ in range(10):
        net = SancScreenNet(c)
        net.fit(dataloaders['train'], dataloaders['val'], verbose=False, early_stopping_lag=50)
        rocs_train.append(roc_auc_score(d.c_train_y.numpy(), net.predict_proba_numpy(d.c_train_x.numpy())))
        rocs_test.append(roc_auc_score(d.c_test_y.numpy(), net.predict_proba_numpy(d.c_test_x.numpy())))

    print(f"ROC Train {np.mean(rocs_train)} +- {np.std(rocs_train)}")
    print(f"ROC Test {np.mean(rocs_test)} +- {np.std(rocs_test)}")

if __name__ == "__main__":
    train_balanced()
