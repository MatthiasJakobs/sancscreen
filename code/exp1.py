# Check model performance

import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from datasets import load_sancscreen
from model import SancScreenNet
from preprocessing import evaluate
from parameters import sancscreen_config

if __name__ == "__main__":
    # Load pretrained model, dataset and configs
    d = load_sancscreen()
    c = sancscreen_config
    rng = np.random.RandomState(123)

    X_train = d.c_train_x.numpy()
    y_train = d.c_train_y.numpy()
    X_test = d.c_test_x.numpy()
    y_test = d.c_test_y.numpy()

    print("NEURAL NETWORK")
    print("-"*30)
    model = SancScreenNet(c)
    model.load_state_dict(torch.load("results/checkpoints/sancscreen_net.pth"))
    evaluate(model, d.c_test_x, d.c_test_y, verbose=True)

    print(" ")

    print("-"*30)
    print("RANDOM FOREST")
    model = RandomForestClassifier(n_estimators=300, random_state=123)
    model.fit(X_train, y_train)
    evaluate(model, d.c_test_x.numpy(), d.c_test_y.numpy(), verbose=True)

    rocs_train = []
    rocs_test = []
    for _ in range(10):
        model = RandomForestClassifier(n_estimators=300)
        model.fit(X_train, y_train)
        rocs_train.append(roc_auc_score(d.c_train_y.numpy(), model.predict_proba(d.c_train_x.numpy())[:, 1]))
        rocs_test.append(roc_auc_score(d.c_test_y.numpy(), model.predict_proba(d.c_test_x.numpy())[:, 1]))

    print(f"ROC Train {np.mean(rocs_train)} +- {np.std(rocs_train)}")
    print(f"ROC Test {np.mean(rocs_test)} +- {np.std(rocs_test)}")