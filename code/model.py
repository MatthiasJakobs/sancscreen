import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import copy

from sklearn.metrics import roc_auc_score, roc_curve

class SancScreenNet(nn.Module):
    def __init__(self, config):
        super(SancScreenNet, self).__init__()

        self.c = config

        layers = [
            nn.Linear(self.c.input_size, self.c.hidden_size),
            nn.ReLU()
        ]

        for _ in range(self.c.depth):
            layers.append(nn.Linear(self.c.hidden_size, self.c.hidden_size))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(self.c.hidden_size, 1))

        self.model = nn.Sequential(*layers)
        self.threshold = 0.5

        self.sigmoid = nn.Sigmoid()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.c.lr)

        self.criterion = nn.BCELoss()


    def forward(self, x):
        pred = self.model(x)
        return self.sigmoid(pred)

    def logits(self, x):
        return self.model(x)

    def predict_proba(self, x, requires_grad=False):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        is_training = self.training
        self.eval()
        with torch.set_grad_enabled(requires_grad):
            output = self.forward(x)

        if is_training:
            self.train()

        return output

    def predict_proba_numpy(self, x):
        return self.predict_proba(x, requires_grad=False).numpy()

    def predict(self, x):
        pred = self.predict_proba(x)
        pred = (pred >= self.threshold).byte()
        return pred

    def fit(self, train_loader, val_loader, verbose=True, early_stopping_lag=100):

        consecutive_epochs_not_better = 0
        last_val_auroc = 0
        val_auroc = 0

        best_model = copy.deepcopy(self.model)
        for e in range(self.c.max_epochs):
            self.model.train()
            epoch_loss = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                self.optimizer.zero_grad()
                output = self.forward(data)
                loss = self.criterion(output, target)

                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            if val_loader is None:
                if verbose:
                    print(f"{e+1} - train loss {epoch_loss*100:.2f}")
                continue

            _, _, val_auroc = self.test(val_loader)
            if verbose:
                print(f"{e+1} - train loss {epoch_loss*100:.2f} val_acc {val_auroc*100:.2f}")

            if val_auroc <= last_val_auroc:
                consecutive_epochs_not_better += 1
            else:
                consecutive_epochs_not_better = 0
                last_val_auroc = val_auroc
                best_model = copy.deepcopy(self.model)

            if consecutive_epochs_not_better >= early_stopping_lag:
                if verbose:
                    print(f"Early stopping after {e+1} epochs")
                self.model = best_model
                break

        return 0, val_auroc

    def test(self, test_loader):
        self.model.eval()

        with torch.no_grad():
            all_targets = []
            all_predictions = []
            for (data, target) in test_loader:
                output = self.predict_proba(data).squeeze().numpy()
                target = target.squeeze().numpy()
                all_targets.append(target)
                all_predictions.append(output)

        return 0, 0, roc_auc_score(np.concatenate(all_targets), np.concatenate(all_predictions))