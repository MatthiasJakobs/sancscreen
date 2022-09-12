import numpy as np
import torch

from os.path import join
from utils import objectview

def load_sancscreen(root_path=None):
    if root_path is None:
        root_path = "data/"
    else:
        root_path = join(root_path, "data/")

    ds_name = "sancscreen"

    c_train_x = np.load(f"data/{ds_name}_c_train_x.npy")
    c_test_x = np.load(f"data/{ds_name}_c_test_x.npy")
    c_train_y = np.load(f"data/{ds_name}_c_train_y.npy")
    c_test_y = np.load(f"data/{ds_name}_c_test_y.npy")
    e_x = np.load(f"data/{ds_name}_e_x.npy")
    e_y = np.load(f"data/{ds_name}_e_y.npy")
    annot = np.load(f"data/{ds_name}_annot.npy")
    with open("data/feature_names.txt", "r") as f:
        feature_names = f.readline().split(",")

    numeric_features = np.array([0, 1, 16, 17, 18])
    binary_features = np.array(list(set(np.arange(19)) - set(numeric_features)))

    return objectview({ 
            "c_train_x": torch.from_numpy(c_train_x).float(), 
            "c_train_y": torch.from_numpy(c_train_y),
            "c_test_x":  torch.from_numpy(c_test_x).float(),
            "c_test_y":  torch.from_numpy(c_test_y),
            "e_x": torch.from_numpy(e_x).float(),
            "e_y": torch.from_numpy(e_y),
            "annot": torch.from_numpy(annot).float(),
            "feature_names": feature_names,
            "numeric_features": numeric_features,
            "binary_features": binary_features,
            })
