from utils import objectview
import pandas as pd

sancscreen_config = objectview({
    "batch_size": 100,
    "lr": 3e-4,
    "out_path": "results/checkpoints/sancscreen_net.pth",
    "report_every": 100,
    "hidden_size": 50,
    "dropout": None,
    "max_epochs": 1000,
    "depth": 2,
    "input_size": 19,
})